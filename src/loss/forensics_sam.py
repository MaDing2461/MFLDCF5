"""
ForensicsSAM损失函数
参考MVSS的BCELoss + SoftDiceLoss组合方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np


class SoftDiceLoss(_Loss):
    """
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    用于掩码预测的像素级损失
    """
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "predict和target的大小必须相等"
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        return dice_loss


class ForgeryLocalizationLoss(nn.Module):
    """
    伪造定位损失 - 兼容MVSS的损失函数设计
    组合BCELoss (20%)和SoftDiceLoss (80%)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.soft_dice = SoftDiceLoss()

    def forward(self, mask_pred, mask_gt):
        """
        Args:
            mask_pred: (B, 1, H, W) 或 (B, H, W) SAM模型预测的掩码
            mask_gt: (B, H, W) 或 (B, 1, H, W) 真实掩码标签
            
        Returns:
            loss: 标量损失值
        """
        # 标准化输入形状
        if mask_pred.dim() == 3:
            mask_pred = mask_pred.unsqueeze(1)
        if mask_gt.dim() == 4:
            mask_gt = mask_gt.squeeze(1)
        
        # 调整 mask_gt 大小以匹配 mask_pred（在应用sigmoid之前）
        if mask_gt.shape[1:] != mask_pred.shape[1:]:
            mask_gt = torch.nn.functional.interpolate(
                mask_gt.unsqueeze(1).float(),
                size=mask_pred.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # 确保预测值在[0, 1]范围内
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        
        # 将掩码展平
        mask_pred_flat = mask_pred_sigmoid.view(mask_pred_sigmoid.size(0), -1)
        mask_gt_flat = mask_gt.view(mask_gt.size(0), -1)
        
        # 计算BCE + SoftDice
        bce_loss = self.bce(mask_pred_flat, mask_gt_flat.float())
        dice_loss = self.soft_dice(mask_pred_flat, mask_gt_flat.float())
        
        # 加权组合（20% BCE + 80% Dice）
        loss = 0.2 * bce_loss + 0.8 * dice_loss
        return loss


class ForensicsDetectionLoss(nn.Module):
    """
    伪造检测损失 - 图像级二分类
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cls_pred, cls_gt):
        """
        Args:
            cls_pred: (B, 2) 分类预测（真实/伪造）
            cls_gt: (B,) 二值标签 (0=真实, 1=伪造)
            
        Returns:
            loss: 标量损失值
        """
        return self.ce(cls_pred, cls_gt.long())


class ForensicsSAMLoss(nn.Module):
    """
    完整的ForensicsSAM损失函数
    组合掩码定位损失和分类检测损失
    """
    def __init__(self, alpha=0.5, beta=0.5, with_detection=True):
        """
        Args:
            alpha: 掩码定位损失权重
            beta: 分类检测损失权重
            with_detection: 是否包含检测损失
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_detection = with_detection
        
        self.localization_loss = ForgeryLocalizationLoss()
        if with_detection:
            self.detection_loss = ForensicsDetectionLoss()

    def forward(self, mask_pred, mask_gt, cls_pred=None, cls_gt=None):
        """
        计算总损失
        
        Args:
            mask_pred: (B, 1, H, W) 掩码预测
            mask_gt: (B, H, W) 掩码真实标签
            cls_pred: (B, 2) 分类预测（可选）
            cls_gt: (B,) 分类真实标签（可选）
            
        Returns:
            loss: 标量总损失
            loss_dict: 包含各项损失的字典
        """
        loc_loss = self.localization_loss(mask_pred, mask_gt)
        
        loss_dict = {'localization': loc_loss.item()}
        total_loss = self.alpha * loc_loss
        
        if self.with_detection and cls_pred is not None and cls_gt is not None:
            det_loss = self.detection_loss(cls_pred, cls_gt)
            loss_dict['detection'] = det_loss.item()
            total_loss = total_loss + self.beta * det_loss
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# 兼容现有代码的包装类
class Loss_forensics_sam(nn.Module):
    """兼容现有Loss_fake的包装类"""
    def __init__(self):
        super().__init__()
        self.loss_fn = ForensicsSAMLoss(alpha=0.5, beta=0.5, with_detection=True)

    def loss_calc(self, out, label, out_label, model='ForensicsSAM'):
        """
        Args:
            out: tuple (mask_pred, cls_pred) 或 mask_pred
            label: 掩码真实标签 (B, H, W)
            out_label: 分类真实标签 (B,)
            model: 模型名称
            
        Returns:
            loss: 标量损失值
        """
        if isinstance(out, tuple):
            mask_pred, cls_pred = out
        else:
            mask_pred = out
            cls_pred = None

        if cls_pred is not None and out_label is not None:
            loss, _ = self.loss_fn(mask_pred, label, cls_pred, out_label)
        else:
            loss, _ = self.loss_fn(mask_pred, label)
        
        return loss

    def test_handle(self, out, model='ForensicsSAM'):
        """
        处理测试输出，返回掩码预测和分类预测
        
        Args:
            out: tuple (mask_pred, cls_pred) 或 mask_pred
            model: 模型名称
            
        Returns:
            pred: 掩码预测 (H, W) 或 None
            cls_pred: 分类预测或 None
        """
        if isinstance(out, tuple):
            mask_pred, cls_pred = out
        else:
            mask_pred = out
            cls_pred = None
        
        # 处理掩码预测
        if mask_pred is not None:
            # 转换为numpy并处理
            pred = mask_pred.cpu().data[0].numpy() if mask_pred.dim() == 4 else mask_pred.cpu().numpy()[0]
            pred[np.where(pred < 0.5)] = int(0)
            pred[np.where(pred >= 0.5)] = int(1)
            pred = np.asarray(pred, dtype=int)
        else:
            pred = None
        
        # 处理分类预测
        if cls_pred is not None:
            _, cls_predicted = torch.max(cls_pred.data, 1)
            return pred, cls_predicted
        else:
            return pred, None

