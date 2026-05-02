"""
ForensicsSAM: Forensic-focused Segment Anything Model
参考：https://github.com/siriusPRX/ForensicsSAM
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Optional


class ForgeryDetector(nn.Module):
    """图像级伪造检测器"""
    def __init__(self, input_channels=256, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 自适应处理不同维度的输入
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # 二分类：真实/伪造
        )
        self.input_channels = input_channels

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W) SAM编码器输出的特征图
        returns: logits (B, 2)
        """
        # 自适应处理：如果不是4D，则reshape
        if feat_map.dim() == 2:
            # 如果已经是扁平的，直接通过分类器
            logits = self.classifier(feat_map)
        else:
            x = self.pool(feat_map)
            logits = self.classifier(x)
        return logits


class LoRA_QKV(nn.Module):
    """共享伪造专家 - LoRA QKV层"""
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = True
        self.ori_qkv = ori_qkv
        self.dim = dim
        
        # LoRA层
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        
        # 权重初始化
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        if self.activate:
            qkv[:, :, :, :self.dim] += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim] += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:] += self.b_v(self.a_v(x))
        return qkv


class LoRA_FFN(nn.Module):
    """共享伪造专家 - LoRA FFN层"""
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = True
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out


class Adv_LoRA_QKV(nn.Module):
    """自适应对抗专家 - LoRA QKV层"""
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = False
        self.ori_qkv = ori_qkv
        self.dim = dim
        
        # LoRA层
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        
        # 权重初始化
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        if self.activate:
            qkv[:, :, :, :self.dim] += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim] += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:] += self.b_v(self.a_v(x))
        return qkv


class Adv_LoRA_FFN(nn.Module):
    """自适应对抗专家 - LoRA FFN层"""
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = False
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out


class ForensicsSAM(nn.Module):
    """
    ForensicsSAM: 集成伪造检测和定位的SAM模型
    
    包含三个核心组件：
    1. 共享伪造专家 - 补偿冻结编码器的伪造知识
    2. 伪造检测器 - 图像级伪造分类
    3. 自适应对抗专家 - 抵抗对抗攻击
    """
    def __init__(self, sam_model, r=8, lora_layer=None, with_detector=True):
        super().__init__()
        assert r > 0
        self.lora_layer = lora_layer or list(range(len(sam_model.image_encoder.blocks)))
        self.with_detector = with_detector

        # 冻结SAM图像编码器
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # 确定全局注意力层索引
        blk_num = len(sam_model.image_encoder.blocks)
        if blk_num == 32:  # ViT-H
            self.global_attn_index = [7, 15, 23, 31]
        elif blk_num == 24:  # ViT-L
            self.global_attn_index = [5, 11, 17, 23]
        elif blk_num == 12:  # ViT-B
            self.global_attn_index = [2, 5, 8, 11]
        else:
            self.global_attn_index = []

        # ============= 1. 共享伪造专家 =============
        dim = None
        in_features = None
        out_features = None
        
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            
            w_qkv_linear = blk.attn.qkv
            if dim is None:
                dim = w_qkv_linear.in_features
            blk.attn.qkv = LoRA_QKV(w_qkv_linear, dim, r)

            w_lin1 = blk.mlp.lin1
            if in_features is None:
                in_features = w_lin1.in_features
                out_features = w_lin1.out_features
            blk.mlp.lin1 = LoRA_FFN(w_lin1, in_features, out_features, r)

            # 冻结共享专家的LoRA参数
            for param in blk.attn.qkv.parameters():
                param.requires_grad = False
            for param in blk.mlp.lin1.parameters():
                param.requires_grad = False

        # 冻结prompt_encoder和mask_decoder
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        # ============= 2. 伪造检测器 =============
        # SAM image_encoder 输出的通道数总是 256（由 neck 投影）
        detector_input_channels = 256
        self.detector = ForgeryDetector(input_channels=detector_input_channels) if with_detector else None
        
        # 注意：不冻结检测器，它需要梯度进行训练

        self.sam = sam_model

        # ============= 3. 自适应对抗专家 =============
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            w_qkv_linear = blk.attn.qkv
            blk.attn.qkv = Adv_LoRA_QKV(w_qkv_linear, dim, 8*r)

            w_lin1 = blk.mlp.lin1
            blk.mlp.lin1 = Adv_LoRA_FFN(w_lin1, in_features, out_features, 8*r)

    def activate_adv(self, activate):
        """激活/禁用对抗专家"""
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            blk.attn.qkv.set_activate(activate)
            blk.mlp.lin1.set_activate(activate)

    def forward(self, images, activate_adv=False):
        """
        前向传播
        
        Args:
            images: (B, 3, H, W) 输入图像
            activate_adv: bool 是否激活对抗专家
            
        Returns:
            mask_prediction: (B, 1, H//4, W//4) 掩码预测
            cls_prediction: (B, 2) 分类预测 或 None
        """
        self.activate_adv(activate_adv)
        
        # 保存原始尺寸用于后续恢复
        original_h, original_w = images.shape[2:]
        
        # SAM要求输入为1024x1024，如果不是则调整
        if original_h != 1024 or original_w != 1024:
            images = torch.nn.functional.interpolate(
                images, 
                size=(1024, 1024), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 图像编码 - 使用梯度检查点减少内存占用
        if self.training:
            image_embeddings = torch.utils.checkpoint.checkpoint(
                self.sam.image_encoder, 
                images, 
                use_reentrant=False
            )
        else:
            image_embeddings = self.sam.image_encoder(images)
        
        # 伪造检测
        cls_prediction = None
        if self.detector is not None:
            cls_prediction = self.detector(image_embeddings)
        
        # 掩码解码
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        device = image_embeddings.device
        mask_prediction, iou_prediction = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe().to(device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # 恢复到原始尺寸
        if original_h != 1024 or original_w != 1024:
            mask_prediction = torch.nn.functional.interpolate(
                mask_prediction,
                size=(original_h // 4, original_w // 4),
                mode='bilinear',
                align_corners=False
            )
        
        return mask_prediction, cls_prediction

    # ============= 参数加载/保存 =============
    def load_all_parameters(self, path):
        """一次性加载所有可训练参数"""
        all_params = torch.load(path, map_location='cpu')
        if "sam" in all_params:
            self._load_state_dict(self.sam, all_params["sam"])
        if "detector" in all_params and self.detector is not None:
            self._load_state_dict(self.detector, all_params["detector"])

    def save_all_parameters(self, path):
        """一次性保存所有可训练参数"""
        all_params = {}
        all_params["sam"] = {
            k: v for k, v in self._get_named_params(self.sam) if v.requires_grad
        }
        if self.detector is not None:
            all_params["detector"] = {
                k: v for k, v in self._get_named_params(self.detector) if v.requires_grad
            }
        torch.save(all_params, path)

    def _get_named_params(self, module):
        if isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return module.module.named_parameters()
        return module.named_parameters()

    def _load_state_dict(self, module, state_dict):
        if isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module.module.load_state_dict(state_dict, strict=False)
        else:
            module.load_state_dict(state_dict, strict=False)
