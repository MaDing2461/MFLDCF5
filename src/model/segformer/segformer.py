import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleSegFormer(nn.Module):
    """
    轻量 SegFormer 分割网络，与本项目 FLDCF_multiModal_TransUNet 兼容。
    返回格式：(seg_pred, cls_pred)
    - seg_pred: (B, 2, H, W) 分割logits（二分类）
    - cls_pred: (B, 2) 分类logits（图像级真伪分类）
    """

    def __init__(self, in_ch=3, num_classes=1, base_ch=32):
        super().__init__()
        # Encoder: downsampling path
        self.enc1 = ConvBlock(in_ch, base_ch)                  # 256 x 256, ch=32
        self.enc2 = ConvBlock(base_ch, base_ch * 2)            # 256 x 256, ch=64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)        # 256 x 256, ch=128
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)        # 256 x 256, ch=256

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder: upsampling path with skip connections
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 4 + base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch + base_ch * 2, base_ch)

        # Segmentation head: output 2-class logits
        self.seg_head = nn.Conv2d(base_ch, 2, kernel_size=1)
        
        # Classification head: global average pooling + FC
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(base_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)        # 256 x 256, ch=32
        p1 = self.pool(e1)       # 128 x 128, ch=32

        e2 = self.enc2(p1)       # 128 x 128, ch=64
        p2 = self.pool(e2)       # 64 x 64, ch=64

        e3 = self.enc3(p2)       # 64 x 64, ch=128
        p3 = self.pool(e3)       # 32 x 32, ch=128

        e4 = self.enc4(p3)       # 32 x 32, ch=256
        p4 = self.pool(e4)       # 16 x 16, ch=256

        b = self.bottleneck(p4)  # 16 x 16, ch=256

        # decoder
        u3 = self.up3(b)         # 32 x 32, ch=128
        u3 = torch.cat([u3, e4], dim=1)  # 32 x 32, ch=128+256=384
        d3 = self.dec3(u3)       # 32 x 32, ch=128

        u2 = self.up2(d3)        # 64 x 64, ch=64
        u2 = torch.cat([u2, e3], dim=1)  # 64 x 64, ch=64+128=192
        d2 = self.dec2(u2)       # 64 x 64, ch=64

        u1 = self.up1(d2)        # 128 x 128, ch=32
        u1 = torch.cat([u1, e2], dim=1)  # 128 x 128, ch=32+64=96
        d1 = self.dec1(u1)       # 128 x 128, ch=32

        # Segmentation output: upsample to original size
        seg_out = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        seg_out = self.seg_head(seg_out)  # (B, 2, 256, 256)
        
        # Classification output: global average pooling
        cls_feat = self.cls_pool(d1)  # (B, 32, 1, 1)
        cls_feat = cls_feat.view(cls_feat.size(0), -1)  # (B, 32)
        cls_out = self.cls_head(cls_feat)  # (B, 2)
        
        return seg_out, cls_out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum(dim=(2, 3)) + self.smooth
        den = (probs + target).sum(dim=(2, 3)) + self.smooth
        loss = 1 - (num / den)
        return loss.mean()


class SegFormerLoss(nn.Module):
    """BCEWithLogits + Dice"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # target expected shape [B, 1, H, W]
        bce_loss = self.bce(logits, target.float())
        dice_loss = self.dice(logits, target.float())
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def create_segformer(args=None, num_classes=1):
    model = SimpleSegFormer(in_ch=3, num_classes=num_classes, base_ch=32)
    return model
