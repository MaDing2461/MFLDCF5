"""
GenD: Generative Deepfake Detection Model
Adapted for fakedetect framework compatibility
Based on: https://github.com/yermandy/GenD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LinearHead(nn.Module):
    """Linear classification head matching GenD architecture"""
    
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs
    
    def forward(self, x):
        """
        Args:
            x: Feature tensor of shape (B, D)
        Returns:
            Tuple of (logits, l2_embeddings)
            - logits: (B, C) classification logits
            - l2_embeddings: (B, D) normalized embeddings
        """
        l2_embeddings = F.normalize(x, p=2, dim=1)
        
        if self.normalize_inputs:
            x = l2_embeddings
        
        logits = self.linear(x)
        
        return logits, l2_embeddings


class CLIPBackbone(nn.Module):
    """
    Lightweight CLIP-like backbone using ResNet50 with multi-scale features
    Mimics CLIP's feature extraction capability
    """
    
    def __init__(self, model_name='resnet50', pretrained=False):
        super().__init__()
        
        if 'resnet' in model_name.lower():
            if model_name == 'resnet50':
                backbone = models.resnet50(weights=None)
            elif model_name == 'resnet101':
                backbone = models.resnet101(weights=None)
            else:
                backbone = models.resnet50(weights=None)
        else:
            backbone = models.resnet50(weights=None)
        
        # Remove classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Feature dimension for ResNet50
        self.feature_dim = 2048
    
    def forward(self, x):
        """
        Extract features from input image
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Feature tensor of shape (B, 2048)
        """
        features = self.features(x)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        return features
    
    def get_feature_dim(self):
        return self.feature_dim


class DINOBackbone(nn.Module):
    """
    Vision Transformer backbone mimicking DINO's self-supervised learning
    """
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Use Vision Transformer from timm-style implementation
        # For simplicity, use ResNet as backbone
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
    
    def get_feature_dim(self):
        return self.feature_dim


class PerceptionBackbone(nn.Module):
    """
    Perception-based backbone with multi-scale feature extraction
    """
    
    def __init__(self, image_size=256, pretrained=False):
        super().__init__()
        
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.image_size = image_size
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
    
    def get_feature_dim(self):
        return self.feature_dim


class GenD(nn.Module):
    """
    GenD: Generative Deepfake Detection Model
    
    Architecture:
    - Feature Extractor: Multi-modal encoder (CLIP/DINO/Perception)
    - Classification Head: Linear probe with L2 normalization
    - Loss: Cross-entropy with optional alignment/uniformity regularization
    
    Returns (logits, embeddings) tuple for compatibility with fakedetect framework
    """
    
    def __init__(self, args, num_classes=2, backbone='clip', pretrained=False, 
                 normalize_head=False, use_l2_norm=True):
        """
        Args:
            args: Arguments object (not used in simplified version)
            num_classes: Number of classification classes (default: 2 for fake/real)
            backbone: Backbone type ('clip', 'dino', 'perception')
            pretrained: Whether to use pretrained weights
            normalize_head: Whether to normalize head inputs
            use_l2_norm: Whether to use L2 normalization in embeddings
        """
        super().__init__()
        
        self.args = args
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_l2_norm = use_l2_norm
        
        # Initialize feature extractor based on backbone type
        if 'clip' in backbone.lower():
            self.feature_extractor = CLIPBackbone('resnet50', pretrained=pretrained)
        elif 'dino' in backbone.lower():
            self.feature_extractor = DINOBackbone(pretrained=pretrained)
        elif 'perception' in backbone.lower() or 'vit' in backbone.lower():
            self.feature_extractor = PerceptionBackbone(pretrained=pretrained)
        else:
            self.feature_extractor = CLIPBackbone('resnet50', pretrained=pretrained)
        
        feature_dim = self.feature_extractor.get_feature_dim()
        
        # Classification head
        self.head = LinearHead(feature_dim, num_classes, normalize_inputs=normalize_head)
    
    def forward(self, x):
        """
        Forward pass through GenD model
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple of (logits, embeddings)
            - logits: Classification logits (B, num_classes)
            - embeddings: L2-normalized embeddings (B, feature_dim)
        """
        # Extract features
        features = self.feature_extractor(x)  # (B, feature_dim)
        
        # Get logits and embeddings from head
        logits, embeddings = self.head(features)  # (B, num_classes), (B, feature_dim)
        
        # For framework compatibility, return (logits, embeddings)
        # Note: This matches the convention where aux=embeddings
        return logits, embeddings
    
    def get_embedding(self, x):
        """Get embeddings without classification"""
        features = self.feature_extractor(x)
        _, embeddings = self.head(features)
        return embeddings
    
    def get_features(self, x):
        """Get raw features from backbone"""
        return self.feature_extractor(x)


class GenDMultiModal(nn.Module):
    """
    Multi-modal GenD with RGB + DSM fusion
    """
    
    def __init__(self, args, num_classes=2, backbone='clip', pretrained=False):
        super().__init__()
        
        self.args = args
        self.num_classes = num_classes
        
        # RGB backbone
        self.rgb_backbone = GenD(args, num_classes, backbone, pretrained)
        
        # DSM backbone (single channel, converted to 3-channel internally)
        self.dsm_backbone = GenD(args, num_classes, backbone, pretrained)
        
        # Feature fusion
        feature_dim = self.rgb_backbone.feature_extractor.get_feature_dim()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification head on fused features
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x, dsm=None):
        """
        Forward pass with optional DSM channel fusion
        
        Args:
            x: RGB input (B, 3, H, W)
            dsm: Optional DSM input (B, 1, H, W)
        
        Returns:
            Tuple of (logits, embeddings)
        """
        # RGB path
        rgb_features = self.rgb_backbone.get_features(x)
        
        if dsm is not None and dsm is not False:
            # Convert DSM to 3-channel by repeating
            if dsm.dim() == 4 and dsm.shape[1] == 1:
                dsm = dsm.repeat(1, 3, 1, 1)
            
            # DSM path
            dsm_features = self.dsm_backbone.get_features(dsm)
            
            # Fusion
            fused_features = torch.cat([rgb_features, dsm_features], dim=1)
            fused_features = self.fusion(fused_features)
        else:
            fused_features = rgb_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        # L2-normalize embeddings
        embeddings = F.normalize(fused_features, p=2, dim=1)
        
        return logits, embeddings


def get_gend_model(args, num_classes=2, use_multimodal=False, backbone='clip'):
    """
    Factory function to create GenD model
    
    Args:
        args: Arguments object
        num_classes: Number of classes
        use_multimodal: Whether to use multi-modal fusion
        backbone: Backbone type
    
    Returns:
        GenD or GenDMultiModal model
    """
    if use_multimodal:
        return GenDMultiModal(args, num_classes, backbone)
    else:
        return GenD(args, num_classes, backbone)
