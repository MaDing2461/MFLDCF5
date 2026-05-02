"""
Simplified GenD wrapper compatible with the existing training framework.
GenD is primarily a research model that requires multiple specialized encoders
and loss functions. This wrapper provides a minimal classification interface.
"""

import torch
import torch.nn as nn
from torchvision import models


class SimpleGenD(nn.Module):
    """
    Simplified GenD implementation using ResNet50 as backbone.
    Returns (logits, None) tuple for compatibility with training framework.
    """
    
    def __init__(self, args=None, num_classes=2, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        
        # Use ResNet50 as feature extractor (no pretrained weights to save time)
        backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Add custom classification head
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            tuple of (logits, None) for compatibility with loss functions
        """
        features = self.features(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)
        
        return logits, None
