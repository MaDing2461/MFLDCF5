#!/usr/bin/env python3
# coding=utf-8
"""
EfficientNet Model for Fake Image Detection
Compatible with FLDCF_multiModal_TransUNet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
import numpy as np

class EfficientNet(nn.Module):
    """
    EfficientNet model for fake image detection and localization
    Compatible with FLDCF_multiModal_TransUNet output structure
    """
    def __init__(self, args, num_classes=2, pretrained=False):
        """
        Initialize EfficientNet model
        
        Args:
            args: Model arguments
            num_classes: Number of output classes (default: 2 for real/fake)
            pretrained: Whether to use pretrained weights (default: False)
        """
        super(EfficientNet, self).__init__()
        print("Initializing EfficientNet model (FLDCF_multiModal_TransUNet compatible)...")
        
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.clip_visual = None
        
        # Primary implementation: Use timm EfficientNet as the backbone
        try:
            import timm
            print("  Loading EfficientNet backbone from timm...")
            
            # Try different EfficientNet variants
            efficientnet_variants = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
            
            for model_name in efficientnet_variants:
                try:
                    # Create model with custom input size
                    self.clip_model = timm.create_model(
                        model_name, 
                        pretrained=pretrained,
                        in_chans=3,
                        num_classes=num_classes
                    )
                    self.clip_visual = self.clip_model
                    
                    # Get feature dimension from the backbone
                    # For EfficientNet, the feature dimension before the final head is the output of the last block
                    self.feature_dim = self.clip_model.blocks[-1].conv_pwl.bn2.num_features
                    
                    print(f"  ✓ {model_name} loaded (feature_dim={self.feature_dim})")
                    break
                except Exception as e:
                    print(f"  - {model_name} failed: {e}")
            else:
                # If none of the models work, use a simple CNN as fallback
                print("  WARNING: Could not load EfficientNet model, using CNN fallback...")
                # Create a simple CNN backbone
                self.clip_model = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((16, 16))  # Output 16x16 feature map
                )
                self.clip_visual = self.clip_model
                self.feature_dim = 512
                print(f"  ✓ CNN fallback loaded (feature_dim={self.feature_dim})")
            
        except Exception as e:
            print(f"  WARNING: Backbone initialization failed: {e}")
            raise ImportError("Failed to initialize model backbone")
        
        # Setup visual encoder - keep some layers trainable for fine-tuning
        print("  Configuring visual encoder...")
        
        # Configure trainable layers based on backbone type
        if hasattr(self.clip_visual, 'blocks'):
            # EfficientNet-like model
            trainable_layers = ['blocks.5', 'blocks.6', 'conv_head', 'bn2', 'fc']
            for name, param in self.clip_visual.named_parameters():
                should_train = any(layer_name in name for layer_name in trainable_layers)
                param.requires_grad = should_train
        else:
            # CNN-like model - make all layers trainable for simplicity
            for param in self.clip_visual.parameters():
                param.requires_grad = True
        
        # Segmentation head (similar to FLDCF_multiModal_TransUNet)
        print(f"  Building segmentation head...")
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        )
        
        # Classification head (similar to FLDCF_multiModal_TransUNet)
        print(f"  Building classification head...")
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ EfficientNet model initialized successfully")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Input size: 256x256")
        print(f"  Output: (segmentation result, classification logits)")
    
    def encode_image(self, image):
        """
        Extract visual features from image using either EfficientNet or CNN backbone
        
        Args:
            image: Tensor of shape (B, 3, 256, 256)
            
        Returns:
            cls_features: Tensor of shape (B, feature_dim) - features for classification
            spatial_features: Tensor of shape (B, feature_dim, H, W) - spatial features for segmentation
        """
        import torch.nn.functional as F
        
        # Forward pass based on backbone type
        if hasattr(self.clip_visual, 'forward_features'):
            # EfficientNet with forward_features method
            x = self.clip_visual.forward_features(image)
        elif hasattr(self.clip_visual, 'blocks'):
            # Standard EfficientNet structure
            # Process through the early layers
            x = self.clip_visual.conv_stem(image)
            x = self.clip_visual.bn1(x)
            x = self.clip_visual.act1(x)
            
            # Process through the blocks
            for i, block in enumerate(self.clip_visual.blocks):
                x = block(x)
        else:
            # CNN-like model
            x = self.clip_visual(image)
            
        # Process features based on shape
        if len(x.shape) == 3:
            # EfficientNet with transformer-like output: (B, num_patches, feature_dim)
            cls_features = x[:, 0]  # CLS token for classification
            
            # Convert remaining tokens to spatial feature map
            spatial_tokens = x[:, 1:]  # Remove CLS token if present
            b, n_patches, c = spatial_tokens.shape
            
            # Calculate spatial dimensions
            h = w = int(n_patches ** 0.5)
            
            # Reshape to (B, C, H, W)
            spatial_features = spatial_tokens.transpose(1, 2).reshape(b, c, h, w)
            
        elif len(x.shape) == 4:
            # CNN-like model: x shape (B, C, H, W)
            b, c, h, w = x.shape
            
            # Global average pooling for classification
            cls_features = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
            
            # Spatial features are directly the CNN output
            spatial_features = x
            
        else:
            # Fallback if shape is different
            print(f"WARNING: Unexpected feature shape: {x.shape}")
            cls_features = x
            spatial_features = None
        
        return cls_features, spatial_features
    
    def forward(self, x):
        """
        Forward pass for image segmentation and classification
        Compatible with FLDCF_multiModal_TransUNet output structure
        
        Args:
            x: Input images, shape (B, 3, 256, 256)
            
        Returns:
            segmentation_result: Segmentation map, shape (B, num_classes, 256, 256)
            classification_logits: Classification logits, shape (B, num_classes)
        """
        # Extract visual features
        cls_features, spatial_features = self.encode_image(x)
        
        # Classification branch (fake/real detection)
        classification_logits = self.classification_head(cls_features)
        
        # Segmentation branch (localization)
        if spatial_features is not None:
            segmentation_result = self.segmentation_head(spatial_features)
            # Ensure output size is 256x256
            if segmentation_result.shape[-2:] != (256, 256):
                segmentation_result = F.interpolate(segmentation_result, size=(256, 256), 
                                                   mode='bilinear', align_corners=False)
        else:
            # Fallback: create segmentation map from classification
            b, c = classification_logits.shape
            segmentation_result = classification_logits.unsqueeze(-1).unsqueeze(-1)
            segmentation_result = F.interpolate(segmentation_result, size=(256, 256), 
                                               mode='bilinear', align_corners=False)
        
        # Match FLDCF_multiModal_TransUNet output format
        return segmentation_result, classification_logits
    
    def compute_loss(self, outputs, labels):
        """
        Compute loss (compatible with FLDCF_multiModal_TransUNet)
        
        Args:
            outputs: Tuple of (segmentation_result, classification_logits)
            labels: Tuple of (segmentation_labels, classification_labels)
            
        Returns:
            loss: Combined loss value
        """
        from src.loss import CrossEntropy2d, CrossEntropyLoss
        
        # Separate outputs and labels
        segmentation_result, classification_logits = outputs
        segmentation_labels, classification_labels = labels
        
        # Segmentation loss (CrossEntropy2d)
        seg_loss_fn = CrossEntropy2d(ignore_label=255)
        seg_loss = seg_loss_fn(segmentation_result, segmentation_labels)
        
        # Classification loss (CrossEntropyLoss)
        cls_loss_fn = CrossEntropyLoss()
        cls_loss = cls_loss_fn(classification_logits, classification_labels)
        
        # Combined loss (similar to FLDCF_multiModal_TransUNet)
        loss = seg_loss + cls_loss
        
        return loss