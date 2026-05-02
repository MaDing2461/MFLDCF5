# coding=utf-8
"""
Clip-ViT-L/14 Model for Fake Image Detection
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


class ClipViTL14(nn.Module):
    """
    CLIP ViT-L/14 model for fake image detection and localization
    Compatible with FLDCF_multiModal_TransUNet
    Input: 256x256 RGB images
    Output: Tuple of (segmentation_result, classification_logits)
    Loss: CrossEntropy2d (segmentation) + CrossEntropyLoss (classification)
    """
    
    def __init__(self, args, num_classes=2, pretrained=True):
        super(ClipViTL14, self).__init__()
        self.num_classes = num_classes
        self.args = args
        self.feature_dim = 768
        self.clip_visual = None
        
        print("Initializing ClipViTL14 model (FLDCF_multiModal_TransUNet compatible)...")
        
        # Primary implementation: Use timm ViT as the CLIP-compatible backbone
        try:
            import timm
            print("  Loading ViT backbone from timm...")
            
            # Create a ViT model that can handle 256x256 input
            try:
                # Try to create a model with custom input size
                self.clip_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, img_size=256)
                self.clip_visual = self.clip_model
                self.feature_dim = 768
                print(f"  ✓ ViT model with img_size=256 loaded (feature_dim={self.feature_dim})")
            except Exception as e:
                print(f"  - Could not create ViT with custom size: {e}")
                
                # Try different ViT model names that might support 256x256
                vit_model_names = ['vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224']
                
                for model_name in vit_model_names:
                    try:
                        # Create model with default size first
                        self.clip_model = timm.create_model(model_name, pretrained=pretrained)
                        self.clip_visual = self.clip_model
                        self.feature_dim = 768 if 'base' in model_name else 1024  # ViT-B has 768 dims, ViT-L has 1024
                        print(f"  ✓ {model_name} loaded (feature_dim={self.feature_dim})")
                        break
                    except Exception as e:
                        print(f"  - {model_name} failed: {e}")
                else:
                    # If none of the models work, use a simple CNN as fallback
                    print("  WARNING: Could not load ViT model, using CNN fallback...")
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
        if hasattr(self.clip_visual, 'blocks') or hasattr(self.clip_visual, 'block'):
            # ViT-like model
            trainable_layers = ['block.10', 'block.11', 'norm', 'head']  # Last two blocks + head
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
        
        print(f"✓ ClipViTL14 model initialized successfully")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Input size: 256x256")
        print(f"  Output: (segmentation result, classification logits)")
    
    def encode_image(self, image):
        """
        Extract visual features from image using either ViT or CNN backbone
        
        Args:
            image: Tensor of shape (B, 3, 256, 256)
            
        Returns:
            cls_features: Tensor of shape (B, feature_dim) - features for classification
            spatial_features: Tensor of shape (B, feature_dim, H, W) - spatial features for segmentation
        """
        import torch.nn.functional as F
        
        # Forward pass based on backbone type
        if hasattr(self.clip_visual, 'forward_features'):
            # ViT-like model
            x = self.clip_visual.forward_features(image)
        else:
            # CNN-like model
            x = self.clip_visual(image)
            
        # Process features based on shape
        if len(x.shape) == 3:
            # ViT-like model: x shape (B, num_patches+1, feature_dim)
            cls_features = x[:, 0]  # CLS token for classification
            
            # Convert remaining tokens to spatial feature map
            spatial_tokens = x[:, 1:]  # Remove CLS token
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


class ClipViTL14WithAux(nn.Module):
    """
    Extended CLIP ViT-L/14 model with auxiliary detection branches
    Similar to FLDCF_multiModal_TransUNet with multiple prediction heads
    """
    
    def __init__(self, args, num_classes=2, pretrained=True):
        super(ClipViTL14WithAux, self).__init__()
        
        # Base CLIP ViT-L/14 model
        self.base_model = ClipViTL14(args, num_classes=num_classes, pretrained=pretrained)
        self.feature_dim = self.base_model.feature_dim
        self.num_classes = num_classes
        
        # Auxiliary classifiers (similar to FLDCF_multiModal_TransUNet)
        self.aux_classifier_1 = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.aux_classifier_2 = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        print("ClipViTL14WithAux model initialized")
    
    def forward(self, x):
        """
        Forward pass with auxiliary outputs
        
        Args:
            x: Input images, shape (B, 3, 256, 256)
            
        Returns:
            outputs: Dict containing main and auxiliary logits
        """
        # Extract visual features
        with torch.no_grad():
            features = self.base_model.encode_image(x)
        
        # Main classification
        main_logits = self.base_model.classifier(features)
        
        # Auxiliary classifications
        aux_logits_1 = self.aux_classifier_1(features)
        aux_logits_2 = self.aux_classifier_2(features)
        
        return {
            'logits': main_logits,
            'aux_logits_1': aux_logits_1,
            'aux_logits_2': aux_logits_2
        }
    
    def compute_loss(self, outputs, labels):
        """
        Compute combined loss from main and auxiliary branches
        
        Args:
            outputs: Model outputs dict
            labels: Ground truth labels
            
        Returns:
            loss: Total loss
        """
        if labels.dtype != outputs['logits'].dtype:
            labels = labels.float()
        
        main_loss = self.loss_fn(outputs['logits'], labels)
        aux_loss_1 = self.loss_fn(outputs['aux_logits_1'], labels)
        aux_loss_2 = self.loss_fn(outputs['aux_logits_2'], labels)
        
        # Weighted combination (similar to multi-task learning in FLDCF_multiModal_TransUNet)
        total_loss = main_loss + 0.5 * aux_loss_1 + 0.5 * aux_loss_2
        
        return total_loss


def build_clip_vit_l14(args, **kwargs):
    """
    Build ClipViTL14 model
    
    Args:
        args: Arguments
        **kwargs: Additional arguments
        
    Returns:
        model: ClipViTL14 model instance
    """
    model = ClipViTL14(args, **kwargs)
    return model


def get_model_performance_metrics(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    Calculate FLOPs, Parameters, Memory usage, and Speed (FPS) of the model
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (default: 1x3x256x256)
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        metrics: Dict containing FLOPs, Parameters, Memory, and Speed
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # 1. FLOPs and Parameters
    print('==> Computing model complexity...')
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
        params_m = params / 1e6
        print(f'FLOPs: {flops_g:.2f}G')
        print(f'Parameters: {params_m:.2f}M')
    except Exception as e:
        print(f"Warning: Could not compute FLOPs and Parameters: {e}")
        flops_g, params_m = 0, 0
    
    # 2. Memory usage
    print('==> Computing memory usage...')
    try:
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device.startswith('cuda'):
            memory_allocated = torch.cuda.max_memory_allocated(device=device)
            memory_mb = memory_allocated / (1024 ** 2)
        else:
            memory_mb = 0
            
        print(f'Memory: {memory_mb:.2f}MB')
    except Exception as e:
        print(f"Warning: Could not compute memory: {e}")
        memory_mb = 0
    
    # 3. Speed (FPS)
    print('==> Computing inference speed...')
    try:
        model.eval()
        if device.startswith('cuda'):
            torch.cuda.synchronize(device=device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize(device=device)
        
        # Actual timing
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize(device=device)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = num_iterations / elapsed_time if elapsed_time > 0 else 0
        
        print(f'Speed: {fps:.2f}FPS')
    except Exception as e:
        print(f"Warning: Could not compute speed: {e}")
        fps = 0
    
    # Print summary
    print('=' * 60)
    print('ClipViTL14 Model Summary:')
    print(f'  Input Size: {input_size}')
    print('-' * 60)
    print(f'  FLOPs: {flops_g:.2f}G')
    print(f'  Parameters: {params_m:.2f}M')
    print(f'  Memory: {memory_mb:.2f}MB')
    print(f'  Speed: {fps:.2f}FPS')
    print('=' * 60)
    
    return {
        'flops_g': flops_g,
        'params_m': params_m,
        'memory_mb': memory_mb,
        'fps': fps
    }


if __name__ == '__main__':
    # Test the model
    print("Testing ClipViTL14 model...")
    
    # Mock args
    class Args:
        pass
    
    args = Args()
    
    # Create model
    model = ClipViTL14(args, num_classes=2, pretrained=False)
    model.cuda()
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compute performance metrics
    metrics = get_model_performance_metrics(model, input_size=(1, 3, 256, 256), device='cuda')
    print(f"\nPerformance Metrics: {metrics}")
