#!/usr/bin/env python3
# coding=utf-8
"""
测试 ClipViTL14 模型
测试 FLOPs, Memory, Parameters, Speed 的计算
"""

import sys
import os
import torch

# Add the model directory to path
sys.path.insert(0, '/media/lscsc/nas2/mading/fakedetect/src/model')

from ClipViTL14.ClipViTL14 import ClipViTL14, get_model_performance_metrics


def test_clip_vit_l14():
    """Test ClipViTL14 model"""
    
    print("=" * 70)
    print("Testing ClipViTL14 Model")
    print("=" * 70)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Mock args
    class Args:
        pass
    
    args = Args()
    
    # Create model
    print("\n[1/4] Creating ClipViTL14 model...")
    try:
        model = ClipViTL14(args, num_classes=2, pretrained=False)
        model = model.to(device)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Test forward pass
    print("\n[2/4] Testing forward pass...")
    try:
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        if output.shape == (2, 2):
            print("✓ Forward pass successful")
        else:
            print(f"✗ Unexpected output shape: {output.shape}, expected (2, 2)")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("\n[3/4] Testing loss computation...")
    try:
        labels = torch.randn(2, 2).to(device)
        loss = model.compute_loss(output, labels)
        
        print(f"  Loss value: {loss.item():.6f}")
        print("✓ Loss computation successful")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    # Compute performance metrics
    print("\n[4/4] Computing performance metrics...")
    try:
        metrics = get_model_performance_metrics(
            model, 
            input_size=(1, 3, 256, 256),
            device=device
        )
        
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 70)
        print(f"  FLOPs (G):     {metrics['flops_g']:.2f}G")
        print(f"  Parameters (M): {metrics['params_m']:.2f}M")
        print(f"  Memory (MB):    {metrics['memory_mb']:.2f}MB")
        print(f"  Speed (FPS):    {metrics['fps']:.2f}FPS")
        print("=" * 70)
        
        print("✓ Performance metrics computed successfully")
    except Exception as e:
        print(f"✗ Performance metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = test_clip_vit_l14()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("✗ Some tests failed!")
        print("=" * 70)
        sys.exit(1)
