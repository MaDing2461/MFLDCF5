#!/usr/bin/env python3
"""Debug script to verify ClipViTL14 has trainable parameters"""

import sys
sys.path.insert(0, '/media/lscsc/nas2/mading/fakedetect/src/model')

import torch
from ClipViTL14 import ClipViTL14

class Args:
    pass

args = Args()

print("=" * 70)
print("ClipViTL14 Model Initialization Test")
print("=" * 70)

# Create model
print("\n[1] Creating model...")
model = ClipViTL14(args, num_classes=2, pretrained=True)
model.cuda()

# Check trainable parameters
print("\n[2] Analyzing trainable parameters...")
total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"  ✓ {name}: {param.numel():,} params (requires_grad=True)")

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

if trainable_params == 0:
    print("\n❌ ERROR: No trainable parameters! Optimizer will fail.")
else:
    print(f"\n✓ {trainable_params:,} trainable parameters available for training")

# Test forward pass
print("\n[3] Testing forward pass...")
try:
    dummy_input = torch.randn(2, 3, 256, 256).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✓ Forward pass successful")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
