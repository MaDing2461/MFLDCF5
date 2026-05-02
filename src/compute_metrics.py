"""
ForensicsSAM性能指标计算和打印脚本
包含完整的FLOPs、Memory、Parameters、Speed计算
"""

import torch
import time
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))

from model.ForensicsSAM import ForensicsSAM
from segment_anything import sam_model_registry
from thop import profile


def print_model_metrics(model, model_type='vit_h', lora_r=8, 
                        with_detector=True, input_size=(1, 3, 1024, 1024)):
    """
    计算并打印ForensicsSAM的完整性能指标
    
    Args:
        model: ForensicsSAM模型实例
        model_type: SAM类型 (vit_b/l/h)
        lora_r: LoRA秩
        with_detector: 是否包含检测器
        input_size: 输入大小 (B, C, H, W)
    """
    device = next(model.parameters()).device
    
    print("\n" + "=" * 70)
    print("ForensicsSAM Performance Metrics Computation")
    print("=" * 70)
    
    # 准备输入
    dummy_input = torch.randn(input_size).to(device)
    
    # 1. 计算FLOPs和Parameters
    print("\n[1/4] Computing FLOPs and Parameters...")
    try:
        flops, params = profile(model, inputs=(dummy_input, False), verbose=False)
        flops_g = flops / 1e9
        params_m = params / 1e6
        print(f"      ✓ FLOPs: {flops_g:.2f}G")
        print(f"      ✓ Parameters: {params_m:.2f}M")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        flops_g, params_m = 0, 0
    
    # 2. 计算内存占用
    print("\n[2/4] Computing Memory Usage...")
    try:
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input, activate_adv=False)
        
        memory_allocated = torch.cuda.max_memory_allocated(device=device)
        memory_mb = memory_allocated / (1024 ** 2)
        print(f"      ✓ Memory: {memory_mb:.2f}MB")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        memory_mb = 0
    
    # 3. 计算推理速度
    print("\n[3/4] Computing Inference Speed...")
    try:
        model.eval()
        torch.cuda.synchronize(device=device)
        
        # 预热 (3次)
        print("      Warming up...", end=" ")
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input, activate_adv=False)
        torch.cuda.synchronize(device=device)
        print("✓")
        
        # 计时 (10次迭代)
        print("      Timing...", end=" ")
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input, activate_adv=False)
        
        torch.cuda.synchronize(device=device)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        fps = num_iterations / elapsed_time
        avg_latency_ms = (elapsed_time / num_iterations) * 1000
        print("✓")
        print(f"      ✓ Speed: {fps:.2f}FPS")
        print(f"      ✓ Latency: {avg_latency_ms:.2f}ms")
    except Exception as e:
        print(f"\n      ✗ Error: {e}")
        fps, avg_latency_ms = 0, 0
    
    # 4. 模型配置信息
    print("\n[4/4] Model Configuration")
    print(f"      SAM Type: {model_type}")
    print(f"      LoRA Rank: {lora_r}")
    print(f"      With Detector: {with_detector}")
    print(f"      Input Size: {input_size}")
    print(f"      Device: {device}")
    
    # 打印最终摘要
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Value':<15} {'Unit'}")
    print("-" * 70)
    print(f"{'FLOPs':<20} {flops_g:<15.2f} {'G'}")
    print(f"{'Parameters':<20} {params_m:<15.2f} {'M'}")
    print(f"{'Memory':<20} {memory_mb:<15.2f} {'MB'}")
    print(f"{'Speed':<20} {fps:<15.2f} {'FPS'}")
    print(f"{'Latency':<20} {avg_latency_ms:<15.2f} {'ms'}")
    print("=" * 70 + "\n")
    
    return {
        'flops_g': flops_g,
        'params_m': params_m,
        'memory_mb': memory_mb,
        'fps': fps,
        'latency_ms': avg_latency_ms
    }


def main():
    """主函数 - 演示使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ForensicsSAM性能指标计算')
    parser.add_argument('--sam_type', type=str, default='vit_h',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM模型类型')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA秩')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='SAM预训练权重路径')
    parser.add_argument('--with_detector', action='store_true', default=True,
                        help='是否包含检测器')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 1024, 1024],
                        help='输入大小 [B, C, H, W]')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ForensicsSAM Performance Metrics Calculator")
    print("=" * 70)
    
    # 构建SAM模型
    print(f"\nBuilding {args.sam_type} SAM model...")
    sam, _ = sam_model_registry[args.sam_type](
        image_size=1024,
        checkpoint=args.checkpoint
    )
    
    # 包装为ForensicsSAM
    print("Wrapping as ForensicsSAM...")
    model = ForensicsSAM(
        sam_model=sam,
        r=args.lora_r,
        with_detector=args.with_detector
    ).cuda()
    
    # 计算和打印指标
    metrics = print_model_metrics(
        model=model,
        model_type=args.sam_type,
        lora_r=args.lora_r,
        with_detector=args.with_detector,
        input_size=tuple(args.input_size)
    )
    
    return metrics


if __name__ == '__main__':
    main()
