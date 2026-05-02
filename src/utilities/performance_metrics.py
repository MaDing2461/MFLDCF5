"""
ForensicsSAM模型性能指标计算工具
计算FLOPs、Parameters、Memory和Speed
"""

import torch
import time
from thop import profile


class ModelPerformanceMetrics:
    """模型性能指标计算器"""
    
    def __init__(self, model, device=None):
        """
        初始化指标计算器
        
        Args:
            model: PyTorch模型
            device: 计算设备 (默认auto-detect)
        """
        self.model = model
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
    
    def compute_flops_and_params(self, input_size=(1, 3, 1024, 1024)):
        """
        计算FLOPs和参数量
        
        Args:
            input_size: 输入大小 (B, C, H, W)
            
        Returns:
            flops_g: GFLOPs
            params_m: 参数量(M)
        """
        try:
            dummy_input = torch.randn(input_size).to(self.device)
            
            # 根据模型类型选择输入
            if hasattr(self.model, '__class__'):
                model_name = self.model.__class__.__name__
                if model_name == 'ForensicsSAM':
                    flops, params = profile(self.model, inputs=(dummy_input, False), verbose=False)
                else:
                    flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            else:
                flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            
            flops_g = flops / 1e9
            params_m = params / 1e6
            return flops_g, params_m
        except Exception as e:
            print(f"Error computing FLOPs and Parameters: {e}")
            return 0, 0
    
    def compute_memory(self, input_size=(1, 3, 1024, 1024)):
        """
        计算GPU内存占用
        
        Args:
            input_size: 输入大小 (B, C, H, W)
            
        Returns:
            memory_mb: 内存占用(MB)
        """
        try:
            torch.cuda.reset_peak_memory_stats(device=self.device)
            torch.cuda.empty_cache()
            
            dummy_input = torch.randn(input_size).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                # 根据模型类型选择调用方式
                if hasattr(self.model, '__class__'):
                    model_name = self.model.__class__.__name__
                    if model_name == 'ForensicsSAM':
                        _ = self.model(dummy_input, activate_adv=False)
                    else:
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
            
            memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            memory_mb = memory_allocated / (1024 ** 2)
            return memory_mb
        except Exception as e:
            print(f"Error computing memory: {e}")
            return 0
    
    def compute_speed(self, input_size=(1, 3, 1024, 1024), num_iterations=10, warmup=3):
        """
        计算推理速度 (FPS)
        
        Args:
            input_size: 输入大小 (B, C, H, W)
            num_iterations: 测试迭代次数
            warmup: 预热迭代次数
            
        Returns:
            fps: 每秒帧数
        """
        try:
            dummy_input = torch.randn(input_size).to(self.device)
            
            self.model.eval()
            torch.cuda.synchronize(device=self.device)
            
            # 预热
            with torch.no_grad():
                for _ in range(warmup):
                    if hasattr(self.model, '__class__'):
                        model_name = self.model.__class__.__name__
                        if model_name == 'ForensicsSAM':
                            _ = self.model(dummy_input, activate_adv=False)
                        else:
                            _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
            
            torch.cuda.synchronize(device=self.device)
            
            # 计时
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    if hasattr(self.model, '__class__'):
                        model_name = self.model.__class__.__name__
                        if model_name == 'ForensicsSAM':
                            _ = self.model(dummy_input, activate_adv=False)
                        else:
                            _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
            
            torch.cuda.synchronize(device=self.device)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            fps = num_iterations / elapsed_time
            return fps
        except Exception as e:
            print(f"Error computing speed: {e}")
            return 0
    
    def compute_all(self, input_size=(1, 3, 1024, 1024)):
        """
        计算所有性能指标
        
        Args:
            input_size: 输入大小 (B, C, H, W)
            
        Returns:
            dict: 包含所有指标的字典
        """
        metrics = {
            'flops_g': 0,
            'params_m': 0,
            'memory_mb': 0,
            'fps': 0
        }
        
        print('Computing model performance metrics...')
        
        # FLOPs和Parameters
        print('  Computing FLOPs and Parameters...', end=' ')
        flops_g, params_m = self.compute_flops_and_params(input_size)
        metrics['flops_g'] = flops_g
        metrics['params_m'] = params_m
        print(f'✓ ({flops_g:.2f}G, {params_m:.2f}M)')
        
        # Memory
        print('  Computing Memory...', end=' ')
        memory_mb = self.compute_memory(input_size)
        metrics['memory_mb'] = memory_mb
        print(f'✓ ({memory_mb:.2f}MB)')
        
        # Speed
        print('  Computing Speed...', end=' ')
        fps = self.compute_speed(input_size)
        metrics['fps'] = fps
        print(f'✓ ({fps:.2f}FPS)')
        
        return metrics
    
    def print_summary(self, input_size=(1, 3, 1024, 1024), model_name='Model'):
        """
        打印模型性能摘要
        
        Args:
            input_size: 输入大小
            model_name: 模型名称（用于显示）
        """
        metrics = self.compute_all(input_size)
        
        print('=' * 70)
        print(f'{model_name} Performance Summary:')
        print(f'  Input Size: {input_size}')
        print('-' * 70)
        print(f'  FLOPs: {metrics["flops_g"]:.2f}G')
        print(f'  Parameters: {metrics["params_m"]:.2f}M')
        print(f'  Memory: {metrics["memory_mb"]:.2f}MB')
        print(f'  Speed: {metrics["fps"]:.2f}FPS')
        print('=' * 70)
        
        return metrics


def compute_forensics_sam_metrics(model, model_type='vit_h', lora_r=8, 
                                   with_detector=True, input_size=(1, 3, 1024, 1024)):
    """
    专门用于计算ForensicsSAM的性能指标
    
    Args:
        model: ForensicsSAM模型实例
        model_type: SAM模型类型 (vit_b/l/h)
        lora_r: LoRA秩
        with_detector: 是否包含检测器
        input_size: 输入大小
        
    Returns:
        dict: 包含所有指标的字典
    """
    calculator = ModelPerformanceMetrics(model)
    metrics = calculator.compute_all(input_size)
    
    # 打印详细摘要
    print('=' * 70)
    print('ForensicsSAM Model Summary:')
    print(f'  Model Type: {model_type}')
    print(f'  LoRA Rank: {lora_r}')
    print(f'  With Detector: {with_detector}')
    print(f'  Input Size: {input_size}')
    print('-' * 70)
    print(f'  FLOPs: {metrics["flops_g"]:.2f}G')
    print(f'  Parameters: {metrics["params_m"]:.2f}M')
    print(f'  Memory: {metrics["memory_mb"]:.2f}MB')
    print(f'  Speed: {metrics["fps"]:.2f}FPS')
    print('=' * 70)
    
    return metrics


# 使用示例
if __name__ == '__main__':
    print("ForensicsSAM Performance Metrics Calculator")
    print("=" * 70)
    print("Usage Example:")
    print("""
from src.model.ForensicsSAM import ForensicsSAM
from segment_anything import sam_model_registry
from utilities.performance_metrics import compute_forensics_sam_metrics

# 构建模型
sam, _ = sam_model_registry['vit_h'](image_size=1024)
model = ForensicsSAM(sam, r=8).cuda()

# 计算指标
metrics = compute_forensics_sam_metrics(
    model,
    model_type='vit_h',
    lora_r=8,
    with_detector=True
)

# 获取具体数值
print(f"FLOPs: {metrics['flops_g']:.2f}G")
print(f"Parameters: {metrics['params_m']:.2f}M")
print(f"Memory: {metrics['memory_mb']:.2f}MB")
print(f"Speed: {metrics['fps']:.2f}FPS")
    """)
