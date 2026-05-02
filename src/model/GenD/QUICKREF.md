# GenD 快速参考卡片

## 🚀 快速开始

### 安装和导入
```python
from src.model.GenD.gend_full import GenD, GenDMultiModal
import torch

# 创建模型
model = GenD(args=None, num_classes=2, backbone='clip')
model.cuda()

# 或者使用工厂函数
from src.model.GenD.gend_full import get_gend_model
model = get_gend_model(args, num_classes=2, use_multimodal=False)
```

## 📋 模型API

### GenD 单模态
```python
model = GenD(
    args,                    # 参数对象（框架兼容）
    num_classes=2,          # 分类类别数
    backbone='clip',        # 'clip', 'dino', 'perception'
    pretrained=False,       # 是否使用预训练权重
    normalize_head=False,   # 是否归一化头部输入
    use_l2_norm=True        # 是否使用L2归一化
)

# 前向传播
logits, embeddings = model(x)  # x: (B, 3, H, W)
# 返回：
#   logits: (B, 2) - 分类logits
#   embeddings: (B, 2048) - L2归一化嵌入
```

### GenDMultiModal 多模态
```python
model = GenDMultiModal(
    args,
    num_classes=2,
    backbone='clip',
    pretrained=False
)

# RGB仅
logits, embeddings = model(rgb_image)

# RGB + DSM
logits, embeddings = model(rgb_image, dsm_image)
# 返回融合的logits和embeddings
```

## 🎯 常见任务

### 任务1：推理和预测
```python
model.eval()
with torch.no_grad():
    logits, embeddings = model(x)
    
predictions = torch.argmax(logits, dim=1)      # 硬预测
probabilities = torch.softmax(logits, dim=1)   # 软概率
```

### 任务2：提取嵌入特征
```python
embeddings = model.get_embedding(x)  # (B, 2048) 
# 嵌入已L2归一化（范数为1）
```

### 任务3：提取原始特征
```python
features = model.get_features(x)  # (B, 2048)
# 原始特征（未归一化）
```

### 任务4：完整训练步骤
```python
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Forward
logits, embeddings = model(x)
loss = criterion(logits, labels)

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 任务5：评估
```python
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in dataloader:
        logits, _ = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds)
        all_labels.append(y)

# 计算指标
accuracy = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean()
```

## 🧬 架构参数

### 骨干网络选择
| 名称 | 特性 | 推荐用途 |
|------|------|---------|
| `clip` | ResNet50，通用 | 标准任务 |
| `dino` | ViT风格，自监督 | 高精度需求 |
| `perception` | 多尺度，增强 | 细节检测 |

### 输出维度
| 层级 | 形状 | 备注 |
|------|------|------|
| 输入 | (B, 3, H, W) | 任何H,W |
| 骨干输出 | (B, 2048) | 固定特征维度 |
| Logits | (B, num_classes) | 分类输出 |
| Embeddings | (B, 2048) | L2归一化 |

## 📊 与fakedetect框架的兼容点

### ✓ Loss计算
```python
# loss/__init__.py 中自动处理
if model == 'GenD':
    pred, aux = output  # (logits, embeddings)
    loss = criterion(pred, labels)  # CrossEntropyLoss
```

### ✓ 测试评估  
```python
# test_handle() 自动处理
if model == 'GenD':
    pred, aux = output
    _, predicted = torch.max(pred, 1)  # 预测标签
```

### ✓ 检查点保存
```python
# 标准PyTorch保存
torch.save(model.state_dict(), 'checkpoint.pth')
model.load_state_dict(torch.load('checkpoint.pth'))
```

## 🔍 调试技巧

### 验证输出形状
```python
x = torch.randn(4, 3, 256, 256).cuda()
logits, embeddings = model(x)
assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
assert embeddings.shape == (4, 2048), f"Expected (4, 2048), got {embeddings.shape}"
```

### 检查梯度
```python
# 确保梯度正常流动
x = torch.randn(4, 3, 256, 256, requires_grad=True, device='cuda')
logits, _ = model(x)
loss = logits.sum()
loss.backward()
assert x.grad is not None, "梯度未正确计算"
```

### 监控模型大小
```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
```

### 内存使用
```python
# 查看GPU内存
print(torch.cuda.memory_allocated() / 1e9, "GB")

# 清理缓存
torch.cuda.empty_cache()
```

## 🚨 常见问题

### Q: 如何使用预训练权重？
A: 设置 `pretrained=True`（需要下载ResNet50权重）
```python
model = GenD(args, pretrained=True)
```

### Q: 如何自定义num_classes？
A: 在创建时指定
```python
model = GenD(args, num_classes=5)  # 5类分类
```

### Q: 如何冻结骨干网络？
A: 创建后冻结特征提取器参数
```python
for param in model.feature_extractor.parameters():
    param.requires_grad = False
```

### Q: 如何使用不同的优化器？
A: 使用标准PyTorch优化器
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

### Q: 多个GPU如何支持？
A: 使用 DataParallel
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Q: 如何混合精度训练？
A: 使用torch.cuda.amp
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    logits, _ = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📈 性能优化

### 批量推理
```python
# 更大的批次更高效
model.eval()
batch_size = 128  # 调整以适应GPU内存
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    with torch.no_grad():
        predictions = model(batch)
```

### 模型蒸馏
```python
# 使用大模型蒸馏到小模型
teacher = GenD(args, backbone='perception')  # 大模型
student = GenD(args, backbone='clip')         # 小模型

# 训练student以匹配teacher输出
```

### 量化
```python
# 动态量化以减小模型大小
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

## 📚 完整示例

运行完整示例脚本：
```bash
cd /media/lscsc/nas2/mading/fakedetect
python src/model/GenD/examples.py
```

示例包括：
1. ✓ 基础推理
2. ✓ 骨干网络变体
3. ✓ 多模态融合
4. ✓ 嵌入提取
5. ✓ 完整训练循环
6. ✓ fakedetect兼容格式
7. ✓ 工厂函数使用

## 🔗 相关资源

| 资源 | 位置 |
|------|------|
| 完整文档 | [README.md](README.md) |
| 集成指南 | [INTEGRATION.md](INTEGRATION.md) |
| 代码示例 | [examples.py](examples.py) |
| 实现代码 | [gend_full.py](gend_full.py) |
| 框架主页 | `../../` |

## 🎓 学习路径

1. **快速入门** → 本文档
2. **理解架构** → README.md
3. **查看示例** → examples.py
4. **深入实现** → gend_full.py
5. **框架集成** → INTEGRATION.md

---

**快速参考卡片 v1.0**  
**最后更新**: 2026-01-08
