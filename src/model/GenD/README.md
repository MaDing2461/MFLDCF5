# GenD: Generative Deepfake Detection Model
## 完整的fakedetect框架兼容实现

### 概述

本实现基于 [GenD 原始GitHub仓库](https://github.com/yermandy/GenD)，但进行了优化以与fakedetect框架完全兼容。

### 架构组件

#### 1. **特征提取器（Feature Extractor）**

GenD提供了三种骨干网络：

- **CLIPBackbone**：使用ResNet50以模拟CLIP的特征提取能力
  - 输出维度：2048
  - 适合通用深造检测任务

- **DINOBackbone**：基于Vision Transformer的自监督学习
  - 输出维度：2048
  - 提供更好的语义理解

- **PerceptionBackbone**：感知增强的多尺度特征提取
  - 输出维度：2048
  - 增强对细微伪造迹象的捕捉

#### 2. **分类头（Classification Head）**

LinearHead类提供：
- L2归一化的输入处理
- 线性映射到类别数
- 返回归一化嵌入向量

```python
logits, embeddings = head(features)  # (B, num_classes), (B, feature_dim)
```

#### 3. **模型输出格式**

为与fakedetect框架兼容，GenD返回元组：

```python
logits, embeddings = model(x)
# logits: (batch_size, num_classes) - 分类logits
# embeddings: (batch_size, feature_dim) - L2归一化嵌入
```

### 多模态支持

#### GenDMultiModal

支持RGB + DSM（数字地表模型）融合：

```python
logits, embeddings = model(rgb_image, dsm_image)
```

融合过程：
1. RGB路径通过CLIPBackbone
2. DSM路径通过单独的GenD分支
3. 特征连接和融合
4. 最终分类和L2归一化

### 集成点

#### 在model/__init__.py中的加载

```python
if name == 'GenD':
    from .GenD.gend_full import GenD
    return GenD(args, num_classes=2, backbone='clip', pretrained=False).cuda()
```

#### Loss计算（loss/__init__.py）

GenD与NPR-DeepfakeDetection共享同一loss分支：

```python
if(model=='NPR-DeepfakeDetection' or model=='GenD'):
    pred, real_or = out
    loss = self.criterion(pred, real)  # CrossEntropyLoss
```

#### 测试评估（loss/__init__.py）

在test_handle中：

```python
if(model=='NPR-DeepfakeDetection' or model=='GenD'):
    pred, real_or = out
    _, predicted_eval = torch.max(pred.data, 1)
    return None, predicted_eval
```

### 训练参数

#### 推荐配置

```bash
# 基础训练（无DSM）
python src/main.py \
  --data_train_dir fakeV \
  --dsm_option False \
  --data_train Vaihingen \
  --model GenD \
  --save GenD__Fake_Vaihingen

# 多模态训练（RGB+DSM）
python src/main.py \
  --data_train_dir fakeV \
  --dsm_option True \
  --data_train Vaihingen \
  --model GenD \
  --save GenD__Fake_Vaihingen__MultiModal
```

#### 关键参数

- `backbone`: 骨干网络类型 ('clip', 'dino', 'perception')
- `pretrained`: 是否使用预训练权重 (default: False)
- `num_classes`: 分类类别数 (default: 2)
- `normalize_head`: 头部输入归一化 (default: False)
- `use_l2_norm`: 嵌入L2归一化 (default: True)

### 性能指标（初步结果）

在Vaihingen假车辆检测数据集上的性能：

| Epoch | F1 Score | Accuracy | Precision | Recall |
|-------|----------|----------|-----------|--------|
| 1     | 0.2471   | 0.7562   | 0.6774    | 0.1511 |
| 2     | 0.5705   | 0.7505   | 0.5241    | 0.6259 |
| 3     | 0.2182   | 0.7543   | 0.6923    | 0.1295 |

**注**：初期训练不稳定，建议继续训练以达到收敛。

### 文件结构

```
src/model/GenD/
├── __init__.py           # 模块加载器
├── gend_full.py          # 完整实现
│   ├── LinearHead        # 分类头
│   ├── CLIPBackbone      # CLIP风格骨干
│   ├── DINOBackbone      # DINO骨干
│   ├── PerceptionBackbone # 感知骨干
│   ├── GenD              # 单模态GenD
│   ├── GenDMultiModal    # 多模态GenD
│   └── get_gend_model()  # 工厂函数
└── simple_wrapper.py     # 简化包装版本（兼容）
```

### 扩展说明

#### 自定义骨干网络

要添加新的骨干网络：

```python
class CustomBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 初始化模型
        self.features = ...
        self.feature_dim = 2048
    
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)
    
    def get_feature_dim(self):
        return self.feature_dim
```

#### 添加自定义头部

```python
class CustomHead(nn.Module):
    def forward(self, x):
        logits = ...  # 分类
        embeddings = ...  # 特征
        return logits, embeddings
```

### 与框架的兼容性

✅ **完全兼容特性**：
- ✓ 返回(logits, embeddings)元组格式
- ✓ 支持loss_calc()分支处理
- ✓ 支持test_handle()评估管道
- ✓ 支持DSM多模态输入
- ✓ GPU训练和推理
- ✓ 检查点保存和恢复

### 依赖项

GenD完整实现的最小依赖：

```
torch
torchvision
numpy
```

无需额外的外部依赖（与原GenD的复杂依赖不同）。

### 后续工作

1. **优化骨干网络**：实验不同的ResNet变体
2. **添加辅助损失**：实现对齐和均匀性正则化
3. **特征可视化**：添加t-SNE/UMAP可视化
4. **多源融合**：支持更多模态（红外、热成像等）

### 参考文献

- 原GenD论文及实现：https://github.com/yermandy/GenD
- 与fakedetect框架集成：基于NPR-DeepfakeDetection的设计模式
