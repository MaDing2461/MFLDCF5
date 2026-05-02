#!/usr/bin/env python3
"""
GenD Complete Integration Examples
示例脚本展示如何在fakedetect框架中使用GenD模型
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加fakedetect路径
fakedetect_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(fakedetect_root / 'fakedetect'))

from src.model.GenD.gend_full import GenD, GenDMultiModal, get_gend_model


def example_1_basic_inference():
    """示例1：基础推理"""
    print("=" * 60)
    print("示例 1: GenD 基础推理")
    print("=" * 60)
    
    # 创建模型
    model = GenD(
        args=None,
        num_classes=2,
        backbone='clip',
        pretrained=False
    )
    model.eval()
    model.cuda()
    
    # 创建虚拟输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256).cuda()
    
    # 前向传播
    with torch.no_grad():
        logits, embeddings = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出logits形状: {logits.shape}")
    print(f"✓ 输出embeddings形状: {embeddings.shape}")
    
    # 预测
    predictions = torch.argmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    
    print(f"✓ 预测: {predictions}")
    print(f"✓ 概率 (real/fake): {probabilities}")
    print()


def example_2_backbone_variants():
    """示例2：不同骨干网络"""
    print("=" * 60)
    print("示例 2: 骨干网络变体")
    print("=" * 60)
    
    backbones = ['clip', 'dino', 'perception']
    x = torch.randn(2, 3, 256, 256).cuda()
    
    for backbone_name in backbones:
        model = GenD(
            args=None,
            num_classes=2,
            backbone=backbone_name,
            pretrained=False
        )
        model.eval()
        model.cuda()
        
        with torch.no_grad():
            logits, embeddings = model(x)
        
        print(f"✓ {backbone_name:12} -> logits: {logits.shape}, embeddings: {embeddings.shape}")
    print()


def example_3_multimodal_fusion():
    """示例3：多模态RGB+DSM融合"""
    print("=" * 60)
    print("示例 3: 多模态RGB+DSM融合")
    print("=" * 60)
    
    model = GenDMultiModal(
        args=None,
        num_classes=2,
        backbone='clip',
        pretrained=False
    )
    model.eval()
    model.cuda()
    
    # 创建虚拟输入
    rgb = torch.randn(2, 3, 256, 256).cuda()
    dsm = torch.randn(2, 1, 256, 256).cuda()
    
    # 仅RGB
    with torch.no_grad():
        logits_rgb, emb_rgb = model(rgb, dsm=None)
    print(f"✓ 仅RGB输入 -> logits: {logits_rgb.shape}")
    
    # RGB+DSM
    with torch.no_grad():
        logits_fused, emb_fused = model(rgb, dsm)
    print(f"✓ RGB+DSM融合 -> logits: {logits_fused.shape}")
    print(f"✓ 融合嵌入特征维度: {emb_fused.shape}")
    print()


def example_4_embedding_extraction():
    """示例4：提取嵌入向量用于检索"""
    print("=" * 60)
    print("示例 4: 嵌入提取")
    print("=" * 60)
    
    model = GenD(
        args=None,
        num_classes=2,
        backbone='clip',
        pretrained=False
    )
    model.eval()
    model.cuda()
    
    # 模拟图像batch
    images = torch.randn(8, 3, 256, 256).cuda()
    
    # 提取特征
    with torch.no_grad():
        embeddings = model.get_embedding(images)
    
    print(f"✓ 输入图像数: {images.shape[0]}")
    print(f"✓ 嵌入维度: {embeddings.shape[1]}")
    print(f"✓ 嵌入向量的L2范数（应为1）:")
    
    for i in range(min(3, embeddings.shape[0])):
        norm = torch.norm(embeddings[i]).item()
        print(f"  样本 {i}: {norm:.6f}")
    print()


def example_5_training_loop():
    """示例5：完整训练循环示例"""
    print("=" * 60)
    print("示例 5: 训练循环框架")
    print("=" * 60)
    
    # 模型
    model = GenD(args=None, num_classes=2, backbone='clip')
    model.cuda()
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("✓ 模型创建完毕")
    print("✓ 损失函数: CrossEntropyLoss")
    print("✓ 优化器: Adam(lr=1e-4)")
    print()
    
    # 模拟训练步骤
    print("模拟训练步骤:")
    
    for step in range(3):
        # 虚拟batch
        images = torch.randn(4, 3, 256, 256).cuda()
        labels = torch.tensor([0, 1, 0, 1]).cuda()
        
        # 前向
        logits, embeddings = model(images)
        loss = criterion(logits, labels)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        
        print(f"  Step {step}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    print()


def example_6_compatible_format():
    """示例6：与fakedetect框架的兼容格式"""
    print("=" * 60)
    print("示例 6: fakedetect框架兼容格式")
    print("=" * 60)
    
    model = GenD(args=None, num_classes=2, backbone='clip')
    model.eval()
    model.cuda()
    
    # 模拟fakedetect的输入形式
    batch = {
        'image': torch.randn(4, 3, 256, 256).cuda(),
        'label': torch.tensor([0, 1, 0, 1])
    }
    
    # 前向传播
    with torch.no_grad():
        logits, embeddings = model(batch['image'])
    
    # fakedetect loss分支期望的格式
    output_tuple = (logits, embeddings)
    
    print("✓ 输入格式（fakedetect兼容）:")
    print(f"  - batch['image']: {batch['image'].shape}")
    
    print("✓ 输出格式（fakedetect兼容）:")
    print(f"  - output[0] (logits): {output_tuple[0].shape}")
    print(f"  - output[1] (embeddings): {output_tuple[1].shape}")
    
    # loss计算示例
    criterion = nn.CrossEntropyLoss()
    pred, aux = output_tuple
    loss = criterion(pred, batch['label'].cuda())
    print(f"✓ Loss计算: CrossEntropyLoss(pred, label) = {loss.item():.4f}")
    
    # test_handle评估示例
    _, predicted_eval = torch.max(pred.data, 1)
    accuracy = (predicted_eval == batch['label'].cuda()).float().mean().item()
    print(f"✓ 评估指标: Accuracy = {accuracy:.4f}")
    print()


def example_7_model_factory():
    """示例7：使用工厂函数"""
    print("=" * 60)
    print("示例 7: 模型工厂函数")
    print("=" * 60)
    
    # 单模态
    model_single = get_gend_model(
        args=None,
        num_classes=2,
        use_multimodal=False,
        backbone='clip'
    )
    print(f"✓ 单模态模型: {type(model_single).__name__}")
    
    # 多模态
    model_multi = get_gend_model(
        args=None,
        num_classes=2,
        use_multimodal=True,
        backbone='dino'
    )
    print(f"✓ 多模态模型: {type(model_multi).__name__}")
    
    print("✓ 工厂函数支持简便的模型创建")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("GenD 完整集成示例")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_inference()
        example_2_backbone_variants()
        example_3_multimodal_fusion()
        example_4_embedding_extraction()
        example_5_training_loop()
        example_6_compatible_format()
        example_7_model_factory()
        
        print("=" * 60)
        print("✅ 所有示例执行成功!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
