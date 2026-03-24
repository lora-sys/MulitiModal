#!/usr/bin/env python
"""快速测试 baseline_c 训练"""
import sys
import os
sys.path.insert(0, '/home/lora/repos/MulitiModal/experiment/model')
sys.path.insert(0, '/home/lora/repos/MulitiModal/experiment')
sys.path.insert(0, '/home/lora/repos/MulitiModal/experiment/dataset')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import get_model
from unified_source import UnifiedNPZDataSource
from unified_dataset import UnifiedMultimodalDataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
source = UnifiedNPZDataSource('experiment/model/unified_dataset_expanded.npz')
source.initialize()
dataset = UnifiedMultimodalDataset(source, preprocessor=None)

# 划分数据集
print(f"数据集总大小: {len(dataset)}")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
indices = list(range(len(dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]
print(f"训练集索引数: {len(train_indices)}, 验证集索引数: {len(val_indices)}")
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
print(f"训练集子集大小: {len(train_dataset)}, 验证集子集大小: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(val_dataset)} 样本")

# 创建模型
model = get_model(
    model_type="baseline_c",
    num_classes=3,
    dyn_channels=2,
    static_dim=4
).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 添加 CosineAnnealingWarmup 调度器
from warmup_scheduler import CosineAnnealingWarmupScheduler

warmup_epochs = 10
max_epochs = 50
eta_min = 1e-6

scheduler = CosineAnnealingWarmupScheduler(
    optimizer,
    warmup_epochs=warmup_epochs,
    max_epochs=max_epochs,
    eta_min=eta_min
)

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        dynamic = batch['dynamic'].to(device)
        static_basic = batch['static_basic'].to(device)
        static_scores = batch['static_scores'].to(device)
        constitution = batch['constitution'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(dynamic, static_basic, static_scores, constitution)
        loss = criterion(outputs, labels)
        loss.backward()

        # 检查梯度
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_acc = 100. * train_correct / train_total

    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            labels = batch['label'].to(device)

            outputs = model(dynamic, static_basic, static_scores, constitution)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}% | "
          f"Grad Norm: {grad_norm:.4f}")

print("\n测试完成！")