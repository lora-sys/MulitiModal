"""
训练脚本 - 模型控制中心
通过配置文件切换不同的模型架构 (CNN / LSTM / Inception)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("experiment/dataset")
sys.path.append("experiment/model")

from csv_source import NPZDataSource
from nk2_processor import NK2Preprocessor
from massage_dataset import MassageDataset
from model import get_model
from config import MODEL_CONFIG, TRAIN_CONFIG


def load_dataset_config():
    """加载数据集配置"""
    import yaml

    config_path = "experiment/dataset/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataset(dataset_config):
    """创建数据集"""
    npz_path = "experiment/model/processed_data.npz"

    source = NPZDataSource(npz_path)
    source.initialize()

    preprocessor = NK2Preprocessor(dataset_config)
    dataset = MassageDataset(source, preprocessor)

    return dataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        x_dynamic = batch["dynamic"].to(device)
        x_static = batch["static"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(x_dynamic, x_static)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x_dynamic = batch["dynamic"].to(device)
            x_static = batch["static"].to(device)
            y = batch["label"].to(device)

            outputs = model(x_dynamic, x_static)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    # 1. 加载配置
    dataset_config = load_dataset_config()
    model_config = MODEL_CONFIG
    train_config = TRAIN_CONFIG

    print("=" * 60)
    print("🧠 按摩椅舒适度分类 - 训练")
    print("=" * 60)
    print(f"📊 模型类型: {model_config['type']}")
    print(f"📦 批次大小: {train_config['batch_size']}")
    print(f"🔢 类别数: {model_config['params']['num_classes']}")

    # 2. 创建数据集
    print("\n📂 加载数据集...")
    dataset = create_dataset(dataset_config)

    # 划分训练/验证集
    n_samples = len(dataset)
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"训练集: {n_train} 样本")
    print(f"验证集: {n_val} 样本")

    # 3. 创建模型
    print("\n🏗️ 创建模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = get_model(
        model_type=model_config["type"],
        num_classes=model_config["params"]["num_classes"],
        dyn_channels=model_config["params"]["dyn_channels"],
        static_dim=model_config["params"]["static_dim"],
    )
    model = model.to(device)

    # 4. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_config["scheduler"]["patience"],
        factor=train_config["scheduler"]["factor"],
    )

    # 5. 训练循环
    print("\n🚀 开始训练...")
    print("-" * 60)

    num_epochs = train_config["num_epochs"]
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1:2d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "experiment/model/best_model.pth")
            print(f"  💾 保存最佳模型 (Acc: {val_acc:.2f}%)")

    print("-" * 60)
    print(f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
