"""
训练脚本 - 模型控制中心
通过配置文件切换不同的模型架构 (CNN / LSTM / Inception)
支持日志记录和多种学习率调度器
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

sys.path.append("experiment/dataset")
sys.path.append("experiment/model")

from csv_source import NPZDataSource
from nk2_processor import NK2Preprocessor
from self_healing_processor import SelfHealingPreprocessor
from massage_dataset import MassageDataset
from model import get_model
from config import MODEL_CONFIG, TRAIN_CONFIG, SCHEDULER_CONFIGS, CURRENT_SCHEDULER


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

    # 根据配置选择预处理器
    preprocessor_type = dataset_config.get("preprocessor", {}).get("type", "nk2")

    if preprocessor_type == "self_healing":
        print("🔧 使用自研信号自愈预处理器")
        preprocessor = SelfHealingPreprocessor(dataset_config)
    else:
        print("🔧 使用 NK2 预处理器")
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


def plot_training_history(
    history, save_path="experiment/test/result/test_result_inception.png"
):
    """绘制训练曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r--", label="Val Acc", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📈 训练曲线已保存: {save_path}")
    plt.close()


def save_experiment_log(
    model_type, train_config, best_acc, training_time, scheduler_type="CosineAnnealing"
):
    """保存实验日志"""
    log_path = "experiment/model/log.txt"

    log_entry = f"""================================================================================
训练实验日志 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================
模型: {model_type}
学习率: {train_config["learning_rate"]}
批次大小: {train_config["batch_size"]}
Epochs: {train_config["num_epochs"]}
学习率调度器: {scheduler_type}
最佳验证准确率: {best_acc:.2f}%
训练时间: {training_time:.1f}秒
--------------------------------------------------------------------------------
"""

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"📝 实验日志已保存: {log_path}")


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
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 1e-4),
    )

    # 根据配置选择调度器
    scheduler_cfg = SCHEDULER_CONFIGS[CURRENT_SCHEDULER]
    scheduler_type = scheduler_cfg.get("type")

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            patience=scheduler_cfg.get("patience", 5),
            factor=scheduler_cfg.get("factor", 0.5),
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", train_config["num_epochs"]),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get("T_0", 10),
            T_mult=scheduler_cfg.get("T_mult", 2),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    elif scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # 5. 训练循环
    print("\n🚀 开始训练...")
    print("-" * 60)

    start_time = time.time()

    num_epochs = train_config["num_epochs"]
    best_val_acc = 0

    # 记录训练历史
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        if scheduler_type in ["ReduceLROnPlateau"]:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:2d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {lr:.6f}"
        )

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_type = model_config["type"]
            torch.save(
                model.state_dict(), f"experiment/model/best_model_{model_type}.pth"
            )
            print(f"  💾 保存最佳模型 (Acc: {val_acc:.2f}%)")

    print("-" * 60)
    training_time = time.time() - start_time
    print(
        f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}% | 耗时: {training_time:.1f}秒"
    )

    # 6. 保存实验日志
    model_type = model_config["type"]
    save_experiment_log(
        model_type, train_config, best_val_acc, training_time, scheduler_type
    )

    # 7. 绘制训练曲线
    print("\n📈 生成训练曲线...")
    result_path = f"experiment/test/result/test_result_{model_type}.png"
    plot_training_history(history, result_path)


if __name__ == "__main__":
    main()
