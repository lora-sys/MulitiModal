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

from unified_source import UnifiedNPZDataSource
from nk2_processor import NK2Preprocessor
from self_healing_processor import SelfHealingPreprocessor
from unified_dataset import UnifiedMultimodalDataset
from model import get_model
from config import MODEL_CONFIG, MODEL_PARAMS, TRAIN_CONFIG, SCHEDULER_CONFIGS, CURRENT_SCHEDULER


def load_dataset_config():
    """加载数据集配置"""
    import yaml

    config_path = "experiment/dataset/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataset(dataset_config):
    """创建数据集 - 使用统一NPZ数据源"""
    # 从配置文件读取数据集路径
    npz_path = dataset_config.get('unified_npz', {}).get('path', "experiment/model/unified_dataset_100.npz")
    
    # 创建数据源
    source = UnifiedNPZDataSource(npz_path)
    source.initialize()

    # 创建数据集（不使用预处理器，NPZ数据已预处理）
    dataset = UnifiedMultimodalDataset(source, preprocessor=None)

    print(f"[*] Created dataset: {len(dataset)} samples")
    return dataset


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        # 新数据集返回字典格式
        dynamic = batch['dynamic'].to(device)
        static_basic = batch['static_basic'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # 根据模型类型调用不同的 forward
        if model_type == "multimodal":
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            outputs = model(dynamic, static_basic, static_scores, constitution)
        else:
            outputs = model(dynamic, static_basic)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device, model_type):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            labels = batch['label'].to(device)

            # 根据模型类型调用不同的 forward
            if model_type == "multimodal":
                static_scores = batch['static_scores'].to(device)
                constitution = batch['constitution'].to(device)
                outputs = model(dynamic, static_basic, static_scores, constitution)
            else:
                outputs = model(dynamic, static_basic)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_labels, all_preds, all_probs


def detailed_evaluation(labels, preds, probs, num_classes=4, class_names=['很差(0)', '一般(1)', '正常(2)', '良好(3)']):
    """详细评估 - 打印分类报告和混淆矩阵"""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    import seaborn as sns

    print("\n" + "=" * 60)
    print("📊 详细评估报告")
    print("=" * 60)

    # 1. 分类报告
    print("\n📋 分类报告:")
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)

    # 2. 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print("\n📊 混淆矩阵:")
    print(cm)

    # 3. 每类指标
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)

    print("\n📈 每类详细指标:")
    for i, name in enumerate(class_names):
        print(f"  {name}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall:    {recall[i]:.4f}")
        print(f"    F1-Score:  {f1[i]:.4f}")
        print(f"    Support:   {support[i]}")

    # 4. 宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    print("\n📊 总体指标:")
    print(f"  Accuracy:         {accuracy_score(labels, preds):.4f}")
    print(f"  Macro Precision:  {precision_macro:.4f}")
    print(f"  Macro Recall:     {recall_macro:.4f}")
    print(f"  Macro F1-Score:   {f1_macro:.4f}")
    print(f"  Weighted F1-Score:{f1_weighted:.4f}")

    return cm


def plot_confusion_matrix(cm, class_names=['很差(0)', '一般(1)', '正常(2)', '良好(3)'], save_path="experiment/test/result/confusion_matrix_multimodal.png"):
    """绘制混淆矩阵热力图"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': '样本数'})
    plt.title('混淆矩阵 - Multi-Expert Fusion', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 混淆矩阵已保存: {save_path}")
    plt.close()


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
    model_params = MODEL_PARAMS.get(model_config["type"], {})
    model = get_model(
        model_type=model_config["type"],
        num_classes=model_config["params"]["num_classes"],
        dyn_channels=model_config["params"]["dyn_channels"],
        static_dim=model_config["params"]["static_dim"],
        **model_params # 传入模型专属参数 (如 transformer 的 d_model, nhead 等)
        
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
        scheduler = optim.lr_scheduler.ReduceLoroPlateau(
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
    elif scheduler_type == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.get("max_lr", 2e-3),
            total_steps=scheduler_cfg.get("total_steps", 4000),
            pct_start=scheduler_cfg.get("pct_start", 0.3),
            anneal_strategy=scheduler_cfg.get("anneal_strategy", "cos"),
        )
    elif scheduler_type == "CosineAnnealingWarmup":
        # Cosine Annealing + Warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config["num_epochs"] - scheduler_cfg.get("warmup_epochs", 5),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
        # 创建 warmup 调度器
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # warmup 从 0.1x 开始
            end_factor=1.0,
            total_iters=scheduler_cfg.get("warmup_epochs", 5)
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            warmup_scheduler,
            scheduler
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
            model, train_loader, criterion, optimizer, device, model_config["type"]
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, model_config["type"])

        # 更新学习率
        if scheduler_type in ["ReduceLROnPlateau"]:
            scheduler.step(val_loss)
        elif scheduler_type in ["OneCycleLR"]:
            scheduler.step()
        elif scheduler_type in ["CosineAnnealingWarmup", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR"]:
            scheduler.step()
        else:
            scheduler.step(val_loss)

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
                model.state_dict(),
                f"experiment/model/bfoundation_model_{model_type}.pth",
            )
            print(f"  💾 保存最佳模型 (Acc: {val_acc:.2f}%)")

    print("-" * 60)
    training_time = time.time() - start_time
    print(
        f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}% | 耗时: {training_time:.1f}秒"
    )

    # 6. 详细评估
    print("\n" + "=" * 60)
    print("🔍 加载最佳模型进行详细评估...")
    print("=" * 60)

    # 加载最佳模型
    best_model_path = f"experiment/model/best_model_{model_config['type']}.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    # 在验证集上评估
    val_loss, val_acc, all_labels, all_preds, all_probs = evaluate(
        model, val_loader, criterion, device, model_config["type"]
    )

    # 调试信息
    print(f"\n[调试] 验证集大小: {len(all_labels)}")
    print(f"[调试] 预测数量: {len(all_preds)}")
    print(f"[调试] 标签分布: {np.bincount(all_labels)}")

    # 详细评估报告
    cm = detailed_evaluation(all_labels, all_preds, all_probs)

    # 绘制混淆矩阵
    cm_path = f"experiment/test/result/confusion_matrix_{model_config['type']}.png"
    plot_confusion_matrix(cm, save_path=cm_path)

    # 7. 保存实验日志
    model_type = model_config["type"]
    save_experiment_log(
        model_type, train_config, best_val_acc, training_time, scheduler_type
    )

    # 8. 绘制训练曲线
    print("\n📈 生成训练曲线...")
    result_path = f"experiment/test/result/test_result_{model_type}.png"
    plot_training_history(history, result_path)


if __name__ == "__main__":
    main()
