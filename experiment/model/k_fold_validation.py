"""
5-Fold 交叉验证脚本
验证模型在不同数据划分下的稳定性
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'recorder'))

from unified_source import UnifiedNPZDataSource
from unified_dataset import UnifiedMultimodalDataset
from model import get_model
from config import MODEL_CONFIG, TRAIN_CONFIG, SCHEDULER_CONFIGS, CURRENT_SCHEDULER
from recorder import ExperimentRecorder, compute_metrics

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 配置
N_FOLDS = 5
NUM_EPOCHS = 50
MODEL_TYPE = "baseline_c"  # 可以改为 baseline_a, baseline_b, baseline_c
TASK_TYPE = "classification"  # 可以改为 classification 或 regression


def load_dataset():
    """加载数据集"""
    npz_path = "experiment/model/unified_dataset_expanded.npz"
    source = UnifiedNPZDataSource(npz_path)
    source.initialize()
    dataset = UnifiedMultimodalDataset(source, preprocessor=None)
    print(f"[*] 加载数据集: {len(dataset)} 样本")
    return dataset


def train_fold(model, train_loader, val_loader, device, fold_num, task_type="classification"):
    """训练单个 fold"""
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler_config = SCHEDULER_CONFIGS.get(CURRENT_SCHEDULER, {})
    if scheduler_config["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 20),
            eta_min=scheduler_config.get("eta_min", 1e-6)
        )
    
    best_metric = 0 if task_type == "classification" else float('inf')
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if MODEL_TYPE in ["multimodal", "simple_concat", "late_fusion", "baseline_a", "baseline_b", "baseline_c"]:
                static_scores = batch['static_scores'].to(device)
                constitution = batch['constitution'].to(device)
                outputs = model(dynamic, static_basic, static_scores, constitution)
            else:
                outputs = model(dynamic, static_basic)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if task_type == "classification":
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch['dynamic'].to(device)
                static_basic = batch['static_basic'].to(device)
                labels = batch['label'].to(device)

                if MODEL_TYPE in ["multimodal", "simple_concat", "late_fusion", "baseline_a", "baseline_b", "baseline_c"]:
                    static_scores = batch['static_scores'].to(device)
                    constitution = batch['constitution'].to(device)
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                else:
                    outputs = model(dynamic, static_basic)

                if task_type == "classification":
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                else:
                    # 回归任务：计算 MAE
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    val_correct += torch.abs(outputs - labels).sum().item()
        
        # 计算指标
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        val_mae = val_correct / val_total if val_total > 0 else 0  # 回归任务的 MAE

        # 更新最佳模型
        current_metric = val_acc if task_type == "classification" else val_mae
        is_better = current_metric > best_metric if task_type == "classification" else current_metric < best_metric
        
        if is_better:
            best_metric = current_metric
            best_epoch = epoch
    
    return best_metric, best_epoch


def main():
    """主函数"""
    print("=" * 60)
    print(f"🚀 {N_FOLDS}-Fold 交叉验证")
    print(f"模型: {MODEL_TYPE}")
    print(f"任务: {TASK_TYPE}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    
    # 加载数据集
    dataset = load_dataset()
    
    # 准备标签用于分层抽样
    labels = np.array([dataset[i]['label'] for i in range(len(dataset))])

    # 创建 K-Fold 分割
    if TASK_TYPE == "regression":
        # 回归任务使用普通 KFold
        from sklearn.model_selection import KFold
        skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    else:
        # 分类任务使用 StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    total_start_time = time.time()

    print(f"\n开始交叉验证...")
    print("-" * 60)

    # 根据任务类型选择分割参数
    if TASK_TYPE == "regression":
        # 回归任务：KFold 只需要 X 参数
        folds = list(skf.split(np.arange(len(dataset))))
    else:
        # 分类任务：StratifiedKFold 需要 X 和 y 参数
        folds = list(skf.split(np.arange(len(dataset)), labels))

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n📦 Fold {fold + 1}/{N_FOLDS}")
        print(f"  训练集: {len(train_idx)} 样本")
        print(f"  验证集: {len(val_idx)} 样本")

        # 创建子数据集
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        # 创建模型
        num_classes = 3 if TASK_TYPE == "classification" else 1
        model = get_model(model_type=MODEL_TYPE, num_classes=num_classes, num_constitutions=38).to(device)
        
        # 训练
        fold_start_time = time.time()
        best_metric, best_epoch = train_fold(model, train_loader, val_loader, device, fold, TASK_TYPE)
        fold_time = time.time() - fold_start_time
        
        fold_results.append(best_metric)
        
        metric_name = "Acc" if TASK_TYPE == "classification" else "MAE"
        print(f"  最佳 {metric_name}: {best_metric:.4f} (Epoch {best_epoch})")
        print(f"  训练时间: {fold_time:.1f}秒")
    
    # 统计结果
    total_time = time.time() - total_start_time
    fold_results = np.array(fold_results)
    mean_metric = np.mean(fold_results)
    std_metric = np.std(fold_results)
    
    print("\n" + "=" * 60)
    print("📊 交叉验证结果汇总")
    print("=" * 60)
    
    metric_name = "Accuracy" if TASK_TYPE == "classification" else "MAE"
    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: {result:.4f}")
    
    print(f"\n平均 {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")
    print(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print("=" * 60)
    
    # 保存结果
    results = {
        "model_type": MODEL_TYPE,
        "task_type": TASK_TYPE,
        "n_folds": N_FOLDS,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
        "fold_results": fold_results.tolist(),
        f"mean_{metric_name.lower()}": float(mean_metric),
        f"std_{metric_name.lower()}": float(std_metric),
        "total_time_sec": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_dir = f"experiment/results/k_fold_{MODEL_TYPE}_{TASK_TYPE}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_dir}/results.json")


if __name__ == "__main__":
    main()