#!/usr/bin/env python3
"""训练单个fold的回归任务"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import argparse
import time
import json
import sys
import gc

sys.path.insert(0, 'experiment/model')
from model import get_model

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 配置
NUM_EPOCHS = 5  # 从15减少到5以加快训练
MODEL_TYPE = "baseline_c"
batch_size = 16
num_workers = 0


class RegressionDataset(torch.utils.data.Dataset):
    """回归数据集包装器 - 优化版：预先转换为tensor"""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.n_samples = len(data['label'])
        # 预先将所有数据转换为tensor
        self.dynamic = torch.from_numpy(data['dynamic']).float()
        self.static_basic = torch.from_numpy(data['static_basic']).float()
        self.static_scores = torch.from_numpy(data['static_scores']).float()
        self.constitution = torch.from_numpy(data['constitution']).long()
        self.label = torch.from_numpy(data['label']).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'dynamic': self.dynamic[idx],
            'static_basic': self.static_basic[idx],
            'static_scores': self.static_scores[idx],
            'constitution': self.constitution[idx],
            'label': self.label[idx]
        }


def train_single_fold(fold_num):
    """训练单个fold"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 加载数据集
    npz_path = "experiment/model/unified_dataset_regression.npz"
    dataset = RegressionDataset(npz_path)
    print(f"[*] 加载数据集: {len(dataset)} 样本")

    # 创建 K-Fold 分割
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_indices = np.arange(len(dataset))

    # 获取指定fold的分割
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        if fold == fold_num:
            break

    print(f"\n📦 Fold {fold_num + 1}/5")
    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(val_idx)} 样本")

    # 创建子数据集
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 创建模型
    model = get_model(model_type=MODEL_TYPE, num_classes=1, num_constitutions=39).to(device)

    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    best_mae = float('inf')
    best_epoch = 0

    fold_start_time = time.time()

    print(f"[*] 开始训练，共 {NUM_EPOCHS} 个epoch")
    print(f"[*] 训练集batch数: {len(train_loader)}")
    print(f"[*] 验证集batch数: {len(val_loader)}")

    for epoch in range(NUM_EPOCHS):
        print(f"\n  -> Epoch {epoch+1}/{NUM_EPOCHS} 开始")
        # 训练
        model.train()
        train_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(dynamic, static_basic, static_scores, constitution)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # 每10个batch输出一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"    Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / batch_count
        print(f"  -> Epoch {epoch+1} 训练完成, 平均Loss: {avg_train_loss:.4f}")

        scheduler.step()

        # 验证
        model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch['dynamic'].to(device)
                static_basic = batch['static_basic'].to(device)
                static_scores = batch['static_scores'].to(device)
                constitution = batch['constitution'].to(device)
                labels = batch['label'].to(device)

                outputs = model(dynamic, static_basic, static_scores, constitution)
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(labels.squeeze().cpu().numpy())

        # 计算 MAE
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))

        # 更新最佳模型
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, MAE: {mae:.4f}, Best: {best_mae:.4f} (Epoch {best_epoch})")

    fold_time = time.time() - fold_start_time

    print(f"\n✓ Fold {fold_num + 1} 完成")
    print(f"  最佳 MAE: {best_mae:.4f} (Epoch {best_epoch})")
    print(f"  训练时间: {fold_time:.1f}秒")

    # 清理内存
    del model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()

    # 保存结果到JSON
    results_file = "experiment/results/k_fold_baseline_c_regression/results.json"

    # 读取现有结果
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "model_type": "baseline_c",
            "task_type": "regression",
            "n_folds": 5,
            "num_epochs": 15,
            "random_seed": 42,
            "fold_results": []
        }

    # 添加当前fold结果
    results["fold_results"].append({
        "fold": fold_num + 1,
        "best_mae": float(best_mae),
        "best_epoch": best_epoch,
        "training_time": fold_time
    })

    # 如果所有fold都完成了，计算统计结果
    if len(results["fold_results"]) == 5:
        fold_maes = [r["best_mae"] for r in results["fold_results"]]
        results["mean_mae"] = float(np.mean(fold_maes))
        results["std_mae"] = float(np.std(fold_maes))
        results["total_time"] = float(sum(r["training_time"] for r in results["fold_results"]))
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return best_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练单个fold')
    parser.add_argument('--fold', type=int, required=True, help='Fold编号 (0-4)')
    args = parser.parse_args()

    if args.fold < 0 or args.fold >= 5:
        print("错误: fold编号必须在0-4之间")
        sys.exit(1)

    import os
    train_single_fold(args.fold)