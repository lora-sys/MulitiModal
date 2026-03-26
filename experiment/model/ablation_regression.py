#!/usr/bin/env python3
"""回归任务消融实验 - 验证各模态对回归性能的贡献度"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import argparse
import json
import time
import gc
import sys

sys.path.insert(0, 'experiment/model')
from model import get_model

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 配置
NUM_EPOCHS = 3  # 减少epoch以加快速度
MODEL_TYPE = "baseline_c"
batch_size = 16
num_workers = 0


class RegressionDataset(torch.utils.data.Dataset):
    """回归数据集包装器 - 优化版：预先转换为tensor"""
    def __init__(self, npz_path, ablation_config="full"):
        data = np.load(npz_path)
        self.n_samples = len(data['label'])
        self.ablation_config = ablation_config

        # 预先将所有数据转换为tensor
        self.dynamic = torch.from_numpy(data['dynamic']).float()
        self.static_basic = torch.from_numpy(data['static_basic']).float()
        self.static_scores = torch.from_numpy(data['static_scores']).float()
        self.constitution = torch.from_numpy(data['constitution']).long()
        self.label = torch.from_numpy(data['label']).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        result = {
            'dynamic': self.dynamic[idx],
            'static_basic': self.static_basic[idx],
            'static_scores': self.static_scores[idx],
            'constitution': self.constitution[idx],
            'label': self.label[idx]
        }

        # 消融处理
        if self.ablation_config == "no_dynamic":
            # 去掉动态波形：置零
            result['dynamic'] = torch.zeros_like(result['dynamic'])
        elif self.ablation_config == "no_static_basic":
            # 去掉身体特征：置零
            result['static_basic'] = torch.zeros_like(result['static_basic'])
        elif self.ablation_config == "no_static_scores":
            # 去掉舌面诊：置零
            result['static_scores'] = torch.zeros_like(result['static_scores'])
        elif self.ablation_config == "no_constitution":
            # 去掉体质：设为0（第一个体质）
            # 注意：这种方法使用索引0的embedding，不是完全消除
            # 更好的方法是修改ConstitutionEmbedding来返回零向量
            result['constitution'] = torch.zeros_like(result['constitution'], dtype=torch.long)

        return result


def run_ablation(ablation_config):
    """运行单个消融配置的单次训练验证"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 加载数据集
    npz_path = "experiment/model/unified_dataset_regression.npz"
    dataset = RegressionDataset(npz_path, ablation_config=ablation_config)
    print(f"[*] 加载数据集: {len(dataset)} 样本")
    print(f"[*] 消融配置: {ablation_config}")

    # 数据划分：80%训练，20%验证（使用shuffle确保分布一致）
    n_samples = len(dataset)
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size

    # 生成随机排列的索引
    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(n_samples)
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")

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

    print(f"\n[*] 开始训练 ({NUM_EPOCHS} epochs)")
    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        batch_count = 0

        for batch in train_loader:
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
                predictions.extend(outputs.detach().cpu().numpy().ravel())
                targets.extend(labels.detach().cpu().numpy().ravel())

        # 计算 MAE
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))

        # 更新最佳模型
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch

        avg_train_loss = train_loss / batch_count
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val MAE: {mae:.4f}, Best: {best_mae:.4f} (Epoch {best_epoch})")

    total_time = time.time() - total_start_time

    print(f"\n{'='*60}")
    print(f"消融配置: {ablation_config}")
    print(f"最佳 MAE: {best_mae:.4f} (Epoch {best_epoch})")
    print(f"总训练时间: {total_time:.1f}秒")
    print(f"{'='*60}")

    # 清理内存
    del model, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()

    # 返回结果
    return {
        "config": ablation_config,
        "best_mae": float(best_mae),
        "best_epoch": best_epoch,
        "total_time": total_time
    }


def save_results(result, results_file):
    """保存结果到JSON文件"""
    if results_file == "none":
        return

    # 读取现有结果
    if results_file and results_file != "none":
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[警告] 无法读取结果文件: {e}")
            results = {
                "model_type": "baseline_c",
                "task_type": "regression_ablation",
                "ablations": []
            }
    else:
        results = {
            "model_type": "baseline_c",
            "task_type": "regression_ablation",
            "ablations": []
        }

    # 添加当前结果
    results["ablations"].append(result)

    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='回归任务消融实验')
    parser.add_argument('--model', type=str, default='baseline_c',
                        choices=['baseline_a', 'baseline_b', 'baseline_c'],
                        help='模型类型')
    parser.add_argument('--config', type=str, required=True, 
                        choices=['full', 'no_dynamic', 'no_static_basic', 'no_static_scores', 'no_constitution'],
                        help='消融配置')
    parser.add_argument('--results_file', type=str, 
                        default='experiment/results/k_fold_baseline_c_regression_ablation/results.json',
                        help='结果文件路径')
    args = parser.parse_args()

    # 使用命令行参数
    MODEL_TYPE = args.model

    print("="*60)
    print("回归任务消融实验")
    print(f"模型: {MODEL_TYPE}")
    print(f"消融配置: {args.config}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*60)

    # 运行消融实验
    result = run_ablation(args.config)

    # 保存结果
    save_results(result, args.results_file)

    print("\n✓ 实验完成！")