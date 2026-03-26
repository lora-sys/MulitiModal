"""
5-Fold 交叉验证脚本 - 回归任务 (修复版)
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
from sklearn.model_selection import KFold
from datetime import datetime

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'recorder'))

from unified_source import UnifiedNPZDataSource
from model import get_model

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 配置（减少epoch以加快速度）
N_FOLDS = 5
NUM_EPOCHS = 15  # 加速版
MODEL_TYPE = "baseline_c"


class RegressionDataset(torch.utils.data.Dataset):
    """回归数据集包装器"""
    def __init__(self, npz_path):
        self.data = np.load(npz_path)
        self.n_samples = len(self.data['label'])
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Constitution值范围: 0-38 (共39个值)
        # ConstitutionEmbedding 期望整数索引（LongTensor），形状为 (B,)
        constitution_idx = int(self.data['constitution'][idx])

        return {
            'dynamic': torch.FloatTensor(self.data['dynamic'][idx]),
            'static_basic': torch.FloatTensor(self.data['static_basic'][idx]),
            'static_scores': torch.FloatTensor(self.data['static_scores'][idx]),
            'constitution': torch.LongTensor([constitution_idx]),
            'label': torch.FloatTensor([self.data['label'][idx, 0]])
        }


def load_dataset():
    """加载数据集"""
    npz_path = "experiment/model/unified_dataset_regression.npz"
    dataset = RegressionDataset(npz_path)
    print(f"[*] 加载数据集: {len(dataset)} 样本")
    return dataset


def train_fold(model, train_loader, val_loader, device, fold_num):
    """训练单个 fold"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        
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
    
    return best_mae, best_epoch


def main():
    """主函数"""
    print("=" * 60)
    print(f"🚀 {N_FOLDS}-Fold 交叉验证 (回归任务 - 修复版)")
    print(f"模型: {MODEL_TYPE}")
    print(f"Epochs: {NUM_EPOCHS} (加速版)")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    
    # 加载数据集
    dataset = load_dataset()
    
    # 创建 K-Fold 分割
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    total_start_time = time.time()
    
    print(f"\n开始交叉验证...")
    print("-" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n📦 Fold {fold + 1}/{N_FOLDS}")
        print(f"  训练集: {len(train_idx)} 样本")
        print(f"  验证集: {len(val_idx)} 样本")
        
        # 创建子数据集
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # 创建模型
        model = get_model(model_type=MODEL_TYPE, num_classes=1, num_constitutions=39).to(device)
        
        # 训练
        fold_start_time = time.time()
        best_mae, best_epoch = train_fold(model, train_loader, val_loader, device, fold)
        fold_time = time.time() - fold_start_time
        
        fold_results.append(best_mae)
        
        print(f"  最佳 MAE: {best_mae:.4f} (Epoch {best_epoch})")
        print(f"  训练时间: {fold_time:.1f}秒")
    
    # 统计结果
    total_time = time.time() - total_start_time
    fold_results = np.array(fold_results)
    mean_mae = np.mean(fold_results)
    std_mae = np.std(fold_results)
    
    print("\n" + "=" * 60)
    print("📊 交叉验证结果汇总")
    print("=" * 60)
    
    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: {result:.4f}")
    
    print(f"\n平均 MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print("=" * 60)
    
    # 保存结果
    results = {
        "model_type": MODEL_TYPE,
        "task_type": "regression",
        "n_folds": N_FOLDS,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
        "fold_results": fold_results.tolist(),
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "total_time_sec": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_dir = f"experiment/results/k_fold_{MODEL_TYPE}_regression"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_dir}/results.json")


if __name__ == "__main__":
    main()