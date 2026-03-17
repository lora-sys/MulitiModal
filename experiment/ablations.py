"""
消融实验脚本

支持的消融实验：
1. Ablation 1: 去掉 Cross-Attention Gate (C vs B)
2. Ablation 2: 去掉某一模态 (modality ablation)
3. Ablation 3: 替换 waveform encoder
4. Ablation 4: 标签噪声鲁棒性
5. Ablation 5: 可解释性分析 (attention/gate 统计)
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional, List, Dict

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'dataset'))
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'recorder'))

from recorder import ExperimentRecorder, compute_metrics
from model import get_model, MultiExpertFusionModel, SimpleConcatModel
from unified_source import UnifiedNPZDataSource
from unified_dataset import UnifiedMultimodalDataset


# =========================================================================
# Ablation 2: 模态消融 - 包装 Dataset
# =========================================================================
class ModalityAblationDataset(Dataset):
    """模态消融数据集 - 隐藏指定模态"""
    
    def __init__(self, base_dataset, drop_modality: Optional[str] = None):
        """
        Args:
            base_dataset: 原始数据集
            drop_modality: 要丢弃的模态 ('dynamic', 'static_basic', 'static_scores', 'constitution', None)
        """
        self.base_dataset = base_dataset
        self.drop_modality = drop_modality
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if self.drop_modality == 'dynamic':
            item['dynamic'] = torch.zeros_like(item['dynamic'])
        elif self.drop_modality == 'static_basic':
            item['static_basic'] = torch.zeros_like(item['static_basic'])
        elif self.drop_modality == 'static_scores':
            item['static_scores'] = torch.zeros_like(item['static_scores'])
        elif self.drop_modality == 'constitution':
            item['constitution'] = torch.zeros_like(item['constitution'])
            
        return item


# =========================================================================
# Ablation 4: 标签噪声注入
# =========================================================================
class NoisyLabelDataset(Dataset):
    """标签噪声数据集"""
    
    def __init__(self, base_dataset, noise_ratio: float = 0.1, seed: int = 42):
        """
        Args:
            base_dataset: 原始数据集
            noise_ratio: 噪声比例 (0.1 = 10%)
            seed: 随机种子
        """
        self.base_dataset = base_dataset
        self.noise_ratio = noise_ratio
        
        # 生成噪声标签
        np.random.seed(seed)
        n = len(base_dataset)
        n_noisy = int(n * noise_ratio)
        noisy_indices = np.random.choice(n, n_noisy, replace=False)
        
        self.noisy_labels = {}
        for idx in noisy_indices:
            original_label = base_dataset[idx]['label'].item()
            # 随机选择其他标签
            other_labels = [l for l in range(3) if l != original_label]
            self.noisy_labels[idx] = np.random.choice(other_labels)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if idx in self.noisy_labels:
            item['label'] = torch.tensor(self.noisy_labels[idx], dtype=torch.long)
            
        return item


# =========================================================================
# 可替换的 Waveform Encoder
# =========================================================================
class SimpleConvEncoder(nn.Module):
    """简单的 1D Conv 编码器"""
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_dim, 3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # x: (B, 2, 1000)
        x = self.conv(x)  # (B, out_dim, T)
        return x.mean(dim=-1)  # GAP -> (B, out_dim)


class TCNEncoder(nn.Module):
    """TCN 风格编码器"""
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=8, dilation=4),
            nn.ReLU(),
            nn.Conv1d(128, out_dim, 3, padding=16, dilation=8),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x.mean(dim=-1)


# =========================================================================
# Ablation 模型变体
# =========================================================================
class ModalityAblationModel(nn.Module):
    """模态消融模型 - 继承 Baseline C，支持忽略特定模态"""
    
    def __init__(self, base_model, ignore_modality: Optional[str] = None):
        super().__init__()
        self.base_model = base_model
        self.ignore_modality = ignore_modality
        
    def forward(self, dynamic, static_basic, static_scores, constitution):
        if self.ignore_modality == 'dynamic':
            dynamic = torch.zeros_like(dynamic)
        elif self.ignore_modality == 'static_basic':
            static_basic = torch.zeros_like(static_basic)
        elif self.ignore_modality == 'static_scores':
            static_scores = torch.zeros_like(static_scores)
        elif self.ignore_modality == 'constitution':
            constitution = torch.zeros_like(constitution)
            
        return self.base_model(dynamic, static_basic, static_scores, constitution)


# =========================================================================
# 工具函数
# =========================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def load_data(npz_path, seed=42):
    source = UnifiedNPZDataSource(npz_path)
    source.initialize()
    dataset = UnifiedMultimodalDataset(source, preprocessor=None)
    
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def train_and_eval(model, train_loader, val_loader, test_loader, device, epochs=30):
    """训练并评估"""
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_test_metrics = None
    
    for epoch in range(epochs):
        # Train
        model.train()
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
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch['dynamic'].to(device)
                static_basic = batch['static_basic'].to(device)
                static_scores = batch['static_scores'].to(device)
                constitution = batch['constitution'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(dynamic, static_basic, static_scores, constitution)
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
        
        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            
            # Test evaluation
            test_preds, test_labels = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    dynamic = batch['dynamic'].to(device)
                    static_basic = batch['static_basic'].to(device)
                    static_scores = batch['static_scores'].to(device)
                    constitution = batch['constitution'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                    _, predicted = outputs.max(1)
                    
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            best_test_metrics = compute_metrics(np.array(test_labels), np.array(test_preds))
    
    return best_test_metrics


# =========================================================================
# 消融实验运行器
# =========================================================================
def run_ablation_2_modality(npz_path, output_dir, base_seed=42):
    """Ablation 2: 模态消融"""
    print("\n" + "="*60)
    print("Ablation 2: Modality Ablation")
    print("="*60)
    
    modalities = [None, 'dynamic', 'static_basic', 'static_scores', 'constitution']
    results = {}
    
    for modality in modalities:
        set_seed(base_seed)
        
        name = "full" if modality is None else f"no_{modality}"
        print(f"\n--- Drop: {name} ---")
        
        train_set, val_set, test_set = load_data(npz_path, seed=base_seed)
        
        if modality is not None:
            train_set = ModalityAblationDataset(train_set, modality)
            val_set = ModalityAblationDataset(val_set, modality)
            test_set = ModalityAblationDataset(test_set, modality)
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model('baseline_c', num_classes=3).to(device)
        
        metrics = train_and_eval(model, train_loader, val_loader, test_loader, device)
        results[name] = metrics
        
        print(f"Test Acc: {metrics.accuracy*100:.2f}%, F1: {metrics.macro_f1:.4f}")
    
    # 打印对比
    print("\n" + "-"*50)
    print("Modality Ablation Results:")
    print(f"{'Config':<20} {'Acc':>10} {'F1':>10}")
    for name, m in results.items():
        print(f"{name:<20} {m.accuracy*100:>10.2f} {m.macro_f1:>10.4f}")
    
    return results


def run_ablation_4_noise(npz_path, output_dir, base_seed=42):
    """Ablation 4: 标签噪声鲁棒性"""
    print("\n" + "="*60)
    print("Ablation 4: Label Noise Robustness")
    print("="*60)
    
    noise_ratios = [0.0, 0.05, 0.10, 0.15, 0.20]
    baselines = ['baseline_a', 'baseline_b', 'baseline_c']
    results = {}
    
    for noise in noise_ratios:
        results[noise] = {}
        
        for baseline in baselines:
            set_seed(base_seed)
            
            print(f"\n--- Noise: {noise*100:.0f}%, Model: {baseline} ---")
            
            train_set, val_set, test_set = load_data(npz_path, seed=base_seed)
            
            if noise > 0:
                train_set = NoisyLabelDataset(train_set, noise_ratio=noise, seed=base_seed)
            
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=32)
            test_loader = DataLoader(test_set, batch_size=32)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(baseline, num_classes=3).to(device)
            
            metrics = train_and_eval(model, train_loader, val_loader, test_loader, device)
            results[noise][baseline] = metrics
            
            print(f"Test Acc: {metrics.accuracy*100:.2f}%, F1: {metrics.macro_f1:.4f}")
    
    # 打印对比表
    print("\n" + "-"*60)
    print("Noise Robustness Results:")
    print(f"{'Noise':<10} {'A':<15} {'B':<15} {'C':<15}")
    for noise, baselines_dict in results.items():
        a_acc = baselines_dict['baseline_a'].accuracy * 100
        b_acc = baselines_dict['baseline_b'].accuracy * 100
        c_acc = baselines_dict['baseline_c'].accuracy * 100
        print(f"{noise*100:.0f}%{'':<6} {a_acc:<15.2f} {b_acc:<15.2f} {c_acc:<15.2f}")
    
    return results


# =========================================================================
# 主函数
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['modality', 'noise', 'all'],
                        help='消融实验类型')
    parser.add_argument('--npz_path', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='experiment/results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    npz_path = args.npz_path or "experiment/dataset/unified_dataset.npz"
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.ablation in ['modality', 'all']:
        run_ablation_2_modality(npz_path, args.output_dir, args.seed)
    
    if args.ablation in ['noise', 'all']:
        run_ablation_4_noise(npz_path, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
