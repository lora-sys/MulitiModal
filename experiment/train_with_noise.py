"""
噪声注入训练脚本 - 保存模型权重
对3个模型进行噪声注入训练并保存最佳模型
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'dataset'))
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'recorder'))

from unified_source import UnifiedNPZDataSource
from unified_dataset import UnifiedMultimodalDataset
from model import get_model
from recorder import ExperimentRecorder, compute_metrics


class NoisyDataset:
    """噪声注入数据集"""
    
    def __init__(self, base_dataset, noise_types, noise_prob=0.5, seed=42):
        """
        Args:
            base_dataset: 基础数据集
            noise_types: 噪声类型列表
            noise_prob: 添加噪声的概率
            seed: 随机种子
        """
        self.base_dataset = base_dataset
        self.noise_types = noise_types
        self.noise_prob = noise_prob
        np.random.seed(seed)
        
        # 生成噪声掩码
        n = len(base_dataset)
        n_noisy = int(n * noise_prob)
        self.noisy_indices = np.random.choice(n, n_noisy, replace=False)
        
        # 为每个噪声样本分配噪声类型
        self.noise_type_map = {}
        for idx in self.noisy_indices:
            self.noise_type_map[idx] = np.random.choice(noise_types)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # 如果是噪声样本，添加噪声
        if idx in self.noisy_indices:
            noise_type = self.noise_type_map[idx]
            item['dynamic'] = self._add_noise(item['dynamic'], noise_type)
        
        return item
    
    def _add_noise(self, signal, noise_type):
        """添加噪声到动态特征"""
        signal_np = signal.numpy()
        
        if noise_type == 'baseline':
            # 基线偏移 ±5%
            offset = np.random.uniform(-0.05, 0.05) * np.abs(signal_np).max()
            signal_np = signal_np + offset
        
        elif noise_type == 'gaussian':
            # 高斯噪声 SNR=30-40dB
            signal_power = np.mean(signal_np ** 2)
            noise_power = signal_power / (10 ** (np.random.uniform(30, 40) / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), signal_np.shape)
            signal_np = signal_np + noise
        
        elif noise_type == 'amplitude':
            # 幅度缩放 ±10-15%
            scale = np.random.uniform(0.85, 1.15)
            signal_np = signal_np * scale
        
        elif noise_type == 'motion':
            # 低频运动伪影 <200ms
            n_samples = signal_np.shape[-1]
            artifact_length = int(np.random.uniform(10, 100))
            start_idx = np.random.randint(0, n_samples - artifact_length)
            
            artifact = np.random.randn(*signal_np.shape) * 0.3
            for i in range(signal_np.shape[0]):
                signal_np[i, start_idx:start_idx+artifact_length] += artifact[i, :artifact_length]
        
        elif noise_type == 'channel_dropout':
            # 随机通道丢失 <10%
            for i in range(signal_np.shape[0]):
                if np.random.random() < 0.1:
                    signal_np[i, :] = 0
        
        return torch.from_numpy(signal_np).float()


def train_with_noise(
    model_type,
    run_id,
    seed,
    npz_path,
    output_dir,
    epochs=20,
    noise_prob=0.5,
    noise_types=['baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout'],
    verbose=True
):
    """噪声注入训练"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 模型配置
    model_configs = {
        'baseline_a': {'shared_dim': 64, 'hidden_dim': 128},
        'baseline_b': {'shared_dim': 64, 'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2},
        'baseline_c': {'shared_dim': 128, 'hidden_dim': 256}
    }
    
    config = model_configs[model_type]
    model_names = {
        'baseline_a': 'Simple Concatenation',
        'baseline_b': 'Late Fusion Transformer',
        'baseline_c': 'Cross-Attention Gate Fusion'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建 Recorder
    experiment_id = f"{model_type}_noise"
    recorder = ExperimentRecorder(
        output_dir=output_dir,
        experiment_id=experiment_id,
        run_id=run_id,
        seed=seed,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"Run: {run_id}, Seed: {seed}")
        print(f"Model: {model_names[model_type]}")
        print(f"Device: {device}")
        print(f"Noise Prob: {noise_prob}")
        print(f"{'='*60}")
    
    # 保存配置
    recorder.save_config(
        model=model_names[model_type],
        fusion_type=model_type,
        batch_size=32,
        lr=0.001,
        optimizer="AdamW",
        weight_decay=0.0001,
        num_epochs=epochs,
        num_workers=0,
        device=str(device),
        scheduler="CosineAnnealingLR",
        notes=f"Noise training with {noise_prob} probability",
    )
    
    # 加载数据
    source = UnifiedNPZDataSource(npz_path)
    source.initialize()
    full_dataset = UnifiedMultimodalDataset(source, preprocessor=None)
    
    # 划分数据集
    n = len(full_dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    # 创建带噪声的训练集
    noisy_train_set = NoisyDataset(train_set, noise_types, noise_prob, seed)
    
    train_loader = DataLoader(noisy_train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    if verbose:
        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # 创建模型
    model = get_model(
        model_type=model_type,
        num_classes=3,
        num_constitutions=38,
        **config,
        dropout=0.3,
    )
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        metrics = compute_metrics(all_labels, all_preds)
        scheduler.step()
        
        # 记录
        is_best = recorder.log_epoch(
            epoch, train_loss, val_loss,
            metrics.accuracy * 100, metrics.macro_f1,
            print_log=verbose
        )
        
        if is_best:
            recorder.save_checkpoint(model, optimizer, epoch, is_best=True)
    
    train_time = (time.time() - start_time) / 60  # 分钟
    
    # 测试评估
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(dynamic, static_basic, static_scores, constitution)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_metrics = compute_metrics(all_labels, all_preds)
    
    # 保存结果
    recorder.save_result(test_metrics, train_time)
    recorder.save_confusion_matrix(all_labels, all_preds)
    recorder.save_training_curves()
    
    if verbose:
        print(f"\n{recorder.get_summary()}")
    
    return {
        'val_accuracy': recorder.best_val_acc,
        'val_f1': recorder.best_val_f1,
        'test_accuracy': test_metrics.accuracy * 100,
        'test_f1': test_metrics.macro_f1,
        'train_time_min': train_time,
    }


def main():
    parser = argparse.ArgumentParser(description='噪声注入训练')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['A', 'B', 'C'],
                        help='选择 baseline (A/B/C)')
    parser.add_argument('--runs', type=int, default=1,
                        help='运行次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--npz_path', type=str, default='experiment/model/unified_dataset_expanded.npz',
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='experiment/results',
                        help='结果输出目录')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练epoch数')
    
    args = parser.parse_args()
    
    # 模型映射
    baseline_map = {
        'A': 'baseline_a',
        'B': 'baseline_b',
        'C': 'baseline_c'
    }
    
    model_type = baseline_map[args.baseline]
    
    # 训练
    result = train_with_noise(
        model_type=model_type,
        run_id='r1',
        seed=args.seed,
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        noise_prob=0.5,
        noise_types=['baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout'],
        verbose=True
    )
    
    print(f"\n✅ {args.baseline} 噪声注入训练完成!")
    print(f"测试准确率: {result['test_accuracy']:.2f}%")
    print(f"测试F1分数: {result['test_f1']:.4f}")


if __name__ == "__main__":
    main()
