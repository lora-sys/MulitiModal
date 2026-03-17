"""
统一实验运行脚本
支持三个 Baseline 的对比实验，自动记录实验结果

使用方法:
    python run_experiments.py --baseline A --runs 3
    python run_experiments.py --baseline all --runs 3
    python run_experiments.py --baseline C --seed 42 --epochs 30
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
from torch.utils.data import DataLoader, random_split

# 添加路径 - 确保能找到所有模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'dataset'))
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'recorder'))

from recorder import ExperimentRecorder, compute_metrics
from model import get_model
from unified_source import UnifiedNPZDataSource
from unified_dataset import UnifiedMultimodalDataset


# =========================================================================
# 配置
# =========================================================================
BASELINE_CONFIGS = {
    'A': {
        'model_type': 'baseline_a',
        'name': 'Simple Concatenation',
        'fusion_type': 'concat',
        'shared_dim': 64,
        'hidden_dim': 128,
    },
    'B': {
        'model_type': 'baseline_b',
        'name': 'Late Fusion Transformer',
        'fusion_type': 'transformer',
        'shared_dim': 64,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
    },
    'C': {
        'model_type': 'baseline_c',
        'name': 'Cross-Attention Gate Fusion',
        'fusion_type': 'cross_attention',
        'shared_dim': 128,
        'hidden_dim': 256,
    },
}

# 统一训练超参
TRAIN_HYPERPARAMS = {
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'num_workers': 0,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'T_max': 20,
    'eta_min': 1e-6,
}


# =========================================================================
# 数据加载
# =========================================================================
def load_dataset(npz_path):
    """加载数据集"""
    source = UnifiedNPZDataSource(npz_path)
    source.initialize()
    dataset = UnifiedMultimodalDataset(source, preprocessor=None)
    return dataset


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """划分数据集 (70/15/15)"""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    
    return train_set, val_set, test_set


# =========================================================================
# 训练与评估
# =========================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
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
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(dynamic, static_basic, static_scores, constitution)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_preds)
    
    return total_loss / len(dataloader), metrics, all_labels, all_preds


# =========================================================================
# 运行单次实验
# =========================================================================
def run_single_experiment(
    baseline_key: str,
    run_id: str,
    seed: int,
    npz_path: str,
    output_dir: str,
    epochs: int = None,
    verbose: bool = True,
):
    """运行单次实验"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 获取配置
    config = BASELINE_CONFIGS[baseline_key]
    hyperparams = TRAIN_HYPERPARAMS.copy()
    if epochs is not None:
        hyperparams['num_epochs'] = epochs
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建 Recorder
    experiment_id = f"{baseline_key}_{config['fusion_type']}"
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
        print(f"Model: {config['name']}")
        print(f"Device: {device}")
        print(f"{'='*60}")
    
    # 保存配置
    recorder.save_config(
        model=config['name'],
        fusion_type=config['fusion_type'],
        batch_size=hyperparams['batch_size'],
        lr=hyperparams['lr'],
        optimizer=hyperparams['optimizer'],
        weight_decay=hyperparams['weight_decay'],
        num_epochs=hyperparams['num_epochs'],
        num_workers=hyperparams['num_workers'],
        device=str(device),
        scheduler=hyperparams['scheduler'],
    )
    
    # 加载数据
    dataset = load_dataset(npz_path)
    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    
    train_loader = DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hyperparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=hyperparams['batch_size'], shuffle=False)
    
    if verbose:
        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # 创建模型
    model = get_model(
        model_type=config['model_type'],
        num_classes=3,
        num_constitutions=38,
        shared_dim=config['shared_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=0.3,
    )
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hyperparams['T_max'],
        eta_min=hyperparams['eta_min']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    start_time = time.time()
    
    for epoch in range(hyperparams['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        is_best = recorder.log_epoch(
            epoch, train_loss, val_loss,
            val_metrics.accuracy * 100, val_metrics.macro_f1,
            print_log=verbose
        )
        
        if is_best:
            recorder.save_checkpoint(model, optimizer, epoch, is_best=True)
        
        scheduler.step()
    
    train_time = (time.time() - start_time) / 60  # 分钟
    
    # 测试评估
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    # 保存结果
    recorder.save_result(test_metrics, train_time)
    recorder.save_confusion_matrix(y_true, y_pred)
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


# =========================================================================
# 运行多次实验并汇总
# =========================================================================
def run_baseline_experiments(
    baseline_key: str,
    num_runs: int = 3,
    base_seed: int = 42,
    npz_path: str = None,
    output_dir: str = None,
    epochs: int = None,
):
    """运行多次实验取均值"""
    if npz_path is None:
        npz_path = "experiment/model/unified_dataset_realonly.npz"
    if output_dir is None:
        output_dir = "experiment/results"
    
    results = []
    
    for run_idx in range(num_runs):
        run_id = f"r{run_idx + 1}"
        seed = base_seed + run_idx * 1000  # 不同 seed
        
        result = run_single_experiment(
            baseline_key=baseline_key,
            run_id=run_id,
            seed=seed,
            npz_path=npz_path,
            output_dir=output_dir,
            epochs=epochs,
        )
        results.append(result)
    
    # 计算统计
    summary = {
        'baseline': baseline_key,
        'num_runs': num_runs,
        'val_acc_mean': np.mean([r['val_accuracy'] for r in results]),
        'val_acc_std': np.std([r['val_accuracy'] for r in results]),
        'test_acc_mean': np.mean([r['test_accuracy'] for r in results]),
        'test_acc_std': np.std([r['test_accuracy'] for r in results]),
        'test_f1_mean': np.mean([r['test_f1'] for r in results]),
        'test_f1_std': np.std([r['test_f1'] for r in results]),
        'train_time_mean': np.mean([r['train_time_min'] for r in results]),
    }
    
    print(f"\n{'='*60}")
    print(f"Baseline {baseline_key} Summary ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"Val Acc:  {summary['val_acc_mean']:.2f} ± {summary['val_acc_std']:.2f}")
    print(f"Test Acc: {summary['test_acc_mean']:.2f} ± {summary['test_acc_std']:.2f}")
    print(f"Test F1:  {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
    print(f"Time:     {summary['train_time_mean']:.1f} min")
    
    return summary


# =========================================================================
# 主函数
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='运行多模态融合实验')
    parser.add_argument('--baseline', type=str, default='A',
                        choices=['A', 'B', 'C', 'all'],
                        help='选择 baseline (A/B/C/all)')
    parser.add_argument('--runs', type=int, default=3,
                        help='每个 baseline 运行次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='基础随机种子')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练 epoch 数 (None 使用默认值)')
    parser.add_argument('--npz_path', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='experiment/results',
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    baselines = ['A', 'B', 'C'] if args.baseline == 'all' else [args.baseline]
    
    all_summaries = []
    
    for baseline in baselines:
        summary = run_baseline_experiments(
            baseline_key=baseline,
            num_runs=args.runs,
            base_seed=args.seed,
            npz_path=args.npz_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
        all_summaries.append(summary)
    
    # 打印对比表
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("Baseline Comparison Summary")
        print(f"{'='*70}")
        print(f"{'Baseline':<10} {'Test Acc':<20} {'Test F1':<20} {'Time':<10}")
        print(f"{'-'*60}")
        for s in all_summaries:
            acc_str = f"{s['test_acc_mean']:.2f} ± {s['test_acc_std']:.2f}"
            f1_str = f"{s['test_f1_mean']:.4f} ± {s['test_f1_std']:.4f}"
            time_str = f"{s['train_time_mean']:.1f} min"
            print(f"{s['baseline']:<10} {acc_str:<20} {f1_str:<20} {time_str:<10}")


if __name__ == "__main__":
    main()
