"""
回归任务综合实时预测测试脚本
测试6个模型在干净和噪声条件下的实时预测性能
"""

import numpy as np
import torch
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import get_model

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoiseInjector:
    """噪声注入器"""
    
    @staticmethod
    def add_gaussian(waveform, std=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(waveform) * std
        return waveform + noise
    
    @staticmethod
    def add_drift(waveform, max_drift=0.05):
        """添加漂移噪声"""
        batch_size, channels, seq_len = waveform.shape
        drift_factor = torch.linspace(0, max_drift, seq_len, device=waveform.device)
        return waveform * (1 + drift_factor).view(1, 1, -1)
    
    @staticmethod
    def add_dropout(waveform, dropout_prob=0.1):
        """添加丢失噪声"""
        batch_size, channels, seq_len = waveform.shape
        mask = torch.rand(batch_size, channels, seq_len, device=waveform.device) > dropout_prob
        return waveform * mask.float()


class RealTimeTester:
    """实时预测测试器"""
    
    def __init__(self, model_type, model_path, num_constitutions=39):
        """初始化测试器"""
        self.model_type = model_type
        self.model_path = model_path
        
        print(f"[*] 加载模型: {model_type}")
        print(f"    权重路径: {model_path}")
        
        # 加载模型
        self.model = get_model(
            model_type=model_type,
            num_classes=1,
            num_constitutions=num_constitutions
        )
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        print(f"    设备: {DEVICE}")
    
    def test_clean(self, test_loader):
        """干净数据测试"""
        print(f"\n[*] 干净数据测试...")
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                dynamic, static_basic, static_scores, constitution, labels = batch
                
                # 转移到设备
                dynamic = dynamic.to(DEVICE)
                static_basic = static_basic.to(DEVICE)
                static_scores = static_scores.to(DEVICE)
                constitution = constitution.to(DEVICE)
                
                # 预测
                outputs = self.model(dynamic, static_basic, static_scores, constitution)
                
                # 记录
                preds = outputs.squeeze().cpu().numpy()
                trues = labels.squeeze().cpu().numpy()
                
                predictions.extend(preds)
                targets.extend(trues)
        
        return np.array(predictions), np.array(targets)
    
    def test_noisy(self, test_loader, noise_types=['gaussian', 'drift', 'dropout']):
        """噪声数据测试"""
        print(f"\n[*] 噪声数据测试...")
        print(f"    噪声类型: {', '.join(noise_types)}")
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                dynamic, static_basic, static_scores, constitution, labels = batch
                
                # 转移到设备
                dynamic = dynamic.to(DEVICE)
                static_basic = static_basic.to(DEVICE)
                static_scores = static_scores.to(DEVICE)
                constitution = constitution.to(DEVICE)
                
                # 注入噪声
                if 'gaussian' in noise_types:
                    dynamic = NoiseInjector.add_gaussian(dynamic, std=0.1)
                if 'drift' in noise_types:
                    dynamic = NoiseInjector.add_drift(dynamic, max_drift=0.05)
                if 'dropout' in noise_types:
                    dynamic = NoiseInjector.add_dropout(dynamic, dropout_prob=0.1)
                
                # 预测
                outputs = self.model(dynamic, static_basic, static_scores, constitution)
                
                # 记录
                preds = outputs.squeeze().cpu().numpy()
                trues = labels.squeeze().cpu().numpy()
                
                predictions.extend(preds)
                targets.extend(trues)
        
        return np.array(predictions), np.array(targets)
    
    def calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        accuracy_5 = np.sum(np.abs(predictions - targets) < 5) / len(predictions) * 100
        accuracy_3 = np.sum(np.abs(predictions - targets) < 3) / len(predictions) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson': float(correlation),
            'accuracy_5': float(accuracy_5),
            'accuracy_3': float(accuracy_3)
        }


def load_test_data():
    """加载测试数据"""
    print("\n[*] 加载测试数据...")
    data_path = '/home/lora/repos/MulitiModal/experiment/model/unified_dataset_regression.npz'
    data = np.load(data_path)
    
    total_samples = len(data['label'])
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_start = train_size + val_size
    
    # 提取测试集
    test_dynamic = torch.from_numpy(data['dynamic'][test_start:]).float()
    test_static_basic = torch.from_numpy(data['static_basic'][test_start:]).float()
    test_static_scores = torch.from_numpy(data['static_scores'][test_start:]).float()
    test_constitution = torch.from_numpy(data['constitution'][test_start:]).long()
    test_labels = torch.from_numpy(data['label_original'][test_start:]).float()
    
    test_dataset = TensorDataset(test_dynamic, test_static_basic, test_static_scores, 
                              test_constitution, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"    测试集大小: {len(test_dataset)}")
    
    return test_loader


def test_all_models():
    """测试所有模型"""
    print("=" * 70)
    print("综合实时预测测试 - 6个模型 × 2种条件")
    print("=" * 70)
    
    # 加载测试数据
    test_loader = load_test_data()
    
    # 定义模型配置
    models_config = [
        {
            'name': 'baseline_a_clean',
            'type': 'baseline_a',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_a_clean/r1/checkpoints/best_model.pth'
        },
        {
            'name': 'baseline_a_noisy',
            'type': 'baseline_a',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_a_noisy/r1/checkpoints/best_model.pth'
        },
        {
            'name': 'baseline_b_clean',
            'type': 'baseline_b',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_b_clean/r1/checkpoints/best_model.pth'
        },
        {
            'name': 'baseline_b_noisy',
            'type': 'baseline_b',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_b_noisy/r1/checkpoints/best_model.pth'
        },
        {
            'name': 'baseline_c_clean',
            'type': 'baseline_c',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_c_clean/r1/checkpoints/best_model.pth'
        },
        {
            'name': 'baseline_c_noisy',
            'type': 'baseline_c',
            'path': '/home/lora/repos/MulitiModal/experiment/results/regression_c_noisy/r1/checkpoints/best_model.pth'
        },
    ]
    
    # 测试结果存储
    results = {}
    
    # 测试每个模型
    for config in models_config:
        print(f"\n{'=' * 70}")
        print(f"测试模型: {config['name']}")
        print('=' * 70)
        
        # 创建测试器
        tester = RealTimeTester(config['type'], config['path'])
        
        # 干净测试
        pred_clean, target_clean = tester.test_clean(test_loader)
        metrics_clean = tester.calculate_metrics(pred_clean, target_clean)
        
        # 噪声测试
        pred_noisy, target_noisy = tester.test_noisy(test_loader)
        metrics_noisy = tester.calculate_metrics(pred_noisy, target_noisy)
        
        # 存储结果
        results[config['name']] = {
            'clean': {
                'predictions': pred_clean,
                'targets': target_clean,
                'metrics': metrics_clean
            },
            'noisy': {
                'predictions': pred_noisy,
                'targets': target_noisy,
                'metrics': metrics_noisy
            }
        }
        
        # 打印结果
        print(f"\n  干净数据:")
        print(f"    MAE: {metrics_clean['mae']:.4f}")
        print(f"    RMSE: {metrics_clean['rmse']:.4f}")
        print(f"    R²: {metrics_clean['r2']:.4f}")
        print(f"    Pearson: {metrics_clean['pearson']:.4f}")
        print(f"    准确率(<5分): {metrics_clean['accuracy_5']:.2f}%")
        
        print(f"\n  噪声数据:")
        print(f"    MAE: {metrics_noisy['mae']:.4f}")
        print(f"    RMSE: {metrics_noisy['rmse']:.4f}")
        print(f"    R²: {metrics_noisy['r2']:.4f}")
        print(f"    Pearson: {metrics_noisy['pearson']:.4f}")
        print(f"    准确率(<5分): {metrics_noisy['accuracy_5']:.2f}%")
    
    # 生成对比图表
    output_dir = Path('/home/lora/repos/MulitiModal/experiment/results/realtime')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("生成对比图表")
    print('=' * 70)
    
    # 1. 干净 vs 噪声 MAE对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    clean_mae = [results[m]['clean']['metrics']['mae'] for m in models]
    noisy_mae = [results[m]['noisy']['metrics']['mae'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_mae, width, label='Clean Data', color='#4ecdc4')
    bars2 = ax.bar(x + width/2, noisy_mae, width, label='Noisy Data', color='#ff6b6b')
    
    ax.set_ylabel('MAE (Lower is better)', fontsize=12)
    ax.set_title('Clean vs Noisy Data: MAE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clean_vs_noisy_mae_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'clean_vs_noisy_mae_comparison.png'}")
    plt.close()
    
    # 2. 干净 vs 噪声 R²对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    clean_r2 = [results[m]['clean']['metrics']['r2'] for m in models]
    noisy_r2 = [results[m]['noisy']['metrics']['r2'] for m in models]
    
    bars1 = ax.bar(x - width/2, clean_r2, width, label='Clean Data', color='#4ecdc4')
    bars2 = ax.bar(x + width/2, noisy_r2, width, label='Noisy Data', color='#ff6b6b')
    
    ax.set_ylabel('R² (Higher is better)', fontsize=12)
    ax.set_title('Clean vs Noisy Data: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clean_vs_noisy_r2_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'clean_vs_noisy_r2_comparison.png'}")
    plt.close()
    
    # 3. 每个模型的预测精度对比（生成6张子图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        
        pred_clean = results[model_name]['clean']['predictions']
        target_clean = results[model_name]['clean']['targets']
        pred_noisy = results[model_name]['noisy']['predictions']
        target_noisy = results[model_name]['noisy']['targets']
        
        # 绘制散点图
        ax.scatter(target_clean, pred_clean, alpha=0.5, s=10, color='#4ecdc4', label='Clean')
        ax.scatter(target_noisy, pred_noisy, alpha=0.5, s=10, color='#ff6b6b', label='Noisy')
        ax.plot([30, 100], [30, 100], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(30, 100)
        ax.set_ylim(30, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'all_models_prediction_comparison.png'}")
    plt.close()
    
    # 4. 汇总表格
    print(f"\n{'=' * 70}")
    print("结果汇总")
    print('=' * 70)
    
    print(f"\n{'Model':<20} {'Clean MAE':<12} {'Noisy MAE':<12} {'MAE Change':<12} {'Clean R²':<12} {'Noisy R²':<12}")
    print('-' * 70)
    
    for model_name in models:
        clean_mae = results[model_name]['clean']['metrics']['mae']
        noisy_mae = results[model_name]['noisy']['metrics']['mae']
        mae_change = (noisy_mae - clean_mae) / clean_mae * 100
        clean_r2 = results[model_name]['clean']['metrics']['r2']
        noisy_r2 = results[model_name]['noisy']['metrics']['r2']
        
        print(f"{model_name:<20} {clean_mae:.4f}        {noisy_mae:.4f}        {mae_change:+.2f}%      {clean_r2:.4f}      {noisy_r2:.4f}")
    
    # 保存JSON结果
    summary = {
        model_name: {
            'clean': results[model_name]['clean']['metrics'],
            'noisy': results[model_name]['noisy']['metrics']
        }
        for model_name in models
    }
    
    with open(output_dir / 'realtime_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[*] 结果已保存: {output_dir / 'realtime_test_summary.json'}")
    print(f"[*] 图表已保存: {output_dir}")
    
    print(f"\n{'=' * 70}")
    print("✅ 综合实时预测测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_all_models()