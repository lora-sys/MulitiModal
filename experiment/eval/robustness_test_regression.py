"""
回归模型鲁棒性测试脚本
测试最佳模型在不同噪声条件下的性能
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys
from torch.utils.data import DataLoader, TensorDataset

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.unified_source import UnifiedNPZDataSource
from dataset.unified_dataset import UnifiedMultimodalDataset
from model.model import get_model


def add_gaussian_noise(waveform, std=0.1):
    """添加高斯噪声"""
    noise = torch.randn_like(waveform) * std
    return waveform + noise


def add_drift_noise(waveform, max_drift=0.05):
    """添加漂移噪声"""
    batch_size, channels, seq_len = waveform.shape
    drift_factor = torch.linspace(0, max_drift, seq_len, device=waveform.device)
    return waveform * (1 + drift_factor).view(1, 1, -1)


def add_dropout_noise(waveform, dropout_prob=0.1):
    """添加数据丢失噪声"""
    batch_size, channels, seq_len = waveform.shape
    mask = torch.rand(batch_size, channels, seq_len, device=waveform.device) > dropout_prob
    return waveform * mask.float()


def evaluate_with_noise(model, test_loader, device, noise_func=None, **noise_kwargs):
    """在指定噪声条件下评估模型"""
    model.eval()
    total_mae = 0
    total_rmse = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # TensorDataset返回的是元组
            dynamic, static_basic, static_scores, constitution, labels = batch
            
            dynamic = dynamic.to(device)
            static_basic = static_basic.to(device)
            static_scores = static_scores.to(device)
            constitution = constitution.to(device)
            labels = labels.to(device)
            
            # 添加噪声
            if noise_func is not None:
                dynamic = noise_func(dynamic, **noise_kwargs)
            
            # 预测
            outputs = model(dynamic, static_basic, static_scores, constitution)
            
            # 计算指标
            mae = torch.mean(torch.abs(outputs.squeeze() - labels.squeeze()))
            rmse = torch.sqrt(torch.mean((outputs.squeeze() - labels.squeeze()) ** 2))
            
            total_mae += mae.item()
            total_rmse += rmse.item()
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.squeeze().cpu().numpy())
    
    n_batches = len(test_loader)
    return {
        'mae': total_mae / n_batches,
        'rmse': total_rmse / n_batches,
        'predictions': np.array(predictions),
        'targets': np.array(targets)
    }


def robustness_test():
    """鲁棒性测试主函数"""
    print("=" * 70)
    print("回归模型鲁棒性测试")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] 使用设备: {device}")
    
    # 加载测试数据
    print("\n[*] 加载测试数据...")
    data_path = '/home/lora/repos/MulitiModal/experiment/model/unified_dataset_regression.npz'
    
    # 直接加载NPZ文件
    data = np.load(data_path)
    
    # 手动创建数据集（回归任务）
    total_samples = len(data['label'])
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_start = train_size + val_size
    
    # 提取测试集数据
    test_dynamic = torch.from_numpy(data['dynamic'][test_start:]).float()
    test_static_basic = torch.from_numpy(data['static_basic'][test_start:]).float()
    test_static_scores = torch.from_numpy(data['static_scores'][test_start:]).float()
    test_constitution = torch.from_numpy(data['constitution'][test_start:]).long()
    test_labels = torch.from_numpy(data['label'][test_start:]).float()
    
    # 创建TensorDataset
    test_dataset = TensorDataset(test_dynamic, test_static_basic, test_static_scores, test_constitution, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"    测试集大小: {len(test_dataset)}")
    
    # 加载最佳模型
    print("\n[*] 加载最佳模型 (baseline_c_clean)...")
    model_path = '/home/lora/repos/MulitiModal/experiment/results/regression_c_clean/r1/checkpoints/best_model.pth'
    model = get_model(model_type='baseline_c', num_classes=1, num_constitutions=39)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"    模型已加载: {model_path}")
    
    # 测试条件
    test_conditions = [
        {'name': '无噪声', 'func': None, 'kwargs': {}},
        {'name': '高斯噪声(std=0.05)', 'func': add_gaussian_noise, 'kwargs': {'std': 0.05}},
        {'name': '高斯噪声(std=0.1)', 'func': add_gaussian_noise, 'kwargs': {'std': 0.1}},
        {'name': '高斯噪声(std=0.2)', 'func': add_gaussian_noise, 'kwargs': {'std': 0.2}},
        {'name': '漂移噪声(0.03)', 'func': add_drift_noise, 'kwargs': {'max_drift': 0.03}},
        {'name': '漂移噪声(0.05)', 'func': add_drift_noise, 'kwargs': {'max_drift': 0.05}},
        {'name': '丢失噪声(5%)', 'func': add_dropout_noise, 'kwargs': {'dropout_prob': 0.05}},
        {'name': '丢失噪声(10%)', 'func': add_dropout_noise, 'kwargs': {'dropout_prob': 0.1}},
        {'name': '丢失噪声(20%)', 'func': add_dropout_noise, 'kwargs': {'dropout_prob': 0.2}},
    ]
    
    # 运行测试
    print("\n[*] 开始鲁棒性测试...")
    results = []
    
    for condition in test_conditions:
        print(f"\n  测试条件: {condition['name']}")
        result = evaluate_with_noise(
            model, test_loader, device,
            condition['func'], **condition['kwargs']
        )
        result['condition'] = condition['name']
        results.append(result)
        print(f"    MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
    
    # 保存结果
    output_dir = Path('/home/lora/repos/MulitiModal/experiment/results/robustness')
    output_dir.mkdir(exist_ok=True)
    
    # 保存JSON结果
    results_json = []
    for result in results:
        results_json.append({
            'condition': result['condition'],
            'mae': result['mae'],
            'rmse': result['rmse']
        })
    
    with open(output_dir / 'robustness_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # 生成可视化
    print("\n[*] Generating robustness test charts...")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE对比
    ax1 = axes[0]
    conditions = [r['condition'] for r in results]
    mae_values = [r['mae'] for r in results]
    colors = ['#4ecdc4' if i == 0 else '#ff6b6b' for i in range(len(conditions))]
    bars = ax1.barh(conditions, mae_values, color=colors)
    ax1.set_xlabel('MAE (Lower is better)', fontsize=12)
    ax1.set_title('Robustness Test - MAE Comparison', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    for bar, val in zip(bars, mae_values):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    # RMSE对比
    ax2 = axes[1]
    rmse_values = [r['rmse'] for r in results]
    bars = ax2.barh(conditions, rmse_values, color=colors)
    ax2.set_xlabel('RMSE (Lower is better)', fontsize=12)
    ax2.set_title('Robustness Test - RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    for bar, val in zip(bars, rmse_values):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_test_results.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_dir / 'robustness_test_results.png'}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("鲁棒性测试总结")
    print("=" * 70)
    print(f"\n无噪声基线: MAE = {results[0]['mae']:.4f}")
    print(f"\n最差情况: {results[-1]['condition']}, MAE = {results[-1]['mae']:.4f}")
    print(f"性能下降: {(results[-1]['mae'] / results[0]['mae'] - 1) * 100:.2f}%")
    
    # 找出最鲁棒的噪声类型
    for i in range(1, len(results)):
        if '高斯' in results[i]['condition']:
            print(f"\n高斯噪声影响: {results[i]['mae'] / results[0]['mae'] * 100:.1f}%")
            break
    
    print(f"\n结果已保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    robustness_test()