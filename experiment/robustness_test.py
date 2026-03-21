"""
鲁棒性测试脚本
测试所有模型（干净训练 vs 噪声训练）在干净和噪声数据上的性能
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import get_model


def add_noise(signal, noise_type, seed=42):
    """
    为信号添加指定类型的噪声
    
    Args:
        signal: 输入信号 (2, 1000)
        noise_type: 噪声类型 ('baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout', 'none')
        seed: 随机种子
    """
    np.random.seed(seed)
    signal_np = signal.copy()
    
    if noise_type == 'none':
        return signal_np
    
    elif noise_type == 'baseline':
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
    
    return signal_np


def test_model(model, test_loader, device, noise_type='none'):
    """
    测试模型性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        noise_type: 噪声类型（'none'表示不添加噪声）
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch.get('static_scores', torch.zeros(dynamic.shape[0], 2)).to(device)
            constitution = batch.get('constitution', torch.zeros(dynamic.shape[0], dtype=torch.long)).to(device)
            labels = batch['labels'].cpu().numpy()
            
            # 添加噪声
            if noise_type != 'none':
                for i in range(dynamic.shape[0]):
                    dynamic_np = dynamic[i].cpu().numpy()
                    noisy_dynamic = add_noise(dynamic_np, noise_type, seed=i)
                    dynamic[i] = torch.FloatTensor(noisy_dynamic).to(device)
            
            # 预测
            outputs = model(dynamic, static_basic, static_scores, constitution)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1


def main():
    print("=" * 80)
    print("鲁棒性测试：干净训练 vs 噪声训练模型")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # 加载数据
    print("\n📂 加载数据...")
    data = np.load('experiment/model/unified_dataset_expanded.npz')
    dynamic_samples = data['dynamic']  # (6684, 2, 1000)
    static_basic = data['static_basic']  # (6684, 4)
    static_scores = data['static_scores']  # (6684, 2)
    constitution = data['constitution']  # (6684,)
    labels = data['labels']  # (6684,)
    
    # 划分训练集和测试集
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
    
    # 准备测试数据
    test_dynamic = dynamic_samples[test_idx]
    test_static_basic = static_basic[test_idx]
    test_static_scores = static_scores[test_idx]
    test_constitution = constitution[test_idx]
    test_labels = labels[test_idx] - 1  # 标签映射：1,2,3 -> 0,1,2
    
    print(f"  测试样本数: {len(test_idx)}")
    
    # 创建测试数据集和加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_dynamic),
        torch.FloatTensor(test_static_basic),
        torch.FloatTensor(test_static_scores),
        torch.LongTensor(test_constitution),
        torch.LongTensor(test_labels)
    )
    
    def collate_fn(batch):
        dynamic = torch.stack([item[0] for item in batch])
        static_basic = torch.stack([item[1] for item in batch])
        static_scores = torch.stack([item[2] for item in batch])
        constitution = torch.stack([item[3] for item in batch])
        labels = torch.stack([item[4] for item in batch])
        return {
            'dynamic': dynamic,
            'static_basic': static_basic,
            'static_scores': static_scores,
            'constitution': constitution,
            'labels': labels
        }
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 定义模型路径
    model_configs = [
        # 干净训练的模型
        {
            'name': 'Simple Concat (Clean)',
            'type': 'baseline_a',
            'path': 'experiment/results/train_simple_concat/r1/checkpoints/best.pth',
            'is_noise': False
        },
        {
            'name': 'Late Fusion Transformer (Clean)',
            'type': 'baseline_b',
            'path': 'experiment/results/train_late_fusion/r1/checkpoints/best.pth',
            'is_noise': False
        },
        {
            'name': 'Cross-Attention Gate Fusion (Clean)',
            'type': 'baseline_c',
            'path': 'experiment/results/train_multimodal/r1/checkpoints/best.pth',
            'is_noise': False
        },
        # 噪声训练的模型
        {
            'name': 'Simple Concat (Noise)',
            'type': 'baseline_a',
            'path': 'experiment/results/baseline_a_noise/r1/checkpoints/best.pth',
            'is_noise': True
        },
        {
            'name': 'Late Fusion Transformer (Noise)',
            'type': 'baseline_b',
            'path': 'experiment/results/baseline_b_noise/r1/checkpoints/best.pth',
            'is_noise': True
        },
        {
            'name': 'Cross-Attention Gate Fusion (Noise)',
            'type': 'baseline_c',
            'path': 'experiment/results/baseline_c_noise/r1/checkpoints/best.pth',
            'is_noise': True
        }
    ]
    
    # 测试噪声类型
    noise_types = ['none', 'baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout']
    
    # 存储结果
    results = []
    
    # 测试每个模型
    for model_config in model_configs:
        print(f"\n{'=' * 80}")
        print(f"测试模型: {model_config['name']}")
        print(f"路径: {model_config['path']}")
        print(f"训练方式: {'噪声注入' if model_config['is_noise'] else '干净训练'}")
        print('=' * 80)
        
        # 加载模型
        model = get_model(model_config['type']).to(device)
        checkpoint = torch.load(model_config['path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试每种噪声
        for noise_type in noise_types:
            print(f"\n  📊 噪声类型: {noise_type}")
            
            accuracy, f1 = test_model(model, test_loader, device, noise_type)
            
            result = {
                'model': model_config['name'],
                'training': 'Noise' if model_config['is_noise'] else 'Clean',
                'noise_type': noise_type,
                'accuracy': accuracy,
                'f1': f1
            }
            results.append(result)
            
            print(f"    准确率: {accuracy*100:.2f}%")
            print(f"    F1分数: {f1:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = 'experiment/results/robustness_test_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'=' * 80}")
    print("📊 鲁棒性测试完成！")
    print(f"结果已保存到: {results_path}")
    print('=' * 80)
    
    # 打印对比表格
    print("\n📋 性能对比表格:")
    print("\n1. 干净数据上的性能（无噪声）:")
    clean_results = results_df[results_df['noise_type'] == 'none']
    print(clean_results[['model', 'training', 'accuracy', 'f1']].to_string(index=False))
    
    print("\n2. 各噪声类型下的性能下降:")
    for noise_type in ['baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout']:
        print(f"\n  噪声类型: {noise_type}")
        noise_results = results_df[results_df['noise_type'] == noise_type]
        clean_acc = clean_results.set_index('model')['accuracy'].to_dict()
        
        for _, row in noise_results.iterrows():
            model_name = row['model']
            clean_acc_model = clean_acc.get(model_name, 0)
            acc_drop = clean_acc_model - row['accuracy']
            print(f"    {model_name}: {row['accuracy']*100:.2f}% (下降: {acc_drop*100:.2f}%)")
    
    print("\n3. 鲁棒性排名（在噪声数据上的平均性能）:")
    noise_avg = results_df[results_df['noise_type'] != 'none'].groupby('model').agg({
        'accuracy': 'mean',
        'f1': 'mean'
    }).sort_values('accuracy', ascending=False)
    print(noise_avg.to_string())
    
    # 分析噪声训练的效果
    print("\n4. 噪声训练 vs 干净训练的性能对比:")
    model_names = ['Simple Concat', 'Late Fusion Transformer', 'Cross-Attention Gate Fusion']
    
    for model_name in model_names:
        print(f"\n  {model_name}:")
        clean_model = results_df[(results_df['model'].str.contains(model_name)) & 
                                 (results_df['training'] == 'Clean')]
        noise_model = results_df[(results_df['model'].str.contains(model_name)) & 
                                 (results_df['training'] == 'Noise')]
        
        # 干净数据上的对比
        clean_acc_clean = clean_model[clean_model['noise_type'] == 'none']['accuracy'].values[0]
        noise_acc_clean = noise_model[noise_model['noise_type'] == 'none']['accuracy'].values[0]
        print(f"    干净数据: Clean训练={clean_acc_clean*100:.2f}%, Noise训练={noise_acc_clean*100:.2f}%")
        
        # 噪声数据上的对比（平均）
        noise_avg_clean = clean_model[clean_model['noise_type'] != 'none']['accuracy'].mean()
        noise_avg_noise = noise_model[noise_model['noise_type'] != 'none']['accuracy'].mean()
        print(f"    噪声数据(平均): Clean训练={noise_avg_clean*100:.2f}%, Noise训练={noise_avg_noise*100:.2f}%")
        
        # 鲁棒性提升
        robustness_gain = noise_avg_noise - noise_avg_clean
        print(f"    鲁棒性提升: {robustness_gain*100:.2f}%")


if __name__ == "__main__":
    main()