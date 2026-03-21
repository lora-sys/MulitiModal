"""
生成带噪声的实时测试数据
基于训练数据的真实特征，添加真实生产环境下的噪声
"""

import os
import numpy as np
import pandas as pd

def generate_noisy_realtime_test(
    output_path='experiment/streamdata/noisy_realtime_test.csv',
    noise_types=['baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout'],
    noise_prob=0.5,
    duration=180,  # 3分钟
    sample_rate=50,
    seed=42
):
    """
    生成带噪声的实时测试数据
    
    Args:
        output_path: 输出文件路径
        noise_types: 噪声类型列表
        noise_prob: 添加噪声的概率
        duration: 持续时间（秒）
        sample_rate: 采样率（Hz）
        seed: 随机种子
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, duration * sample_rate)
    n_samples = len(t)
    
    # 从训练数据中采样样本（模拟真实场景）
    data = np.load('experiment/model/unified_dataset_expanded.npz')
    dynamic_samples = data['dynamic']  # (6684, 2, 1000)
    labels = data['labels']  # (6684,)
    
    # 选择测试集样本（使用StratifiedSplit）
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=labels, random_state=42)
    
    print("=" * 70)
    print("生成带噪声的实时测试数据")
    print("=" * 70)
    print(f"从训练数据中采样: {len(test_idx)} 个样本")
    print(f"噪声类型: {noise_types}")
    print(f"噪声概率: {noise_prob}")
    print(f"输出文件: {output_path}")
    print("=" * 70)
    
    # 生成测试数据流（通过拼接多个训练样本）
    num_segments = 9  # 大约3分钟 / 每个样本20秒
    samples_per_segment = len(test_idx) // num_segments
    
    all_p1 = []
    all_p2 = []
    all_labels = []
    all_noise_types = []
    
    # 创建噪声生成器
    class NoiseGenerator:
        def __init__(self, noise_types, noise_prob, seed):
            self.noise_types = noise_types
            self.noise_prob = noise_prob
            np.random.seed(seed)
        
        def add_noise(self, signal):
            signal_np = signal.copy()
            
            # 随机选择是否添加噪声
            if np.random.random() < self.noise_prob:
                noise_type = np.random.choice(self.noise_types)
                
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
                
                return signal_np, noise_type
            
            return signal_np, None
    
    noise_gen = NoiseGenerator(noise_types, noise_prob, seed)
    
    # 构建测试数据流
    current_idx = 0
    samples_in_segment = samples_per_segment + 1  # 多加一个以避免索引越界
    
    for seg in range(num_segments):
        if current_idx + samples_in_segment >= len(test_idx):
            samples_in_segment = len(test_idx) - current_idx
            if samples_in_segment <= 0:
                break
        
        segment_indices = test_idx[current_idx:current_idx+samples_in_segment]
        
        for idx in segment_indices:
            sample = dynamic_samples[idx]
            label = labels[idx]
            
            # 添加噪声
            noisy_sample, noise_type = noise_gen.add_noise(sample)
            
            # 添加到流
            all_p1.extend(noisy_sample[0])
            all_p2.extend(noisy_sample[1])
            all_labels.extend([label] * len(noisy_sample[0]))
            
            # 记录噪声类型（每个时间点记录一次）
            all_noise_types.extend([noise_type if noise_type else 'none'] * len(noisy_sample[0]))
        
        current_idx += samples_in_segment
    
    # 截断到目标长度
    all_p1 = all_p1[:n_samples]
    all_p2 = all_p2[:n_samples]
    all_labels = all_labels[:n_samples]
    all_noise_types = all_noise_types[:n_samples]
    
    # 保存
    df = pd.DataFrame({
        "时间戳": t,
        "压力传感器1": all_p1,
        "压力传感器2": all_p2,
        "True_Label": all_labels,
        "Noise_Type": all_noise_types
    })
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ 噪声实时测试数据已生成: {output_path}")
    print(f"  总时间点: {n_samples}")
    print(f"  采样率: {sample_rate} Hz")
    print(f"  时长: {duration} 秒")
    
    # 统计噪声分布
    noise_counts = {}
    for noise_type in ['none'] + noise_types:
        count = all_noise_types.count(noise_type)
        noise_counts[noise_type] = count
        percentage = count / n_samples * 100
        print(f"  {noise_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n📊 标签分布:")
    for label in [1, 2, 3]:
        count = all_labels.count(label)
        percentage = count / n_samples * 100
        label_names = {1: '一般', 2: '正常', 3: '良好'}
        print(f"  标签{label}（{label_names[label]}）: {count} ({percentage:.1f}%)")
    
    return output_path


if __name__ == "__main__":
    # 生成噪声实时测试数据
    generate_noisy_realtime_test(
        output_path='experiment/streamdata/noisy_realtime_test.csv',
        noise_types=['baseline', 'gaussian', 'amplitude', 'motion', 'channel_dropout'],
        noise_prob=0.5,
        duration=180,
        sample_rate=50,
        seed=42
    )