"""
生成回归数据集的动态波形
为integrated_health_dataset.csv中的每个样本生成动态波形
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'generate'))

from synthesizer import WaveformGenerator


def generate_waveform_for_score(score, seq_len=1000, sampling_rate=50):
    """
    根据中医诊断分数生成波形

    Args:
        score: 中医诊断分数（30-100）
        seq_len: 序列长度
        sampling_rate: 采样率

    Returns:
        waveform: (2, seq_len) 波形数据
    """
    # 分数归一化到0-1
    score_norm = (score - 30) / (100 - 30)  # 30->0, 100->1

    # 初始化生成器
    waveform_gen = WaveformGenerator(base_freq=0.5)

    # 生成时间序列
    t = np.arange(seq_len) / sampling_rate

    # 根据分数生成参数
    # 分数越高：压力越大，振幅越稳定
    base_pressure = 30 + score_norm * 50  # 30->80
    amplitude = 20 + score_norm * 40  # 20->60

    # 生成两个通道的波形
    channel1 = waveform_gen.generate(t, base_pressure, amplitude, phase_shift=0)
    channel2 = waveform_gen.generate(t, base_pressure * 0.8, amplitude * 0.9, phase_shift=np.pi/4)

    # 添加一些随机噪声（分数越低，噪声越大）
    noise_level = 0.05 * (1 - score_norm)
    channel1 += np.random.normal(0, noise_level, channel1.shape)
    channel2 += np.random.normal(0, noise_level, channel2.shape)

    # Z-score标准化
    channel1 = (channel1 - channel1.mean()) / (channel1.std() + 1e-8)
    channel2 = (channel2 - channel2.mean()) / (channel2.std() + 1e-8)

    # 组合成 (2, seq_len)
    waveform = np.stack([channel1, channel2], axis=0)

    return waveform


def load_and_process_csv(csv_path):
    """
    加载并处理CSV数据

    Returns:
        data_dict: 包含所有数据的字典
    """
    print(f"[*] 加载CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[*] 总样本数: {len(df)}")

    # 提取回归目标
    scores = df['中医诊断分数_对齐'].values.astype(np.float32)

    # 提取基础静态特征（年龄、BMI、心率、血氧）
    static_basic = df[['年龄', 'BMI', '心率', '血氧']].values.astype(np.float32)

    # 提取舌面诊特征（BMI数值、健康指数）
    static_scores = df[['BMI 数值', '健康指数']].values.astype(np.float32)

    # 编码体质类型
    constitution_encoder = LabelEncoder()
    constitutions = constitution_encoder.fit_transform(df['体质类型名称'].values)

    print(f"[*] 回归目标范围: {scores.min():.1f} - {scores.max():.1f}")
    print(f"[*] 体质类型数量: {len(constitution_encoder.classes_)}")

    return {
        'scores': scores,
        'static_basic': static_basic,
        'static_scores': static_scores,
        'constitutions': constitutions,
        'constitution_classes': constitution_encoder.classes_,
    }


def generate_regression_npz(csv_path, output_path, seq_len=1000):
    """
    生成回归数据集的NPZ文件

    Args:
        csv_path: CSV文件路径
        output_path: 输出NPZ文件路径
        seq_len: 波形序列长度
    """
    print("=" * 60)
    print("生成回归数据集")
    print("=" * 60)

    # 加载CSV数据
    data = load_and_process_csv(csv_path)

    num_samples = len(data['scores'])
    print(f"[*] 将为 {num_samples} 个样本生成波形...")

    # 初始化数组
    dynamics = np.zeros((num_samples, 2, seq_len), dtype=np.float32)
    labels = data['scores'].reshape(-1, 1).astype(np.float32)

    # 生成波形
    print("[*] 生成波形中...")
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"    进度: {i+1}/{num_samples}")

        score = data['scores'][i]
        waveform = generate_waveform_for_score(score, seq_len=seq_len)
        dynamics[i] = waveform

    # 标准化静态特征
    print("[*] 标准化静态特征...")
    scaler_basic = StandardScaler()
    static_basic = scaler_basic.fit_transform(data['static_basic']).astype(np.float32)

    scaler_scores = StandardScaler()
    static_scores = scaler_scores.fit_transform(data['static_scores']).astype(np.float32)

    # 归一化回归目标到0-1
    labels_normalized = (labels - 30) / (100 - 30)

    print("[*] 数据形状:")
    print(f"    dynamics: {dynamics.shape}")
    print(f"    static_basic: {static_basic.shape}")
    print(f"    static_scores: {static_scores.shape}")
    print(f"    constitutions: {data['constitutions'].shape}")
    print(f"    labels: {labels.shape}")
    print(f"    labels_normalized: {labels_normalized.shape}")

    # 保存NPZ文件
    print(f"[*] 保存NPZ文件: {output_path}")
    np.savez_compressed(
        output_path,
        dynamic=dynamics,  # 注意：使用单数形式
        static_basic=static_basic,
        static_scores=static_scores,
        constitution=data['constitutions'],  # 注意：使用单数形式
        label=labels_normalized,  # 归一化的标签用于训练
        label_original=labels,    # 原始标签用于评估
    )

    print("=" * 60)
    print("✅ 回归数据集生成完成！")
    print("=" * 60)


def main():
    # 文件路径
    csv_path = "experiment/rawdata/train_data/integrated_health_dataset.csv"
    output_path = "experiment/model/unified_dataset_regression.npz"

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"[!] 错误: CSV文件不存在: {csv_path}")
        sys.exit(1)

    # 生成数据集
    generate_regression_npz(csv_path, output_path)


if __name__ == '__main__':
    main()