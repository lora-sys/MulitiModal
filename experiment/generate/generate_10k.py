"""
生成10000人NPZ格式预训练数据（增强版）
使用增强版合成器生成更真实的波形，包含多种噪声类型
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from synthesizer import MassageDataSynthesizer

SAVE_PATH = "experiment/model/pretrain_10k_enhanced.npz"
CATEGORIES = {
    "身体表征很差": 0,
    "身体表征一般": 1,
    "身体表征正常": 2,
    "身体表征良好": 3,
}

PEOPLE_PER_CAT = 2500
FS = 50
DURATION = 20
TOTAL_POINTS = DURATION * FS


def handle_nan_values(signal):
    """
    处理NaN值（数据丢失）
    使用线性插值修复丢失的采样点
    """
    s = pd.Series(signal)

    # 线性插值修复NaN
    s_clean = s.interpolate(method='linear')

    # 如果开头或结尾还有NaN，用前后值填充
    s_clean = s_clean.bfill().ffill()

    return s_clean.values


def self_heal_signal(signal, fs=50, window_size=15, sigma_threshold=3, highcut=10):
    """
    自研信号自愈 Pipeline

    步骤：
    1. 处理NaN值（数据丢失）
    2. 3-Sigma 滚动窗口异常检测
    3. 样条插值修复异常点
    4. NeuroKit2 低通滤波
    5. Z-Score 归一化
    """
    # 1. 处理NaN值
    signal = handle_nan_values(signal)

    s = pd.Series(signal)

    rolling_mean = s.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = s.rolling(window=window_size, center=True, min_periods=1).std()
    rolling_std = rolling_std.bfill().ffill()

    is_anomaly = (s > rolling_mean + sigma_threshold * rolling_std) | (
        s < rolling_mean - sigma_threshold * rolling_std
    )

    s_clean = s.copy()
    s_clean[is_anomaly] = np.nan
    s_clean = s_clean.interpolate(method="cubic")
    s_clean = s_clean.bfill().ffill()

    s_filtered = nk.signal_filter(
        s_clean.values,
        sampling_rate=fs,
        highcut=highcut,
        method="butterworth",
        order=4,
    )

    s_norm = (s_filtered - np.mean(s_filtered)) / (np.std(s_filtered) + 1e-6)

    return s_norm.astype(np.float32)


def generate_realistic_signal(label, synthesizer, global_id):
    """
    使用增强版合成器生成信号
    """
    folder_name = list(CATEGORIES.keys())[label]

    # 使用合成器生成数据
    df, params, _ = synthesizer.generate_person(folder_name, global_id)

    # 提取双通道压力波形
    p1 = df['压力传感器1'].values
    p2 = df['压力传感器2'].values

    # 自愈处理（包括NaN处理、异常检测、滤波、归一化）
    p1_norm = self_heal_signal(p1, fs=FS, highcut=10)
    p2_norm = self_heal_signal(p2, fs=FS, highcut=10)

    # 从合成器参数中提取静态特征
    weight = params['weight']
    hr_base = params['hr']
    spo2_base = params['spo2']
    height = params['height']

    # 静态特征（归一化）
    static = np.array(
        [weight / 100.0, hr_base / 120.0, spo2_base / 100.0, height / 200.0],
        dtype=np.float32,
    )

    # 动态特征
    dynamic = np.vstack([p1_norm, p2_norm]).astype(np.float32)

    return dynamic, static


def generate_10k_npz():
    """生成10000人NPZ数据（使用增强版合成器）"""
    print("=" * 60)
    print("生成10000人NPZ数据（增强版）")
    print("=" * 60)

    # 初始化合成器（使用增强版噪声）
    synthesizer = MassageDataSynthesizer(
        seed=42,
        overlap_ratio=0.15,
        output_dir="experiment/model/temp_data"
    )

    X_dynamic = []
    X_static = []
    Y_labels = []

    global_id = 1

    for folder_name, label in CATEGORIES.items():
        print(f"🌊 生成数据: {folder_name} ({PEOPLE_PER_CAT}条)...")

        for _ in range(PEOPLE_PER_CAT):
            dynamic, static = generate_realistic_signal(label, synthesizer, global_id)

            X_dynamic.append(dynamic)
            X_static.append(static)
            Y_labels.append(label)

            global_id += 1

            # 进度显示
            if global_id % 500 == 0:
                print(f"   已生成: {global_id} 人")

    # 打包保存
    print("\n📦 打包压缩...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(
        SAVE_PATH,
        dynamic=np.array(X_dynamic),  # (10000, 2, 1000)
        static=np.array(X_static),  # (10000, 4)
        labels=np.array(Y_labels),  # (10000,)
    )

    print(f"✅ 生成完成! 路径: {SAVE_PATH}")
    print(f"\n📊 数据统计:")
    print(f"   动态特征: {np.array(X_dynamic).shape}")
    print(f"   静态特征: {np.array(X_static).shape}")
    print(f"   标签: {np.array(Y_labels).shape}")
    print(f"\n📊 标签分布:")
    unique, counts = np.unique(np.array(Y_labels), return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   标签 {label}: {count} ({count/len(Y_labels)*100:.1f}%)")


if __name__ == "__main__":
    generate_10k_npz()
