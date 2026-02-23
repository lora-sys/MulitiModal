"""
生成10000人NPZ格式预训练数据
整合更真实的生理信号生成逻辑 + 自研信号自愈处理
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk

SAVE_PATH = "experiment/model/pretrain_10k.npz"
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


def self_heal_signal(signal, fs=50, window_size=15, sigma_threshold=3, highcut=10):
    """
    自研信号自愈 Pipeline

    步骤：
    1. 3-Sigma 滚动窗口异常检测
    2. 样条插值修复异常点
    3. NeuroKit2 低通滤波
    4. Z-Score 归一化
    """
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


def generate_realistic_signal(label):
    """生成更符合生理规律的信号"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)

    # 1. 生理参数
    height = np.random.randint(155, 190)
    bmi = np.random.uniform(18.5, 28)
    weight = round(bmi * (height / 100) ** 2)

    # 心率（与类别关联）
    if label == 3:  # 良好
        hr_base = np.random.normal(65, 3)
    elif label == 2:  # 正常
        hr_base = np.random.normal(70, 5)
    elif label == 1:  # 一般
        hr_base = np.random.normal(80, 5)
    else:  # 很差
        hr_base = np.random.normal(90, 8)
    hr_base = np.clip(hr_base, 50, 120)

    # 血氧
    if label >= 2:
        spo2_base = np.random.uniform(96, 99)
    else:
        spo2_base = np.random.uniform(90, 95)

    # 2. 脉搏频率与心率同步
    hr_freq = hr_base / 60
    freq = hr_freq + np.random.uniform(-0.05, 0.05)

    # 3. 信号幅值
    p_amplitude = 20 + (weight / 5) + label * 2 + np.random.uniform(-2, 2)
    p_offset = weight * 0.5 + np.random.uniform(-3, 3)

    # 4. 噪声模型
    noise_level = 8 - label * 1.2
    noise = np.random.normal(0, noise_level, TOTAL_POINTS)
    drift = np.linspace(
        np.random.uniform(-5, 5), np.random.uniform(-5, 5), TOTAL_POINTS
    )

    # 尖峰噪声
    spikes = np.zeros(TOTAL_POINTS)
    spike_prob = 0.001 if label >= 2 else 0.003
    spike_idx = np.random.choice(TOTAL_POINTS, int(TOTAL_POINTS * spike_prob))
    spikes[spike_idx] = np.random.uniform(30, 60, len(spike_idx)) * np.random.choice(
        [-1, 1], len(spike_idx)
    )

    # 5. 生成双通道信号
    harmonic = 0.3 * p_amplitude * np.sin(2 * np.pi * (freq * 2) * t)
    p1 = (
        p_offset
        + p_amplitude * np.sin(2 * np.pi * freq * t)
        + harmonic
        + noise
        + drift
        + spikes
    )

    time_delay = np.random.uniform(0.01, 0.03)
    p2 = (
        (p_offset * 0.95)
        + (p_amplitude * 0.98) * np.sin(2 * np.pi * freq * (t - time_delay))
        + harmonic * 0.95
        + noise * 1.02
        + drift
        + spikes
    )

    # 传感器差异
    p1 *= np.random.uniform(0.9, 1.1)
    p2 *= np.random.uniform(0.9, 1.1)

    # 6. 自愈处理 (3-Sigma异常检测 + 样条插值 + 滤波 + 归一化)
    p1_norm = self_heal_signal(p1, fs=FS, highcut=10)
    p2_norm = self_heal_signal(p2, fs=FS, highcut=10)

    # 静态特征
    static = np.array(
        [weight / 100.0, hr_base / 120.0, spo2_base / 100.0, height / 200.0],
        dtype=np.float32,
    )

    # 动态特征
    dynamic = np.vstack([p1_norm, p2_norm]).astype(np.float32)

    return dynamic, static


def generate_10k_npz():
    """生成10000人NPZ数据"""
    X_dynamic = []
    X_static = []
    Y_labels = []

    for folder_name, label in CATEGORIES.items():
        print(f"🌊 生成数据: {folder_name} (2500条)...")

        for _ in range(PEOPLE_PER_CAT):
            dynamic, static = generate_realistic_signal(label)

            X_dynamic.append(dynamic)
            X_static.append(static)
            Y_labels.append(label)

    # 打包保存
    print("📦 打包压缩...")
    np.savez_compressed(
        SAVE_PATH,
        dynamic=np.array(X_dynamic),  # (10000, 2, 1000)
        static=np.array(X_static),  # (10000, 4)
        labels=np.array(Y_labels),  # (10000,)
    )

    print(f"✅ 生成完成! 路径: {SAVE_PATH}")


if __name__ == "__main__":
    generate_10k_npz()
