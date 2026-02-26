import numpy as np
import pandas as pd
import os

DURATION = 300  # 5分钟
FS = 50  # 采样率
TOTAL_POINTS = DURATION * FS
SAVE_DIR = "experiment/streamdata"


def generate_normal():
    """生成 stream_001_normal.csv - 平平淡淡，完全舒服"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr_base, spo2_base, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    # 全程"良好(3)"，无异常
    true_labels = np.full(TOTAL_POINTS, 3, dtype=int)
    amplitudes = np.full(TOTAL_POINTS, 35.0)
    noises = np.full(TOTAL_POINTS, 1.0)

    # 生成波形
    dynamic_noise1 = np.random.normal(0, noises)
    dynamic_noise2 = np.random.normal(0, noises)

    p1 = p_offset + amplitudes * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1
    p2 = (
        (p_offset * 0.9)
        + amplitudes * np.sin(2 * np.pi * 0.5 * t + 0.1)
        + dynamic_noise2
    )

    # 保存
    filename = f"stream_001_{weight}_{hr_base}_{spo2_base}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(
        {"时间戳": t, "压力传感器1": p1, "压力传感器2": p2, "True_Label": true_labels}
    )
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")


def generate_pain():
    """生成 stream_002_pain.csv - 中间突然剧痛"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr_base, spo2_base, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    amplitudes = np.zeros(TOTAL_POINTS)
    noises = np.zeros(TOTAL_POINTS)
    true_labels = np.zeros(TOTAL_POINTS, dtype=int)

    # 阶段1: 0-60秒 → 一般(1)
    idx_1 = 60 * FS
    amplitudes[0:idx_1] = 23
    noises[0:idx_1] = 3.5
    true_labels[0:idx_1] = 1

    # 阶段2: 60-180秒 → 良好(3)
    idx_2 = 180 * FS
    amplitudes[idx_1:idx_2] = 39
    noises[idx_1:idx_2] = 0.8
    true_labels[idx_1:idx_2] = 3

    # 阶段3: 180-210秒 → 很差(0) - 剧痛！
    idx_3 = 210 * FS
    amplitudes[idx_2:idx_3] = 15
    noises[idx_2:idx_3] = 8.0
    true_labels[idx_2:idx_3] = 0

    # 阶段4: 210-300秒 → 正常(2)
    amplitudes[idx_3:] = 31
    noises[idx_3:] = 2.0
    true_labels[idx_3:] = 2

    # 平滑过渡
    window_size = 3 * FS
    amps_smooth = (
        pd.Series(amplitudes).rolling(window_size, min_periods=1).mean().values
    )
    noises_smooth = pd.Series(noises).rolling(window_size, min_periods=1).mean().values

    # 生成波形
    dynamic_noise1 = np.random.normal(0, noises_smooth)
    dynamic_noise2 = np.random.normal(0, noises_smooth)

    # 剧痛阶段注入尖峰
    spike_mask = (t > 185) & (t < 205) & (np.random.rand(TOTAL_POINTS) > 0.95)
    dynamic_noise1[spike_mask] += np.random.uniform(20, 50, spike_mask.sum())

    p1 = p_offset + amps_smooth * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1
    p2 = (
        (p_offset * 0.9)
        + amps_smooth * np.sin(2 * np.pi * 0.5 * t + 0.1)
        + dynamic_noise2
    )

    # 保存
    filename = f"stream_002_{weight}_{hr_base}_{spo2_base}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(
        {"时间戳": t, "压力传感器1": p1, "压力传感器2": p2, "True_Label": true_labels}
    )
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")


def generate_noise():
    """生成 stream_003_noise.csv - 全程干扰"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr_base, spo2_base, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    # 全程"一般(1)"，但噪声极大
    true_labels = np.full(TOTAL_POINTS, 1, dtype=int)
    amplitudes = np.full(TOTAL_POINTS, 25.0)
    noises = np.full(TOTAL_POINTS, 8.0)  # 高噪声

    # 生成波形
    dynamic_noise1 = np.random.normal(0, noises)
    dynamic_noise2 = np.random.normal(0, noises)

    # 添加随机尖峰（模拟坐姿移动）
    spike_mask = np.random.rand(TOTAL_POINTS) > 0.98
    dynamic_noise1[spike_mask] += np.random.uniform(30, 60, spike_mask.sum())
    dynamic_noise2[spike_mask] += np.random.uniform(30, 60, spike_mask.sum())

    # 添加电磁干扰（高频噪声）
    emi_noise = 5 * np.sin(2 * np.pi * 10 * t)  # 10Hz电磁干扰
    dynamic_noise1 += emi_noise
    dynamic_noise2 += emi_noise

    p1 = p_offset + amplitudes * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1
    p2 = (
        (p_offset * 0.9)
        + amplitudes * np.sin(2 * np.pi * 0.5 * t + 0.1)
        + dynamic_noise2
    )

    # 保存
    filename = f"stream_003_{weight}_{hr_base}_{spo2_base}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(
        {"时间戳": t, "压力传感器1": p1, "压力传感器2": p2, "True_Label": true_labels}
    )
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")


def generate_hardcore():
    """生成 stream_004_hardcore.csv - 极端环境测试"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr_base, spo2_base, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    # 全程"良好(3)"
    true_labels = np.full(TOTAL_POINTS, 3, dtype=int)
    amplitudes = np.full(TOTAL_POINTS, 35.0)

    # === 核心：注入极端干扰 ===
    # 1. 基线提高 15Pa
    p_offset += 15.0

    # 2. 5Hz 高频震动
    real_noise = np.random.normal(0, 2.0, TOTAL_POINTS) + 3 * np.sin(
        2 * np.pi * 5.0 * t
    )

    dynamic_noise1 = np.random.normal(0, 1.0, TOTAL_POINTS)
    dynamic_noise2 = np.random.normal(0, 1.0, TOTAL_POINTS)

    p1 = (
        p_offset
        + amplitudes * np.sin(2 * np.pi * 0.5 * t)
        + real_noise
        + dynamic_noise1
    )
    p2 = (
        (p_offset * 0.9)
        + amplitudes * np.sin(2 * np.pi * 0.5 * t + 0.1)
        + real_noise
        + dynamic_noise2
    )

    filename = f"stream_004_{weight}_{hr_base}_{spo2_base}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(
        {"时间戳": t, "压力传感器1": p1, "压力传感器2": p2, "True_Label": true_labels}
    )
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")


def generate_diff_static():
    """生成 stream_005_diff_static.csv - 不同静态特征，测试模型是否依赖捷径"""
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    # 使用完全不同的静态特征！
    weight, hr_base, spo2_base, height = 85, 95, 96, 180
    p_offset = weight * 0.6

    # 全程"良好(3)" - 与 stream_001 相同的波形模式
    true_labels = np.full(TOTAL_POINTS, 3, dtype=int)
    amplitudes = np.full(TOTAL_POINTS, 35.0)
    noises = np.full(TOTAL_POINTS, 1.0)

    # 生成波形 - 与 generate_normal() 相同
    dynamic_noise1 = np.random.normal(0, noises)
    dynamic_noise2 = np.random.normal(0, noises)

    p1 = p_offset + amplitudes * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1
    p2 = (
        (p_offset * 0.9)
        + amplitudes * np.sin(2 * np.pi * 0.5 * t + 0.1)
        + dynamic_noise2
    )

    filename = f"stream_005_{weight}_{hr_base}_{spo2_base}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame(
        {"时间戳": t, "压力传感器1": p1, "压力传感器2": p2, "True_Label": true_labels}
    )
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功! (用于测试静态特征捷径)")


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("🚀 开始生成测试数据文件...")
    print("-" * 40)

    generate_normal()  # 平平淡淡 (weight=70)
    generate_pain()  # 中间剧痛
    generate_noise()  # 全程干扰
    generate_hardcore()  # 极端环境
    generate_diff_static()  # 不同静态特征 (weight=85)

    print("-" * 40)
    print("✅ 全部生成完成!")
    print(f"📁 保存路径: {SAVE_DIR}")


if __name__ == "__main__":
    main()
