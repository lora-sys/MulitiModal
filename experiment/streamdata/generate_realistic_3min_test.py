"""
生成真实的3分钟实时测试数据
基于训练数据的特征分布，模拟真实的按摩椅使用场景
"""

import numpy as np
import pandas as pd
import os

DURATION = 180  # 3分钟
FS = 50  # 采样率
TOTAL_POINTS = DURATION * FS
SAVE_DIR = "experiment/streamdata"


def generate_realistic_scenario():
    """
    生成真实的3分钟按摩椅使用场景
    模拟：从正常开始，到舒适享受，再到轻微不适，最后回到正常
    """
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr, spo2, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    # 基于训练数据的波形特征（振幅）
    # 标签1（一般）：振幅最大（约25）
    # 标签2（正常）：振幅中等（约20）
    # 标签3（良好）：振幅最小（约18）
    amplitudes = np.zeros(TOTAL_POINTS)
    noises = np.zeros(TOTAL_POINTS)
    true_labels = np.zeros(TOTAL_POINTS, dtype=int)

    # 场景设计：
    # 0-30秒：正常按摩 (标签2)
    # 30-90秒：舒适享受 (标签3) - 用户完全放松
    # 90-120秒：轻微不适 (标签1) - 按摩力度稍大
    # 120-150秒：舒适享受 (标签3) - 调整后很舒服
    # 150-180秒：正常按摩 (标签2) - 按摩结束

    # 阶段1: 0-30秒 → 正常(2)
    idx_1 = 30 * FS
    amplitudes[0:idx_1] = 20.0
    noises[0:idx_1] = 2.0
    true_labels[0:idx_1] = 2

    # 阶段2: 30-90秒 → 良好(3) - 完全放松
    idx_2 = 90 * FS
    amplitudes[idx_1:idx_2] = 18.0
    noises[idx_1:idx_2] = 1.5
    true_labels[idx_1:idx_2] = 3

    # 阶段3: 90-120秒 → 一般(1) - 轻微不适
    idx_3 = 120 * FS
    amplitudes[idx_2:idx_3] = 25.0
    noises[idx_2:idx_3] = 3.5
    true_labels[idx_2:idx_3] = 1

    # 阶段4: 120-150秒 → 良好(3) - 调整后很舒服
    idx_4 = 150 * FS
    amplitudes[idx_3:idx_4] = 18.0
    noises[idx_3:idx_4] = 1.5
    true_labels[idx_3:idx_4] = 3

    # 阶段5: 150-180秒 → 正常(2) - 按摩结束
    amplitudes[idx_4:] = 20.0
    noises[idx_4:] = 2.0
    true_labels[idx_4:] = 2

    # 平滑过渡（防止突变）
    window_size = 5 * FS  # 5秒过渡
    amps_smooth = pd.Series(amplitudes).rolling(window_size, min_periods=1).mean().values
    noises_smooth = pd.Series(noises).rolling(window_size, min_periods=1).mean().values

    # 生成基础波形
    dynamic_noise1 = np.random.normal(0, noises_smooth)
    dynamic_noise2 = np.random.normal(0, noises_smooth)

    # 呼吸节律（模拟正常呼吸频率，约0.2Hz）
    breathing = 2.0 * np.sin(2 * np.pi * 0.2 * t)

    # 心跳节律（模拟正常心率，约1.2Hz，对应72bpm）
    heartbeat = 0.5 * np.sin(2 * np.pi * 1.2 * t)

    p1 = p_offset + amps_smooth * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1 + breathing + heartbeat
    p2 = (p_offset * 0.9) + amps_smooth * np.sin(2 * np.pi * 0.5 * t + 0.1) + dynamic_noise2 + breathing + heartbeat

    # 保存
    filename = f"stream_3min_test_{weight}_{hr}_{spo2}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame({
        "时间戳": t,
        "压力传感器1": p1,
        "压力传感器2": p2,
        "True_Label": true_labels
    })
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")
    print(f"📊 场景说明：")
    print(f"  0-30秒：正常按摩 (标签2)")
    print(f"  30-90秒：舒适享受（完全放松）(标签3)")
    print(f"  90-120秒：轻微不适（按摩力度稍大）(标签1)")
    print(f"  120-150秒：舒适享受（调整后很舒服）(标签3)")
    print(f"  150-180秒：正常按摩（按摩结束）(标签2)")
    print(f"📊 标签分布：")
    for label in [1, 2, 3]:
        count = (true_labels == label).sum()
        duration = count / FS
        percentage = 100 * count / TOTAL_POINTS
        label_name = {1: "一般", 2: "正常", 3: "良好"}[label]
        print(f"  标签{label}（{label_name}）：{duration:.0f}秒 ({percentage:.1f}%)")

    return filepath


def generate_realistic_with_noise():
    """
    生成包含真实噪声干扰的3分钟测试数据
    模拟：用户移动、传感器干扰、外界震动等
    """
    t = np.linspace(0, DURATION, TOTAL_POINTS)
    weight, hr, spo2, height = 70, 75, 98, 175
    p_offset = weight * 0.6

    amplitudes = np.zeros(TOTAL_POINTS)
    noises = np.zeros(TOTAL_POINTS)
    true_labels = np.zeros(TOTAL_POINTS, dtype=int)

    # 场景设计：包含各种真实干扰
    # 0-45秒：正常按摩 (标签2)
    # 45-105秒：舒适享受 (标签3) - 中间有用户移动
    # 105-135秒：正常按摩 (标签2) - 有传感器干扰
    # 135-180秒：舒适享受 (标签3) - 恢复平静

    # 阶段1: 0-45秒 → 正常(2)
    idx_1 = 45 * FS
    amplitudes[0:idx_1] = 20.0
    noises[0:idx_1] = 2.0
    true_labels[0:idx_1] = 2

    # 阶段2: 45-105秒 → 良好(3)
    idx_2 = 105 * FS
    amplitudes[idx_1:idx_2] = 18.0
    noises[idx_1:idx_2] = 1.5
    true_labels[idx_1:idx_2] = 3

    # 阶段3: 105-135秒 → 正常(2)
    idx_3 = 135 * FS
    amplitudes[idx_2:idx_3] = 20.0
    noises[idx_2:idx_3] = 2.0
    true_labels[idx_2:idx_3] = 2

    # 阶段4: 135-180秒 → 良好(3)
    amplitudes[idx_3:] = 18.0
    noises[idx_3:] = 1.5
    true_labels[idx_3:] = 3

    # 平滑过渡
    window_size = 5 * FS
    amps_smooth = pd.Series(amplitudes).rolling(window_size, min_periods=1).mean().values
    noises_smooth = pd.Series(noises).rolling(window_size, min_periods=1).mean().values

    # 生成基础波形
    dynamic_noise1 = np.random.normal(0, noises_smooth)
    dynamic_noise2 = np.random.normal(0, noises_smooth)

    # 呼吸和心跳
    breathing = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    heartbeat = 0.5 * np.sin(2 * np.pi * 1.2 * t)

    # === 注入真实干扰 ===

    # 1. 用户移动干扰（60-70秒，持续10秒）
    movement_mask = (t > 60) & (t < 70)
    dynamic_noise1[movement_mask] += np.random.uniform(-15, 15, movement_mask.sum())
    dynamic_noise2[movement_mask] += np.random.uniform(-15, 15, movement_mask.sum())

    # 2. 传感器尖峰干扰（120-125秒，持续5秒）
    spike_mask = (t > 120) & (t < 125) & (np.random.rand(TOTAL_POINTS) > 0.9)
    dynamic_noise1[spike_mask] += np.random.uniform(20, 40, spike_mask.sum())
    dynamic_noise2[spike_mask] += np.random.uniform(20, 40, spike_mask.sum())

    # 3. 外界震动干扰（150-160秒，持续10秒）
    vibration_mask = (t > 150) & (t < 160)
    vibration = 8 * np.sin(2 * np.pi * 5.0 * t[vibration_mask])  # 5Hz震动
    dynamic_noise1[vibration_mask] += vibration
    dynamic_noise2[vibration_mask] += vibration

    p1 = p_offset + amps_smooth * np.sin(2 * np.pi * 0.5 * t) + dynamic_noise1 + breathing + heartbeat
    p2 = (p_offset * 0.9) + amps_smooth * np.sin(2 * np.pi * 0.5 * t + 0.1) + dynamic_noise2 + breathing + heartbeat

    # 保存
    filename = f"stream_3min_realistic_noise_{weight}_{hr}_{spo2}_{height}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    df = pd.DataFrame({
        "时间戳": t,
        "压力传感器1": p1,
        "压力传感器2": p2,
        "True_Label": true_labels
    })
    df.to_csv(filepath, index=False)
    print(f"✅ {filename} 生成成功!")
    print(f"📊 场景说明：包含真实噪声干扰")
    print(f"  60-70秒：用户移动干扰")
    print(f"  120-125秒：传感器尖峰干扰")
    print(f"  150-160秒：外界震动干扰")
    print(f"📊 标签分布：")
    for label in [1, 2, 3]:
        count = (true_labels == label).sum()
        duration = count / FS
        percentage = 100 * count / TOTAL_POINTS
        label_name = {1: "一般", 2: "正常", 3: "良好"}[label]
        print(f"  标签{label}（{label_name}）：{duration:.0f}秒 ({percentage:.1f}%)")

    return filepath


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("🚀 开始生成真实的3分钟测试数据...")
    print("-" * 50)

    # 生成基础场景数据
    print("\n📝 生成场景1：标准按摩流程")
    file1 = generate_realistic_scenario()

    # 生成包含噪声的数据
    print("\n📝 生成场景2：包含真实干扰")
    file2 = generate_realistic_with_noise()

    print("-" * 50)
    print("✅ 全部生成完成!")
    print(f"📁 保存路径: {SAVE_DIR}")
    print(f"📄 生成文件:")
    print(f"  1. {os.path.basename(file1)}")
    print(f"  2. {os.path.basename(file2)}")


if __name__ == "__main__":
    main()
