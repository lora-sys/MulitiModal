"""
工业级鲁棒性测试数据生成器 (Simulation 3.0)

目标：生成1000人规模的工业级噪声数据，注入3种恶性噪声：
1. 强力底噪：方差加大的高斯噪声（3倍）
2. 接触不良跳变：随机产生瞬间爆表的极值（0.2%概率）
3. 基线缓慢漂移：模拟传感器温漂或用户坐姿缓慢挪动

"""

import os
import numpy as np
import pandas as pd

# ==================== 配置参数 ====================
BASE_PATH = "./data/industrial"  # 数据输出路径
CATEGORIES = {"身体表征很差": 0, "身体表征一般": 1, "身体表征正常": 2, "身体表征良好": 3}
PEOPLE_PER_CAT = 250  # 每类人数，总计1000人
FS = 50  # 采样率 50Hz
DURATION = 20  # 持续时间 20秒

# ==================== 物理模型参数 ====================
def generate_robust_data():
    """生成工业级噪声数据"""

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"✓ 创建数据目录: {BASE_PATH}")

    global_id = 1

    for folder_name, label in CATEGORIES.items():
        # 创建类别子目录
        path = os.path.join(BASE_PATH, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"\n{'='*60}")
        print(f"正在生成加噪数据: {folder_name} (标签={label})")
        print(f"{'='*60}")

        for i in range(PEOPLE_PER_CAT):
            # ========== 步骤1: 生成基础物理参数 ==========
            height = np.random.randint(155, 190)  # 身高 155-190cm
            weight = int(height - 105 + np.random.normal(0, 5))  # 体重基于身高计算
            hr_base = 70 + (3 - label) * 10 + np.random.randint(-5, 5)  # 心率基准
            spo2_base = 98 - (3 - label) + np.random.randint(-1, 1)  # 血氧基准

            # ========== 步骤2: 生成时间序列 ==========
            t = np.linspace(0, DURATION, DURATION * FS)  # 0-20秒，1000个点

            # ========== 步骤3: 计算压力信号基础参数 ==========
            p_offset = weight * 0.6  # 压力基准 = 体重 × 0.6
            p_amplitude = 15 + label * 8  # 振幅与身体表征正相关

            # ========== 步骤4: 注入工业级噪声 ==========

            # 噪声1: 强力底噪（高斯噪声，方差8.0，随身体表征递减）
            noise = np.random.normal(0, 8.0 - label * 1.5, len(t))
            # 解释：身体表征好的人，按摩力度大，底噪相对较小

            # 噪声2: 随机跳点（0.2%概率出现瞬间爆表）
            spikes = np.zeros(len(t))
            spike_idx = np.random.choice(len(t), int(len(t) * 0.002))
            spikes[spike_idx] = np.random.uniform(50, 100, len(spike_idx))
            # 解释：模拟接触不良导致的瞬间信号爆表（50-100个单位）

            # 噪声3: 基线漂移（正弦缓慢波动）
            drift = 5 * np.sin(2 * np.pi * 0.01 * t)
            # 解释：模拟传感器温漂或用户坐姿缓慢挪动（幅度5，频率0.01Hz）

            # ========== 步骤5: 生成双传感器压力信号 ==========
            # 传感器1：主传感器
            p1 = (p_offset +
                  p_amplitude * np.sin(2 * np.pi * 0.5 * t) +  # 0.5Hz正弦波
                  noise + spikes + drift)  # 注入所有噪声

            # 传感器2：从传感器（相位偏移+幅度缩放）
            p2 = (p_offset * 0.9 +
                  p_amplitude * np.sin(2 * np.pi * 0.5 * t + 0.1) +  # 相位偏移0.1
                  noise + spikes + drift)  # 相同噪声

            # ========== 步骤6: 保存数据 ==========
            filename = f"{global_id:03d}_{weight}_{hr_base}_{spo2_base}_{height}.csv"
            df = pd.DataFrame({
                '时间戳': t,
                '压力传感器1': p1,
                '压力传感器2': p2
            })
            df.to_csv(os.path.join(path, filename), index=False)

            # 进度显示
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{PEOPLE_PER_CAT} 样本已生成")

            global_id += 1

        print(f"✓ {folder_name} 完成 ({PEOPLE_PER_CAT} 人)")

    print(f"\n{'='*60}")
    print(f"🎉 数据生成完成！总计 {global_id-1} 人")
    print(f"{'='*60}")
    print(f"输出路径: {os.path.abspath(BASE_PATH)}")
    print(f"噪声类型: 强力底噪、接触不良跳点、基线漂移")
    print(f"数据规模: {PEOPLE_PER_CAT * len(CATEGORIES)} 人 × {DURATION*FS} 点/人")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_robust_data()
