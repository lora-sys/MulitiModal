import numpy as np
import pandas as pd
from processor import MassageSignalProcessor


def generate_sample(mode, duration=2):
    """模拟不同模式下的 2 秒传感器原始信号"""
    fs = 50
    t = np.linspace(0, duration, duration * fs)
    if mode == 0:  # 空载
        base, amp, freq = 20, 2, 0.05
    elif mode == 1:  # 柔和
        base, amp, freq = 40, 10, 0.1
    else:  # 深度
        base, amp, freq = 60, 30, 0.5

    signal = base + amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 2, len(t))
    # 随机注入点刺噪声 (模拟 0.6 的投毒)
    if np.random.rand() > 0.8:
        signal[np.random.randint(0, len(t))] += 50
    return signal


# 开始制作数据集
processor = MassageSignalProcessor()
dataset = []

print("🧪 正在模拟录制数据并提取特征...")
for mode_id, mode_name in enumerate(["空载", "柔和", "深度"]):
    for _ in range(200):  # 每种模式生成 200 组样本
        raw = generate_sample(mode_id)
        clean = processor.clean_signal(raw)
        feats = processor.extract_features(clean)
        feats["label"] = mode_id
        dataset.append(feats)

df = pd.DataFrame(dataset)
df.to_csv("D:/repos/mulitModal/iteration0.8/training_dataset.csv", index=False)
print(f"✅ 数据集制作完成！共 {len(df)} 组带标签的数据。已保存至 training_dataset.csv")
