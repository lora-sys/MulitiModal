import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from processor import MassageSignalProcessor

# 1. 加载模型和处理器
# 确保你已经运行了 train_model.py 并生成了模型文件
model_path = "D:/repos/mulitModal/iteration0.8/massage_ai_model.pkl"
try:
    model = joblib.load(model_path)
except:
    print("❌ 找不到模型文件，请先运行 train_model.py")
    exit()

processor = MassageSignalProcessor(fs=50)
mode_names = ["空载", "柔和", "深度"]

# 2. 模拟生成一段“动态实战”信号 (15秒)
# 我们让信号在 5s 和 10s 的时候发生突变
fs = 50
t = np.linspace(0, 15, 15 * fs)
signal = np.zeros_like(t)

print("📡 正在生成实时传感器模拟信号...")
# 0-5秒: 空载 (基础20 + 噪声)
signal[0 : 5 * fs] = 20 + np.random.normal(0, 1.5, 5 * fs)

# 5-10秒: 柔和 (基础40 + 0.1Hz波动)
t_soft = t[5 * fs : 10 * fs]
signal[5 * fs : 10 * fs] = (
    40 + 10 * np.sin(2 * np.pi * 0.1 * t_soft) + np.random.normal(0, 1.5, 5 * fs)
)

# 10-15秒: 深度 (基础60 + 0.5Hz剧烈波动)
t_deep = t[10 * fs : 15 * fs]
signal[10 * fs : 15 * fs] = (
    60 + 30 * np.sin(2 * np.pi * 0.5 * t_deep) + np.random.normal(0, 1.5, 5 * fs)
)

# 随机注入 5 个刺头干扰（模拟碰撞或传感器抖动）
for _ in range(5):
    signal[np.random.randint(0, len(t))] += 40

# 3. 模拟滑动窗口预测 (每 0.5 秒预测一次)
window_sec = 2
step_sec = 0.5
predictions = []
timeline = []

print("🧠 AI 正在实时监测中...")
for start_t in np.arange(0, 15 - window_sec, step_sec):
    start_idx = int(start_t * fs)
    end_idx = int((start_t + window_sec) * fs)

    # 截取当前 2 秒的信号片段
    chunk = signal[start_idx:end_idx]

    # 清洗并提取特征
    clean_chunk = processor.clean_signal(chunk)
    feats = processor.extract_features(clean_chunk)

    # 将字典转为 DataFrame 喂给模型
    feat_df = pd.DataFrame([feats])
    pred_id = model.predict(feat_df)[0]

    # 记录结果
    predictions.append(pred_id)
    current_time = start_t + window_sec
    timeline.append(current_time)

    # 控制台实时打印
    print(f" [时间: {current_time:>4.1f}s] AI 判断结果: 【{mode_names[pred_id]}】")

# 4. 可视化“实战结果”
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示问题
plt.figure(figsize=(12, 8))

# 子图1: 原始信号流
plt.subplot(2, 1, 1)
plt.plot(t, signal, color="#bdc3c7", alpha=0.8, label="原始传感器波形")
plt.title("按摩模式实战检测: 信号流 vs AI 决策")
plt.ylabel("压力值")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# 子图2: AI 预测阶梯图
plt.subplot(2, 1, 2)
plt.step(
    timeline, predictions, where="post", color="#e74c3c", lw=2.5, label="AI 识别模式"
)
plt.yticks([0, 1, 2], mode_names)
plt.ylim(-0.5, 2.5)
plt.xlabel("时间 (秒)")
plt.ylabel("AI 决策结果")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("D:/repos/mulitModal/iteration0.8/realtime_test_report.png")
print("\n✅ 实战模拟完成！结果图表已保存至: realtime_test_report.png")
plt.show()
