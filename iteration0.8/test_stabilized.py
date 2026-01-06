from collections import deque

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from processor import MassageSignalProcessor

# 1. 加载模型
model_path = "D:/repos/mulitModal/iteration0.8/massage_ai_model.pkl"
model = joblib.load(model_path)
processor = MassageSignalProcessor(fs=50)
mode_names = ["空载", "柔和", "深度"]

# 2. 模拟生成同样的 15s 信号 (包含那个刺头噪声)
fs = 50
t = np.linspace(0, 15, 15 * fs)
signal = np.zeros_like(t)
signal[0 : 5 * fs] = 20 + np.random.normal(0, 1.5, 5 * fs)  # 空载
signal[5 * fs : 10 * fs] = (
    40
    + 10 * np.sin(2 * np.pi * 0.1 * t[5 * fs : 10 * fs])
    + np.random.normal(0, 1.5, 5 * fs)
)  # 柔和
signal[10 * fs : 15 * fs] = (
    60
    + 30 * np.sin(2 * np.pi * 0.5 * t[10 * fs : 15 * fs])
    + np.random.normal(0, 1.5, 5 * fs)
)  # 深度

# 在 8 秒处注入那个致命的“刺头”噪声
signal[int(8.0 * fs)] += 60

# 3. 核心：带防抖的实时监测
window_sec = 2
step_sec = 0.5
raw_preds = []  # 原始 AI 结果
stable_preds = []  # 经过防抖后的结果
timeline = []

# 防抖缓存：记录最近 3 次的预测结果
debounce_buffer = deque(maxlen=3)
current_confirmed_mode = 0  # 初始认定为空载

print("🧠 AI 正在通过【防抖逻辑】进行监测...")

for start_t in np.arange(0, 15 - window_sec, step_sec):
    start_idx = int(start_t * fs)
    end_idx = int((start_t + window_sec) * fs)
    chunk = signal[start_idx:end_idx]

    # AI 原始预测
    clean_chunk = processor.clean_signal(chunk)
    feats = processor.extract_features(clean_chunk)
    raw_pred = model.predict(pd.DataFrame([feats]))[0]
    raw_preds.append(raw_pred)

    # --- 防抖逻辑开始 ---
    debounce_buffer.append(raw_pred)

    # 逻辑：只有当缓存满了，且里面 3 个结果全一样时，才更新“确认模式”
    if len(debounce_buffer) == 3:
        # 如果缓存里所有元素都等于同一个值
        if all(x == debounce_buffer[0] for x in debounce_buffer):
            current_confirmed_mode = debounce_buffer[0]

    stable_preds.append(current_confirmed_mode)
    # --- 防抖逻辑结束 ---

    timeline.append(start_t + window_sec)

# 4. 可视化对比：原始 AI vs 防抖 AI
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(figsize=(12, 10))

# 子图1: 信号
plt.subplot(3, 1, 1)
plt.plot(t, signal, color="#bdc3c7", label="原始信号 (含 8s 处尖峰)")
plt.title("防抖逻辑效果对比")
plt.legend()

# 子图2: 原始 AI 预测 (会跳变)
plt.subplot(3, 1, 2)
plt.step(
    timeline,
    raw_preds,
    where="post",
    color="#9b59b6",
    alpha=0.5,
    label="原始 AI (易受干扰)",
)
plt.yticks([0, 1, 2], mode_names)
plt.ylabel("未处理结果")
plt.legend()

# 子图3: 防抖后的 AI 预测 (稳如老狗)
plt.subplot(3, 1, 3)
plt.step(
    timeline,
    stable_preds,
    where="post",
    color="#2ecc71",
    lw=3,
    label="防抖 AI (平滑稳定)",
)
plt.yticks([0, 1, 2], mode_names)
plt.ylabel("确认结果")
plt.xlabel("时间 (秒)")
plt.legend()

plt.tight_layout()
plt.savefig("D:/repos/mulitModal/iteration0.8/stabilized_test_report.png")
print("\n✅ 防抖模拟完成！请查看生成的对比图：stabilized_test_report.png")
plt.show()
