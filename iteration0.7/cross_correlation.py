import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# 1. 读取数据（确保有样条插值后的心率）
aligned = pd.read_csv("sensor_aligned_filled.csv")
aligned['hr_cubic'] = aligned['hr'].interpolate(method='cubic')
# 2. 准备数据
p = aligned['pressure'].values
h = aligned['hr_cubic'].bfill().ffill().values  # 头尾NaN用bfill/ffill处理
# 3. 标准化（必须做）
p_norm = (p - np.mean(p)) / np.std(p)
h_norm = (h - np.mean(h)) / np.std(h)
# 4. 计算互相关
corr = signal.correlate(p_norm, h_norm, mode='full')
lags = signal.correlation_lags(len(p_norm), len(h_norm))
# 5. 找最大相关点
best_lag_samples = lags[np.argmax(corr)]
delay_sec = best_lag_samples / 50.0  # 50Hz
print(f"最大相关系数点: 滞后 {best_lag_samples} 个采样点")
print(f"心率相对于压力的延迟: {delay_sec:.2f} 秒")
# 6. 可视化
plt.figure(figsize=(10, 4))
plt.plot(lags / 50.0, corr)
plt.axvline(delay_sec, color='r', linestyle='--', label=f'Best Lag: {delay_sec}s')
plt.xlabel("Delay (seconds)")
plt.ylabel("Correlation Strength")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("cross_correlation.png")
plt.show()