import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


class FeatureExtractor:
    def __init__(self, sampling_rate=50):
        self.fs = sampling_rate

    def extract_from_window(self, window_data):
        """对一个窗口的数据提取所有特征"""
        # --- 1. 时域特征 ---
        feat_mean = np.mean(window_data)
        feat_std = np.std(window_data)
        feat_ptp = np.ptp(window_data)  # 峰峰值
        feat_rms = np.sqrt(np.mean(window_data**2))  # 均值方差

        # --- 2. 频域特征 (FFT) ---
        n = len(window_data)
        # 减去均值防止直流分量(0Hz)干扰频率识别
        yf = fft(window_data - feat_mean)
        xf = fftfreq(n, 1 / self.fs)

        # 找到振幅最大的正频率点
        idx = np.argmax(np.abs(yf[: n // 2]))
        dom_freq = np.abs(xf[idx])

        return {
            "mean": feat_mean,
            "std": feat_std,
            "ptp": feat_ptp,
            "rms": feat_rms,
            "dom_freq": dom_freq,
        }

    def transform(self, signal, window_sec=2, step_sec=1):
        """滑动窗口遍历整个信号"""
        window_size = int(window_sec * self.fs)
        step_size = int(step_sec * self.fs)

        feature_list = []
        for start in range(0, len(signal) - window_size, step_size):
            end = start + window_size
            window = signal[start:end]

            features = self.extract_from_window(window)
            features["start_time"] = start / self.fs
            feature_list.append(features)

        return pd.DataFrame(feature_list)


# 1. 检查并加载数据
data_path = "../iteration0.5/cleaned_pressure_final.csv"
if os.path.exists(data_path):
    df_recovered = pd.read_csv(data_path)
    signal = df_recovered["final_recovered"].values

    # 2. 初始化提取器
    extractor = FeatureExtractor(sampling_rate=50)

    # 3. 提取特征矩阵
    feature_matrix = extractor.transform(
        signal, window_sec=2, step_sec=0.5
    )  # 缩小步长让图表更细致

    # --- 4. 导出特征矩阵 CSV ---
    output_csv = "signal_features_matrix.csv"
    feature_matrix.to_csv(output_csv, index=False, float_format="%.4f")
    print(f"✅ 特征矩阵已导出至: {output_csv}")

    # --- 5. 直观可视化：特征仪表盘 ---
    plt.figure(figsize=(14, 10))

    # 子图1: 原始清洗后的信号
    plt.subplot(4, 1, 1)
    plt.plot(df_recovered["timestamp"], signal, color="green", label="Cleaned Signal")
    plt.title("Time Domain: Cleaned Pressure Signal")
    plt.ylabel("Pressure")
    plt.grid(alpha=0.3)

    # 子图2: 均值特征 (反映基本趋势)
    plt.subplot(4, 1, 2)
    plt.plot(feature_matrix["start_time"], feature_matrix["mean"], color="blue", lw=2)
    plt.title("Feature: Mean (Overall Trend)")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)

    # 子图3: 标准差特征 (反映波动的剧烈程度)
    plt.subplot(4, 1, 3)
    plt.plot(feature_matrix["start_time"], feature_matrix["std"], color="purple", lw=2)
    plt.title("Feature: Standard Deviation (Stability/Noise Level)")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)

    # 子图4: 主频特征 (反映是否有周期性震动)
    plt.subplot(4, 1, 4)
    plt.scatter(
        feature_matrix["start_time"], feature_matrix["dom_freq"], color="red", s=10
    )
    plt.title("Feature: Dominant Frequency (Vibration Analysis)")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_dashboard.png")
    plt.show()

else:
    print(f"❌ 找不到数据文件，请确保路径正确: {data_path}")
