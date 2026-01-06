import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


class MassageSignalProcessor:
    """信号处理器：整合了 0.6 的清洗和特征提取逻辑"""

    def __init__(self, fs=50):
        self.fs = fs

    def clean_signal(self, raw_signal):
        """专业清洗逻辑 (来自 0.6 的 3-Sigma + 插值)"""
        df = pd.DataFrame({"raw": raw_signal})
        rolling = df["raw"].rolling(window=15, center=True)
        mu, std = rolling.mean(), rolling.std()

        # 异常检测
        is_anomaly = (df["raw"] > mu + 3 * std) | (df["raw"] < mu - 3 * std)

        # 修复
        df["clean"] = df["raw"].copy()
        df.loc[is_anomaly, "clean"] = np.nan
        df["clean"] = (
            df["clean"].interpolate().fillna(method="bfill").fillna(method="ffill")
        )

        # 平滑
        return df["clean"].rolling(window=15, center=True, min_periods=1).mean().values

    def extract_features(self, signal, window_sec=2):
        """特征提取逻辑 (来自 0.6 的时域+频域分析)"""
        window_size = int(window_sec * self.fs)
        # 提取一个窗口内的核心特征
        feat_mean = np.mean(signal)
        feat_std = np.std(signal)
        feat_ptp = np.ptp(signal)

        # 频域特征 (FFT)
        yf = fft(signal - feat_mean)
        xf = fftfreq(len(signal), 1 / self.fs)
        idx = np.argmax(np.abs(yf[: len(signal) // 2]))
        dom_freq = np.abs(xf[idx])

        return {
            "mean": feat_mean,
            "std": feat_std,
            "ptp": feat_ptp,
            "dom_freq": dom_freq,
        }
