"""
自研信号自愈预处理器
整合 3-Sigma 异常检测 + 样条插值修复 + NK2滤波
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import torch
from typing import Dict, Any
from interfaces import IPreprocessor, ProcessedData


class SelfHealingPreprocessor(IPreprocessor):
    """自研信号自愈预处理器

    特点：
    1. 3-Sigma 异常检测
    2. 样条插值修复
    3. NeuroKit2 低通滤波
    4. Z-Score 标准化
    """

    def __init__(self, config: Dict):
        self.config = config
        self.sampling_rate = config.get("sampling_rate", 50)
        self.target_length = config.get("target_length", 1000)

        # 自愈参数
        self.healing_config = config.get("healing", {})
        self.window_size = self.healing_config.get("window_size", 15)
        self.sigma_threshold = self.healing_config.get("sigma_threshold", 3)

        # 滤波参数
        self.filter_config = config.get("filter", {})
        self.highcut = self.filter_config.get("highcut", 10)

        # 归一化参数
        self.norm_config = config.get("normalization", {})
        self.dynamic_method = self.norm_config.get("dynamic", "zscore")

    def self_healing_pipeline(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        信号自愈 Pipeline

        步骤：
        1. 3-Sigma 异常检测
        2. NaN 隔离与样条插值
        3. NeuroKit2 低通平滑
        4. Z-Score 标准化
        """
        # 转换为 Pandas Series 方便处理
        s = pd.Series(raw_signal)

        # 1. 动态阈值检测 (Rolling 3-Sigma)
        rolling_mean = s.rolling(
            window=self.window_size, center=True, min_periods=1
        ).mean()

        rolling_std = s.rolling(
            window=self.window_size, center=True, min_periods=1
        ).std()

        # 填补 std 边缘的 NaN
        rolling_std = rolling_std.bfill().ffill()

        # 识别异常点
        is_anomaly = (s > rolling_mean + self.sigma_threshold * rolling_std) | (
            s < rolling_mean - self.sigma_threshold * rolling_std
        )

        # 统计异常点数量
        anomaly_count = is_anomaly.sum()

        # 2. 隔离与修复 (NaN Masking & Interpolation)
        s_clean = s.copy()
        s_clean[is_anomaly] = np.nan

        # 使用三次样条插值恢复波形
        s_clean = s_clean.interpolate(method="cubic")

        # 边缘用 ffill/bfill 兜底
        s_clean = s_clean.bfill().ffill()

        # 3. NeuroKit2 专业滤波 (消除剩余的高频底噪)
        s_filtered = nk.signal_filter(
            s_clean.values,
            sampling_rate=self.sampling_rate,
            highcut=self.highcut,
            method="butterworth",
        )

        # 4. Z-Score 归一化
        s_norm = (s_filtered - np.mean(s_filtered)) / (np.std(s_filtered) + 1e-6)

        return s_norm.astype(np.float32)

    def process(self, raw_data) -> ProcessedData:
        """执行完整预处理流程"""
        # 1. 提取波形
        s1 = raw_data.raw_data["压力传感器1"].values
        s2 = raw_data.raw_data["压力传感器2"].values

        # 2. 信号自愈处理
        s1_healed = self.self_healing_pipeline(s1)
        s2_healed = self.self_healing_pipeline(s2)

        # 3. 重采样（确保长度一致）
        s1_resampled = nk.signal_resample(
            s1_healed,
            sampling_rate=self.sampling_rate,
            desired_length=self.target_length,
        )
        s2_resampled = nk.signal_resample(
            s2_healed,
            sampling_rate=self.sampling_rate,
            desired_length=self.target_length,
        )

        # 4. 组合动态特征
        dynamic = np.stack([s1_resampled, s2_resampled])

        # 5. 静态特征归一化
        static_raw = raw_data.metadata["static"]
        static = self._normalize_static(static_raw)

        # 6. 标签
        label = raw_data.metadata["label"]

        return ProcessedData(
            dynamic=torch.tensor(dynamic, dtype=torch.float32),
            static=torch.tensor(static, dtype=torch.float32),
            label=torch.tensor(label, dtype=torch.long),
        )

    def _normalize_static(self, static: Dict) -> np.ndarray:
        """静态特征归一化"""
        stats = {
            "weight": (65.0, 15.0),
            "hr": (75.0, 15.0),
            "spo2": (97.0, 2.0),
            "height": (170.0, 10.0),
        }

        normalized = []
        for key in ["weight", "hr", "spo2", "height"]:
            mean, std = stats[key]
            normalized.append((static[key] - mean) / (std + 1e-8))

        return np.array(normalized, dtype=np.float32)

    def get_config(self) -> Dict:
        return self.config
