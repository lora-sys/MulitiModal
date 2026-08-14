"""
NK2预处理器实现
"""

import numpy as np
import neurokit2 as nk
import torch
from typing import Dict, Any
from interfaces import IPreprocessor, ProcessedData


class NK2Preprocessor(IPreprocessor):
    """NeuroKit2预处理器

    参数从配置文件读取，便于调整
    """

    def __init__(self, config: Dict):
        self.config = config
        self.sampling_rate = config.get("sampling_rate", 50)
        self.target_length = config.get("target_length", 1000)

        # 滤波参数
        self.filter_config = config.get("filter", {})
        self.lowcut = self.filter_config.get("lowcut", 0.1)
        self.highcut = self.filter_config.get("highcut", 15)
        self.order = self.filter_config.get("order", 2)

        # 归一化参数
        self.norm_config = config.get("normalization", {})
        self.dynamic_method = self.norm_config.get("dynamic", "zscore")
        self.static_method = self.norm_config.get("static", "standard")

    def process(self, raw_data) -> ProcessedData:
        """执行完整预处理流程"""
        # 1. 提取波形
        s1 = raw_data.raw_data["压力传感器1"].values
        s2 = raw_data.raw_data["压力传感器2"].values

        # 2. NK2滤波
        s1_filtered = nk.signal_filter(
            s1,
            sampling_rate=self.sampling_rate,
            lowcut=self.lowcut,
            highcut=self.highcut,
            method="butterworth",
            order=self.order,
        )
        s2_filtered = nk.signal_filter(
            s2,
            sampling_rate=self.sampling_rate,
            lowcut=self.lowcut,
            highcut=self.highcut,
            method="butterworth",
            order=self.order,
        )

        # 3. 重采样（确保长度一致）
        s1_resampled = nk.signal_resample(
            s1_filtered,
            sampling_rate=self.sampling_rate,
            desired_length=self.target_length,
        )
        s2_resampled = nk.signal_resample(
            s2_filtered,
            sampling_rate=self.sampling_rate,
            desired_length=self.target_length,
        )

        # 4. 动态特征归一化
        s1_norm = self._normalize(s1_resampled, self.dynamic_method)
        s2_norm = self._normalize(s2_resampled, self.dynamic_method)
        dynamic = np.stack([s1_norm, s2_norm])

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

    def _normalize(self, signal: np.ndarray, method: str) -> np.ndarray:
        """动态特征归一化"""
        if method == "zscore":
            return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        elif method == "minmax":
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
        return signal

    def _normalize_static(self, static: Dict) -> np.ndarray:
        """静态特征归一化"""
        # 使用经验值（后续可以从source获取准确统计）
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

        return np.array(normalized)

    def get_config(self) -> Dict:
        return self.config
