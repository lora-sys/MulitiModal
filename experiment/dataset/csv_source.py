"""
CSV数据源实现
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from interfaces import IDataSource, Sample


class CSVDataSource(IDataSource):
    """CSV文件数据源"""

    def __init__(self, base_path: str, label_map: Dict[str, int]):
        self.base_path = base_path
        self.label_map = label_map
        self._sample_list: List[Tuple[str, int, Dict]] = []
        self._statistics = {}

    def initialize(self) -> bool:
        """扫描文件夹，建立样本索引"""
        for folder_name, label_idx in self.label_map.items():
            folder_path = os.path.join(self.base_path, folder_name)
            if not os.path.exists(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(folder_path, file_name)
                    # 提取静态特征
                    static = self._parse_filename(file_name)
                    self._sample_list.append((file_path, label_idx, static))

        print(f"📦 CSVDataSource initialized: {len(self._sample_list)} samples")
        return True

    def _parse_filename(self, filename: str) -> Dict[str, float]:
        """从文件名提取静态特征"""
        numbers = re.findall(r"\d+", filename)
        if len(numbers) >= 5:
            return {
                "weight": float(numbers[1]),
                "hr": float(numbers[2]),
                "spo2": float(numbers[3]),
                "height": float(numbers[4]),
            }
        return {"weight": 0, "hr": 0, "spo2": 0, "height": 0}

    def get_sample_list(self) -> List:
        return self._sample_list

    def load_sample(self, sample_id: Tuple) -> Sample:
        """加载单个CSV文件"""
        file_path, label, static = sample_id
        df = pd.read_csv(file_path)

        return Sample(
            sample_id=file_path,
            raw_data=df,
            metadata={"label": label, "static": static},
        )

    def get_statistics(self) -> Dict:
        """计算全局统计量"""
        if not self._statistics:
            # 懒加载计算
            weights, hrs, spo2s, heights = [], [], [], []
            for _, _, static in self._sample_list:
                weights.append(static["weight"])
                hrs.append(static["hr"])
                spo2s.append(static["spo2"])
                heights.append(static["height"])

            self._statistics = {
                "weight": {"mean": np.mean(weights), "std": np.std(weights)},
                "hr": {"mean": np.mean(hrs), "std": np.std(hrs)},
                "spo2": {"mean": np.mean(spo2s), "std": np.std(spo2s)},
                "height": {"mean": np.mean(heights), "std": np.std(heights)},
            }
        return self._statistics
    
class NPZDataSource(IDataSource):
    """npz文件数据源"""
    
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        self._data = None
        self._sample_list = []
        self._statistics = {}
    
    def initialize(self) -> bool:
        # TODO: 加载 npz 文件
        self._data = np.load(self.npz_path)
        
        # TODO: 构建样本列表（索引从 0 到 N-1）
        n_samples = self._data['Y'].shape[0]
        self._sample_list = list(range(n_samples))
        
        print(f"📦 NPZDataSource initialized: {len(self._sample_list)} samples")
        return True
    
    def get_sample_list(self) -> List:
        return self._sample_list
    
    def load_sample(self, sample_id: int) -> Sample:
        # TODO: 根据索引提取数据，包装成 Sample
        X_dynamic = self._data['X_dynamic']  # (N, 2, 1000)
        X_static = self._data['X_static']   # (N, 4)
        Y = self._data['Y']                   # (N,)
        
        # 提取单个样本
        s1 = X_dynamic[sample_id, 0, :]  # 传感器1
        s2 = X_dynamic[sample_id, 1, :]  # 传感器2
        static = X_static[sample_id]       # 静态特征
        label = Y[sample_id]              # 标签
        
        # 包装成 DataFrame（兼容 NK2Preprocessor）
        df = pd.DataFrame({
            '压力传感器1': s1,
            '压力传感器2': s2
        })
        
        # 静态特征字典
        static_dict = {
            'weight': static[0],
            'hr': static[1],
            'spo2': static[2],
            'height': static[3]
        }
        
        return Sample(
            sample_id=str(sample_id),
            raw_data=df,
            metadata={'label': int(label), 'static': static_dict}
        )
    
    def get_statistics(self) -> Dict:
        # TODO: 计算静态特征的统计量
        if not self._statistics and self._data is not None:
            X_static = self._data['X_static']  # (N, 4)
            
            self._statistics = {
                'weight': {'mean': float(X_static[:, 0].mean()), 'std': float(X_static[:, 0].std())},
                'hr': {'mean': float(X_static[:, 1].mean()), 'std': float(X_static[:, 1].std())},
                'spo2': {'mean': float(X_static[:, 2].mean()), 'std': float(X_static[:, 2].std())},
                'height': {'mean': float(X_static[:, 3].mean()), 'std': float(X_static[:, 3].std())},
            }
        return self._statistics
