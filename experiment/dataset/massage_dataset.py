"""
MassageDataset - 具体实现
"""

import torch
from typing import Dict
from interfaces import BaseDataset, IDataSource, IPreprocessor


class MassageDataset(BaseDataset):
    """按摩椅数据集 - 具体实现"""

    def __init__(
        self, source: IDataSource, preprocessor: IPreprocessor, transform=None
    ):
        # 初始化父类
        super().__init__(source, preprocessor)

        # 初始化数据源
        source.initialize()

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本"""
        # 1. 获取样本ID
        sample_id = self._sample_ids[idx]

        # 2. 加载原始数据
        raw_sample = self.source.load_sample(sample_id)

        # 3. 预处理
        if self.preprocessor is None:
            
            raw_df = raw_sample.raw_data  # DataFrame with 压力传感器1, 压力传感器2
            static = raw_sample.metadata["static"]
            
            item = {
                "dynamic": torch.tensor(raw_df.values.T, dtype=torch.float32),  # (2, 1000)
                "static": torch.tensor([
                    static["weight"], static["hr"], static["spo2"], static["height"]
                ], dtype=torch.float32),
                "label": torch.tensor(raw_sample.metadata["label"], dtype=torch.long),
            }
        else:
            processed = self.preprocessor.process(raw_sample)
            item = {
                "dynamic": processed.dynamic,
                "static": processed.static,
                "label": processed.label,
            }
        # 5. 可选变换
        if self.transform:
            item = self.transform(item)

        return item
