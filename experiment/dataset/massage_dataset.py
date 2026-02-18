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
        processed = self.preprocessor.process(raw_sample)

        # 4. 构建输出
        item = {
            "dynamic": processed.dynamic,  # [2, 1000]
            "static": processed.static,  # [4]
            "label": processed.label,  # scalar
        }

        # 5. 可选变换
        if self.transform:
            item = self.transform(item)

        return item
