"""
UnifiedMultimodalDataset - 统一多模态数据集实现
继承 BaseDataset，使用 UnifiedNPZDataSource，返回字典格式的 Tensor 数据
"""

import numpy as np
import torch
from typing import Dict
from interfaces import BaseDataset, IDataSource, IPreprocessor


class UnifiedMultimodalDataset(BaseDataset):
    """统一多模态数据集

    使用 UnifiedNPZDataSource 加载 NPZ 数据
    在 __getitem__ 中将 numpy 数组转换为 Tensor，返回字典格式

    返回格式:
    {
        'dynamic': Tensor (2, 1000),
        'static_basic': Tensor (4,),
        'static_scores': Tensor (2,),
        'constitution': Tensor (scalar),
        'label': Tensor (scalar)
    }
    """

    def __init__(self, source: IDataSource, preprocessor: IPreprocessor = None):
        """初始化数据集

        Args:
            source: UnifiedNPZDataSource 实例
            preprocessor: 可选的预处理器（NPZ数据已预处理，通常为None）
        """
        super().__init__(source, preprocessor)

        # 初始化数据源
        source.initialize()

        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本（字典格式的 Tensor）"""
        # 1. 获取样本ID
        sample_id = self._sample_ids[idx]

        # 2. 加载原始数据
        raw_sample = self.source.load_sample(sample_id)

        # 3. 预处理（可选，NPZ数据通常已预处理）
        if self.preprocessor is not None:
            processed = self.preprocessor.process(raw_sample)
            item = {
                'dynamic': processed.dynamic,
                'static_basic': processed.static,
                'label': processed.label - 1,  # 标签映射：1, 2, 3 -> 0, 1, 2
            }
        else:
            # 直接从 NPZ 数据转换为 Tensor
            raw_data = raw_sample.raw_data

            # 转换为 Tensor
            dynamic = torch.from_numpy(raw_data['dynamic'].astype(np.float32))
            static_basic = torch.from_numpy(raw_data['static_basic'].astype(np.float32))
            static_scores = torch.from_numpy(raw_data['static_scores'].astype(np.float32))
            constitution = torch.tensor(raw_data['constitution'], dtype=torch.long)
            
            # 标签映射：1, 2, 3 -> 0, 1, 2（PyTorch CrossEntropyLoss 要求标签从 0 开始）
            label = torch.tensor(raw_sample.metadata['label'] - 1, dtype=torch.long)

            item = {
                'dynamic': dynamic,
                'static_basic': static_basic,
                'static_scores': static_scores,
                'constitution': constitution,
                'label': label,
            }

        return item