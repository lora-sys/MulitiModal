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
    在初始化时预加载所有数据到内存，提高访问速度

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

        # 预加载所有数据到内存（大幅提高访问速度）
        if hasattr(source, '_data') and source._data is not None:
            print("📦 预加载数据到内存...")
            self._dynamic = torch.from_numpy(source._data['dynamic'].astype(np.float32))
            self._static_basic = torch.from_numpy(source._data['static_basic'].astype(np.float32))
            self._static_scores = torch.from_numpy(source._data['static_scores'].astype(np.float32))
            self._constitution = torch.from_numpy(source._data['constitution'].astype(np.int64))
            self._labels = torch.from_numpy(source._data['labels'].astype(np.int64)) - 1  # 标签映射：1,2,3 -> 0,1,2
            print(f"✅ 预加载完成: {len(self._labels)} 样本")
        else:
            self._dynamic = None
            self._static_basic = None
            self._static_scores = None
            self._constitution = None
            self._labels = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本（字典格式的 Tensor）"""
        # 使用预加载的数据（快速访问）
        if self._dynamic is not None:
            return {
                'dynamic': self._dynamic[idx],
                'static_basic': self._static_basic[idx],
                'static_scores': self._static_scores[idx],
                'constitution': self._constitution[idx],
                'label': self._labels[idx],
            }

        # 回退到原始方法（慢速访问）
        sample_id = self._sample_ids[idx]
        raw_sample = self.source.load_sample(sample_id)

        if self.preprocessor is not None:
            processed = self.preprocessor.process(raw_sample)
            item = {
                'dynamic': processed.dynamic,
                'static_basic': processed.static,
                'label': processed.label - 1,
            }
        else:
            raw_data = raw_sample.raw_data
            dynamic = torch.from_numpy(raw_data['dynamic'].astype(np.float32))
            static_basic = torch.from_numpy(raw_data['static_basic'].astype(np.float32))
            static_scores = torch.from_numpy(raw_data['static_scores'].astype(np.float32))
            constitution = torch.tensor(raw_data['constitution'], dtype=torch.long)
            label = torch.tensor(raw_sample.metadata['label'] - 1, dtype=torch.long)

            item = {
                'dynamic': dynamic,
                'static_basic': static_basic,
                'static_scores': static_scores,
                'constitution': constitution,
                'label': label,
            }

        return item