"""
UnifiedNPZDataSource - 统一NPZ数据源实现
实现 IDataSource 接口，加载包含完整特征的 NPZ 文件
"""

import numpy as np
from typing import List, Dict
from interfaces import IDataSource, Sample


# 体质类型映射表 (38种体质 -> ID 0-37)
CONSTITUTION_MAP = {
    '痰湿': 0, '气虚痰湿': 1, '平和': 2, '气虚': 3, '湿热': 4,
    '气虚湿热': 5, '痰瘀互结': 6, '血虚': 7, '阳虚': 8, '实热': 9,
    '气虚痰湿夹瘀': 10, '气郁': 11, '湿热血瘀': 12, '瘀热互结': 13,
    '阴虚': 14, '气血两虚': 15, '气滞血瘀': 16, '阴虚痰湿': 17,
    '气虚湿热夹瘀': 18, '血瘀': 19, '阳虚湿热': 20, '阳虚痰湿夹瘀': 21,
    '气虚血瘀': 22, '阳虚血瘀': 23, '阴虚湿热': 24, '阳虚湿热夹瘀': 25,
    '血虚血瘀': 26, '气血两虚夹瘀': 27, '阳虚痰湿': 28, '阴虚血瘀': 29,
    '阴虚湿热夹瘀': 30, '阴虚痰湿夹瘀': 31, '表寒里热': 32, '阴虚阳亢': 33,
    '特禀': 34, '阴虚阳亢夹瘀': 35, '表热里寒': 36, '表寒里热夹瘀': 37,
}


class UnifiedNPZDataSource(IDataSource):
    """统一NPZ数据源

    NPZ 文件结构:
    - dynamic:       (N, 2, 1000)  压力波形
    - static_basic:  (N, 4)       [年龄, BMI, 血氧, 心率]
    - static_scores: (N, 2)       [健康指数, 诊断得分]
    - constitution:  (N,)         体质ID
    - labels:        (N,)         分类标签
    """

    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        self._data = None
        self._sample_list = []
        self._statistics = {}

    def initialize(self) -> bool:
        """初始化数据源，加载 NPZ 文件"""
        self._data = np.load(self.npz_path)

        # 样本ID列表
        n_samples = self._data['dynamic'].shape[0]
        self._sample_list = list(range(n_samples))

        print(f"📦 UnifiedNPZDataSource initialized: {len(self._sample_list)} samples")
        return True

    def get_sample_list(self) -> List:
        """获取样本ID列表"""
        return self._sample_list

    def load_sample(self, sample_id: int) -> Sample:
        """根据ID加载单个样本"""
        dynamic = self._data['dynamic'][sample_id]          # (2, 1000)
        static_basic = self._data['static_basic'][sample_id]  # (4,)
        static_scores = self._data['static_scores'][sample_id] # (2,)
        constitution = self._data['constitution'][sample_id]   # scalar
        label = self._data['labels'][sample_id]               # scalar

        return Sample(
            sample_id=str(sample_id),
            raw_data={
                'dynamic': dynamic,
                'static_basic': static_basic,
                'static_scores': static_scores,
                'constitution': constitution,
            },
            metadata={'label': int(label)},
        )

    def get_statistics(self) -> Dict:
        """获取数据统计信息（用于归一化）"""
        if not self._statistics and self._data is not None:
            static_basic = self._data['static_basic']
            static_scores = self._data['static_scores']

            self._statistics = {
                'static_basic': {
                    'age': {'mean': float(static_basic[:, 0].mean()), 'std': float(static_basic[:, 0].std())},
                    'bmi': {'mean': float(static_basic[:, 1].mean()), 'std': float(static_basic[:, 1].std())},
                    'spo2': {'mean': float(static_basic[:, 2].mean()), 'std': float(static_basic[:, 2].std())},
                    'hr': {'mean': float(static_basic[:, 3].mean()), 'std': float(static_basic[:, 3].std())},
                },
                'static_scores': {
                    'health': {'mean': float(static_scores[:, 0].mean()), 'std': float(static_scores[:, 0].std())},
                    'diagnosis': {'mean': float(static_scores[:, 1].mean()), 'std': float(static_scores[:, 1].std())},
                },
            }
        return self._statistics