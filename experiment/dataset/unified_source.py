"""
UnifiedNPZDataSource - 统一NPZ数据源实现
实现 IDataSource 接口，加载包含完整特征的 NPZ 文件
"""

import numpy as np
from typing import List, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        try:
            # 检查文件是否存在
            if not os.path.exists(self.npz_path):
                raise FileNotFoundError(f"NPZ文件不存在: {self.npz_path}")

            # 加载NPZ文件
            self._data = np.load(self.npz_path)

            # 验证必需的字段
            required_keys = ['dynamic', 'static_basic', 'static_scores', 'constitution', 'labels']
            for key in required_keys:
                if key not in self._data:
                    raise KeyError(f"NPZ文件缺少必需字段: {key}")

            # 验证数据形状
            n_samples = self._data['dynamic'].shape[0]

            # 验证各字段形状一致
            if self._data['static_basic'].shape[0] != n_samples:
                raise ValueError(f"static_basic样本数不一致: {self._data['static_basic'].shape[0]} != {n_samples}")
            if self._data['static_scores'].shape[0] != n_samples:
                raise ValueError(f"static_scores样本数不一致: {self._data['static_scores'].shape[0]} != {n_samples}")
            if self._data['constitution'].shape[0] != n_samples:
                raise ValueError(f"constitution样本数不一致: {self._data['constitution'].shape[0]} != {n_samples}")
            if self._data['labels'].shape[0] != n_samples:
                raise ValueError(f"labels样本数不一致: {self._data['labels'].shape[0]} != {n_samples}")

            # 样本ID列表
            self._sample_list = list(range(n_samples))

            print(f"📦 UnifiedNPZDataSource initialized: {len(self._sample_list)} samples")
            return True

        except FileNotFoundError as e:
            print(f"❌ 错误: {e}")
            return False
        except KeyError as e:
            print(f"❌ 错误: {e}")
            return False
        except ValueError as e:
            print(f"❌ 错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            return False

    def get_sample_list(self) -> List:
        """获取样本ID列表"""
        return self._sample_list

    def load_sample(self, sample_id: int) -> Sample:
        """根据ID加载单个样本"""
        try:
            # 验证样本ID
            if sample_id < 0 or sample_id >= len(self._sample_list):
                raise IndexError(f"样本ID超出范围: {sample_id} (0-{len(self._sample_list)-1})")

            if self._data is None:
                raise RuntimeError("数据源未初始化，请先调用initialize()")

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

        except IndexError as e:
            print(f"❌ 加载样本错误: {e}")
            raise
        except RuntimeError as e:
            print(f"❌ 加载样本错误: {e}")
            raise
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            raise

    def get_statistics(self) -> Dict:
        """获取数据统计信息（用于归一化）"""
        try:
            if self._data is None:
                raise RuntimeError("数据源未初始化，请先调用initialize()")

            if not self._statistics:
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

        except RuntimeError as e:
            print(f"❌ 获取统计信息错误: {e}")
            raise
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            raise