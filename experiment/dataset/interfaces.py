"""
Dataset模块 - 抽象接口定义
为未来扩展预留接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    """单个样本的抽象表示"""

    sample_id: str
    raw_data: Any  # 原始数据
    metadata: Optional[Dict[str, Any]] = None  # 元数据


@dataclass
class ProcessedData:
    """预处理后的数据"""

    dynamic: torch.Tensor  # [2, 1000] 动态特征（波形）
    static: torch.Tensor  # [4] 静态特征
    label: torch.Tensor  # 标签


class IDataSource(ABC):
    """数据源抽象接口

    未来新增数据源只需实现此接口：
    - CSVDataSource: 当前CSV文件
    - DatabaseDataSource: 数据库
    - StreamDataSource: 实时流
    """

    @abstractmethod
    def initialize(self) -> bool:
        """初始化数据源"""
        pass

    @abstractmethod
    def get_sample_list(self) -> List:
        """获取样本ID列表"""
        pass

    @abstractmethod
    def load_sample(self, sample_id) -> Sample:
        """根据ID加载单个样本"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """获取数据统计信息（用于归一化）"""
        pass


class IPreprocessor(ABC):
    """预处理器抽象接口

    未来新增处理方法只需实现此接口：
    - NK2Preprocessor: 当前NeuroKit2处理
    - CustomPreprocessor: 自定义处理
    """

    @abstractmethod
    def process(self, raw_data: Any) -> ProcessedData:
        """处理原始数据"""
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """获取当前配置"""
        pass


class BaseDataset(Dataset):
    """Dataset抽象基类

    定义标准接口，具体实现由子类完成
    """

    def __init__(self, source: IDataSource, preprocessor: IPreprocessor):
        self.source = source
        self.preprocessor = preprocessor
        self._sample_ids = source.get_sample_list()

    def __len__(self) -> int:
        return len(self._sample_ids)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本"""
        pass
