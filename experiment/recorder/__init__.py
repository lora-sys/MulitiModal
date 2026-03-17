"""
实验记录模块 - 可复用的实验追踪系统
支持 CSV 记录、JSON 配置保存、可视化输出
"""

from .recorder import ExperimentRecorder
from .metrics import compute_metrics, MetricsResult

__all__ = ['ExperimentRecorder', 'compute_metrics', 'MetricsResult']
