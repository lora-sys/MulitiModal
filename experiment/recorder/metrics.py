"""
指标计算工具
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


@dataclass
class MetricsResult:
    """存储所有计算好的指标"""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=lambda: ['一般', '正常', '良好'])
    
    def to_dict(self) -> dict:
        """转换为字典格式，便于保存"""
        return {
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'weighted_f1': self.weighted_f1,
            'balanced_accuracy': self.balanced_accuracy,
            'per_class_precision': self.per_class_precision,
            'per_class_recall': self.per_class_recall,
            'per_class_f1': self.per_class_f1,
        }


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: List[str] = None,
    compute_confusion_matrix: bool = True
) -> MetricsResult:
    """
    计算完整的分类指标
    
    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        class_names: 类别名称列表
        compute_confusion_matrix: 是否计算混淆矩阵
        
    Returns:
        MetricsResult: 包含所有指标的结果对象
    """
    if class_names is None:
        class_names = ['一般', '正常', '良好']
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class 指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_precision = {name: float(p) for name, p in zip(class_names, precision_per_class)}
    per_class_recall = {name: float(r) for name, r in zip(class_names, recall_per_class)}
    per_class_f1 = {name: float(f) for name, f in zip(class_names, f1_per_class)}
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred) if compute_confusion_matrix else None
    
    return MetricsResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        balanced_accuracy=balanced_acc,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        class_names=class_names,
    )


def format_per_class_metrics(metrics: MetricsResult) -> str:
    """格式化 per-class 指标为字符串"""
    lines = ["Per-class Metrics:"]
    for cls_name in metrics.class_names:
        lines.append(
            f"  {cls_name}: P={metrics.per_class_precision.get(cls_name, 0):.4f}, "
            f"R={metrics.per_class_recall.get(cls_name, 0):.4f}, "
            f"F1={metrics.per_class_f1.get(cls_name, 0):.4f}"
        )
    return "\n".join(lines)