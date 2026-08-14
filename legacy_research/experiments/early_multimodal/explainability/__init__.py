"""
模型解释性模块

提供模型可解释性功能，包括特征重要性分析和预测结果解释
"""

from .feature_importance import (
    compute_feature_importance,
    visualize_feature_importance,
)
from .prediction_explainer import (
    PredictionExplainer,
    explain_prediction,
)

__all__ = [
    'compute_feature_importance',
    'visualize_feature_importance',
    'PredictionExplainer',
    'explain_prediction',
]