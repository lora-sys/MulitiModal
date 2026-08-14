"""
模型模块 - 按摩椅舒适度分类模型
支持多种融合策略：简单拼接、Transformer 晚融合、多专家融合、自注意力融合、门控融合
"""

from .model import (
    InceptionModule,
    InceptionEncoder,
    LSTMEncoder,
    SimpleCNNEncoder,
    TransformerEncoder,
    SimpleConcatModel,
    LateFusionTransformerModel,
    MultiExpertFusionModel,
    SimpleAttentionFusion,
    GatedFusion,
    get_model,
)

__all__ = [
    # 基础组件
    "InceptionModule",
    "InceptionEncoder",
    "LSTMEncoder",
    "SimpleCNNEncoder",
    "TransformerEncoder",
    # 融合模型
    "SimpleConcatModel",
    "LateFusionTransformerModel",
    "MultiExpertFusionModel",
    "SimpleAttentionFusion",
    "GatedFusion",
    # 工厂函数
    "get_model",
]
