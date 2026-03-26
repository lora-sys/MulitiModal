"""
特征重要性分析模块

提供多种特征重要性计算方法
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiment.utils.logger import get_logger


class FeatureImportanceAnalyzer:
    """特征重要性分析器

    支持多种特征重要性计算方法：
    1. 置换重要性（Permutation Importance）
    2. 梯度重要性（Gradient Importance）
    3. 注意力权重（Attention Weights）
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        feature_names: Optional[List[str]] = None
    ):
        """初始化特征重要性分析器

        Args:
            model: 训练好的模型
            device: 训练设备
            feature_names: 特征名称列表
        """
        self.model = model
        self.device = device
        self.logger = get_logger(__name__)

        # 特征名称（如果不提供则使用默认名称）
        if feature_names is None:
            self.feature_names = [
                'Dynamic Waveform',  # 动态波形
                'Static Basic',       # 身体特征
                'Static Scores',      # 舌面诊评分
                'Constitution',       # 体质类型
            ]
        else:
            self.feature_names = feature_names

        # 模型类型检测
        self.model_type = self._detect_model_type()

        self.logger.info(f"初始化特征重要性分析器，模型类型: {self.model_type}")

    def _detect_model_type(self) -> str:
        """检测模型类型

        Returns:
            str: 模型类型
        """
        model_name = self.model.__class__.__name__

        if "SimpleConcat" in model_name:
            return "baseline_a"
        elif "LateFusion" in model_name:
            return "baseline_b"
        elif "MultiExpert" in model_name:
            return "baseline_c"
        elif "SimpleAttention" in model_name:
            return "baseline_d"
        elif "GatedFusion" in model_name:
            return "baseline_e"
        else:
            return "unknown"

    def compute_permutation_importance(
        self,
        data_loader: torch.utils.data.DataLoader,
        metric: str = "accuracy",
        n_repeats: int = 5,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """计算置换重要性

        Args:
            data_loader: 数据加载器
            metric: 评估指标
            n_repeats: 重复次数
            random_seed: 随机种子

        Returns:
            Dict: 特征重要性字典
        """
        self.logger.info("开始计算置换重要性...")

        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 计算基准性能
        baseline_score = self._compute_score(data_loader, metric)
        self.logger.info(f"基准性能: {baseline_score:.4f}")

        # 定义特征组
        feature_groups = self._get_feature_groups()

        # 计算每个特征组的重要性
        importance_scores = {}

        for feature_name, feature_indices in feature_groups.items():
            self.logger.info(f"计算 {feature_name} 的重要性...")

            scores = []
            for repeat in range(n_repeats):
                score = self._compute_permutation_score(
                    data_loader,
                    feature_indices,
                    metric,
                    random_seed=random_seed + repeat
                )
                scores.append(score)

            # 重要性 = 基准性能 - 置换后性能
            # 对于损失指标（越小越好），需要反转符号
            delta = baseline_score - np.mean(scores)
            if metric == "loss" or not self._greater_is_better(metric):
                importance = -delta
            else:
                importance = delta
            importance_scores[feature_name] = importance

            self.logger.info(f"  {feature_name}: {importance:.4f}")

        self.logger.info("置换重要性计算完成")

        return importance_scores

    def _compute_score(
        self,
        data_loader: torch.utils.data.DataLoader,
        metric: str
    ) -> float:
        """计算模型性能

        Args:
            data_loader: 数据加载器
            metric: 评估指标

        Returns:
            float: 性能分数
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                dynamic = batch['dynamic'].to(self.device)
                static_basic = batch['static_basic'].to(self.device)
                static_scores = batch['static_scores'].to(self.device)
                constitution = batch['constitution'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(dynamic, static_basic, static_scores, constitution)

                if metric == "accuracy":
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                elif metric == "loss":
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

        if metric == "accuracy":
            return 100. * correct / total
        elif metric == "loss":
            return total_loss / len(data_loader)
        else:
            raise ValueError(f"未知的指标: {metric}")

    def _compute_permutation_score(
        self,
        data_loader: torch.utils.data.DataLoader,
        feature_indices: Dict[str, int],
        metric: str,
        random_seed: int
    ) -> float:
        """计算置换后的性能

        Args:
            data_loader: 数据加载器
            feature_indices: 特征索引
            metric: 评估指标
            random_seed: 随机种子

        Returns:
            float: 置换后的性能分数
        """
        torch.manual_seed(random_seed)

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                # 创建数据的副本
                dynamic = batch['dynamic'].clone().to(self.device)
                static_basic = batch['static_basic'].clone().to(self.device)
                static_scores = batch['static_scores'].clone().to(self.device)
                constitution = batch['constitution'].clone().to(self.device)
                labels = batch['label'].to(self.device)

                # 置换特征
                if 'dynamic' in feature_indices:
                    # 置换动态波形
                    batch_size = dynamic.size(0)
                    perm_indices = torch.randperm(dynamic.size(0))
                    dynamic = dynamic[perm_indices]

                if 'static_basic' in feature_indices:
                    # 置换身体特征
                    perm_indices = torch.randperm(static_basic.size(0))
                    static_basic = static_basic[perm_indices]

                if 'static_scores' in feature_indices:
                    # 置换舌面诊评分
                    perm_indices = torch.randperm(static_scores.size(0))
                    static_scores = static_scores[perm_indices]

                if 'constitution' in feature_indices:
                    # 置换体质类型
                    perm_indices = torch.randperm(constitution.size(0))
                    constitution = constitution[perm_indices]

                outputs = self.model(dynamic, static_basic, static_scores, constitution)

                if metric == "accuracy":
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                elif metric == "loss":
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

        if metric == "accuracy":
            return 100. * correct / total
        elif metric == "loss":
            return total_loss / len(data_loader)
        else:
            raise ValueError(f"未知的指标: {metric}")

    def _greater_is_better(self, metric: str) -> bool:
        """判断指标是否越大越好

        Args:
            metric: 指标名称

        Returns:
            bool: True表示越大越好，False表示越小越好
        """
        greater_is_better_metrics = ['accuracy', 'acc', 'f1', 'r2', 'precision', 'recall', 'pearson']
        return metric.lower() in greater_is_better_metrics

    def _get_feature_groups(self) -> Dict[str, Dict[str, int]]:
        """获取特征组

        Returns:
            Dict: 特征组字典
        """
        # 优先使用调用者提供的feature_names
        if hasattr(self, 'feature_names') and self.feature_names:
            return {
                self.feature_names[0]: {'dynamic': 0},
                self.feature_names[1]: {'static_basic': 0},
                self.feature_names[2]: {'static_scores': 0},
                self.feature_names[3]: {'constitution': 0},
            }
        else:
            # 回退到默认名称
            return {
                'Dynamic Waveform': {'dynamic': 0},
                'Static Basic': {'static_basic': 0},
                'Static Scores': {'static_scores': 0},
                'Constitution': {'constitution': 0},
            }

    def visualize_importance(
        self,
        importance_scores: Dict[str, float],
        save_path: Optional[str] = None,
        title: str = "Feature Importance"
    ) -> plt.Figure:
        """可视化特征重要性

        Args:
            importance_scores: 特征重要性字典
            save_path: 保存路径
            title: 图表标题

        Returns:
            plt.Figure: 图表对象
        """
        # 排序
        sorted_scores = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [item[0] for item in sorted_scores]
        scores = [item[1] for item in sorted_scores]

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['red' if score < 0 else 'green' for score in scores]
        bars = ax.barh(names, scores, color=colors, alpha=0.7)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score, i, f'{score:.4f}',
                    ha='left' if score > 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"图表已保存到: {save_path}")

        return fig


def compute_feature_importance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    method: str = "permutation",
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """计算特征重要性

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        method: 计算方法
        device: 训练设备

    Returns:
        Dict: 特征重要性字典
    """
    analyzer = FeatureImportanceAnalyzer(model, device)

    if method == "permutation":
        return analyzer.compute_permutation_importance(data_loader)
    else:
        raise ValueError(f"未知的计算方法: {method}")


def visualize_feature_importance(
    importance_scores: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> plt.Figure:
    """可视化特征重要性

    Args:
        importance_scores: 特征重要性字典
        save_path: 保存路径
        title: 图表标题

    Returns:
        plt.Figure: 图表对象
    """
    analyzer = FeatureImportanceAnalyzer(None, torch.device('cpu'))
    return analyzer.visualize_importance(importance_scores, save_path, title)