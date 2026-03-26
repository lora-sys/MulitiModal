"""
Objective 函数实现

定义Optuna的优化目标函数，包括超参数采样、模型训练和Hyperband剪枝
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import dataclasses
import optuna

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiment.utils.logger import get_logger
from experiment.config.base_config import ExperimentConfig
from experiment.dataset.unified_source import UnifiedNPZDataSource
from experiment.dataset.unified_dataset import UnifiedMultimodalDataset
from experiment.model.model import get_model


class ObjectiveFunction:
    """Objective 函数类

    负责实现Optuna的优化目标函数，包括：
    - 超参数采样
    - 模型配置
    - 训练循环
    - Hyperband 剪枝
    - 返回优化指标
    """

    def __init__(
        self,
        config: 'HyperoptConfig',
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """初始化 Objective 函数

        Args:
            config: 超参数优化配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = get_logger(__name__)

    def __call__(self, trial) -> float:
        """Optuna 目标函数

        Args:
            trial: Optuna trial 对象

        Returns:
            float: 优化指标值
        """
        try:
            # 1. 采样超参数
            params = self.sample_hyperparams(trial)

            # 2. 配置模型
            model = self.create_model(params)

            # 3. 配置优化器和损失函数
            optimizer, criterion = self.configure_training(model, params)

            # 4. 训练模型（带Hyperband剪枝）
            best_metric = self.train_with_pruning(trial, model, optimizer, criterion, params)

            return best_metric

        except optuna.TrialPruned:
            # 重新抛出剪枝异常，保持PRUNED状态
            raise
        except Exception as e:
            self.logger.error(f"Trial 失败: {e}")
            # 返回最差值，让Optuna跳过这个trial
            if self.config.get_direction() == "minimize":
                return float('inf')
            else:
                return float('-inf')

    def sample_hyperparams(self, trial) -> Dict[str, Any]:
        """采样超参数

        Args:
            trial: Optuna trial 对象

        Returns:
            Dict: 采样到的超参数字典
        """
        params = {}

        # 获取搜索空间
        search_space_list = self.config.get_search_space_list()

        for param_config in search_space_list:
            param_name = param_config["name"]
            param_type = param_config["type"]

            try:
                if param_type == "uniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "log_uniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=True
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
                elif param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "discrete_uniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1.0)
                    )
                else:
                    self.logger.warning(f"未知的参数类型: {param_type}")
            except Exception as e:
                self.logger.warning(f"采样参数 {param_name} 失败: {e}")
                # 使用默认值
                if "choices" in param_config:
                    params[param_name] = param_config["choices"][0]
                else:
                    params[param_name] = param_config.get("low", 0.0)

        return params

    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """创建模型

        Args:
            params: 超参数字典

        Returns:
            nn.Module: 模型实例
        """
        # 提取模型参数
        num_classes = 3 if self.config.task_type == "classification" else 1

        # 创建模型
        model = get_model(
            model_type=self.config.model_type,
            num_classes=num_classes,
            num_constitutions=38,
            shared_dim=params.get("shared_dim", 64),
            hidden_dim=params.get("hidden_dim", 128),
            dropout=params.get("dropout", 0.3),
            # 编码器特定参数（使用模型工厂期望的参数名）
            num_heads=params.get("num_heads", params.get("transformer_num_heads", 4)),
            num_layers=params.get("num_layers", params.get("transformer_num_layers", 2)),
            dim_feedforward=params.get("dim_feedforward", params.get("transformer_dim_feedforward", 256)),
        )

        model = model.to(self.device)

        return model

    def configure_training(
        self,
        model: nn.Module,
        params: Dict[str, Any]
    ) -> Tuple[torch.optim.Optimizer, nn.Module]:
        """配置训练

        Args:
            model: 模型实例
            params: 超参数字典

        Returns:
            Tuple: (优化器, 损失函数)
        """
        # 优化器
        optimizer_name = params.get("optimizer", "adam")
        learning_rate = params.get("learning_rate", 0.001)
        weight_decay = params.get("weight_decay", 1e-4)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # 损失函数
        if self.config.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        return optimizer, criterion

    def train_with_pruning(
        self,
        trial,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        params: Dict[str, Any]
    ) -> float:
        """带Hyperband剪枝的训练

        Args:
            trial: Optuna trial 对象
            model: 模型实例
            optimizer: 优化器
            criterion: 损失函数
            params: 超参数字典

        Returns:
            float: 最优指标值
        """
        # 训练参数
        num_epochs = 50  # 超参数优化使用较少的epoch
        patience = params.get("patience", 10)
        gradient_clip = params.get("gradient_clip", 5.0)

        best_metric = float('inf') if self.config.get_direction() == "minimize" else float('-inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in self.train_loader:
                optimizer.zero_grad()

                # 获取数据
                dynamic = batch['dynamic'].to(self.device)
                static_basic = batch['static_basic'].to(self.device)
                static_scores = batch['static_scores'].to(self.device)
                constitution = batch['constitution'].to(self.device)

                if self.config.task_type == "classification":
                    labels = batch['label'].to(self.device)

                    # 前向传播
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                    loss = criterion(outputs, labels)

                    # 计算准确率
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                else:
                    scores = batch.get('scores', batch.get('label', None))
                    if scores is None:
                        scores = batch['label'].to(self.device)
                    else:
                        scores = scores.to(self.device)

                    # 前向传播
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                    loss = criterion(outputs.squeeze(-1), scores)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_labels = []
            val_preds = []

            with torch.no_grad():
                for batch in self.val_loader:
                    dynamic = batch['dynamic'].to(self.device)
                    static_basic = batch['static_basic'].to(self.device)
                    static_scores = batch['static_scores'].to(self.device)
                    constitution = batch['constitution'].to(self.device)

                    if self.config.task_type == "classification":
                        labels = batch['label'].to(self.device)
                        outputs = model(dynamic, static_basic, static_scores, constitution)
                        loss = criterion(outputs, labels)

                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                        # 收集标签和预测用于F1计算
                        val_labels.extend(labels.cpu().numpy())
                        val_preds.extend(predicted.cpu().numpy())
                    else:
                        scores = batch.get('scores', batch.get('label', None))
                        if scores is None:
                            scores = batch['label'].to(self.device)
                        else:
                            scores = scores.to(self.device)

                        outputs = model(dynamic, static_basic, static_scores, constitution)
                        loss = criterion(outputs.squeeze(-1), scores)

                        # 收集标签和预测用于R2计算
                        val_labels.extend(scores.cpu().numpy())
                        val_preds.extend(outputs.squeeze(-1).cpu().numpy())

                    val_loss += loss.item()

            # 计算验证指标
            avg_val_loss = val_loss / len(self.val_loader)

            # 根据配置的优化指标计算current_metric
            objective_metric = self.config.get_objective_metric()

            if self.config.task_type == "classification":
                val_acc = 100. * val_correct / val_total
                if objective_metric == "accuracy":
                    current_metric = val_acc
                elif objective_metric == "f1":
                    # 计算F1分数
                    from sklearn.metrics import f1_score
                    val_f1 = f1_score(val_labels, val_preds, average='weighted')
                    current_metric = val_f1 * 100  # 转换为百分比
                else:
                    current_metric = val_acc  # 默认使用准确率
            else:
                # 回归任务支持多种指标
                if objective_metric == "mae":
                    current_metric = avg_val_loss
                elif objective_metric == "rmse":
                    current_metric = np.sqrt(avg_val_loss)
                elif objective_metric == "r2":
                    # 计算R²
                    from sklearn.metrics import r2_score
                    current_metric = -r2_score(val_labels, val_preds)  # R²越大越好，取负值使其越小越好
                else:
                    current_metric = avg_val_loss  # 默认使用MAE

            # 更新最佳指标
            if self.config.get_direction() == "maximize":
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Hyperband 剪枝：报告中间结果
            if self.config.pruning.enabled:
                trial.report(current_metric, epoch)

                # 检查是否应该剪枝
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} 在 epoch {epoch} 被剪枝")
                    raise optuna.TrialPruned()

            # 早停
            if patience_counter >= patience:
                self.logger.info(f"Trial {trial.number} 在 epoch {epoch} 早停")
                break

        return best_metric


def create_objective_function(
    config: 'HyperoptConfig',
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> ObjectiveFunction:
    """创建 Objective 函数

    Args:
        config: 超参数优化配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备

    Returns:
        ObjectiveFunction: Objective 函数实例
    """
    return ObjectiveFunction(config, train_loader, val_loader, device)


# 导入optuna（延迟导入）
import optuna