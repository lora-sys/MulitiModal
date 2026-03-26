"""
超参数优化配置模块

定义Optuna的搜索空间、优化目标和Hyperband剪枝参数
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import yaml
from pathlib import Path


@dataclass
class SearchSpaceConfig:
    """搜索空间配置"""

    # 通用超参数
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "log_uniform",
        "low": 1e-5,
        "high": 1e-2,
        "name": "learning_rate"
    })

    weight_decay: Dict[str, Any] = field(default_factory=lambda: {
        "type": "log_uniform",
        "low": 1e-6,
        "high": 1e-3,
        "name": "weight_decay"
    })

    batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [16, 32, 64, 128],
        "name": "batch_size"
    })

    dropout: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.1,
        "high": 0.5,
        "name": "dropout"
    })

    gradient_clip: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 1.0,
        "high": 10.0,
        "name": "gradient_clip"
    })

    # 模型架构超参数
    shared_dim: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [32, 64, 128, 256],
        "name": "shared_dim"
    })

    hidden_dim: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [64, 128, 256, 512],
        "name": "hidden_dim"
    })

    encoder_lr_ratio: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.05,
        "high": 0.2,
        "name": "encoder_lr_ratio"
    })

    # Inception 编码器特定超参数
    inception_depth: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [2, 3, 4],
        "name": "inception_depth",
        "condition": {
            "model": ["baseline_c", "baseline_b"]
        }
    })

    inception_bottleneck_channels: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [16, 32, 64],
        "name": "inception_bottleneck_channels",
        "condition": {
            "model": ["baseline_c", "baseline_b"]
        }
    })

    # Transformer 编码器特定超参数
    transformer_num_heads: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [2, 4, 8],
        "name": "transformer_num_heads",
        "condition": {
            "model": ["baseline_b"]
        }
    })

    transformer_num_layers: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [1, 2, 3],
        "name": "transformer_num_layers",
        "condition": {
            "model": ["baseline_b"]
        }
    })

    transformer_dim_feedforward: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [128, 256, 512],
        "name": "transformer_dim_feedforward",
        "condition": {
            "model": ["baseline_b"]
        }
    })

    # LSTM 编码器特定超参数
    lstm_hidden_size: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [32, 64, 128],
        "name": "lstm_hidden_size",
        "condition": {
            "model": ["baseline_a"]
        }
    })

    lstm_num_layers: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [1, 2, 3],
        "name": "lstm_num_layers",
        "condition": {
            "model": ["baseline_a"]
        }
    })

    # 训练超参数
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": ["adam", "adamw", "sgd"],
        "name": "optimizer"
    })

    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": ["cosine_annealing", "reduce_on_plateau", "one_cycle"],
        "name": "scheduler"
    })

    warmup_epochs: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [5, 10, 15],
        "name": "warmup_epochs"
    })

    patience: Dict[str, Any] = field(default_factory=lambda: {
        "type": "categorical",
        "choices": [10, 15, 20],
        "name": "patience"
    })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PruningConfig:
    """剪枝配置"""

    enabled: bool = True
    pruner_type: str = "median"  # median, successive_halving
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1

    # MedianPruner 特定参数
    n_min_trials: int = 5

    # SuccessiveHalvingPruner 特定参数
    reduction_factor: int = 3
    min_early_stopping_rate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ObjectiveConfig:
    """优化目标配置"""

    metric: str = "val_loss"  # val_loss, val_acc, mae, f1
    direction: str = "minimize"  # minimize, maximize

    # 分类任务特定指标
    classification_metrics: List[str] = field(default_factory=lambda: ["val_acc", "f1"])

    # 回归任务特定指标
    regression_metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "r2"])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class HyperoptConfig:
    """超参数优化配置"""

    # 优化设置
    n_trials: int = 100
    timeout: Optional[int] = 7200  # 秒
    n_jobs: int = 1

    # 模型配置
    model_type: str = "baseline_c"
    task_type: str = "classification"  # classification, regression

    # 子配置
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # 输出配置
    output_dir: str = "experiment/results/hyperopt"
    study_name: Optional[str] = None

    # 随机种子
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = asdict(self)
        config_dict["search_space"] = self.search_space.to_dict()
        config_dict["pruning"] = self.pruning.to_dict()
        config_dict["objective"] = self.objective.to_dict()
        return config_dict

    def save_yaml(self, path: str) -> None:
        """保存为YAML文件

        Args:
            path: 保存路径
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str) -> 'HyperoptConfig':
        """从YAML文件加载配置

        Args:
            path: YAML文件路径

        Returns:
            HyperoptConfig: 配置实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 递归构建配置对象
        def build_config(config_dict: Dict, config_class: type) -> Any:
            if hasattr(config_class, '__dataclass_fields__'):
                field_dict = {}
                for field_name, field_info in config_class.__dataclass_fields__.items():
                    if field_name in config_dict:
                        field_value = config_dict[field_name]
                        field_type = field_info.type

                        # 处理嵌套的dataclass
                        if hasattr(field_type, '__dataclass_fields__'):
                            field_dict[field_name] = build_config(field_value, field_type)
                        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                            field_dict[field_name] = field_value
                        else:
                            field_dict[field_name] = field_value
                    else:
                        # 使用默认值
                        if field_info.default is not dataclasses.MISSING:
                            field_dict[field_name] = field_info.default
                        elif field_info.default_factory is not dataclasses.MISSING:
                            field_dict[field_name] = field_info.default_factory()
                return config_class(**field_dict)
            else:
                return config_dict

        return build_config(config_dict, cls)

    def get_search_space_list(self) -> List[Dict[str, Any]]:
        """获取搜索空间列表

        Returns:
            List[Dict]: 搜索空间配置列表
        """
        search_space_dict = self.search_space.to_dict()
        search_space_list = []

        for param_name, param_config in search_space_dict.items():
            if isinstance(param_config, dict) and "type" in param_config:
                # 检查条件是否满足
                if "condition" in param_config:
                    condition = param_config["condition"]
                    if condition.get("model"):
                        if self.model_type not in condition["model"]:
                            continue
                search_space_list.append(param_config)

        return search_space_list

    def get_objective_metric(self) -> str:
        """获取优化指标

        Returns:
            str: 优化指标名称
        """
        metric = self.objective.metric

        # 根据任务类型选择默认指标
        if metric == "default":
            if self.task_type == "classification":
                metric = "val_acc"  # 分类任务默认优化准确率
            else:
                metric = "mae"  # 回归任务默认优化MAE

        return metric

    def get_direction(self) -> str:
        """获取优化方向

        Returns:
            str: 优化方向 (minimize/maximize)
        """
        direction = self.objective.direction

        # 根据指标自动确定方向
        if direction == "auto":
            metric = self.get_objective_metric()
            if metric in ["val_acc", "f1", "r2"]:
                direction = "maximize"  # 这些指标越大越好
            else:
                direction = "minimize"  # 这些指标越小越好

        return direction


# 默认配置实例
_default_config: Optional[HyperoptConfig] = None


def get_default_config() -> HyperoptConfig:
    """获取默认配置实例"""
    global _default_config
    if _default_config is None:
        _default_config = HyperoptConfig()
    return _default_config


def create_classification_config(model_type: str = "baseline_c") -> HyperoptConfig:
    """创建分类任务配置

    Args:
        model_type: 模型类型

    Returns:
        HyperoptConfig: 配置实例
    """
    config = HyperoptConfig(
        model_type=model_type,
        task_type="classification",
        objective=ObjectiveConfig(
            metric="val_acc",
            direction="maximize"
        )
    )
    return config


def create_regression_config(model_type: str = "baseline_c") -> HyperoptConfig:
    """创建回归任务配置

    Args:
        model_type: 模型类型

    Returns:
        HyperoptConfig: 配置实例
    """
    config = HyperoptConfig(
        model_type=model_type,
        task_type="regression",
        objective=ObjectiveConfig(
            metric="mae",
            direction="minimize"
        )
    )
    return config


# 导入dataclasses（用于类型检查）
import dataclasses