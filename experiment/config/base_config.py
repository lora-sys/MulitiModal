"""
统一配置管理系统

使用dataclass定义配置类，提供类型安全和默认值
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集路径
    classification_dataset: str = 'experiment/dataset/unified_dataset_expanded.npz'
    regression_dataset: str = 'experiment/dataset/unified_dataset_regression.npz'

    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 数据特征
    dynamic_channels: int = 2  # 动态波形通道数
    dynamic_length: int = 1000  # 动态波形长度
    static_basic_dim: int = 4  # 基本静态特征维度 [年龄, BMI, 血氧, 心率]
    static_scores_dim: int = 2  # 评分特征维度 [健康指数, 诊断得分]
    num_constitutions: int = 39  # 体质类型数量

    # 数据加载
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    # 通用配置
    model_type: str = 'baseline_c'  # 模型类型
    num_classes: int = 3  # 分类任务类别数
    task_type: str = 'classification'  # 任务类型: classification 或 regression

    # 编码器配置
    shared_dim: int = 64  # 共享特征维度
    hidden_dim: int = 128  # 隐藏层维度
    dropout: float = 0.3  # Dropout概率

    # Inception编码器配置
    inception_kernel_sizes: List[int] = field(default_factory=lambda: [9, 19, 39])
    inception_bottleneck_channels: int = 32
    inception_depth: int = 3

    # Transformer编码器配置
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256

    # LSTM编码器配置
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True

    # CNN编码器配置
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])

    # 融合策略配置
    use_cross_attention: bool = True  # 是否使用交叉注意力
    use_gating: bool = True  # 是否使用门控机制


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0

    # 优化器配置
    optimizer: str = 'adam'  # adam, sgd, adamw
    encoder_lr_ratio: float = 0.1  # 编码器学习率比例

    # 学习率调度器配置
    scheduler: str = 'cosine_annealing_warmup'  # cosine_annealing, cosine_annealing_warmup, reduce_on_plateau, one_cycle
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # 早停配置
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001

    # 检查点配置
    save_best_only: bool = True
    save_period: int = 10

    # 设备配置
    device: str = 'cuda'  # cuda, cpu
    mixed_precision: bool = False  # 混合精度训练


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1'])

    # K-Fold交叉验证
    use_kfold: bool = False
    k_folds: int = 5
    stratified: bool = True

    # 鲁棒性测试
    robustness_test: bool = False
    noise_types: List[str] = field(default_factory=lambda: ['gaussian', 's&p', 'speckle'])
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验信息
    experiment_name: str = 'multimodal_experiment'
    description: str = '多模态按摩椅舒适度预测实验'

    # 运行配置
    seed: int = 42
    deterministic: bool = True

    # 输出配置
    output_dir: str = 'experiment/results'
    log_dir: str = 'experiment/logs'
    checkpoint_dir: str = 'experiment/checkpoints'

    # 保存配置
    save_predictions: bool = True
    save_attention_maps: bool = False
    save_confusion_matrix: bool = True

    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """保存为YAML文件

        Args:
            path: 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置

        Args:
            path: YAML文件路径

        Returns:
            ExperimentConfig: 配置实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 递归构建配置对象
        def build_config(config_dict: Dict[str, Any], config_class: type) -> Any:
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
                            # 处理List类型
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


# 全局配置实例
_global_config: Optional[ExperimentConfig] = None


def get_config() -> ExperimentConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ExperimentConfig()
    return _global_config


def set_config(config: ExperimentConfig) -> None:
    """设置全局配置实例

    Args:
        config: 配置实例
    """
    global _global_config
    _global_config = config


def load_config_from_yaml(path: str) -> ExperimentConfig:
    """从YAML文件加载配置并设置为全局配置

    Args:
        path: YAML文件路径

    Returns:
        ExperimentConfig: 配置实例
    """
    config = ExperimentConfig.load_yaml(path)
    set_config(config)
    return config


# 导入dataclasses（用于类型检查）
import dataclasses