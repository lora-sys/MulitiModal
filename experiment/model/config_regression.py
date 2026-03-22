"""
回归任务配置文件
定义回归模型的架构、超参数等
"""

# 回归任务配置
REGRESSION_CONFIG = {
    "task_type": "regression",  # 任务类型：regression
    "target_column": "中医诊断分数_对齐",  # 回归目标列名
    "target_range": [30, 100],  # 回归目标范围
    "model_type": "baseline_a",  # 默认模型类型
    "noise_augmentation": False,  # 是否使用噪声增强
}

# 回归模型配置
REGRESSION_MODEL_CONFIG = {
    "type": "baseline_a",  # 可选: baseline_a / baseline_b / baseline_c
    "params": {
        "dyn_channels": 2,  # 动态特征通道数
        "static_dim": 4,  # 静态特征维度
        "num_classes": 1,  # 回归输出1个连续值
    },
}

# 回归训练超参数
REGRESSION_TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "loss": "MSE",  # 损失函数：MSE / MAE / HuberLoss
    "metrics": ["MAE", "RMSE", "R2", "Pearson"],  # 回归评估指标
    # 噪声增强配置
    "noise_augmentation": False,
    "noise_probability": 0.5,  # 噪声注入概率
    "noise_types": ["gaussian", "drift", "dropout"],  # 噪声类型
    "noise_intensity": "medium",  # 噪声强度：low / medium / high
    # Parameter Groups 优化
    "encoder_lr_ratio": 0.1,
    "smoothing_window": 5,
}

# 调度器配置（与分类任务相同）
SCHEDULER_CONFIGS = {
    "CosineAnnealingWarmup": {
        "type": "CosineAnnealingWarmup",
        "warmup_epochs": 5,
        "max_epochs": 50,
        "eta_min": 1e-6,
    },
}

# 当前使用的调度器
CURRENT_SCHEDULER = "CosineAnnealingWarmup"