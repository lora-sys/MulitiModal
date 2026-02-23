"""
模型配置文件
定义模型架构、超参数等
"""

# 模型配置
MODEL_CONFIG = {
    "type": "inception",  # 可选: cnn / lstm / inception / transformer
    "params": {
        "dyn_channels": 2,  # 动态特征通道数（传感器数量）
        "static_dim": 4,  # 静态特征维度
        "num_classes": 4,  # 分类类别数
    },
}

# 模型特定参数
MODEL_PARAMS = {
    "inception": {
        "out_channels": 32,
        "depth": 3,
        "kernel_sizes": [9, 19, 39],
        "bottleneck_channels": 32,
    },
    "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
    },
    "cnn": {
        # 使用默认配置
    },
}

# 训练超参数
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout": 0.5,
}

# 调度器配置 (选择其中一个)
SCHEDULER_CONFIGS = {
    "ReduceLROnPlateau": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 5,
        "factor": 0.5,
    },
    "CosineAnnealingWarmRestarts": {
        "type": "CosineAnnealingWarmRestarts",
        "T_0": 10,
        "T_mult": 2,
        "eta_min": 1e-6,
    },
    "CosineAnnealingLR": {
        "type": "CosineAnnealingLR",
        "T_max": 20,
        "eta_min": 1e-6,
    },
    "StepLR": {
        "type": "StepLR",
        "step_size": 10,
        "gamma": 0.1,
    },
}

# 当前使用的调度器
CURRENT_SCHEDULER = "ReduceLROnPlateau"
