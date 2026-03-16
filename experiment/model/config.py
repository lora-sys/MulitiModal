"""
模型配置文件
定义模型架构、超参数等
"""

# 模型配置
MODEL_CONFIG = {
    "type": "multimodal",  # 可选: cnn / lstm / inception / transformer / multimodal
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
    "transformer": {
        "d_model": 64,
        "nhead": 8,
        "num_layers": 3,
    },
    "multimodal": {
        "num_constitutions": 38,  # 38种体质
        "shared_dim": 128,
        "hidden_dim": 256,
        "dropout": 0.3,
    },
}

# 训练超参数
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,  # 配合 warmup 使用
    "weight_decay": 1e-4,
    "dropout": 0.5,
}

# 调度器配置 (选择其中一个)
SCHEDULER_CONFIGS = {
    "ReduceLROnPlateau": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 3,
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
    "OneCycleLR": {
        "type": "OneCycleLR",
        "max_lr": 0.002,
        "total_steps": 4000,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
    },
    "CosineAnnealingWarmup": {  # 推荐：Cosine Annealing + Warmup
        "type": "CosineAnnealingWarmup",
        "warmup_epochs": 5,
        "max_epochs": 50,
        "eta_min": 1e-6,
    },
}

# 当前使用的调度器
CURRENT_SCHEDULER = "CosineAnnealingWarmup"  # 使用推荐的调度器
