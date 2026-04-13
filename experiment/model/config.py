"""
模型配置文件
定义模型架构、超参数等
"""

# 模型配置
MODEL_CONFIG = {
    "type": "dual_gating",  # 可选: simple_concat / late_fusion / multimodal / baseline_a / baseline_b / baseline_c / dual_gating
    "params": {
        "dyn_channels": 2,  # 动态特征通道数（传感器数量）
        "static_dim": 8,  # 静态特征维度（TCM 8个体征）
        "num_classes": 3,  # 分类类别数（移除标签0，只使用1、2、3）
        "projector_dim": 128,  # 投影层维度
        "gate_dim": 128,  # 门控网络维度
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
    "dual_gating": {
        "num_constitutions": 9,  # 9种中医体质
        "shared_dim": 128,
        "projector_dim": 128,
        "gate_dim": 128,
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
    # Parameter Groups 优化
    "encoder_lr_ratio": 0.1,  # encoder 学习率相对于主学习率的比例（0.1 = 1/10）
    # 滑动平均优化
    "smoothing_window": 5,    # val 指标滑动平均窗口大小
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
}

# 当前使用的调度器
CURRENT_SCHEDULER = "CosineAnnealingLR"  # 使用标准调度器（无 warmup）
