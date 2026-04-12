"""
全局配置文件
定义模型架构、超参数、Optuna 搜索空间等
"""

import os

# =========================================================================
# 项目路径配置
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# 确保目录存在
for dir_path in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =========================================================================
# 数据配置
# =========================================================================
DATA_CONFIG = {
    "input_path": os.path.join(PROJECT_ROOT, "data", "vital_signs_dataset_final.csv"),
    "n_features": 8,  # 输入特征维度
    "n_classes": 9, 
    "test_split": 0.1,  # 测试集比例
    "random_state": 42,  # 随机种子
    "epsilon": 0.01,  # 标签平滑参数
}

# =========================================================================
# 模型配置
# =========================================================================
MODEL_CONFIG = {
    "d_token": 64,  # Token 嵌入维度（固定）
    "n_heads": 4,  # 注意力头数（固定）
    "n_layers": 3,  # Transformer 层数（可搜索）
    "dropout": 0.3,  # Dropout 比例（可搜索）
}

# =========================================================================
# 训练配置
# =========================================================================
TRAIN_CONFIG = {
    "batch_size": 256,
    "num_epochs": 50,
    "learning_rate": 1e-3,  # 初始学习率（可搜索）
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,  # Warmup 比例（前5%）
    "grad_clip_max_norm": 1.0,  # 梯度裁剪
    "patience": 5,  # 早停耐心值
    "device": "cuda",  # 设备
}

# =========================================================================
# Optuna 超参数搜索配置
# =========================================================================
OPTUNA_CONFIG = {
    "n_trials": 20,  # 试验次数
    "n_jobs": 1,  # 并行任务数（每个 trial 内部有 5-fold CV，并行可能导致资源竞争）
    "direction": "minimize",  # 最小化目标
    "sampler": "TPESampler",  # 采样器
    "pruner": "MedianPruner",  # 剪枝器
    # 注意：实际搜索空间在 optuna_search.py 中使用连续分布（suggest_float/suggest_int）
    # 这里仅保留历史记录和可视化参考
    "search_space": {
        "n_layers": [2, 3, 4],  # Transformer 层数（实际使用 suggest_int(2, 4)）
        "learning_rate": [1e-5, 1e-2],  # 学习率范围（实际使用 log 空间连续搜索）
        "dropout": [0.1, 0.5],  # Dropout 范围（实际使用 step=0.05 连续搜索）
    },
}

# =========================================================================
# 交叉验证配置
# =========================================================================
CV_CONFIG = {
    "n_splits": 5,  # 5折交叉验证
    "shuffle": True,
    "random_state": 42,
}

# =========================================================================
# 体质名称映射
# =========================================================================
CONSTITUTION_NAMES = [
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "痰湿质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
]

# =========================================================================
# 输出文件名
# =========================================================================
OUTPUT_FILES = {
    "best_model": "best_model.pth",
    "scaler_params": "scaler_params.npz",
    "training_history": "training_history.png",
    "cv_comparison": "cv_comparison.png",
    "cv_results": "cv_results.json",
    "test_results": "test_results.json",
    "fixed_test_ids": "fixed_test_ids.npy",
}