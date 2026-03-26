# 超参数优化框架使用说明

基于 **Optuna** 实现的贝叶斯优化 + Hyperband 剪枝的超参数优化框架。

---

## 📦 安装依赖

```bash
pip install optuna optuna-dashboard
```

---

## 🚀 快速开始

### 1. 分类任务优化

```bash
python experiment/hyperopt/run_hyperopt.py \
  --model baseline_c \
  --task classification \
  --n_trials 50 \
  --timeout 3600
```

### 2. 回归任务优化

```bash
python experiment/hyperopt/run_hyperopt.py \
  --model baseline_b \
  --task regression \
  --n_trials 100 \
  --timeout 7200
```

### 3. 启动 Optuna Dashboard

```bash
# 查看Study列表
optuna-dashboard experiment/results/hyperopt/study_*/db.sqlite3

# 或者指定具体的Study
optuna-dashboard experiment/results/hyperopt/study_baseline_c_classification_20260326_130000/db.sqlite3
```

访问 http://localhost:8080 查看实时优化进度。

---

## 📋 可优化的超参数

### 通用超参数

| 参数名称 | 类型 | 搜索范围 |
|---------|------|---------|
| `learning_rate` | float | [1e-5, 1e-2] |
| `weight_decay` | float | [1e-6, 1e-3] |
| `batch_size` | int | [16, 32, 64, 128] |
| `dropout` | float | [0.1, 0.5] |
| `gradient_clip` | float | [1.0, 10.0] |

### 模型架构超参数

| 参数名称 | 类型 | 搜索范围 |
|---------|------|---------|
| `shared_dim` | int | [32, 64, 128, 256] |
| `hidden_dim` | int | [64, 128, 256, 512] |
| `encoder_lr_ratio` | float | [0.05, 0.2] |

### 编码器特定超参数

#### Inception 编码器（baseline_c, baseline_b）
- `inception_depth`: [2, 3, 4]
- `inception_bottleneck_channels`: [16, 32, 64]

#### Transformer 编码器（baseline_b）
- `transformer_num_heads`: [2, 4, 8]
- `transformer_num_layers`: [1, 2, 3]
- `transformer_dim_feedforward`: [128, 256, 512]

#### LSTM 编码器（baseline_a）
- `lstm_hidden_size`: [32, 64, 128]
- `lstm_num_layers`: [1, 2, 3]

### 训练超参数

| 参数名称 | 类型 | 搜索范围 |
|---------|------|---------|
| `optimizer` | str | ['adam', 'adamw', 'sgd'] |
| `scheduler` | str | ['cosine_annealing', 'reduce_on_plateau', 'one_cycle'] |
| `warmup_epochs` | int | [5, 10, 15] |
| `patience` | int | [10, 15, 20] |

---

## 💻 编程接口

### 示例 1：使用配置类

```python
from experiment.hyperopt.config import create_classification_config
from experiment.hyperopt.run_hyperopt import run_hyperopt

# 创建配置
config = create_classification_config("baseline_c")

# 运行优化
run_hyperopt(
    model_type="baseline_c",
    task_type="classification",
    n_trials=50,
    timeout=3600
)
```

### 示例 2：自定义搜索空间

```python
from experiment.hyperopt.config import HyperoptConfig

# 创建自定义配置
config = HyperoptConfig(
    model_type="baseline_c",
    task_type="classification",
    n_trials=100,
    timeout=7200
)

# 修改搜索空间
config.search_space.learning_rate = {
    "type": "log_uniform",
    "low": 1e-4,
    "high": 1e-2
}

# 保存配置
config.save_yaml("experiment/config/hyperopt_custom.yaml")
```

### 示例 3：从YAML文件加载配置

```python
from experiment.hyperopt.config import HyperoptConfig

# 从YAML文件加载配置
config = HyperoptConfig.load_yaml("experiment/config/hyperopt_custom.yaml")

# 使用配置运行优化
run_hyperopt(
    model_type=config.model_type,
    task_type=config.task_type,
    n_trials=config.n_trials,
    timeout=config.timeout
)
```

---

## 📊 结果分析

### 1. 查看最优超参数

优化完成后，结果会保存在 `experiment/results/hyperopt/study_*/best_params.json`：

```json
{
  "model_type": "baseline_c",
  "task_type": "classification",
  "best_value": 0.9888,
  "best_params": {
    "learning_rate": 0.001234,
    "batch_size": 64,
    "dropout": 0.2345,
    ...
  },
  "n_trials": 50,
  "elapsed_time": 1234.56
}
```

### 2. 使用最优超参数训练

```python
import json
from experiment.config.base_config import ExperimentConfig
from experiment.model.train_improved import main as train_main

# 加载最优超参数
from pathlib import Path
import glob

# 查找最佳参数文件
study_dirs = glob.glob("experiment/results/hyperopt/study_*/")
if study_dirs:
    best_params_path = Path(study_dirs[0]) / "best_params.json"
    with open(best_params_path, 'r') as f:
        results = json.load(f)
        best_params = results["best_params"]
else:
    raise FileNotFoundError("未找到超参数优化结果文件")

# 更新配置
config = ExperimentConfig()
config.training.learning_rate = best_params["learning_rate"]
config.data.batch_size = best_params["batch_size"]
# ... 更新其他参数

# 保存配置
config.save_yaml("experiment/config/experiment_config_best.yaml")

# 使用最优配置训练
train_main(config_path="experiment/config/experiment_config_best.yaml")
```

---

## 🔧 高级功能

### 1. 自定义剪枝策略

```python
from optuna.pruners import HyperbandPruner

# 创建Hyperband剪枝器
pruner = HyperbandPruner(
    min_resource=1,
    max_resource=50,
    reduction_factor=3
)

# 使用自定义剪枝器
study = optuna.create_study(
    direction="maximize",
    pruner=pruner
)
```

### 2. 多目标优化

```python
# 同时优化准确率和训练时间
def objective(trial):
    # ... 训练代码 ...
    return val_acc, training_time

study = optuna.create_study(
    directions=["maximize", "minimize"]
)
```

### 3. 并行优化

```bash
# 使用多个进程并行优化
python -m optuna.study optimize \
  --study-name "baseline_c_classification" \
  --direction maximize \
  --n-trials 100 \
  --storage "sqlite:///db.sqlite3" \
  --n-jobs 4 \
  experiment.hyperopt.objective
```

---

## 📈 性能优化技巧

### 1. 渐进式优化

```python
# 第一阶段：粗粒度搜索
config.n_trials = 30
config.timeout = 1800  # 30分钟

# 第二阶段：细粒度搜索（基于第一阶段结果）
config.n_trials = 50
config.timeout = 3600  # 60分钟
config.search_space.learning_rate = {
    "type": "log_uniform",
    "low": 0.0005,
    "high": 0.002  # 缩小搜索范围
}
```

### 2. 使用早停

```python
config.pruning.enabled = True
config.pruning.pruner_type = "median"
config.pruning.n_startup_trials = 5
config.pruning.n_warmup_steps = 10
```

### 3. 缓存数据

```python
# 一次性加载数据
train_loader, val_loader, device = prepare_data(config)

# 复用数据加载器
objective = create_objective_function(config, train_loader, val_loader, device)
```

---

## 🎯 最佳实践

1. **从简单开始**：先用默认配置测试，再逐步调整
2. **监控优化进度**：使用Optuna Dashboard实时查看
3. **合理设置试验次数**：50-100次通常足够
4. **使用剪枝**：Hyperband可以减少50-70%的训练时间
5. **保存配置**：记录每次优化的配置和结果
6. **分析超参数重要性**：理解哪些参数最关键

---

## ❓ 常见问题

### Q: 优化时间太长怎么办？

A:
- 减少 `n_trials`（从100降到50）
- 启用剪枝（Hyperband）
- 使用更少的epoch进行优化

### Q: 如何选择优化目标？

A:
- 分类任务：优化 `val_acc`（准确率）
- 回归任务：优化 `mae`（平均绝对误差）
- 可以同时优化多个指标（多目标优化）

### Q: 如何处理模型训练失败？

A:
- Optuna会自动跳过失败的trial
- 在objective函数中使用try-except捕获异常
- 检查日志文件查看失败原因

### Q: 如何复现优化结果？

A:
- 设置固定的随机种子：`config.seed = 42`
- 使用相同的配置文件
- 保存Study数据库以便重新分析

---

## 📚 更多资源

- [Optuna官方文档](https://optuna.readthedocs.io/)
- [Optuna Dashboard](https://github.com/optuna/optuna-dashboard)
- [Hyperband论文](https://arxiv.org/abs/1603.06560)
- [TPE算法](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)

---

**更新日期**：2026-03-26
**版本**：1.0.0