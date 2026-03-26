# 超参数优化计划

## 📋 项目概述

为多模态按摩椅舒适度预测系统添加超参数优化功能，使用 **Optuna** 实现贝叶斯优化 + Hyperband 剪枝。

---

## 🎯 目标

1. **自动化超参数搜索**：自动找到最佳超参数组合
2. **高效资源利用**：使用 Hyperband 剪枝减少无效训练
3. **支持多种模型**：支持 5 种融合策略模型
4. **支持双任务**：分类任务和回归任务
5. **可复现性**：支持实验记录和结果保存

---

## 🏗️ 架构设计

### 1. 整体架构

```
Optuna 框架
├── Objective 函数
│   ├── 超参数采样（Trial.suggest_*）
│   ├── 模型配置
│   ├── 训练循环
│   ├── Hyperband 剪枝
│   └── 返回优化指标
├── Study 管理
│   ├── 创建 Study
│   ├── 运行优化
│   └── 结果分析
└── 可视化
    ├── 超参数重要性
    ├── 优化历史
    └── 并行坐标图
```

### 2. 组件设计

#### 2.1 Objective 函数
- 接收 trial 对象
- 根据超参数范围进行采样
- 配置模型和训练
- 返回验证集性能指标

#### 2.2 Hyperband 剪枝
- 使用 `MedianPruner` 或 `SuccessiveHalvingPruner`
- 在训练过程中提前终止表现差的 trial
- 提高优化效率

#### 2.3 搜索空间定义
- 连续参数：学习率、dropout、weight_decay
- 离散参数：batch_size、hidden_dim、num_layers
- 条件参数：不同模型架构的特定参数

#### 2.4 结果管理
- 保存最优超参数
- 保存优化历史
- 生成优化报告
- 可视化结果

---

## 🔧 可优化的超参数

### 3.1 通用超参数

| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `learning_rate` | float | [1e-5, 1e-2] | 学习率（log均匀分布） |
| `weight_decay` | float | [1e-6, 1e-3] | 权重衰减（log均匀分布） |
| `batch_size` | int | [16, 32, 64, 128] | 批次大小（类别选择） |
| `dropout` | float | [0.1, 0.5] | Dropout 概率 |
| `gradient_clip` | float | [1.0, 10.0] | 梯度裁剪阈值 |

### 3.2 模型架构超参数

| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `shared_dim` | int | [32, 64, 128, 256] | 共享特征维度 |
| `hidden_dim` | int | [64, 128, 256, 512] | 隐藏层维度 |
| `encoder_lr_ratio` | float | [0.05, 0.2] | 编码器学习率比例 |

### 3.3 编码器特定超参数

#### Inception 编码器
| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `inception_depth` | int | [2, 3, 4] | Inception 深度 |
| `inception_bottleneck_channels` | int | [16, 32, 64] | 瓶颈通道数 |

#### Transformer 编码器
| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `transformer_num_heads` | int | [2, 4, 8] | 注意力头数 |
| `transformer_num_layers` | int | [1, 2, 3] | Transformer 层数 |
| `transformer_dim_feedforward` | int | [128, 256, 512] | 前馈网络维度 |

#### LSTM 编码器
| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `lstm_hidden_size` | int | [32, 64, 128] | LSTM 隐藏层大小 |
| `lstm_num_layers` | int | [1, 2, 3] | LSTM 层数 |
| `lstm_bidirectional` | bool | [True, False] | 是否双向 |

### 3.4 训练超参数

| 参数名称 | 类型 | 搜索范围 | 说明 |
|---------|------|---------|------|
| `optimizer` | str | ['adam', 'adamw', 'sgd'] | 优化器类型 |
| `scheduler` | str | ['cosine_annealing', 'reduce_on_plateau', 'one_cycle'] | 学习率调度器 |
| `warmup_epochs` | int | [5, 10, 15] | 预热轮数 |
| `patience` | int | [10, 15, 20] | 早停耐心值 |

---

## 📁 文件结构

```
experiment/
├── hyperopt/
│   ├── __init__.py                    # 超参数优化模块初始化
│   ├── config.py                      # Optuna 配置和搜索空间定义
│   ├── objective.py                   # Objective 函数实现
│   ├── pruner.py                      # Hyperband 剪枝器配置
│   ├── study_manager.py               # Study 管理和运行
│   ├── visualizer.py                  # 结果可视化
│   └── run_hyperopt.py                # 超参数优化运行脚本
├── config/
│   └── hyperopt_config.yaml           # 超参数优化配置文件
└── results/
    └── hyperopt/                      # 超参数优化结果目录
        ├── study_YYYYMMDD_HHMMSS/    # Study 目录
        │   ├── db.sqlite3             # Optuna 数据库
        │   ├── best_params.json       # 最优超参数
        │   ├── optimization_history.csv  # 优化历史
        │   └── figures/                # 可视化图表
        └── ...
```

---

## 🚀 实现步骤

### 步骤 1：安装依赖

```bash
pip install optuna optuna-dashboard
```

### 步骤 2：创建超参数优化模块

#### 2.1 配置文件 (`hyperopt/config.py`)
- 定义搜索空间
- 定义优化目标
- 配置 Hyperband 剪枝参数

#### 2.2 Objective 函数 (`hyperopt/objective.py`)
- 实现 trial.suggest_* 采样
- 配置模型和训练
- 实现 Hyperband 剪枝回调
- 返回优化指标

#### 2.3 Study 管理 (`hyperopt/study_manager.py`)
- 创建和管理 Study
- 运行优化
- 保存结果
- 分析最优超参数

#### 2.4 剪枝器 (`hyperopt/pruner.py`)
- 配置 MedianPruner
- 配置 SuccessiveHalvingPruner
- 支持自定义剪枝策略

#### 2.5 可视化 (`hyperopt/visualizer.py`)
- 超参数重要性分析
- 优化历史可视化
- 并行坐标图
- 其他 Optuna 可视化图表

#### 2.6 运行脚本 (`hyperopt/run_hyperopt.py`)
- 命令行接口
- 支持不同模型和任务
- 支持并行优化
- 实时监控（Optuna Dashboard）

### 步骤 3：创建配置文件

```yaml
# experiment/config/hyperopt_config.yaml
# 超参数优化配置

# 优化设置
optimization:
  n_trials: 100              # 总试验次数
  timeout: 7200             # 总超时时间（秒）
  n_jobs: 1                 # 并行任务数

# 剪枝设置
pruning:
  enabled: true
  pruner_type: "median"     # median, successive_halving
  n_startup_trials: 5       # 启动试验次数
  n_warmup_steps: 10        # 预热步数
  interval_steps: 1         # 检查间隔

# 优化目标
objective:
  metric: "val_loss"        # 优化指标：val_loss, val_acc, mae
  direction: "minimize"     # 最小化/最大化

# 模型配置
model:
  type: "baseline_c"        # 模型类型
  task: "classification"    # 任务类型

# 搜索空间
search_space:
  # 通用超参数
  learning_rate:
    type: "log_uniform"
    low: 1e-5
    high: 1e-2
  
  weight_decay:
    type: "log_uniform"
    low: 1e-6
    high: 1e-3
  
  batch_size:
    type: "categorical"
    choices: [16, 32, 64, 128]
  
  dropout:
    type: "uniform"
    low: 0.1
    high: 0.5
  
  # 模型架构超参数
  shared_dim:
    type: "categorical"
    choices: [32, 64, 128, 256]
  
  hidden_dim:
    type: "categorical"
    choices: [64, 128, 256, 512]
  
  encoder_lr_ratio:
    type: "uniform"
    low: 0.05
    high: 0.2
  
  # 编码器特定超参数
  inception_depth:
    type: "categorical"
    choices: [2, 3, 4]
    condition:
      model: ["baseline_c", "baseline_b"]
  
  transformer_num_heads:
    type: "categorical"
    choices: [2, 4, 8]
    condition:
      model: ["baseline_b"]
```

### 步骤 4：实现示例运行脚本

```bash
# 运行超参数优化
python experiment/hyperopt/run_hyperopt.py \
  --model baseline_c \
  --task classification \
  --n_trials 50 \
  --timeout 3600

# 启动 Optuna Dashboard
optuna-dashboard experiment/results/hyperopt/study_YYYYMMDD_HHMMSS/db.sqlite3
```

### 步骤 5：集成到现有框架

- 在训练脚本中支持加载最优超参数
- 在配置文件中添加超参数优化结果路径
- 创建示例脚本展示如何使用优化结果

---

## 📊 预期效果

### 优化效率
- **Hyperband 剪枝**：减少 50-70% 的无效训练时间
- **贝叶斯优化**：比网格搜索快 5-10 倍
- **并行优化**：支持多 GPU 并行搜索

### 性能提升
- **分类任务**：预期准确率提升 1-3%
- **回归任务**：预期 MAE 降低 5-10%
- **训练稳定性**：减少过拟合和训练失败

### 可维护性
- **模块化设计**：易于扩展和修改
- **配置化**：通过 YAML 文件配置优化参数
- **可视化**：直观展示优化过程和结果

---

## 🎓 使用示例

### 示例 1：分类任务优化

```python
from experiment.hyperopt.study_manager import StudyManager

# 创建 Study 管理器
manager = StudyManager(
    model_type="baseline_c",
    task_type="classification",
    n_trials=50,
    timeout=3600
)

# 运行优化
best_params, best_value = manager.optimize()

# 使用最优超参数训练
print(f"最优超参数: {best_params}")
print(f"最优验证准确率: {best_value}")
```

### 示例 2：回归任务优化

```python
from experiment.hyperopt.study_manager import StudyManager

# 创建 Study 管理器
manager = StudyManager(
    model_type="baseline_b",
    task_type="regression",
    n_trials=100,
    timeout=7200
)

# 运行优化
best_params, best_value = manager.optimize()

# 使用最优超参数训练
print(f"最优超参数: {best_params}")
print(f"最优验证 MAE: {best_value}")
```

### 示例 3：多模型对比优化

```python
from experiment.hyperopt.study_manager import StudyManager

models = ["baseline_a", "baseline_b", "baseline_c"]
results = {}

for model in models:
    manager = StudyManager(
        model_type=model,
        task_type="classification",
        n_trials=30
    )
    best_params, best_value = manager.optimize()
    results[model] = {
        "best_params": best_params,
        "best_value": best_value
    }

# 对比结果
for model, result in results.items():
    print(f"{model}: {result['best_value']}")
```

---

## 📈 监控和可视化

### Optuna Dashboard
```bash
# 启动 Dashboard
optuna-dashboard experiment/results/hyperopt/study_*/db.sqlite3

# 访问 http://localhost:8080
```

### 可视化图表
- 超参数重要性条形图
- 优化历史折线图
- 并行坐标图
- 超参数关系热力图

---

## 🔍 最佳实践

1. **渐进式优化**：
   - 先粗粒度搜索，再细粒度搜索
   - 逐步缩小搜索空间

2. **早停策略**：
   - 使用 Hyperband 剪枝
   - 设置合理的耐心值

3. **并行优化**：
   - 使用多 GPU 并行搜索
   - 分布式优化

4. **结果分析**：
   - 分析超参数重要性
   - 理解模型行为

5. **可复现性**：
   - 保存优化历史
   - 记录最优超参数

---

## ⚙️ 高级特性

### 1. 条件超参数搜索
- 根据模型类型动态调整搜索空间
- 根据任务类型调整优化目标

### 2. 多目标优化
- 同时优化多个指标（准确率、训练时间）
- 使用 Pareto 最优解

### 3. 分布式优化
- 多机多卡并行优化
- 数据库共享 Study

### 4. 持续学习
- 基于历史优化结果调整搜索策略
- 自适应搜索空间

---

## 🎯 预期成果

1. **自动化超参数搜索**：无需手动调参
2. **性能提升**：模型性能提升 1-3%
3. **效率提升**：优化时间减少 50-70%
4. **可复现性**：所有优化结果可追溯
5. **易于扩展**：支持新模型和新任务

---

## 📝 后续扩展

1. **支持更多优化算法**：TPE, CMA-ES, Random Search
2. **支持更多剪枝策略**：ASHA, BOHB
3. **集成到 CI/CD**：自动化超参数优化流程
4. **支持在线学习**：实时优化超参数
5. **支持 AutoML**：自动模型选择和超参数优化

---

**创建日期**：2026-03-26
**状态**：计划中
**下一步**：开始实现