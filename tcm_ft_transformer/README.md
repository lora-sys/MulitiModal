# FT-Transformer 中医体质分类

基于 FT-Transformer 架构的深度学习模型，用于根据 8 维静态体征预测 9 种中医体质的概率分布。

## 项目特点

- **模型架构**: FT-Transformer (Feature Tokenizer Transformer)
- **损失函数**: KL Divergence (衡量概率分布差异)
- **标签处理**: Epsilon 平滑 (0.01) + 行归一化
- **优化策略**: Optuna 贝叶斯超参数搜索 + 5 折交叉验证
- **输出层**: Softmax (强制输出和为 1)

## 项目结构

```
tcm_ft_transformer/
├── config.py              # 全局配置
├── preprocess.py          # 数据预处理
├── ft_transformer.py      # FT-Transformer 模型
├── train.py               # 训练器
├── optuna_search.py       # Optuna 超参数搜索
├── visualize.py           # 可视化工具
├── main.py                # 主入口文件
├── train_all.sh           # 一键训练脚本
├── data/                  # 数据目录
├── checkpoints/           # 模型检查点
├── logs/                  # 训练日志
└── results/               # 结果文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate

# 安装依赖（使用项目根目录的 requirements.txt）
pip install -r ../requirements.txt
```

### 2. 准备数据

数据文件格式: `data/vital_signs_dataset_final.csv`

- 前 8 列: 特征 (Age, Gender, BMI, HeartRate, SBP, DBP, SpO2, Temperature)
- 后 9 列: 标签 (平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质、特禀质)

**注意**: 请确保数据集文件已放置在 `data/vital_signs_dataset_final.csv` 位置。

### 3. 一键训练

```bash
./train_all.sh
```

这将自动完成:
1. 检查环境和依赖
2. 检查数据文件
3. 运行 Optuna 超参数搜索 (20 trials)
4. 运行 5 折交叉验证
5. 在测试集上评估
6. 生成可视化结果

### 4. 手动训练

#### 完整流程

```bash
python3 main.py \
    --mode full \
    --data data/vital_signs_dataset_final.csv \
    --trials 20 \
    --epochs_search 20 \
    --epochs_final 50 \
    --device cuda
```

#### 仅搜索超参数

```bash
python3 main.py \
    --mode search \
    --data data/vital_signs_dataset_final.csv \
    --trials 20 \
    --epochs_search 20
```

## 参数说明

### main.py 参数

- `--mode`: 运行模式
  - `full`: 完整流程 (搜索 + 验证 + 测试)
  - `search`: 仅搜索超参数
- `--data`: 数据文件路径
- `--trials`: Optuna 试验次数 (默认: 20)
- `--epochs_search`: 搜索阶段每个 trial 的训练轮数 (默认: 20)
- `--epochs_final`: 最终训练的轮数 (默认: 50)
- `--device`: 设备 (cuda/cpu, 默认: cuda)

## 交付物清单

训练完成后，将生成以下文件:

1. **模型权重**: `checkpoints/best_model.pth`
2. **标准化参数**: `data/scaler_params.npz`
3. **训练历史**: `checkpoints/training_history.png`
4. **交叉验证对比**: `checkpoints/cv_comparison.png`
5. **交叉验证结果**: `results/cv_results.json`
6. **Optuna 搜索结果**: `checkpoints/optuna_results.json`
7. **Optuna 可视化**: `checkpoints/optuna_results.png`
8. **预测分布**: `checkpoints/prediction_distribution.png`
9. **混淆矩阵**: `checkpoints/confusion_matrix.png`
10. **测试结果**: `checkpoints/test_results.json`

## 核心技术要点

### 1. 数据预处理

- **Gender 编码**: Male → 0, Female → 1
- **数据类型转换**: 所有列转为 float32
- **动态划分**: 90% 训练验证池 + 10% 锁定测试集
- **特征标准化**: 仅在训练验证池上计算 mean/std
- **标签预处理**: Epsilon (0.01) 平滑 + 行归一化

### 2. 模型架构

- **NumericalFeatureTokenizer**: 8 个独立的 Linear 层，将每个特征映射为 64 维向量
- **CLS Token**: 用于聚合全局信息
- **Transformer Encoder**: 3 层，4 个注意力头
- **输出层**: Linear + Softmax

### 3. 训练配置

- **优化器**: AdamW (lr=1e-3, weight_decay=0.01)
- **调度器**: CosineAnnealingLR + Warmup (前 5%)
- **梯度裁剪**: max_norm=1.0
- **早停**: patience=5

### 4. 超参数搜索

- **采样器**: TPESampler
- **剪枝器**: MedianPruner
- **搜索空间**:
  - n_layers: [2, 3, 4]
  - learning_rate: [1e-4, 5e-4, 1e-3, 5e-3]
  - dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
- **固定参数**:
  - d_token: 64
  - n_heads: 4

## 注意事项

1. **数据泄露**: 标准化的 Mean/Std 仅基于训练集计算，测试集使用相同参数
2. **标签归一化**: 每一行标签的和必须严格等于 1.0
3. **随机种子**: 所有随机操作都使用固定的随机种子 (42) 以保证可复现性
4. **GPU 内存**: 如果遇到 OOM 错误，可以减小 batch_size

## 故障排除

### 1. CUDA Out of Memory

```bash
# 在 config.py 中减小 batch_size
TRAIN_CONFIG = {
    "batch_size": 128,  # 从 256 减小到 128
    ...
}
```

### 2. 数据文件不存在

请确保数据集文件已放置在 `data/vital_signs_dataset_final.csv` 位置。

### 3. 依赖缺失

```bash
# 安装所有依赖（使用项目根目录的 requirements.txt）
pip install -r ../requirements.txt
```

## 许可证

MIT License

## 联系方式

如有问题，请联系项目维护者。