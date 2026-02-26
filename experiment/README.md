# MulitiModal - 按摩椅健康监测系统

基于深度学习的按摩椅舒适度分类系统，支持多种模型架构和预处理方案。

---

## 项目结构

```
experiment/
├── dataset/           # 数据集模块
│   ├── config.yaml   # 数据集配置
│   ├── csv_source.py # CSV/NPZ 数据源
│   ├── interfaces.py # 抽象接口定义
│   ├── massage_dataset.py # 数据集实现
│   ├── nk2_processor.py # NK2 预处理器
│   └── self_healing_processor.py # 自研信号自愈预处理器
├── model/            # 模型模块
│   ├── model.py     # 模型架构 (CNN/LSTM/Inception/Transformer)
│   ├── config.py    # 模型配置
│   ├── train.py     # 训练脚本
│   └── *.pth        # 训练好的模型权重
├── generate/         # 数据生成模块
│   ├── generate_10k.py # 10k 数据生成器
│   └── visualize_data.py # 数据可视化
├── eval/            # 评估测试模块
│   └── frameworktest/ # 框架测试
├── predict/         # 预测模块
├── streamdata/       # 流式数据模块
└── test/            # 测试结果
```

---

## 核心功能

### 1. 预处理器

| 预处理器 | 说明 |
|----------|------|
| `NK2Preprocessor` | NeuroKit2 标准预处理 |
| `SelfHealingPreprocessor` | 自研信号自愈处理 (3-Sigma 异常检测 + 样条插值) |

### 2. 模型架构

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| CNN | 简单快速 | 基准线 |
| LSTM | 递归处理时序 | 前后依赖 |
| Inception | 多尺度卷积核 | **推荐** |
| Transformer | 注意力机制 | 大数据 |

### 3. 学习率调度器

支持 4 种调度器：
- `ReduceLROnPlateau`
- `CosineAnnealingLR`
- `CosineAnnealingWarmRestarts`
- `StepLR`

---

## 训练结果

### Inception 模型

| 指标 | 值 |
|------|-----|
| 最佳验证准确率 | **99.00%** |
| 训练时间 | ~5.5 分钟 |
| 数据量 | 10,000 条 |

### Transformer 模型

| 指标 | 值 |
|------|-----|
| 最佳验证准确率 | ~76% (需更多调优) |
| 训练时间 | ~4 小时 |

---

## 快速开始

### 1. 训练模型

```bash
cd /root/repos/MulitiModal
source venv/bin/activate
python experiment/model/train.py
```

### 2. 生成 10k 数据

```bash
python experiment/generate/generate_10k.py
```

### 3. 可视化数据

```bash
python experiment/generate/visualize_data.py
```

### 4. 框架鲁棒性测试

```bash
python experiment/eval/frameworktest/simulate_real_hardware.py
```

---

## 结果路径

### 训练结果

| 文件 | 说明 |
|------|------|
| `model/best_model_inception.pth` | Inception 最佳模型 |
| `model/best_model_transformer.pth` | Transformer 最佳模型 |
| `model/log.txt` | 训练日志 |
| `test/result/test_result_*.png` | 训练曲线 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `model/pretrain_10k.npz` | 10k 预处理数据 |
| `model/processed_data.npz` | 1k 预处理数据 |

### 可视化结果

| 文件 | 说明 |
|------|------|
| `generate/samples_visualization.png` | 随机样本可视化 |
| `generate/class_visualization.png` | 按类别可视化 |
| `generate/signal_comparison.png` | 信号对比图 |

---

## 配置说明

### 切换模型

编辑 `model/config.py`:
```python
MODEL_CONFIG = {
    "type": "inception",  # 可选: cnn / lstm / inception / transformer
    ...
}
```

### 切换调度器

编辑 `model/config.py`:
```python
CURRENT_SCHEDULER = "ReduceLROnPlateau"  # 可选调度器
```

### 切换预处理器

编辑 `dataset/config.yaml`:
```yaml
preprocessor:
  type: "self_healing"  # 可选: nk2 / self_healing
```

---

## 数据格式

### NPZ 数据结构

```
dynamic: (N, 2, 1000)  # N个样本，2通道，1000点/通道
static: (N, 4)         # N个样本，4个静态特征
labels: (N,)            # N个标签
```

### 静态特征

| 索引 | 特征 | 说明 |
|------|------|------|
| 0 | weight | 体重 |
| 1 | hr | 心率 |
| 2 | spo2 | 血氧 |
| 3 | height | 身高 |

---

## 标签定义

| 标签 | 含义 |
|------|------|
| 0 | Poor (很差) |
| 1 | Fair (一般) |
| 2 | Normal (正常) |
| 3 | Good (良好) |
