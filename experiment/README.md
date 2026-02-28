# MulitiModal - 按摩椅健康监测系统

基于深度学习的按摩椅舒适度分类系统，支持多种模型架构和预处理方案。

---

![架构图](./image.png)
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

### 微调模型 (Fine-tuned)

| 模型 | 数据量 | 验证准确率 |
|------|--------|-----------|
| Inception (微调) | 50 样本 | **80%** |

---

## 实时流式测试结果

使用 `stream_engine.py` 对多个测试流进行实时推理测试：

### 基础测试

| 测试流 | 预测次数 | 正确 | 准确率 | 真实标签 |
|--------|----------|------|--------|----------|
| stream_001 (weight=70, hr=75) | 281 | 281 | **100%** | 3 (良好) |
| stream_005 (weight=85, hr=95) | 281 | 281 | **100%** | 3 (良好) |

### 极端环境测试

| 测试流 | 干扰类型 | 预测次数 | 准确率 |
|--------|----------|----------|--------|
| stream_004 | +15Pa 基线偏移 + 5Hz 震动 | 281 | **100%** |

### 测试配置
- 模型：`bfoundation_model_inception.pth` (Inception)
- 预处理器：`SelfHealingPreprocessor` (与训练一致)
- 缓冲区：1000 点 (20秒)
- 推理频率：1Hz

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

### 5. 实时流式推理测试

```bash
python experiment/streamdata/stream_engine.py
```

---

## 结果路径

### 训练结果

| 文件 | 说明 |
|------|------|
| `model/bfoundation_model_inception.pth` | Inception 预训练模型 (99%) |
| `model/best_model_inception.pth` | Inception 最佳模型 (99%) |
| `model/bfoundation_model_inception_finetuned.pth` | Inception 微调模型 (80%) |
| `model/best_model_transformer.pth` | Transformer 最佳模型 |
| `model/log.txt` | 训练日志 |
| `test/result/test_result_*.png` | 训练曲线 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `model/pretrain_10k.npz` | 10k 预处理数据 |
| `model/processed_data.npz` | 1k 预处理数据 |

### 流式测试数据

| 文件 | 说明 |
|------|------|
| `streamdata/stream_001_70_75_98_175.csv` | 测试流1 (weight=70, 标签=3 良好) |
| `streamdata/stream_002_70_75_98_175.csv` | 测试流2 (weight=70, 标签=变化 1→3→0→2) |
| `streamdata/stream_003_70_75_98_175.csv` | 测试流3 (weight=70, 高噪声) |
| `streamdata/stream_004_70_75_98_175.csv` | 测试流4 (+15Pa基线偏移 + 5Hz震动) |
| `streamdata/stream_005_85_95_96_180.csv` | 测试流5 (不同静态特征 weight=85, hr=95) |

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

---

## 关键发现

### 1. 预处理一致性至关重要
训练与推理必须使用相同的预处理流程。使用 `SelfHealingPreprocessor` 确保训练和流式推理的预处理一致，这是实现 100% 准确率的关键。

### 2. Inception 优于 Transformer
在此数据集上，Inception 模型表现更好：
- Inception: **99%** 验证准确率，训练时间短
- Transformer: ~76% 验证准确率，需要更多调优

### 3. 迁移学习可行
使用 50 个真实样本微调预训练模型，达到了 80% 的验证准确率，证明迁移学习在此场景下有效。

### 4. 实时流式推理稳定
在 3 个不同的测试流上均达到 100% 准确率，模型推理稳定可靠。

### 5. Z-Score 归一化的"护盾"效应
在 `stream_engine.py` 中使用了滚动 Z-Score 归一化：
```python
s_norm = (s_filtered - np.mean(s_filtered)) / (np.std(s_filtered) + 1e-6)
```

**物理意义**：当真实硬件基线多了 10Pa 时，`np.mean(s_filtered)` 也跟着多了 10Pa，两者相减，偏差被瞬间抵消。

**实验验证**：使用 `stream_004` (+15Pa 基线偏移 + 5Hz 高频震动) 测试，模型仍能正确识别，证明 Z-Score 归一化形成了"环境适应护盾"。

### 6. 模型学习的是波形模式，而非静态特征捷径
使用不同静态特征的测试文件 `stream_005_85_95_96_180.csv` (weight=85, hr=95) 进行测试：
- 波形模式与 `stream_001` 相同 (amplitude=35, 良好)
- 预测结果：**正确识别为良好**

结论：模型确实在学习波形模式，而不是依赖静态特征的捷径。
