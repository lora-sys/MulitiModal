# Iteration 1.1 实验计划：解决AI领域三大核心痛点

## 🎯 实验核心目标

通过**数据放量（200→1000）+ 强力加噪 + 特征瘦身 + K折验证**，系统性解决AI领域的三大核心痛点：

| 痛点 | 解决方案 | 验证指标 |
|------|---------|---------|
| **真实性** | 1000人工业数据（3种噪声） | 噪声环境下的准确率 |
| **鲁棒性** | 增强版预处理（异常检测+修复） | 性能下降率 < 15% |
| **稳定性** | 5-Fold交叉验证 | 标准差 < 2% |

---

## 📊 实验设计矩阵

### 三组对比实验

| 实验组 | 数据规模 | 特征数量 | 预处理 | 目的 |
|--------|---------|---------|--------|------|
| **实验A** | 1000人 | 16个（全量） | 无 | 建立噪声环境下的基线 |
| **实验B** | 1000人 | 6个（精简） | 无 | 验证特征选择的有效性 |
| **实验C** | 1000人 | 6个（精简） | 增强版 | 验证预处理的价值 |

---

## 🔬 实验详细设计

### 实验 A：全量特征基线（噪声环境）

**目标**：验证在噪声环境下，使用全量特征的性能表现

**配置**：
- 数据：1000人工业数据（3种噪声）
- 特征：16个全量特征
- 预处理：无（直接使用原始噪声数据）
- 模型：RandomForestClassifier (n_estimators=100, max_depth=10)
- 验证：Stratified 5-Fold CV

**全量特征列表**（16个）：
```python
features_full = [
    # 物理参数
    'weight', 'height', 'hr', 'spo2',
    # 传感器1
    'sensor1_mean', 'sensor1_std', 'sensor1_ptp', 'sensor1_min', 'sensor1_max',
    # 传感器2
    'sensor2_mean', 'sensor2_std', 'sensor2_ptp', 'sensor2_min', 'sensor2_max',
    # 相对特征
    'amplitude_ratio', 'offset_ratio'
]
```

**预期结果**：
- 准确率：75-80%（受噪声影响明显）
- 标准差：3-5%（不稳定）
- 问题：mean特征与体重强相关，可能导致过拟合

---

### 实验 B：特征瘦身（噪声环境）

**目标**：验证去掉"猪队友"特征后的性能提升

**配置**：
- 数据：1000人工业数据（3种噪声）
- 特征：6个精简特征
- 预处理：无
- 模型：RandomForestClassifier (n_estimators=100, max_depth=10)
- 验证：Stratified 5-Fold CV

**精简特征列表**（6个）：
```python
features_reduced = [
    # 传感器动态特征
    'sensor1_std',   # 稳定性
    'sensor2_std',   # 稳定性
    'sensor1_ptp',   # 振动幅度
    'sensor2_ptp',   # 振动幅度
    # 生理指标
    'hr',            # 心率
    'spo2'           # 血氧
]
```

**特征选择理由**：
- ❌ 去掉 `mean`：与体重强相关，但与舒适度无关
- ❌ 去掉 `weight`, `height`：静态参数，与动态感知无关
- ✅ 保留 `std`, `ptp`：动态特征，直接反映按摩体验
- ✅ 保留 `hr`, `spo2`：生理指标，反映身体状态

**预期结果**：
- 准确率：80-85%（相比实验A提升）
- 标准差：2-3%（更稳定）
- 训练时间：减少30-40%

---

### 实验 C：增强版预处理（鲁棒性验证）

**目标**：验证异常检测与修复算法的价值

**配置**：
- 数据：1000人工业数据（3种噪声）
- 特征：6个精简特征
- 预处理：增强版（跳点检测 + 漂移校正 + 底噪抑制）
- 模型：RandomForestClassifier (n_estimators=100, max_depth=10)
- 验证：Stratified 5-Fold CV

**增强版预处理流程**：

#### 1. 跳点检测与修复
```python
# 检测：滑动窗口 Z-score
window_size = 10
threshold = 5.0

# 修复：邻域中位数替换
for spike_idx in detected_spikes:
    signal[spike_idx] = np.median(signal[spike_idx-5:spike_idx+5])
```

#### 2. 基线漂移校正
```python
# 检测：移动平均趋势
trend = np.convolve(signal, np.ones(100)/100, mode='same')

# 修复：去趋势
signal_detrended = signal - trend
```

#### 3. 底噪抑制
```python
# 方法：小波去噪或移动平均
signal_denoised = moving_average(signal_detrended, window=3)
```

**预期结果**：
- 准确率：85-90%（相比实验B进一步提升）
- 标准差：< 2%（高稳定性）
- 性能下降率：相比干净数据 < 15%

---

## 📈 成功标准

### 1. 真实性（真实性验证）

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| **噪声环境准确率** | > 80% | 实验 A vs 干净数据对比 |
| **跳点检测率** | > 95% | 实验 C 中的异常检测统计 |
| **漂移校正效果** | > 90% | 修复前后信号质量对比 |

### 2. 鲁棒性（鲁棒性验证）

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| **性能下降率** | < 15% | 实验 C vs 干净数据基线 |
| **抗噪声能力** | SNR-10dB时准确率 > 70% | 不同噪声水平测试 |
| **灾难性故障率** | < 1% | 完全错误预测的比例 |

### 3. 稳定性（稳定性验证）

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| **5折标准差** | < 2% | 5-Fold CV 的标准差 |
| **类别间方差** | < 3% | 4个类别的准确率方差 |
| **重复测试稳定性** | 标准差 < 1% | 多次运行同一实验 |

---

## 📁 输出文件结构

```
iteration1.1/
├── generate_industrial_data.py          # 数据生成（已完成）
├── extract_features_industrial.py       # 特征提取
├── experiment_A_full_features.py        # 实验A
├── experiment_B_reduced_features.py     # 实验B
├── experiment_C_robust_preprocessing.py # 实验C
├── comparison_analysis.py               # 对比分析
├── visualization_report.py              # 可视化报告
├── features/
│   ├── industrial_features_full.csv     # 16个全量特征
│   └── industrial_features_reduced.csv  # 6个精简特征
├── results/
│   ├── experiment_A_results.csv
│   ├── experiment_B_results.csv
│   ├── experiment_C_results.csv
│   ├── comparison_summary.csv
│   ├── robustness_metrics.csv
│   └── visualization/
│       ├── three_experiments_comparison.png
│       ├── feature_importance_A_vs_B.png
│       ├── preprocessing_effect.png
│       ├── confusion_matrices.png
│       ├── accuracy_boxplot.png
│       └── stability_analysis.png
├── models/
│   ├── model_A_full.pkl
│   ├── model_B_reduced.pkl
│   └── model_C_robust.pkl
└── logs/
    ├── experiment_log.json
    └── performance_metrics.json
```

---

## 🎨 可视化报告要点

### 1. 解决"真实性"的证据
- **噪声示例图**：干净 vs 噪声 vs 修复后的波形对比
- **准确率对比**：干净数据 vs 工业数据
- **异常检测效果图**：跳点标注 + 修复结果

### 2. 解决"鲁棒性"的证据
- **性能下降曲线**：不同噪声水平下的准确率
- **预处理效果图**：修复前后的分类准确率对比
- **混淆矩阵热力图**：实验A vs B vs C 的错误分析

### 3. 解决"稳定性"的证据
- **5折准确率箱线图**：三组实验的稳定性对比
- **特征重要性对比**：实验A vs B 的特征权重变化
- **雷达图**：准确率、召回率、F1、稳定性、效率的综合评分

---

## 🚀 执行顺序

```
1. 特征提取（任务3）
   ↓
2. 实验A（任务4）→ 实验B（任务5）→ 实验C（任务6）
   ↓
3. 对比分析（任务7）
   ↓
4. 可视化报告（任务8）
```

---

## 💡 预期核心发现

1. **特征瘦身的价值**：
   - 实验 B > 实验 A（精简特征优于全量特征）
   - 证明：去除"猪队友"特征能提升性能

2. **预处理的价值**：
   - 实验 C > 实验 B（带预处理优于无预处理）
   - 证明：异常检测与修复算法有效

3. **稳定性的提升**：
   - 5-Fold标准差：A > B > C
   - 证明：特征选择 + 预处理能提升稳定性

---

## 📊 最终输出指标

| 指标 | 实验A | 实验B | 实验C | 改进幅度 |
|------|-------|-------|-------|---------|
| **准确率** | ? | ? | ? | A→B→C |
| **标准差** | ? | ? | ? | 越小越好 |
| **训练时间** | ? | ? | ? | B < A |
| **特征数量** | 16 | 6 | 6 | B↓62.5% |
| **预处理时间** | 0 | 0 | +10ms | C的代价 |

---

**计划版本**：v1.0
**最后更新**：2026-01-29
**作者**：Iteration 1.1 团队