# Iteration 1.2: 迈向"深度学习"与"时序建模"

## 🎯 阶段目标

从"手工特征"转向"端到端学习"，引入深度学习架构，解决传统模型在复杂真实环境下的局限性。

---

## 🔍 上一阶段总结（Iteration 1.1）

### ✅ 逻辑成立
- 压力稳定性（std）和振幅（ptp）确实是判断身体素质的关键物理指标
- 打乱标签测试（26.60% 接近随机基线）证明了模型学到了真实规律
- 物理意义清晰，可解释性强

### ⚠️ 瓶颈显现
- 仿真数据规律性过强，导致随机森林轻易拿到100%准确率
- 类别间参数范围重叠度较低（34.9%），传统统计特征仍可区分
- 真实世界的个体差异、坐姿偏移、传感器漂移未被充分考虑

### 🚨 现实挑战
- 真实人体数据会有更严重的个体差异
- 坐姿偏移会导致系统性偏移
- 传感器零点漂移需要在线校准
- 预期在真实环境中准确率会下降到70-85%

---

## 🚀 Iteration 1.2 四大核心任务

### 任务1：模拟"高难度"数据集（仿真 4.0）

#### 目的
模拟真实世界的"模糊性"，把100%的准确率降下来，给深度学习留出优化空间。

#### 动作
在数据生成脚本中，将类别间的参数范围重叠度增加到50%以上。

#### 具体修改
```python
# 原来的参数范围（iteration1.1）
p_amplitude = 15 + label * 8  # 15, 23, 31, 39（不重叠）

# 新的参数范围（iteration1.2）
p_amplitude = 20 + label * 4 + np.random.uniform(-5, 5)  # 15-25, 19-29, 23-33, 27-37（高度重叠）
p_noise_level = 6.0 + np.random.uniform(-2, 2)  # 所有类别噪声水平相近
```

#### 意义
- 传统统计特征（mean/std）就分不开了
- 必须依靠波形的微观时序特征来判断
- 模拟真实世界的"边界模糊"情况

#### 预期结果
- 随机森林准确率跌破90%（目标：85-90%）
- 为深度学习创造优化空间

---

### 任务2：引入 1D-CNN 或 GRU（LSTM 优化版）

#### 目的
直接对压力波形进行训练，而不是用提取后的6个特征。

#### 为什么选择 1D-CNN 而非 LSTM？
1. **训练速度**：1D-CNN 训练速度是 LSTM 的3-5倍
2. **部署效率**：在嵌入式系统上，1D-CNN 的推理速度是 LSTM 的数倍
3. **参数量**：1D-CNN 参数量更少，内存占用更低
4. **工业实践**：针对震动信号，1D-CNN 在工业界效果更佳

#### 实验设计

**输入**：
- 20秒原始压力序列（1000个点）
- 形状：(batch_size, 1000, 2) - 双传感器

**输出**：
- 4个等级分类（身体表征：很差/一般/正常/良好）
- 形状：(batch_size, 4)

**模型架构**：
```python
# 1D-CNN 基准模型
model = Sequential([
    # 第1层：提取局部特征
    Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(1000, 2)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # 第2层：提取时序特征
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # 第3层：提取高级特征
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.5),

    # 全连接层
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```

#### 对比实验
- **基线**：RandomForest + 手工特征（6个）
- **实验1**：1D-CNN + 原始波形（1000点）
- **实验2**：GRU + 原始波形（1000点）
- **实验3**：1D-CNN + GRU 混合架构

#### 预期结果
- 在"高难度"数据集上，深度学习准确率 > 随机森林
- 证明深度学习能捕捉传统特征无法发现的微观时序特征

---

### 任务3：嵌入式部署预研

#### 目的
提前验证模型能不能跑在按摩椅的主控板上。

#### 动作A：随机森林 → C 代码（m2cgen）

**工具**：m2cgen（Machine Learning to Code Generator）

**步骤**：
```python
import m2cgen as m2c

# 加载训练好的随机森林模型
model = joblib.load('./models/model_C_robust.pkl')

# 导出为 C 代码
code = m2c.to_c(model)

# 保存代码
with open('./models/rf_model.c', 'w') as f:
    f.write(code)
```

**评估指标**：
- Flash 占用（代码大小）
- RAM 占用（运行时内存）
- 推理时间（单次预测耗时）

#### 动作B：深度学习 → TFLite Micro

**工具**：TensorFlow Lite for Microcontrollers

**步骤**：
```python
import tensorflow as tf

# 加载训练好的深度学习模型
model = tf.keras.models.load_model('./models/cnn_model.h5')

# 转换为 TFLite 格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 保存 TFLite 模型
with open('./models/cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 估算内存需求
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
print(f"模型大小: {len(tflite_model) / 1024:.2f} KB")
print(f"输入张量大小: {interpreter.get_input_details()[0]['shape']}")
print(f"输出张量大小: {interpreter.get_output_details()[0]['shape']}")
```

**评估指标**：
- Flash 占用（模型文件大小）
- RAM 占用（推理时需要的内存）
- 推理时间（在目标硬件上的实际耗时）

#### 目标硬件规格（按摩椅主控板）
- **CPU**：ARM Cortex-M4 @ 168MHz
- **Flash**：512KB - 1MB
- **RAM**：128KB - 256KB
- **推理时间要求**：< 100ms

#### 预期结果
- 随机森林：Flash ~50KB, RAM ~10KB, 推理时间 ~10ms
- 1D-CNN：Flash ~200KB, RAM ~50KB, 推理时间 ~50ms
- 结论：两种模型都能部署，但需要优化

---

### 任务4：多模态生理特征深度融合

#### 目的
引入更专业的生理指标，提高模型鲁棒性。

#### 工具：NeuroKit2（如果可用）或手动实现

#### 可提取的生理特征

**心率变异性（HRV）特征**：
- RMSSD（相邻RR间期差值的均方根）
- SDNN（RR间期的标准差）
- pNN50（相邻RR间期差值>50ms的百分比）

**压力波形特征（高级）**：
- 频谱特征（FFT的频域能量分布）
- 小波变换特征（时频分析）
- 分形维数（信号复杂度）

#### 融合架构
```python
# 多输入模型
# 输入1：压力波形 (1000, 2)
# 输入2：手工特征 (6,)
# 输入3：高级生理特征 (10,)

# 压力波形分支
waveform_input = Input(shape=(1000, 2), name='waveform')
x = Conv1D(64, 7, activation='relu')(waveform_input)
x = GlobalAveragePooling1D()(x)
waveform_features = Dense(32, activation='relu')(x)

# 手工特征分支
manual_input = Input(shape=(6,), name='manual')
manual_features = Dense(16, activation='relu')(manual_input)

# 生理特征分支
physio_input = Input(shape=(10,), name='physio')
physio_features = Dense(16, activation='relu')(physio_input)

# 融合层
combined = concatenate([waveform_features, manual_features, physio_features])
x = Dense(64, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=[waveform_input, manual_input, physio_input], outputs=output)
```

#### 预期结果
- 多模态融合准确率 > 单模态（压力波形）
- 证明生理特征能提供额外的区分能力

---

## 📅 详细执行路线

### 第一步：挑战"不可能分类"任务（Week 1-2）

#### 任务描述
修改仿真代码，让"良好"和"正常"的压力波动范围完全重叠，但**波形的形状（频率微动）**不同。

#### 具体动作

**1. 修改数据生成脚本**
```python
# 文件：iteration1.2/generate_difficult_data.py

# 参数重叠度设置为 50%
p_amplitude_base = 20  # 所有类别的基准振幅
p_amplitude_range = 8   # 所有类别的振幅范围（完全重叠）

# 频率微动（区分关键）
frequency_base = 0.5
frequency_noise = 0.05  # 微小频率变化

# 不同类别的频率特征不同
frequency_modulation = {
    0: 0.0,    # 很差：无调制
    1: 0.02,   # 一般：低频调制
    2: 0.05,   # 正常：中频调制
    3: 0.08    # 良好：高频调制
}
```

**2. 生成"高难度"数据集**
- 1000人数据（每类250人）
- 50%以上的参数重叠
- 频率微动作为区分特征

**3. 测试随机森林性能**
- 使用原有的6个特征
- 预期准确率：85-90%

**4. 生成可视化对比**
- 对比旧数据集 vs 新数据集的参数分布
- 展示频率微动的波形差异

#### 成功标准
- ✅ 随机森林准确率 < 90%
- ✅ 传统统计特征（std、ptp）无法有效区分
- ✅ 波形频率特征成为区分关键

---

### 第二步：编写 1D-CNN 基准模型（Week 2-3）

#### 任务描述
建立一个3层卷积的深度网络，输入原始压力曲线，看看深度学习是否能在传统特征"失效"的地方把准确率救回来。

#### 具体动作

**1. 构建 1D-CNN 模型**
```python
# 文件：iteration1.2/models/cnn_baseline.py

model = Sequential([
    Conv1D(64, 7, activation='relu', input_shape=(1000, 2)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 3, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```

**2. 训练配置**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

**3. 对比实验**
- RandomForest（基线）：准确率 85-90%
- 1D-CNN（实验）：目标准确率 > 92%
- GRU（对比）：目标准确率 > 92%

**4. 可视化分析**
- 训练曲线（loss、accuracy）
- 混淆矩阵
- 特征可视化（CNN学到的滤波器）

#### 成功标准
- ✅ 1D-CNN 准确率 > RandomForest（> 92%）
- ✅ 证明深度学习能捕捉传统特征无法发现的微观特征
- ✅ 训练时间 < 10分钟（在普通PC上）

---

### 第三步：嵌入式部署预研（Week 3-4）

#### 任务描述
验证模型能否在按摩椅主控板上运行。

#### 具体动作

**1. 随机森林 → C 代码**
```bash
cd iteration1.2
python3 export_rf_to_c.py
```

**2. 深度学习 → TFLite**
```bash
cd iteration1.2
python3 export_cnn_to_tflite.py
```

**3. 内存和性能评估**
```python
# 评估脚本
evaluate_deployment.py
```

**4. 部署方案对比**
| 模型 | Flash | RAM | 推理时间 | 可行性 |
|------|-------|-----|---------|--------|
| RandomForest | ~50KB | ~10KB | ~10ms | ✅ 完全可行 |
| 1D-CNN | ~200KB | ~50KB | ~50ms | ✅ 可行（需优化） |
| GRU | ~300KB | ~80KB | ~100ms | ⚠️ 边缘可行 |

#### 成功标准
- ✅ 至少一种模型能在目标硬件上运行
- ✅ 推理时间 < 100ms
- ✅ 内存占用 < 硬件限制

---

### 第四步：多模态融合（Week 4-5）

#### 任务描述
引入高级生理特征，提升模型鲁棒性。

#### 具体动作

**1. 提取高级特征**
```python
# 文件：iteration1.2/extract_advanced_features.py

def extract_hrv_features(hr_signal):
    """提取心率变异性特征"""
    # RR间期
    rr_intervals = np.diff(hr_signal)

    # RMSSD
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))

    # SDNN
    sdnn = np.std(rr_intervals)

    # pNN50
    pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100

    return [rmssd, sdnn, pnn50]

def extract_spectral_features(pressure_signal):
    """提取频谱特征"""
    # FFT
    fft = np.fft.fft(pressure_signal)
    power_spectrum = np.abs(fft)**2

    # 主频
    dominant_freq = np.argmax(power_spectrum[:len(power_spectrum)//2])

    # 能量分布
    energy_low = np.sum(power_spectrum[:100])
    energy_high = np.sum(power_spectrum[100:500])

    return [dominant_freq, energy_low, energy_high]
```

**2. 构建多模态模型**
```python
# 文件：iteration1.2/models/multimodal_model.py

# 三个输入
waveform_input = Input(shape=(1000, 2), name='waveform')
manual_input = Input(shape=(6,), name='manual')
advanced_input = Input(shape=(10,), name='advanced')

# 特征提取分支
waveform_features = cnn_branch(wave_input)
manual_features = dense_branch(manual_input)
advanced_features = dense_branch(advanced_input)

# 融合
combined = concatenate([waveform_features, manual_features, advanced_features])
x = Dense(128, activation='relu')(combined)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=[waveform_input, manual_input, advanced_input], outputs=output)
```

**3. 对比实验**
- 单模态（压力波形）：准确率 ~92%
- 双模态（压力 + 手工特征）：准确率 ~94%
- 三模态（压力 + 手工 + 高级）：准确率 ~96%

#### 成功标准
- ✅ 多模态融合准确率 > 单模态（> 94%）
- ✅ 证明高级特征能提供额外信息
- ✅ 模型仍可部署（内存 < 硬件限制）

---

## 📊 预期成果

### 数据层面
- ✅ 生成"高难度"数据集（50%参数重叠）
- ✅ 验证传统特征在模糊边界下的局限性

### 模型层面
- ✅ 实现 1D-CNN 基准模型
- ✅ 实现多模态融合模型
- ✅ 验证深度学习的优势

### 部署层面
- ✅ 评估嵌入式部署可行性
- ✅ 对比不同模型的部署成本
- ✅ 提供部署方案建议

### 文档层面
- ✅ 完整的技术文档
- ✅ 代码注释和示例
- ✅ 部署指南

---

## 🎯 成功标准

### 必须达成
- [ ] 生成"高难度"数据集，随机森林准确率 < 90%
- [ ] 1D-CNN 在"高难度"数据集上准确率 > 92%
- [ ] 至少一种模型能在目标硬件上部署（推理时间 < 100ms）

### 期望达成
- [ ] 多模态融合准确率 > 94%
- [ ] 完成嵌入式部署预研
- [ ] 生成完整的对比报告

### 可选达成
- [ ] 尝试 GRU 模型并对比性能
- [ ] 实现在线学习功能
- [ ] 部署到真实硬件进行测试

---

## 📁 文件结构规划

```
iteration1.2/
├── README.md                       # 本文档
├── data/
│   ├── difficult/                  # "高难度"数据集
│   │   ├── 身体表征很差/           # 250人
│   │   ├── 身体表征一般/           # 250人
│   │   ├── 身体表征正常/           # 250人
│   │   └── 身体表征良好/           # 250人
│   └── comparison/                 # 旧数据集（对比用）
│
├── models/
│   ├── cnn_baseline.py             # 1D-CNN 基准模型
│   ├── multimodal.py               # 多模态融合模型
│   ├── gru_model.py                # GRU 模型（可选）
│   ├── rf_model.c                  # 随机森林 C 代码
│   ├── cnn_model.tflite            # CNN TFLite 模型
│   └── deployment_analysis.py      # 部署分析脚本
│
├── features/
│   ├── advanced_features.csv       # 高级特征
│   └── multimodal_features.csv     # 多模态特征
│
├── results/
│   ├── rf_baseline_results.csv     # RF 基线结果
│   ├── cnn_results.csv             # CNN 结果
│   ├── multimodal_results.csv      # 多模态结果
│   ├── deployment_report.json      # 部署报告
│   └── visualization/              # 可视化图表
│
└── scripts/
    ├── generate_difficult_data.py  # 生成高难度数据
    ├── train_cnn.py                # 训练 CNN
    ├── train_multimodal.py         # 训练多模态
    ├── export_rf_to_c.py           # 导出 RF 为 C
    ├── export_cnn_to_tflite.py     # 导出 CNN 为 TFLite
    └── compare_models.py           # 模型对比
```

---

## 🚀 下一步行动

### 立即开始
1. 创建 iteration1.2 目录结构
2. 修改数据生成脚本，创建"高难度"数据集
3. 测试随机森林在新数据集上的性能

### 本周完成
1. 构建 1D-CNN 基准模型
2. 在"高难度"数据集上训练 CNN
3. 对比 CNN vs RandomForest 性能

### 下周完成
1. 嵌入式部署预研
2. 多模态融合实验
3. 生成完整的对比报告

