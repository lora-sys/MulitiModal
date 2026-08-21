# 当前项目状态总结

## 🎯 项目目标

**工业目标**: 多模态按摩椅推荐系统
- 输入: 用户生理状态 (ECG/EDA/EEG/TCM/Profile)
- 输出: 按摩方案 + 三档力度 (轻柔/舒适/强劲) 或 HOLD
- 约束: 实时推理、缺失模态可用、可解释

**当前阶段**: Demo 验证完成 → 待工业代码实现

---

## 🏗️ 模型设计

### 架构: DualGatingModel

```
输入模态:
  ├─ ECG + EDA: [B, 2, 1000] (动态信号)
  ├─ TCM诊断: [B, 4] → [舌/苔/脉/面] (静态)
  ├─ EEG: [B, 1, 1000] (可选,目前Demo未使用)
  └─ Profile: [B, 4] → [年龄/性别/BMI/HR]

编码:
  ├─ Dynamic Encoder (ResNet1D)
  │   ECG + EDA → 128D
  ├─ TCM Encoder (FT-Transformer, 冻结)
  │   [舌/苔/脉/面] → 9D (体质概率) + 128D (内部特征)
  └─ EEG Encoder (CNN)
      EEG → 8D (Demo中未接入主模型)

门控融合:
  ├─ Gate A: TCM概率 [9D] → σ(W_a) → 128D → 调制动态信号
  ├─ Gate B: 动态信号 [128D] → σ(W_b) → 128D → 调制静态特征
  └─ 拼接: [128D ⊕ 128D] → 256D

决策:
  └─ 回归头: 256D → 128D → 1D (放松度/力度值)
  └─ 方案推荐: 基于体质 + 力度 + 血氧
```

### 关键参数

```
总参数量: ~2.2M
├─ Dynamic Encoder: ~2.0M
├─ TCM Encoder: ~200K (冻结)
├─ Gate A: 1,280 (9→128)
├─ Gate B: 16,512 (128→128)
└─ Regressor: 33,025 (256→128→1)

可训练参数: ~50K (Gate A/B + Regressor)
冻结参数: ~200K (TCM)
```

---

## ⚠️ 当前模型的问题

### 1. **任务定义不清晰**

**现状**:
- 回归目标: "放松度" (0-1 连续值)
- 但业务需要: 按摩方案 + 三档力度 (分类/离散决策)

**问题**:
- 回归值如何映射到三档力度? (当前用简单阈值)
- 放松度的真实标签是什么? (用户反馈? 生理指标?)
- 没有明确的训练数据标注

### 2. **疲劳值缺失**

**现状**:
- 模型预测"放松度",但业务关心"疲劳度"
- 没有直接的疲劳评估模块

**竞品做法**:
- OSIM uDream: ECG → HR/HRV/呼吸 → stress 分数
- 荣泰: HR + SpO2 + 疲劳指数

**问题**:
- 当前模型无法输出"疲劳值"
- ECG/EDA/EEG 如何映射到疲劳/放松,缺乏理论支撑

### 3. **模态利用不完整**

**现状**:
- ECG + EDA → 128D ✅
- TCM诊断 → 9D + 128D ✅
- **EEG → 8D** ⚠️ (计算了但没有接入主模型)
- **Profile** ⚠️ (Demo中有但没有接入主模型)

**问题**:
- EEG 和 Profile 信息丢失
- 信息利用率低

### 4. **缺少评估基准**

**现状**:
- Demo 只能做推理测试,无法评估性能
- 没有验证集/测试集
- 没有 MAE/RMSE/R² 指标

**问题**:
- 不知道模型好不好
- 无法做消融实验
- 无法验证 Gate A/B 是否有效

---

## 📊 需要找的论文方向

### 方向 1: 生理信号 → 疲劳/压力评估

**关键词**:
- "PPG fatigue detection"
- "ECG stress assessment"
- "EDA relaxation measurement"
- "multi-modal fatigue recognition"
- "physiological signal mental workload"

**验证问题**:
- ECG/EDA/PPG 能否准确反映疲劳/放松?
- 哪种信号组合最有效?
- 有哪些成熟的算法?

### 方向 2: 门控融合机制

**关键词**:
- "gated fusion multimodal"
- "attention-based fusion physiological"
- "dynamic feature gating"
- "modality-specific gating"

**验证问题**:
- Gate A/B 的设计是否合理?
- 门控机制在生理信号融合中是否有效?

### 方向 3: 按摩椅/康复机器人决策

**关键词**:
- "massage chair recommendation system"
- "rehabilitation robot decision"
- "affective computing massage"
- "personalized wellness recommendation"

**验证问题**:
- 按摩椅行业的标准做法是什么?
- 决策系统的评估指标是什么?

---

## 🔍 评测计划

### Phase 1: 文献调研 (本周)

1. **搜索 PPG/ECG/EDA → 疲劳/压力** 的论文
2. **搜索门控融合** 的有效性研究
3. **搜索按摩椅/康复决策系统** 的工业实践

### Phase 2: 模型诊断 (下周)

1. **分析当前 checkpoint 的 Gate 权重**
   - 是否接近 0 (未学习)?
   - 是否有可解释的模式?

2. **延迟测试**
   - 推理时间是否满足实时性 (<100ms)?

3. **消融实验准备**
   - 准备 M0/M1/M2/M3/M6 的实现
   - 等待公司数据

### Phase 3: 架构优化 (2 周)

根据论文和评测结果决定:
- 是否添加疲劳评估模块?
- 是否接入 EEG/Profile?
- 是否简化 Gate 机制?
- 是否改变任务定义 (回归 → 分类)?

