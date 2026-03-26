# 模型解释性功能设计文档

**状态**：计划中（暂不实现）
**创建日期**：2026-03-26
**优先级**：中（根据项目需求调整）

---

## 📋 功能概述

提供模型可解释性功能，帮助理解模型的决策过程和特征重要性。

---

## 🎯 设计目标

1. **特征重要性分析**：量化各特征对预测结果的贡献
2. **预测结果解释**：解释单个预测结果的原因
3. **可视化展示**：直观展示解释性结果
4. **用户友好**：易于理解的解释方式

---

## 🔧 功能设计

### 1. 特征重要性分析

#### 1.1 置换重要性（Permutation Importance）

**原理**：
- 随机打乱某个特征的值
- 观察模型性能的变化
- 性能下降越多 = 特征越重要

**实现**：
```python
class FeatureImportanceAnalyzer:
    def compute_permutation_importance(self, data_loader, metric="accuracy", n_repeats=5):
        # 计算基准性能
        baseline_score = self._compute_score(data_loader, metric)

        # 计算每个特征的重要性
        for feature in features:
            # 置换特征
            permuted_score = self._compute_permutation_score(data_loader, feature, metric)
            importance = baseline_score - permuted_score
```

**输出**：
- 特征重要性分数
- 可视化图表（条形图）

#### 1.2 梯度重要性（Gradient Importance）

**原理**：
- 计算输出对输入的梯度
- 梯度越大 = 特征越重要

**适用场景**：
- 需要细粒度的特征重要性
- 深度学习模型

### 2. 预测结果解释

#### 2.1 LIME（Local Interpretable Model-agnostic Explanations）

**原理**：
- 在预测点附近生成扰动样本
- 用简单模型（如线性模型）拟合
- 解释简单模型的系数

**实现**：
```python
class LIMEExplainer:
    def explain(self, model, sample, num_samples=1000):
        # 生成扰动样本
        perturbed_samples = self._generate_perturbations(sample, num_samples)

        # 获取预测结果
        predictions = self._get_predictions(model, perturbed_samples)

        # 拟合简单模型
        simple_model = self._fit_simple_model(perturbed_samples, predictions)

        # 返回解释
        return simple_model.coef_
```

**输出**：
- 每个特征对预测的贡献
- 可视化解释（如权重条形图）

#### 2.2 SHAP（SHapley Additive exPlanations）

**原理**：
- 基于博弈论的特征重要性
- 分配每个特征的贡献

**优势**：
- 理论基础扎实
- 全局和局部一致性
- 支持各种模型

### 3. 可视化功能

#### 3.1 特征重要性可视化

- 条形图：显示各特征的重要性分数
- 热力图：显示特征与预测的关系
- 平行坐标图：显示多维特征关系

#### 3.2 预测解释可视化

- 权重条形图：显示每个特征的贡献
- 决策树：显示决策路径
- 力导向图：显示特征交互

---

## 📁 文件结构

```
experiment/
└── explainability/
    ├── __init__.py                 # 模块初始化
    ├── feature_importance.py       # 特征重要性分析
    ├── prediction_explainer.py     # 预测结果解释
    ├── visualization.py            # 可视化工具
    └── run_explainability.py       # 运行脚本
```

---

## 💻 使用示例

### 示例 1：计算特征重要性

```python
from experiment.explainability import compute_feature_importance, visualize_feature_importance

# 计算特征重要性
importance_scores = compute_feature_importance(
    model=trained_model,
    data_loader=val_loader,
    method="permutation",
    device=device
)

# 可视化
visualize_feature_importance(
    importance_scores,
    save_path="results/feature_importance.png"
)
```

### 示例 2：解释预测结果

```python
from experiment.explainability import explain_prediction

# 解释单个预测
explanation = explain_prediction(
    model=trained_model,
    sample=test_sample,
    method="lime"
)

print(explanation)
```

---

## 📊 预期输出

### 1. 特征重要性报告

```
特征重要性分析报告
========================

1. 动态波形 (Dynamic Waveform)
   重要性: 0.3245
   贡献: 32.45%

2. 身体特征 (Static Basic)
   重要性: 0.2891
   贡献: 28.91%

3. 舌面诊评分 (Static Scores)
   重要性: 0.1234
   贡献: 12.34%

4. 体质类型 (Constitution)
   重要性: 0.2630
   贡献: 26.30%
```

### 2. 预测解释示例

```
预测结果解释
============

预测类别: 良好
置信度: 0.95

主要影响因素:
1. 动态波形: +0.45 (正向贡献)
2. 身体特征: +0.30 (正向贡献)
3. 体质类型: +0.20 (正向贡献)
4. 舌面诊评分: +0.05 (轻微正向贡献)

决策路径:
→ 动态波形质量高 (+0.45)
→ 身体特征符合 (+0.30)
→ 体质类型适宜 (+0.20)
→ 舌面诊评分正常 (+0.05)
= 最终预测: 良好 (0.95)
```

---

## ⏱️ 实现估算

| 功能 | 难度 | 时间 | 优先级 |
|------|------|------|--------|
| 特征重要性（置换） | 中等 | 2-3小时 | 高 |
| 预测解释（LIME） | 较难 | 3-4小时 | 中 |
| 可视化工具 | 简单 | 1-2小时 | 中 |
| 完整集成 | 中等 | 1-2小时 | 低 |

**总计**：7-11小时

---

## 🚀 实现计划

### 阶段 1：特征重要性（2-3小时）
- 实现置换重要性算法
- 创建可视化工具
- 生成测试报告

### 阶段 2：预测解释（3-4小时）
- 实现LIME算法
- 集成SHAP（可选）
- 创建解释接口

### 阶段 3：可视化（1-2小时）
- 优化图表样式
- 添加交互功能
- 生成多种可视化

### 阶段 4：集成测试（1-2小时）
- 编写测试用例
- 性能优化
- 文档完善

---

## 📚 参考资料

### 学术论文
- "Why Should I Trust You?": Explaining the Predictions of Any Classifier
- SHAP: A Unified Approach to Interpreting Model Predictions
- Local Interpretable Model-Agnostic Explanations (LIME)

### 开源库
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
- Captum: https://github.com/pytorch/captum

---

## ❓ 何时实现

### 高优先级场景
- ✅ 发表学术论文（必需）
- ✅ 模型审查/审计
- ✅ 用户需要解释

### 中优先级场景
- ⚠️ 产品部署（V2功能）
- ⚠️ 模型优化
- ⚠️ 故障排除

### 低优先级场景
- ❌ 演示/PoC
- ❌ 早期原型
- ❌ 快速验证

---

## 🎯 当前状态

**状态**：计划中（暂不实现）

**原因**：
- 当前模型性能已足够优秀（98.88%准确率）
- 优先完成核心功能（部署、前端）
- 解释性可作为后续增强功能

**触发条件**：
- 需要发表学术论文
- 用户强烈要求解释功能
- 模型出现问题需要诊断

---

## 📝 备注

- 模型解释性功能已设计完成，代码框架已准备
- 可随时根据需求实现
- 预计需要7-11小时完成全部功能
- 优先级根据项目需求动态调整

---

**文档版本**: 1.0.0
**最后更新**: 2026-03-26
**负责人**: 待定