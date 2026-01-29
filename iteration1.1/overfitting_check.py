"""
过拟合检查脚本

目标：通过6个检查项，验证模型是否真的学到了规律还是学到了模拟器的随机种子

作者：Iteration 1.1 团队
日期：2026-01-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import gaussian_kde
import joblib
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('./features/industrial_features_full.csv')

# 加载模型
model = joblib.load('./models/model_C_robust.pkl')

# 特征列表
features_reduced = ['sensor1_std', 'sensor2_std', 'sensor1_ptp', 'sensor2_ptp', 'hr', 'spo2']

X = df[features_reduced]
y = df['label']

print("=" * 80)
print("过拟合检查实验")
print("=" * 80)

results = {
    '检查1_PCA可视化': {'通过': False, '得分': 0, '详情': ''},
    '检查2_特征分布': {'通过': False, '得分': 0, '详情': ''},
    '检查3_边界样本': {'通过': False, '得分': 0, '详情': ''},
    '检查4_混淆分析': {'通过': False, '得分': 0, '详情': ''},
    '检查5_特征重要性': {'通过': False, '得分': 0, '详情': ''},
}

# =============================================================================
# 检查1: PCA降维散点图
# =============================================================================
print("\n检查1: PCA降维散点图...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 计算类别间距离
class_centers = []
for i in range(4):
    class_centers.append(np.mean(X_pca[y == i], axis=0))

# 计算最近邻距离
min_distance = float('inf')
for i in range(4):
    for j in range(i+1, 4):
        dist = np.linalg.norm(np.array(class_centers[i]) - np.array(class_centers[j]))
        min_distance = min(min_distance, dist)

# 计算类内标准差
intra_std = []
for i in range(4):
    intra_std.append(np.std(X_pca[y == i], axis=0).mean())

avg_intra_std = np.mean(intra_std)

# 评分：类间距离 / 类内标准差
separation_score = min_distance / avg_intra_std

# 绘制散点图
plt.figure(figsize=(10, 8))
colors = ['red', 'orange', 'green', 'blue']
labels = ['很差', '一般', '正常', '良好']

for i in range(4):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
               c=colors[i], label=labels[i], alpha=0.6, s=50)

# 绘制类别中心
for i in range(4):
    plt.scatter(class_centers[i][0], class_centers[i][1],
               c=colors[i], marker='x', s=200, linewidths=3)

plt.xlabel(f'主成分1 (方差贡献率: {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'主成分2 (方差贡献率: {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title(f'PCA降维散点图 - 类别分离度: {separation_score:.2f}', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./results/visualization/overfitting_01_pca_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 判断
results['检查1_PCA可视化']['得分'] = separation_score
if separation_score > 3.0:
    results['检查1_PCA可视化']['通过'] = True
    results['检查1_PCA可视化']['详情'] = f'类别分离度高 ({separation_score:.2f} > 3.0)'
else:
    results['检查1_PCA可视化']['详情'] = f'类别分离度低 ({separation_score:.2f} < 3.0)'

print(f"  类别分离度: {separation_score:.2f}")
print(f"  结果: {'✅ 通过' if results['检查1_PCA可视化']['通过'] else '❌ 未通过'}")

# =============================================================================
# 检查2: 特征分布直方图
# =============================================================================
print("\n检查2: 特征分布直方图...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

overlap_scores = []

for idx, feature in enumerate(features_reduced):
    ax = axes[idx]

    # 绘制4个类别的分布
    for i in range(4):
        data = df[df['label'] == i][feature]
        ax.hist(data, bins=30, alpha=0.5, label=labels[i], color=colors[i], density=True)

    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('密度', fontsize=10)
    ax.set_title(f'{feature} 分布', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 计算相邻类别的重叠度
    overlaps = []
    for i in range(3):
        data1 = df[df['label'] == i][feature].values
        data2 = df[df['label'] == i+1][feature].values

        # 计算KDE
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        x = np.linspace(min_val, max_val, 500)

        try:
            kde1 = gaussian_kde(data1)(x)
            kde2 = gaussian_kde(data2)(x)
            overlap = np.minimum(kde1, kde2).sum() / np.maximum(kde1, kde2).sum()
            overlaps.append(overlap)
        except:
            overlaps.append(0.5)

    avg_overlap = np.mean(overlaps) if overlaps else 0.5
    overlap_scores.append(avg_overlap)

plt.suptitle('核心特征在4个类别上的分布', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/visualization/overfitting_02_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 判断
avg_overlap = np.mean(overlap_scores)
results['检查2_特征分布']['得分'] = avg_overlap
if avg_overlap < 0.3:
    results['检查2_特征分布']['通过'] = True
    results['检查2_特征分布']['详情'] = f'分布重叠度低 ({avg_overlap*100:.1f}% < 30%)'
else:
    results['检查2_特征分布']['详情'] = f'分布重叠度高 ({avg_overlap*100:.1f}% > 30%)'

print(f"  平均重叠度: {avg_overlap*100:.1f}%")
print(f"  结果: {'✅ 通过' if results['检查2_特征分布']['通过'] else '❌ 未通过'}")

# =============================================================================
# 检查3: 边界样本分析
# =============================================================================
print("\n检查3: 边界样本分析...")

# 获取预测概率
y_proba = model.predict_proba(X)

# 找到边界样本（预测概率在0.4-0.6之间）
boundary_samples = []
boundary_details = []

for i in range(len(y)):
    prob = y_proba[i, y[i]]  # 正确类别的概率
    if 0.4 < prob < 0.6:
        boundary_samples.append(i)
        boundary_details.append({
            'index': i,
            'true_label': int(y[i]),
            'pred_prob': float(prob),
            'pred_label': int(y_proba[i].argmax())
        })

boundary_ratio = len(boundary_samples) / len(y) * 100

# 绘制预测概率分布
plt.figure(figsize=(10, 6))
plt.hist(y_proba.max(axis=1), bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(0.6, color='red', linestyle='--', linewidth=2, label='边界线 (0.6)')
plt.axvline(0.4, color='orange', linestyle='--', linewidth=2, label='边界线 (0.4)')
plt.xlabel('最大预测概率', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.title(f'预测概率分布 - 边界样本: {len(boundary_samples)}个 ({boundary_ratio:.1f}%)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./results/visualization/overfitting_03_prediction_probability.png', dpi=300, bbox_inches='tight')
plt.close()

# 判断
results['检查3_边界样本']['得分'] = boundary_ratio
if boundary_ratio < 5.0:
    results['检查3_边界样本']['通过'] = True
    results['检查3_边界样本']['详情'] = f'边界样本少 ({boundary_ratio:.1f}% < 5%)'
else:
    results['检查3_边界样本']['详情'] = f'边界样本多 ({boundary_ratio:.1f}% > 5%)'

print(f"  边界样本数量: {len(boundary_samples)} ({boundary_ratio:.1f}%)")
print(f"  结果: {'✅ 通过' if results['检查3_边界样本']['通过'] else '❌ 未通过'}")

# =============================================================================
# 检查4: 混淆分析
# =============================================================================
print("\n检查4: 混淆分析...")

# 进行5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=skf)

# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 计算混淆率（非对角线元素占比）
confusion_rate = (cm.sum() - np.trace(cm)) / cm.sum() * 100

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'混淆矩阵 - 混淆率: {confusion_rate:.2f}%', fontsize=14, fontweight='bold')
plt.colorbar()

tick_marks = np.arange(4)
plt.xticks(tick_marks, labels, fontsize=10)
plt.yticks(tick_marks, labels, fontsize=10)

# 添加数值标注
thresh = cm.max() / 2.
for i in range(4):
    for j in range(4):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold')

plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.savefig('./results/visualization/overfitting_04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 判断
results['检查4_混淆分析']['得分'] = confusion_rate
if confusion_rate < 1.0:
    results['检查4_混淆分析']['通过'] = True
    results['检查4_混淆分析']['详情'] = f'混淆率低 ({confusion_rate:.2f}% < 1%)'
else:
    results['检查4_混淆分析']['详情'] = f'混淆率高 ({confusion_rate:.2f}% > 1%)'

print(f"  混淆率: {confusion_rate:.2f}%")
print(f"  结果: {'✅ 通过' if results['检查4_混淆分析']['通过'] else '❌ 未通过'}")

# =============================================================================
# 检查5: 特征重要性分析
# =============================================================================
print("\n检查5: 特征重要性分析...")

# 获取特征重要性
feature_importance = dict(zip(features_reduced, model.feature_importances_))
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# 绘制特征重要性
plt.figure(figsize=(10, 6))
features_sorted = [item[0] for item in sorted_importance]
importances_sorted = [item[1] for item in sorted_importance]

bars = plt.barh(features_sorted, importances_sorted, color=plt.cm.viridis(np.linspace(0, 1, len(features_sorted))))

# 添加数值标签
for bar, imp in zip(bars, importances_sorted):
    plt.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
            f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')

plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.title('特征重要性排名', fontsize=14, fontweight='bold')
plt.xlim(0, max(importances_sorted) * 1.1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./results/visualization/overfitting_05_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 检查特征重要性是否有物理意义
# 核心特征应该与按摩体验相关
core_features = ['sensor1_std', 'sensor2_std', 'sensor1_ptp', 'sensor2_ptp']
importance_sum = sum(feature_importance[f] for f in core_features)

results['检查5_特征重要性']['得分'] = importance_sum
if importance_sum > 0.5:
    results['检查5_特征重要性']['通过'] = True
    results['检查5_特征重要性']['详情'] = f'核心特征占比高 ({importance_sum*100:.1f}% > 50%)'
else:
    results['检查5_特征重要性']['详情'] = f'核心特征占比低 ({importance_sum*100:.1f}% < 50%)'

print(f"  核心特征占比: {importance_sum*100:.1f}%")
print(f"  结果: {'✅ 通过' if results['检查5_特征重要性']['通过'] else '❌ 未通过'}")

# =============================================================================
# 综合评估
# =============================================================================
print("\n" + "=" * 80)
print("综合评估结果")
print("=" * 80)

passed_count = sum(1 for r in results.values() if r['通过'])
total_count = len(results)

print(f"\n检查项通过情况: {passed_count}/{total_count}")

for check_name, result in results.items():
    status = '✅ 通过' if result['通过'] else '❌ 未通过'
    print(f"  {check_name}: {status}")
    print(f"    详情: {result['详情']}")
    print(f"    得分: {result['得分']:.4f}")
    print()

# 最终判断
if passed_count >= 4:
    overall_result = "✅ 模型学到了真实规律，不存在过拟合"
    confidence = "高"
elif passed_count >= 2:
    overall_result = "⚠️ 需要进一步验证，存在潜在风险"
    confidence = "中"
else:
    overall_result = "❌ 可能存在过拟合，需要重新设计"
    confidence = "低"

print("=" * 80)
print(f"最终评估: {overall_result}")
print(f"置信度: {confidence}")
print("=" * 80)

# 保存结果
results['综合评估'] = {
    '通过数量': passed_count,
    '总数量': total_count,
    '通过率': passed_count / total_count,
    '最终判断': overall_result,
    '置信度': confidence
}

with open('./results/overfitting_check_report.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ 检查报告已保存: ./results/overfitting_check_report.json")
print(f"✓ 可视化图表已保存: ./results/visualization/")
print("=" * 80 + "\n")