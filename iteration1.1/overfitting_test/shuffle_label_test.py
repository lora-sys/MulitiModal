"""
打乱标签测试

目的：验证模型是否真的学到了特征-标签关系，还是记住了数据规律

原理：
- 如果模型学到了真实规律，打乱标签后准确率应该接近随机（25%）
- 如果模型记住了规律，打乱标签后准确率仍然很高（>50%）

作者：Iteration 1.1 团队
日期：2026-01-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('../features/industrial_features_full.csv')

# 特征列表
features_reduced = ['sensor1_std', 'sensor2_std', 'sensor1_ptp', 'sensor2_ptp', 'hr', 'spo2']

X = df[features_reduced].values
y = df['label'].values

print("=" * 80)
print("打乱标签测试")
print("=" * 80)

# 测试1：原始数据（对照组）
print("\n测试1: 原始数据（未打乱标签）...")

model_original = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_original = cross_val_score(model_original, X, y, cv=skf)

print(f"  平均准确率: {scores_original.mean()*100:.2f}%")
print(f"  标准差: {scores_original.std()*100:.2f}%")
print(f"  各折得分: {[f'{s*100:.2f}%' for s in scores_original]}")

# 测试2：完全打乱标签
print("\n测试2: 完全打乱标签...")

y_shuffled = np.random.permutation(y)

model_shuffled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
scores_shuffled = cross_val_score(model_shuffled, X, y_shuffled, cv=skf)

print(f"  平均准确率: {scores_shuffled.mean()*100:.2f}%")
print(f"  标准差: {scores_shuffled.std()*100:.2f}%")
print(f"  各折得分: {[f'{s*100:.2f}%' for s in scores_shuffled]}")

# 测试3：部分打乱标签（测试鲁棒性）
print("\n测试3: 部分打乱标签（50%打乱）...")

y_partial_shuffled = y.copy()
shuffle_indices = np.random.choice(len(y), size=len(y)//2, replace=False)
y_partial_shuffled[shuffle_indices] = np.random.permutation(y[shuffle_indices])

model_partial = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
scores_partial = cross_val_score(model_partial, X, y_partial_shuffled, cv=skf)

print(f"  平均准确率: {scores_partial.mean()*100:.2f}%")
print(f"  标准差: {scores_partial.std()*100:.2f}%")
print(f"  各折得分: {[f'{s*100:.2f}%' for s in scores_partial]}")

# 测试4：多次随机打乱（统计显著性）
print("\n测试4: 多次随机打乱（10次，验证统计显著性）...")

random_scores = []
for i in range(10):
    y_random = np.random.permutation(y)
    model_random = RandomForestClassifier(n_estimators=100, random_state=42+i, n_jobs=-1)
    score = cross_val_score(model_random, X, y_random, cv=skf).mean()
    random_scores.append(score)
    print(f"  第{i+1}次打乱: {score*100:.2f}%")

print(f"\n  随机打乱平均准确率: {np.mean(random_scores)*100:.2f}%")
print(f"  随机打乱标准差: {np.std(random_scores)*100:.2f}%")

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 原始 vs 打乱对比
axes[0, 0].bar(['原始数据', '完全打乱', '50%打乱'],
             [scores_original.mean(), scores_shuffled.mean(), scores_partial.mean()],
             color=['green', 'red', 'orange'], alpha=0.7)
axes[0, 0].axhline(y=0.25, color='blue', linestyle='--', linewidth=2, label='随机基线 (25%)')
axes[0, 0].set_ylabel('准确率', fontsize=12)
axes[0, 0].set_title('标签打乱对准确率的影响', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (label, value) in enumerate(zip(['原始数据', '完全打乱', '50%打乱'],
                                       [scores_original.mean(), scores_shuffled.mean(), scores_partial.mean()])):
    axes[0, 0].text(i, value + 0.02, f'{value*100:.2f}%', ha='center', fontsize=11, fontweight='bold')

# 图2: 5折得分对比
x = np.arange(5)
width = 0.25

axes[0, 1].bar(x - width, scores_original, width, label='原始数据', color='green', alpha=0.7)
axes[0, 1].bar(x, scores_shuffled, width, label='完全打乱', color='red', alpha=0.7)
axes[0, 1].bar(x + width, scores_partial, width, label='50%打乱', color='orange', alpha=0.7)

axes[0, 1].axhline(y=0.25, color='blue', linestyle='--', linewidth=2, label='随机基线 (25%)')
axes[0, 1].set_xlabel('Fold', fontsize=12)
axes[0, 1].set_ylabel('准确率', fontsize=12)
axes[0, 1].set_title('5折交叉验证得分对比', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'Fold {i+1}' for i in range(5)])
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].grid(axis='y', alpha=0.3)

# 图3: 多次随机打乱分布
axes[1, 0].hist(random_scores, bins=10, alpha=0.7, color='red', edgecolor='black')
axes[1, 0].axvline(scores_original.mean(), color='green', linestyle='--', linewidth=2, label=f'原始数据 ({scores_original.mean()*100:.2f}%)')
axes[1, 0].axvline(0.25, color='blue', linestyle=':', linewidth=2, label='随机基线 (25%)')
axes[1, 0].axvline(np.mean(random_scores), color='red', linestyle='-', linewidth=2, label=f'随机打乱平均 ({np.mean(random_scores)*100:.2f}%)')
axes[1, 0].set_xlabel('准确率', fontsize=12)
axes[1, 0].set_ylabel('频次', fontsize=12)
axes[1, 0].set_title('10次随机打乱的准确率分布', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)

# 图4: 综合对比表
axes[1, 1].axis('off')

table_data = [
    ['测试项', '准确率', '标准差', '与随机基线差异'],
    ['原始数据', f'{scores_original.mean()*100:.2f}%', f'{scores_original.std()*100:.2f}%', f'+{(scores_original.mean()-0.25)*100:.2f}%'],
    ['完全打乱', f'{scores_shuffled.mean()*100:.2f}%', f'{scores_shuffled.std()*100:.2f}%', f'+{(scores_shuffled.mean()-0.25)*100:.2f}%'],
    ['50%打乱', f'{scores_partial.mean()*100:.2f}%', f'{scores_partial.std()*100:.2f}%', f'+{(scores_partial.mean()-0.25)*100:.2f}%'],
    ['随机基线', '25.00%', '-', '0.00%']
]

table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置表头样式
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置数据行样式
for i in range(1, 5):
    for j in range(4):
        if i == 1:  # 原始数据
            table[(i, j)].set_facecolor('#E8F5E9')
        elif i == 2:  # 完全打乱
            table[(i, j)].set_facecolor('#FFEBEE')
        elif i == 3:  # 50%打乱
            table[(i, j)].set_facecolor('#FFF3E0')
        elif i == 4:  # 随机基线
            table[(i, j)].set_facecolor('#E3F2FD')

axes[1, 1].set_title('测试结果汇总表', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('./shuffle_label_test_results.png', dpi=300, bbox_inches='tight')
plt.close()

# 判断结果
print("\n" + "=" * 80)
print("结果分析")
print("=" * 80)

# 判断模型是否过拟合
if scores_shuffled.mean() < 0.30:
    result = "✅ 模型学到了真实规律"
    reason = f"打乱标签后准确率降至 {scores_shuffled.mean()*100:.2f}%（接近随机25%），说明模型确实学到了特征-标签关系"
elif scores_shuffled.mean() < 0.40:
    result = "⚠️ 模型部分学到了规律"
    reason = f"打乱标签后准确率为 {scores_shuffled.mean()*100:.2f}%（略高于随机），可能存在轻微过拟合"
else:
    result = "❌ 模型可能存在过拟合"
    reason = f"打乱标签后准确率仍高达 {scores_shuffled.mean()*100:.2f}%（远高于随机），模型可能记住了数据规律"

print(f"\n最终判断: {result}")
print(f"原因: {reason}")

# 保存结果
results = {
    '原始数据': {
        '平均准确率': float(scores_original.mean()),
        '标准差': float(scores_original.std()),
        '各折得分': [float(s) for s in scores_original]
    },
    '完全打乱': {
        '平均准确率': float(scores_shuffled.mean()),
        '标准差': float(scores_shuffled.std()),
        '各折得分': [float(s) for s in scores_shuffled]
    },
    '50%打乱': {
        '平均准确率': float(scores_partial.mean()),
        '标准差': float(scores_partial.std()),
        '各折得分': [float(s) for s in scores_partial]
    },
    '随机基线': 0.25,
    '多次随机打乱': {
        '平均准确率': float(np.mean(random_scores)),
        '标准差': float(np.std(random_scores)),
        '10次得分': [float(s) for s in random_scores]
    },
    '最终判断': result,
    '原因': reason
}

with open('./shuffle_label_test_report.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ 测试报告已保存: ./shuffle_label_test_report.json")
print(f"✓ 可视化图表已保存: ./shuffle_label_test_results.png")
print("=" * 80 + "\n")
