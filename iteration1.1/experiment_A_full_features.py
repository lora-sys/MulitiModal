"""
实验A：全量特征 + 5-Fold交叉验证

目标：在噪声环境下，使用16个全量特征建立基线性能
验证点：真实性（噪声环境下的准确率）

作者：Iteration 1.1 团队
日期：2026-01-29
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# 配置
INPUT_FILE = "./features/industrial_features_full.csv"
OUTPUT_MODEL = "./models/model_A_full.pkl"
OUTPUT_RESULTS = "./results/experiment_A_results.csv"
OUTPUT_LOG = "./results/experiment_A_log.json"

print("=" * 70)
print("实验A：全量特征 + 5-Fold交叉验证")
print("=" * 70)

# 加载数据
print("\n步骤1: 加载数据...")
df = pd.read_csv(INPUT_FILE)
print(f"✓ 数据加载成功: {df.shape[0]} 条 × {df.shape[1]} 列")

# 定义特征集（16个全量特征）
features_full = [
    'weight', 'height', 'hr', 'spo2',
    'sensor1_mean', 'sensor1_std', 'sensor1_ptp', 'sensor1_min', 'sensor1_max',
    'sensor2_mean', 'sensor2_std', 'sensor2_ptp', 'sensor2_min', 'sensor2_max',
    'amplitude_ratio', 'offset_ratio'
]

print(f"\n步骤2: 特征选择")
print(f"✓ 全量特征数量: {len(features_full)}")
print(f"特征列表: {features_full}")

# 准备数据
X = df[features_full]
y = df['label']

print(f"\n步骤3: 数据准备")
print(f"✓ 特征矩阵 X: {X.shape}")
print(f"✓ 标签向量 y: {y.shape}")
print(f"类别分布:")
for i, cat in enumerate(['很差', '一般', '正常', '良好']):
    count = (y == i).sum()
    print(f"  {cat}: {count} ({count/len(y)*100:.1f}%)")

# 5折交叉验证
print(f"\n步骤4: 5-Fold交叉验证")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

print("正在训练和评估...")
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"\n{'='*70}")
print(f"实验A结果：全量特征（16个）")
print(f"{'='*70}")
print(f"\n📊 5次交叉验证得分:")
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\n📈 性能指标:")
print(f"  平均准确率: {scores.mean():.4%}")
print(f"  标准差: {scores.std():.4f}")
print(f"  最高准确率: {scores.max():.4%}")
print(f"  最低准确率: {scores.min():.4%}")

# 在全部数据上训练最终模型
print(f"\n步骤5: 训练最终模型（全部数据）")
model.fit(X, y)

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features_full,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 特征重要性 Top 10:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

# 保存结果
print(f"\n步骤6: 保存结果")

# 保存模型
import os
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
joblib.dump(model, OUTPUT_MODEL)
print(f"✓ 模型已保存: {OUTPUT_MODEL}")

# 保存详细结果
results_df = pd.DataFrame({
    'fold': list(range(1, 6)),
    'accuracy': scores
})
results_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"✓ 结果已保存: {OUTPUT_RESULTS}")

# 保存日志
log = {
    'experiment': 'A',
    'description': '全量特征 + 5-Fold交叉验证',
    'feature_count': len(features_full),
    'features': features_full,
    'mean_accuracy': float(scores.mean()),
    'std_accuracy': float(scores.std()),
    'max_accuracy': float(scores.max()),
    'min_accuracy': float(scores.min()),
    'fold_scores': [float(s) for s in scores],
    'feature_importance': feature_importance.to_dict('records')
}

os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print(f"✓ 日志已保存: {OUTPUT_LOG}")

print(f"\n{'='*70}")
print(f"✅ 实验A完成！")
print(f"{'='*70}")
print(f"\n核心发现:")
print(f"  • 噪声环境下准确率: {scores.mean():.4%}")
print(f"  • 稳定性(标准差): {scores.std():.4f}")
print(f"  • 最重要特征: {feature_importance.iloc[0]['feature']}")
print(f"{'='*70}\n")