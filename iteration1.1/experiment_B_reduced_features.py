"""
实验B：精简特征 + 5-Fold交叉验证

目标：验证去掉"猪队友"特征后的性能提升
验证点：稳定性（去掉冗余特征后的表现）

作者：Iteration 1.1 团队
日期：2026-01-29
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import time

# 配置
INPUT_FILE = "./features/industrial_features_full.csv"
OUTPUT_MODEL = "./models/model_B_reduced.pkl"
OUTPUT_RESULTS = "./results/experiment_B_results.csv"
OUTPUT_LOG = "./results/experiment_B_log.json"

print("=" * 70)
print("实验B：精简特征 + 5-Fold交叉验证")
print("=" * 70)

# 加载数据
print("\n步骤1: 加载数据...")
df = pd.read_csv(INPUT_FILE)
print(f"✓ 数据加载成功: {df.shape[0]} 条 × {df.shape[1]} 列")

# 定义精简特征集（6个核心特征）
features_reduced = [
    'sensor1_std',   # 稳定性
    'sensor2_std',   # 稳定性
    'sensor1_ptp',   # 振动幅度
    'sensor2_ptp',   # 振动幅度
    'hr',            # 心率
    'spo2'           # 血氧
]

print(f"\n步骤2: 特征选择")
print(f"✓ 精简特征数量: {len(features_reduced)}")
print(f"特征列表: {features_reduced}")
print(f"\n特征选择理由:")
print(f"  • 去掉 'mean': 与体重强相关，但与舒适度无关")
print(f"  • 去掉 'weight', 'height': 静态参数，与动态感知无关")
print(f"  • 保留 'std', 'ptp': 动态特征，直接反映按摩体验")
print(f"  • 保留 'hr', 'spo2': 生理指标，反映身体状态")

# 准备数据
X = df[features_reduced]
y = df['label']

print(f"\n步骤3: 数据准备")
print(f"✓ 特征矩阵 X: {X.shape}")
print(f"✓ 标签向量 y: {y.shape}")

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
start_time = time.time()
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
training_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"实验B结果：精简特征（6个）")
print(f"{'='*70}")
print(f"\n📊 5次交叉验证得分:")
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\n📈 性能指标:")
print(f"  平均准确率: {scores.mean():.4%}")
print(f"  标准差: {scores.std():.4f}")
print(f"  最高准确率: {scores.max():.4%}")
print(f"  最低准确率: {scores.min():.4%}")
print(f"  训练时间: {training_time:.2f}秒")

# 在全部数据上训练最终模型
print(f"\n步骤5: 训练最终模型（全部数据）")
model.fit(X, y)

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features_reduced,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 特征重要性:")
for idx, row in feature_importance.iterrows():
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
    'experiment': 'B',
    'description': '精简特征 + 5-Fold交叉验证',
    'feature_count': len(features_reduced),
    'features': features_reduced,
    'mean_accuracy': float(scores.mean()),
    'std_accuracy': float(scores.std()),
    'max_accuracy': float(scores.max()),
    'min_accuracy': float(scores.min()),
    'training_time': training_time,
    'fold_scores': [float(s) for s in scores],
    'feature_importance': feature_importance.to_dict('records')
}

os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print(f"✓ 日志已保存: {OUTPUT_LOG}")

print(f"\n{'='*70}")
print(f"✅ 实验B完成！")
print(f"{'='*70}")
print(f"\n核心发现:")
print(f"  • 噪声环境下准确率: {scores.mean():.4%}")
print(f"  • 稳定性(标准差): {scores.std():.4f}")
print(f"  • 训练时间: {training_time:.2f}秒")
print(f"  • 最重要特征: {feature_importance.iloc[0]['feature']}")
print(f"  • 特征数量减少: 16 → 6 (↓{10/16*100:.1f}%)")
print(f"{'='*70}\n")