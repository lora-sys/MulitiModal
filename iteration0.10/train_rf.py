import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
# 读取标准化后的数据
INPUT_FILE = "processed_features_normalized.pickle"
OUTPUT_DIR = "models"

print(f"正在读取 {INPUT_FILE}...")
df = pd.read_pickle(INPUT_FILE)
print(f"数据形状: {df.shape}")

# 第一步：定义 X 和 y
print("\n=== 第一步：特征选择 ===")
print(f"原始列: {df.columns.tolist()}")

# X: 输入特征（排除 global_id, label, category）
X_cols = [col for col in df.columns if col not in ['global_id', 'label', 'category']]
X = df[X_cols]
y = df['label']

print(f"\n特征集 X (共 {len(X_cols)} 个):")
print(X_cols)

print(f"\n目标集 y (共 {len(y)} 个):")
print(f"类别分布:\n{y.value_counts().sort_index()}")

# 第二步：划分数据集
print("\n=== 第二步：数据集划分 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)} 条 ({len(X_train)/len(X)*100:.1f}%)")
print(f"测试集: {len(X_test)} 条 ({len(X_test)/len(X)*100:.1f}%)")
print(f"训练集类别分布:\n{y_train.value_counts().sort_index()}")
print(f"测试集类别分布:\n{y_test.value_counts().sort_index()}")

# 第三步：随机森林分类
print("\n=== 第三步：随机森林分类 ===")

# 创建并训练模型
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("正在训练...")
rf_model.fit(X_train, y_train)

# 预测
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# 评估
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n=== 模型评估结果 ===")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"准确率提升: {test_acc - train_acc:+.4f}")

print(f"\n=== 测试集分类报告 ===")
print(classification_report(y_test, y_test_pred, target_names=['很差', '一般', '正常', '良好']))

print(f"\n=== 混淆矩阵 ===")
print(confusion_matrix(y_test, y_test_pred))

# 特征重要性分析
print("\n=== 特征重要性分析 ===")
feature_importance = pd.DataFrame({
    'feature': X_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# 保存模型
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, 'random_forest_model.pkl')
joblib.dump(rf_model, model_path)
print(f"\n✅ 模型已保存到: {model_path}")

# 保存特征列表（用于后续推理）
feature_path = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')
joblib.dump(X_cols, feature_path)
print(f"✅ 特征列表已保存到: {feature_path}")
