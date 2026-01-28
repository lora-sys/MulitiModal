import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 读取处理好的特征
INPUT_FILE = "processed_features.pickle"
OUTPUT_FILE = "processed_features_normalized.pickle"

print(f"正在读取 {INPUT_FILE}...")
df = pd.read_pickle(INPUT_FILE)
print(f"原始数据形状: {df.shape}")

# 显示原始数据范围
print("\n=== 原始数据范围 ===")
print(df[['weight', 'hr', 'spo2', 'height', 'sensor1_mean', 'sensor1_std', 'sensor1_ptp']].describe())

# 选择需要标准化的数值特征
# 排除：label (分类标签)、global_id (无意义ID)、category (文本)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['label', 'global_id', 'category']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\n需要标准化的特征: {numeric_cols}")

# 创建 MinMaxScaler
scaler = MinMaxScaler()

# 对数值特征进行标准化（保留 label 和 global_id 不变）
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 显示标准化后的数据范围
print("\n=== 标准化后数据范围 ===")
print(df_scaled[numeric_cols].describe())

# 保存标准化后的数据
df_scaled.to_pickle(OUTPUT_FILE)
print(f"\n✅ 已保存标准化数据到: {OUTPUT_FILE}")

# 保存 scaler 供后续使用
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("✅ 已保存 scaler 到: scaler.pkl")
