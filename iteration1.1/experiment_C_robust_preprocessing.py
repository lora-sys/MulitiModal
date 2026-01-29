"""
实验C：增强版预处理 + 精简特征 + 5-Fold交叉验证

目标：验证异常检测与修复算法的价值
验证点：鲁棒性（预处理后的性能提升）

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
from pathlib import Path

# 配置
INPUT_DATA_DIR = "./data/industrial"
OUTPUT_MODEL = "./models/model_C_robust.pkl"
OUTPUT_RESULTS = "./results/experiment_C_results.csv"
OUTPUT_LOG = "./results/experiment_C_log.json"

# 精简特征集（6个核心特征）
features_reduced = [
    'sensor1_std', 'sensor2_std', 'sensor1_ptp', 'sensor2_ptp', 'hr', 'spo2'
]

def extract_features_from_filename(filename):
    """从文件名提取基本信息"""
    name = filename.replace('.csv', '')
    parts = name.split('_')
    return [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]

def detect_and_repair_spikes(signal, window_size=15, threshold=5.0):
    """
    检测并修复跳点（iteration0.5的自愈算法）
    """
    signal_clean = signal.copy()

    # 计算滑动窗口的均值和标准差
    rolling_mean = np.convolve(signal_clean, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.zeros_like(signal_clean)

    for i in range(len(signal_clean)):
        start = max(0, i - window_size//2)
        end = min(len(signal_clean), i + window_size//2 + 1)
        rolling_std[i] = np.std(signal_clean[start:end])

    # 3-Sigma检测
    upper_bound = rolling_mean + 3 * rolling_std
    lower_bound = rolling_mean - 3 * rolling_std
    is_anomaly = (signal_clean > upper_bound) | (signal_clean < lower_bound)

    # 修复：挖坑 + 线性插值
    signal_clean[is_anomaly] = np.nan
    signal_clean = pd.Series(signal_clean).interpolate(method='linear').ffill().bfill().values

    # 最终平滑
    signal_clean = np.convolve(signal_clean, np.ones(window_size)/window_size, mode='same')

    return signal_clean, is_anomaly

def remove_drift(signal, window_size=100):
    """
    去除基线漂移
    """
    # 计算趋势
    trend = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    # 去趋势
    signal_detrended = signal - trend
    return signal_detrended

def calculate_waveform_features(p1, p2):
    """计算波形特征"""
    return {
        'sensor1_std': np.std(p1),
        'sensor2_std': np.std(p2),
        'sensor1_ptp': np.ptp(p1),
        'sensor2_ptp': np.ptp(p2)
    }

def process_file_with_preprocessing(csv_file, label):
    """处理单个文件，包含增强版预处理"""
    # 读取原始数据
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    p1 = df['压力传感器1'].values
    p2 = df['压力传感器2'].values

    # 从文件名提取生理特征
    id_fields = extract_features_from_filename(csv_file.name)

    # 增强版预处理
    # 1. 检测并修复跳点
    p1_repaired, spikes1 = detect_and_repair_spikes(p1)
    p2_repaired, spikes2 = detect_and_repair_spikes(p2)

    # 2. 去除漂移
    p1_detrended = remove_drift(p1_repaired)
    p2_detrended = remove_drift(p2_repaired)

    # 3. 计算特征
    waveform_features = calculate_waveform_features(p1_detrended, p2_detrended)

    # 构建记录
    record = {
        'global_id': id_fields[0],
        'label': label,
        'hr': id_fields[2],
        'spo2': id_fields[3],
        **waveform_features,
        'spikes_detected': np.sum(spikes1) + np.sum(spikes2)
    }

    return record

print("=" * 70)
print("实验C：增强版预处理 + 精简特征 + 5-Fold交叉验证")
print("=" * 70)

# 步骤1: 处理所有数据（包含预处理）
print("\n步骤1: 增强版预处理 + 特征提取...")
CATEGORIES = {"身体表征很差": 0, "身体表征一般": 1, "身体表征正常": 2, "身体表征良好": 3}

features_list = []
total_spikes = 0

for folder_name, label in CATEGORIES.items():
    folder_path = Path(INPUT_DATA_DIR) / folder_name
    csv_files = sorted(folder_path.glob('*.csv'))

    print(f"  处理 {folder_name}: {len(csv_files)} 个文件")

    for csv_file in csv_files:
        record = process_file_with_preprocessing(csv_file, label)
        features_list.append(record)
        total_spikes += record['spikes_detected']

df = pd.DataFrame(features_list)
print(f"✓ 数据处理完成: {df.shape[0]} 条 × {df.shape[1]} 列")
print(f"✓ 总共检测并修复跳点: {total_spikes} 个")

# 步骤2: 特征选择
print(f"\n步骤2: 特征选择")
X = df[features_reduced]
y = df['label']
print(f"✓ 精简特征数量: {len(features_reduced)}")

# 步骤3: 5折交叉验证
print(f"\n步骤3: 5-Fold交叉验证")
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
print(f"实验C结果：增强版预处理 + 精简特征（6个）")
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
print(f"  修复跳点数: {total_spikes}")

# 步骤4: 训练最终模型
print(f"\n步骤4: 训练最终模型（全部数据）")
model.fit(X, y)

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features_reduced,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 特征重要性:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

# 步骤5: 保存结果
print(f"\n步骤5: 保存结果")

import os
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
joblib.dump(model, OUTPUT_MODEL)
print(f"✓ 模型已保存: {OUTPUT_MODEL}")

results_df = pd.DataFrame({
    'fold': list(range(1, 6)),
    'accuracy': scores
})
results_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"✓ 结果已保存: {OUTPUT_RESULTS}")

log = {
    'experiment': 'C',
    'description': '增强版预处理 + 精简特征 + 5-Fold交叉验证',
    'feature_count': len(features_reduced),
    'features': features_reduced,
    'mean_accuracy': float(scores.mean()),
    'std_accuracy': float(scores.std()),
    'max_accuracy': float(scores.max()),
    'min_accuracy': float(scores.min()),
    'training_time': training_time,
    'total_spikes_repaired': int(total_spikes),
    'spikes_per_sample': float(total_spikes / len(df)),
    'fold_scores': [float(s) for s in scores],
    'feature_importance': feature_importance.to_dict('records')
}

os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print(f"✓ 日志已保存: {OUTPUT_LOG}")

print(f"\n{'='*70}")
print(f"✅ 实验C完成！")
print(f"{'='*70}")
print(f"\n核心发现:")
print(f"  • 预处理后准确率: {scores.mean():.4%}")
print(f"  • 稳定性(标准差): {scores.std():.4f}")
print(f"  • 训练时间: {training_time:.2f}秒")
print(f"  • 修复跳点数: {total_spikes} (平均 {total_spikes/len(df):.2f} 个/样本)")
print(f"  • 最重要特征: {feature_importance.iloc[0]['feature']}")
print(f"{'='*70}\n")