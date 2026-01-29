"""
工业级数据特征提取器

目标：从1000人工业级噪声数据中提取16个全量特征
复用 iteration0.10 的特征提取逻辑

作者：Iteration 1.1 团队
日期：2026-01-29
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# 配置
BASE_PATH = "./data/industrial"
OUTPUT_FILE = "./features/industrial_features_full.csv"

CATEGORIES = {
    "身体表征很差": 0,
    "身体表征一般": 1,
    "身体表征正常": 2,
    "身体表征良好": 3
}

def extract_features_from_filename(filename):
    """
    从文件名提取特征: [global_id, weight, hr, spo2, height]
    文件名格式: 001_70_100_96_170.csv
    """
    name = filename.replace('.csv', '')
    parts = name.split('_')
    return [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]

def calculate_waveform_features(p1, p2):
    """
    计算压力波形的特征（16个特征）
    """
    features = {}

    # Sensor 1 特征（5个）
    features['sensor1_mean'] = np.mean(p1)
    features['sensor1_std'] = np.std(p1)
    features['sensor1_ptp'] = np.ptp(p1)  # peak-to-peak
    features['sensor1_min'] = np.min(p1)
    features['sensor1_max'] = np.max(p1)

    # Sensor 2 特征（5个）
    features['sensor2_mean'] = np.mean(p2)
    features['sensor2_std'] = np.std(p2)
    features['sensor2_ptp'] = np.ptp(p2)
    features['sensor2_min'] = np.min(p2)
    features['sensor2_max'] = np.max(p2)

    # 相对特征（2个）
    features['amplitude_ratio'] = features['sensor2_ptp'] / features['sensor1_ptp'] if features['sensor1_ptp'] > 0 else 0
    features['offset_ratio'] = features['sensor2_mean'] / features['sensor1_mean'] if features['sensor1_mean'] > 0 else 0

    return features

def process_all_data():
    """遍历所有CSV文件，提取特征并汇总"""

    features_list = []

    print("=" * 60)
    print("开始提取工业级数据特征")
    print("=" * 60)

    # 遍历每个类别文件夹
    for folder_name, label in CATEGORIES.items():
        folder_path = Path(BASE_PATH) / folder_name
        csv_files = sorted(folder_path.glob('*.csv'))

        print(f"\n正在处理 {folder_name}: {len(csv_files)} 个文件")

        for csv_file in csv_files:
            # 1. 从文件名提取基本特征
            id_fields = extract_features_from_filename(csv_file.name)
            record = {
                'global_id': id_fields[0],
                'label': label,
                'weight': id_fields[1],
                'hr': id_fields[2],
                'spo2': id_fields[3],
                'height': id_fields[4],
                'category': folder_name
            }

            # 2. 读取CSV并计算波形特征
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            p1 = df['压力传感器1'].values
            p2 = df['压力传感器2'].values

            waveform_features = calculate_waveform_features(p1, p2)
            record.update(waveform_features)

            features_list.append(record)

        print(f"  ✓ 完成 {folder_name} ({len(csv_files)} 个文件)")

    # 转换为DataFrame
    df_result = pd.DataFrame(features_list)

    # 确保列名正确
    print(f"\n生成特征列: {df_result.columns.tolist()}")

    # 保存为CSV（方便后续实验读取）
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_result.to_csv(OUTPUT_FILE, index=False)

    # 重新读取验证
    df_verify = pd.read_csv(OUTPUT_FILE)
    print(f"\n验证CSV读取成功，列数: {len(df_verify.columns)}")

    print("\n" + "=" * 60)
    print("✅ 特征提取完成！")
    print("=" * 60)
    print(f"总计处理: {len(features_list)} 条数据")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"特征数量: {len(df_result.columns)} 个")
    print(f"\n特征列表:")
    for i, col in enumerate(df_result.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\n类别分布:")
    print(f"  身体表征很差: {len(df_result[df_result['label']==0])}")
    print(f"  身体表征一般: {len(df_result[df_result['label']==1])}")
    print(f"  身体表征正常: {len(df_result[df_result['label']==2])}")
    print(f"  身体表征良好: {len(df_result[df_result['label']==3])}")

    print("\n" + "=" * 60)

    return df_result

if __name__ == "__main__":
    process_all_data()
