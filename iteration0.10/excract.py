import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# 从 data.py 导入 CATEGORIES
from data import CATEGORIES

BASE_PATH = "./iteration0.10/massage_physical_data"
OUTPUT_FILE = "processed_features.pickle"

def extract_features_from_filename(filename):
    """
    从文件名提取特征: [global_id, weight, hr, spo2, height]
    文件名格式: 001_65_85_97_170.csv
    """
    # 去掉扩展名
    name = filename.replace('.csv', '')
    parts = name.split('_')
    
    # [global_id, weight, hr, spo2, height]
    return [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]

def calculate_waveform_features(p1, p2):
    """
    计算压力波形的特征
    """
    features = {}
    
    # Sensor 1 特征
    features['sensor1_mean'] = np.mean(p1)
    features['sensor1_std'] = np.std(p1)
    features['sensor1_ptp'] = np.ptp(p1)  # peak-to-peak (max - min)
    features['sensor1_min'] = np.min(p1)
    features['sensor1_max'] = np.max(p1)
    
    # Sensor 2 特征
    features['sensor2_mean'] = np.mean(p2)
    features['sensor2_std'] = np.std(p2)
    features['sensor2_ptp'] = np.ptp(p2)
    features['sensor2_min'] = np.min(p2)
    features['sensor2_max'] = np.max(p2)
    
    # 相对特征
    features['amplitude_ratio'] = features['sensor2_ptp'] / features['sensor1_ptp'] if features['sensor1_ptp'] > 0 else 0
    features['offset_ratio'] = features['sensor2_mean'] / features['sensor1_mean'] if features['sensor1_mean'] > 0 else 0
    
    return features

def process_all_data():
    """遍历所有CSV文件，提取特征并汇总"""
    
    features_list = []
    
    # 遍历每个类别文件夹
    for folder_name, label in CATEGORIES.items():
        folder_path = Path(BASE_PATH) / folder_name
        csv_files = sorted(folder_path.glob('*.csv'))  # 按文件名排序
        
        print(f"正在处理 {folder_name}: {len(csv_files)} 个文件")
        
        for csv_file in csv_files:
            # 1. 从文件名提取基本特征
            id_fields = extract_features_from_filename(csv_file.name)
            record = {
                'global_id': id_fields[0],
                'label': label,  # 类别标签
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
    
    # 转换为DataFrame并保存
    df_result = pd.DataFrame(features_list)
    df_result.to_pickle(OUTPUT_FILE)
    
    print(f"\n✅ 完成！共处理 {len(features_list)} 条数据")
    print(f"✅ 已保存到: {OUTPUT_FILE}")
    
    return df_result

if __name__ == "__main__":
    process_all_data()
