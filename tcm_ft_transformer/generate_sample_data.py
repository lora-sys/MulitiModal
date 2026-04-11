"""
示例数据生成脚本
生成模拟的中医体质数据集用于测试
"""

import os
import numpy as np
import pandas as pd

from config import DATA_CONFIG, CONSTITUTION_NAMES


def generate_sample_dataset(n_samples=200000, output_path=None):
    """
    生成模拟的中医体质数据集
    
    数据格式:
    - 前 8 列: 特征 (Age, Gender, BMI, Heart Rate, SBP, DBP, SpO2, Temperature)
    - 后 9 列: 标签 (9种体质的原始整数分数)
    
    Args:
        n_samples: 样本数量
        output_path: 输出文件路径
    """
    if output_path is None:
        output_path = DATA_CONFIG["input_path"]
    
    print("=" * 60)
    print("生成模拟中医体质数据集")
    print("=" * 60)
    print(f"样本数量: {n_samples}")
    print(f"输出路径: {output_path}")
    print("=" * 60)
    
    np.random.seed(42)
    
    # =====================================================================
    # 生成特征 (8 维)
    # =====================================================================
    print("\n[1/2] 生成特征...")
    
    # Age: 18-100 岁
    age = np.random.uniform(18, 100, n_samples)
    
    # Gender: 0 (Male) 或 1 (Female)
    gender = np.random.randint(0, 2, n_samples)
    
    # BMI: 15-40
    bmi = np.random.uniform(15, 40, n_samples)
    
    # Heart Rate: 55-170 bpm
    heart_rate = np.random.uniform(55, 170, n_samples)
    
    # SBP (收缩压): 90-180 mmHg
    sbp = np.random.uniform(90, 180, n_samples)
    
    # DBP (舒张压): 60-120 mmHg
    dbp = np.random.uniform(60, 120, n_samples)
    
    # SpO2 (血氧): 95-100%
    spo2 = np.random.uniform(95, 100, n_samples)
    
    # Temperature: 36.0-37.5 °C
    temperature = np.random.uniform(36.0, 37.5, n_samples)
    
    # 组合特征矩阵
    features = np.column_stack([
        age, gender, bmi, heart_rate, sbp, dbp, spo2, temperature
    ])
    
    print(f"  特征矩阵形状: {features.shape}")
    print(f"  特征范围:")
    print(f"    Age: {age.min():.1f} - {age.max():.1f}")
    print(f"    Gender: {gender.min()} - {gender.max()}")
    print(f"    BMI: {bmi.min():.1f} - {bmi.max():.1f}")
    print(f"    Heart Rate: {heart_rate.min():.1f} - {heart_rate.max():.1f}")
    print(f"    SBP: {sbp.min():.1f} - {sbp.max():.1f}")
    print(f"    DBP: {dbp.min():.1f} - {dbp.max():.1f}")
    print(f"    SpO2: {spo2.min():.1f} - {spo2.max():.1f}")
    print(f"    Temperature: {temperature.min():.1f} - {temperature.max():.1f}")
    
    # =====================================================================
    # 生成标签 (9 维)
    # =====================================================================
    print("\n[2/2] 生成标签...")
    
    # 生成原始整数分数 (1-10)
    labels = np.random.randint(1, 11, size=(n_samples, 9))
    
    # 为每个样本选择一个主导体质
    dominant_constitutions = np.random.randint(0, 9, n_samples)
    
    # 增强主导体质的分数
    for i in range(n_samples):
        dominant = dominant_constitutions[i]
        labels[i, dominant] = np.random.randint(7, 11)  # 主导体质分数更高
        
        # 其他体质分数较低
        other_indices = [j for j in range(9) if j != dominant]
        labels[i, other_indices] = np.random.randint(1, 6, size=8)
    
    print(f"  标签矩阵形状: {labels.shape}")
    print(f"  标签范围: {labels.min()} - {labels.max()}")
    
    # 统计主导体质分布
    unique, counts = np.unique(dominant_constitutions, return_counts=True)
    print(f"\n  主导体质分布:")
    for cid, cnt in zip(unique, counts):
        print(f"    {CONSTITUTION_NAMES[cid]}: {cnt} ({cnt/n_samples*100:.1f}%)")
    
    # =====================================================================
    # 保存数据
    # =====================================================================
    print("\n[保存] 写入文件...")
    
    # 创建 DataFrame
    feature_names = ['Age', 'Gender', 'BMI', 'HeartRate', 'SBP', 'DBP', 'SpO2', 'Temperature']
    label_names = CONSTITUTION_NAMES
    
    df = pd.DataFrame(
        np.column_stack([features, labels]),
        columns=feature_names + label_names
    )
    
    # 确保 Gender 列为字符串（用于测试编码）
    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    
    # 保存为 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"  已保存: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    print("\n✅ 数据生成完成!")
    print(f"   - 总样本数: {n_samples}")
    print(f"   - 特征维度: 8")
    print(f"   - 标签维度: 9")
    print(f"   - 文件路径: {output_path}")
    
    return df


if __name__ == "__main__":
    # 生成 20 万条样本数据
    generate_sample_dataset(
        n_samples=200000,
        output_path=DATA_CONFIG["input_path"]
    )