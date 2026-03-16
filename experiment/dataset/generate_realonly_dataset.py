"""
生成只使用真实数据的多模态数据集（无虚拟样本）
"""

import os
import numpy as np
import pandas as pd

# 体质映射表
CONSTITUTION_MAP = {
    '痰湿': 0, '气虚痰湿': 1, '平和': 2, '气虚': 3, '湿热': 4,
    '气虚湿热': 5, '痰瘀互结': 6, '血虚': 7, '阳虚': 8, '实热': 9,
    '气虚痰湿夹瘀': 10, '气郁': 11, '湿热血瘀': 12, '瘀热互结': 13,
    '阴虚': 14, '气血两虚': 15, '气滞血瘀': 16, '阴虚痰湿': 17,
    '气虚湿热夹瘀': 18, '血瘀': 19, '阳虚湿热': 20, '阳虚痰湿夹瘀': 21,
    '气虚血瘀': 22, '阳虚血瘀': 23, '阴虚湿热': 24, '阳虚湿热夹瘀': 25,
    '血虚血瘀': 26, '气血两虚夹瘀': 27, '阳虚痰湿': 28, '阴虚血瘀': 29,
    '阴虚湿热夹瘀': 30, '阴虚痰湿夹瘀': 31, '表寒里热': 32, '阴虚阳亢': 33,
    '特禀': 34, '阴虚阳亢夹瘀': 35, '表热里寒': 36, '表寒里热夹瘀': 37,
}

CONSTITUTION_ID_TO_NAME = {v: k for k, v in CONSTITUTION_MAP.items()}


def generate_realonly_dataset(
    output_path='experiment/model/unified_dataset_realonly.npz',
    n_samples_per_class=1250,
    profile_csv='experiment/rawdata/train_data/train_data.csv',
    wave_npz_path='experiment/model/pretrain_10k.npz'
):
    """生成只使用真实数据的多模态数据集（无虚拟样本）"""
    
    print("=" * 60)
    print("生成真实数据集（无虚拟样本）")
    print("=" * 60)
    
    # 动作1: 异常值截断配置
    print("\n[动作1] 异常值截断配置:")
    print("      年龄: 18-100 岁")
    print("      BMI:  15-40")
    print("      心率: 55-170 bpm (标准: 70-170)")
    print("      血氧: 95-100% (正常: 95-100%)")
    
    # [1/6] 加载用户画像
    print(f"\n[1/6] 加载用户画像: {profile_csv}")
    profiles = pd.read_csv(profile_csv)
    print(f"      原始用户数: {len(profiles)}")
    
    # [2/6] 截断异常值
    print("\n[2/6] 截断异常值...")
    
    # 截断年龄
    profiles['年龄'] = profiles['年龄'].clip(18, 100)
    
    # 截断BMI数值
    profiles['BMI 数值'] = profiles['BMI 数值'].clip(15, 40)
    
    # 截断心率
    profiles['心率'] = profiles['心率'].clip(55, 170)
    
    # 截断血氧
    profiles['血氧'] = profiles['血氧'].clip(95, 100)
    
    # 填充缺失值（用中位数）
    profiles['健康指数'] = profiles['健康指数'].fillna(profiles['健康指数'].median())
    profiles['诊断得分'] = profiles['诊断得分'].fillna(profiles['诊断得分'].median())
    
    print("      已截断 & 填充完成")
    
    # [3/6] 加载波形库
    print(f"\n[3/6] 加载波形库: {wave_npz_path}")
    wave_data = np.load(wave_npz_path)
    waves = wave_data['dynamic']
    wave_labels = wave_data['labels']
    
    print(f"      波形数: {len(waves)}")
    
    # 按标签分组波形
    label_to_wave_idx = {}
    for label in range(4):
        indices = np.where(wave_labels == label)[0]
        label_to_wave_idx[label] = indices
        print(f"      标签 {label}: {len(indices)} 条波形")
    
    # [4/6] 采样（只使用真实数据，标签1,2,3）
    print(f"\n[4/6] 采样真实数据（标签1,2,3，每类 {n_samples_per_class} 人）...")
    
    n_samples = n_samples_per_class * 3  # 3类
    dynamic_arr = np.zeros((n_samples, 2, 1000), dtype=np.float32)
    static_basic_arr = np.zeros((n_samples, 4), dtype=np.float32)
    static_scores_arr = np.zeros((n_samples, 2), dtype=np.float32)
    constitution_arr = np.zeros(n_samples, dtype=np.int64)
    labels_arr = np.zeros(n_samples, dtype=np.int64)
    
    idx = 0
    for label in [1, 2, 3]:  # 只处理标签1,2,3
        label_profiles = profiles[profiles['身体状态'] == label]
        available = len(label_profiles)
        
        to_sample = min(n_samples_per_class, available)
        print(f"      标签 {label}: 采样 {to_sample} 人 (原始: {available} 人)")
        
        # 采样真实用户
        sampled_profiles = label_profiles.sample(n=to_sample, replace=False)
        
        for _, row in sampled_profiles.iterrows():
            static_basic_arr[idx] = [row['年龄'], row['BMI 数值'], row['血氧'], row['心率']]
            static_scores_arr[idx] = [row['健康指数'], row['诊断得分']]
            
            constitution_name = row['体质类型名称']
            constitution_arr[idx] = CONSTITUTION_MAP.get(constitution_name, 0)
            labels_arr[idx] = label
            
            # 从同标签波形库随机抽取
            possible_indices = label_to_wave_idx.get(label, np.arange(len(waves)))
            chosen_idx = np.random.choice(possible_indices)
            dynamic_arr[idx] = waves[chosen_idx]
            
            idx += 1
    
    print(f"      总采样: {idx} 人")
    
    # 裁剪数组到实际采样数量
    dynamic_arr = dynamic_arr[:idx]
    static_basic_arr = static_basic_arr[:idx]
    static_scores_arr = static_scores_arr[:idx]
    constitution_arr = constitution_arr[:idx]
    labels_arr = labels_arr[:idx]
    
    print(f"      实际生成: {idx} 人")
    
    # [5/6] 计算归一化参数
    print("\n[5/6] 计算归一化参数...")
    
    # 计算均值和标准差
    age_mean, age_std = static_basic_arr[:, 0].mean(), static_basic_arr[:, 0].std()
    bmi_mean, bmi_std = static_basic_arr[:, 1].mean(), static_basic_arr[:, 1].std()
    hr_mean, hr_std = static_basic_arr[:, 3].mean(), static_basic_arr[:, 3].std()
    spo2_mean, spo2_std = static_basic_arr[:, 2].mean(), static_basic_arr[:, 2].std()
    
    print(f"      年龄: mean={age_mean:.1f}, std={age_std:.1f}")
    print(f"      BMI:  mean={bmi_mean:.1f}, std={bmi_std:.1f}")
    print(f"      心率: mean={hr_mean:.1f}, std={hr_std:.1f}")
    print(f"      血氧: mean={spo2_mean:.1f}, std={spo2_std:.1f}")
    
    # 归一化
    static_basic_arr[:, 0] = (static_basic_arr[:, 0] - age_mean) / (age_std + 1e-8)
    static_basic_arr[:, 1] = (static_basic_arr[:, 1] - bmi_mean) / (bmi_std + 1e-8)
    static_basic_arr[:, 2] = (static_basic_arr[:, 2] - spo2_mean) / (spo2_std + 1e-8)
    static_basic_arr[:, 3] = (static_basic_arr[:, 3] - hr_mean) / (hr_std + 1e-8)
    
    health_mean, health_std = static_scores_arr[:, 0].mean(), static_scores_arr[:, 0].std()
    diagnosis_mean, diagnosis_std = static_scores_arr[:, 1].mean(), static_scores_arr[:, 1].std()
    
    static_scores_arr[:, 0] = (static_scores_arr[:, 0] - health_mean) / (health_std + 1e-8)
    static_scores_arr[:, 1] = (static_scores_arr[:, 1] - diagnosis_mean) / (diagnosis_std + 1e-8)
    
    # [6/6] 保存
    print(f"\n[保存] 写入: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez_compressed(output_path,
        dynamic=dynamic_arr,
        static_basic=static_basic_arr,
        static_scores=static_scores_arr,
        constitution=constitution_arr,
        labels=labels_arr,
    )
    
    print(f"\n✅ 生成完成!")
    print(f"   - dynamic:      {dynamic_arr.shape}")
    print(f"   - static_basic: {static_basic_arr.shape}")
    print(f"   - static_scores:{static_scores_arr.shape}")
    print(f"   - constitution: {constitution_arr.shape}")
    print(f"   - labels:       {labels_arr.shape}")
    
    print(f"\n📊 标签分布:")
    for label in range(4):
        count = (labels_arr == label).sum()
        pct = count / len(labels_arr) * 100
        print(f"   标签 {label}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 体质分布:")
    unique, counts = np.unique(constitution_arr, return_counts=True)
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        name = CONSTITUTION_ID_TO_NAME.get(cid, f"ID:{cid}")
        print(f"   {name}: {cnt}")
    
    return {
        'dynamic': dynamic_arr,
        'static_basic': static_basic_arr,
        'static_scores': static_scores_arr,
        'constitution': constitution_arr,
        'labels': labels_arr
    }


if __name__ == "__main__":
    # 生成真实数据集（无虚拟样本）
    result = generate_realonly_dataset(
        output_path='experiment/model/unified_dataset_realonly.npz',
        n_samples_per_class=1250
    )
    print(f'\n✅ 真实数据集已生成!')
    print(f'样本数: {len(result["labels"])}')