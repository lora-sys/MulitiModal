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
    test_output_path='experiment/model/test_dataset.npz',
    n_samples_per_class=2000,  # 增加到 2000，总计 6000 样本
    test_ratio=0.15,  # 15% 作为独立测试集
    random_seed=42,  # 固定随机种子
    profile_csv='experiment/rawdata/train_data/train_data.csv',
    wave_npz_path='experiment/model/pretrain_10k.npz'
):
    """生成只使用真实数据的多模态数据集（无虚拟样本）
    
    改进：
    1. 增加样本数到 6000
    2. 固定随机种子确保可复现
    3. 创建独立测试集（15%）
    """
    
    # 固定随机种子
    np.random.seed(random_seed)
    
    print("=" * 60)
    print("生成真实数据集（改进版：6000样本 + 独立测试集）")
    print("=" * 60)
    print(f"随机种子: {random_seed}")
    print(f"每类样本: {n_samples_per_class}")
    print(f"测试集比例: {test_ratio*100:.0f}%")
    
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
    # 受限于标签3只有1870人，我们最大化使用数据
    max_samples_per_class = min(n_samples_per_class, 1870)  # 标签3最多1870人
    print(f"\n[4/6] ���样真实数据（标签1,2,3，每类最多 {max_samples_per_class} 人）...")
    print(f"      注意：标签3原始数据仅1870人，将全部使用")
    
    # 先收集所有数据
    all_dynamic = []
    all_static_basic = []
    all_static_scores = []
    all_constitution = []
    all_labels = []
    
    for label in [1, 2, 3]:  # 只处理标签1,2,3
        label_profiles = profiles[profiles['身体状态'] == label]
        available = len(label_profiles)
        
        # 标签3使用全部数据，其他标签采样相同数量
        to_sample = min(available, max_samples_per_class)
        print(f"      标签 {label}: 采样 {to_sample} 人 (原始: {available} 人)")
        
        # 采样真实用户
        sampled_profiles = label_profiles.sample(n=to_sample, replace=False, random_state=random_seed)
        
        for _, row in sampled_profiles.iterrows():
            all_static_basic.append([row['年龄'], row['BMI 数值'], row['血氧'], row['心率']])
            all_static_scores.append([row['健康指数'], row['诊断得分']])
            
            constitution_name = row['体质类型名称']
            all_constitution.append(CONSTITUTION_MAP.get(constitution_name, 0))
            all_labels.append(label)
            
            # 从同标签波形库随机抽取
            possible_indices = label_to_wave_idx.get(label, np.arange(len(waves)))
            chosen_idx = np.random.choice(possible_indices)
            all_dynamic.append(waves[chosen_idx])
    
    # 转换为数组
    dynamic_arr = np.array(all_dynamic, dtype=np.float32)
    static_basic_arr = np.array(all_static_basic, dtype=np.float32)
    static_scores_arr = np.array(all_static_scores, dtype=np.float32)
    constitution_arr = np.array(all_constitution, dtype=np.int64)
    labels_arr = np.array(all_labels, dtype=np.int64)
    
    print(f"      总采样: {len(labels_arr)} 人")
    
    # 划分训练集和独立测试集
    print(f"\n[划分] 创建独立测试集 ({test_ratio*100:.0f}%)...")
    
    # 为每类单独划分，保持平衡
    train_indices = []
    test_indices = []
    
    for label in [1, 2, 3]:
        label_indices = np.where(labels_arr == label)[0]
        np.random.shuffle(label_indices)  # 打乱
        
        n_test = int(len(label_indices) * test_ratio)
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    print(f"      训练集: {len(train_indices)} 人")
    print(f"      测试集: {len(test_indices)} 人")
    
    # [5/6] 计算归一化参数（仅使用训练集）
    print("\n[5/6] 计算归一化参数（仅使用训练集）...")
    
    # 提取训练集数据
    train_static_basic = static_basic_arr[train_indices]
    train_static_scores = static_scores_arr[train_indices]
    
    # 计算均值和标准差（仅训练集）
    age_mean, age_std = train_static_basic[:, 0].mean(), train_static_basic[:, 0].std()
    bmi_mean, bmi_std = train_static_basic[:, 1].mean(), train_static_basic[:, 1].std()
    hr_mean, hr_std = train_static_basic[:, 3].mean(), train_static_basic[:, 3].std()
    spo2_mean, spo2_std = train_static_basic[:, 2].mean(), train_static_basic[:, 2].std()
    
    print(f"      年龄: mean={age_mean:.1f}, std={age_std:.1f}")
    print(f"      BMI:  mean={bmi_mean:.1f}, std={bmi_std:.1f}")
    print(f"      心率: mean={hr_mean:.1f}, std={hr_std:.1f}")
    print(f"      血氧: mean={spo2_mean:.1f}, std={spo2_std:.1f}")
    
    # 归一化（应用全部数据）
    static_basic_arr[:, 0] = (static_basic_arr[:, 0] - age_mean) / (age_std + 1e-8)
    static_basic_arr[:, 1] = (static_basic_arr[:, 1] - bmi_mean) / (bmi_std + 1e-8)
    static_basic_arr[:, 2] = (static_basic_arr[:, 2] - spo2_mean) / (spo2_std + 1e-8)
    static_basic_arr[:, 3] = (static_basic_arr[:, 3] - hr_mean) / (hr_std + 1e-8)
    
    health_mean, health_std = train_static_scores[:, 0].mean(), train_static_scores[:, 0].std()
    diagnosis_mean, diagnosis_std = train_static_scores[:, 1].mean(), train_static_scores[:, 1].std()
    
    static_scores_arr[:, 0] = (static_scores_arr[:, 0] - health_mean) / (health_std + 1e-8)
    static_scores_arr[:, 1] = (static_scores_arr[:, 1] - diagnosis_mean) / (diagnosis_std + 1e-8)
    
    # [6/6] 保存训练集和测试集
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存训练集
    print(f"\n[保存训练集] 写入: {output_path}")
    np.savez_compressed(output_path,
        dynamic=dynamic_arr[train_indices],
        static_basic=static_basic_arr[train_indices],
        static_scores=static_scores_arr[train_indices],
        constitution=constitution_arr[train_indices],
        labels=labels_arr[train_indices],
    )
    
    # 保存独立测试集
    print(f"[保存测试集] 写入: {test_output_path}")
    np.savez_compressed(test_output_path,
        dynamic=dynamic_arr[test_indices],
        static_basic=static_basic_arr[test_indices],
        static_scores=static_scores_arr[test_indices],
        constitution=constitution_arr[test_indices],
        labels=labels_arr[test_indices],
    )
    
    print(f"\n✅ 生成完成!")
    print(f"\n📊 训练集:")
    print(f"   - 样本数: {len(train_indices)}")
    print(f"   - dynamic: {dynamic_arr[train_indices].shape}")
    for label in [1, 2, 3]:
        count = (labels_arr[train_indices] == label).sum()
        print(f"   - 标签 {label}: {count} 人")
    
    print(f"\n📊 测试集:")
    print(f"   - 样本数: {len(test_indices)}")
    for label in [1, 2, 3]:
        count = (labels_arr[test_indices] == label).sum()
        print(f"   - 标签 {label}: {count} 人")
    
    return {
        'train_dynamic': dynamic_arr[train_indices],
        'train_labels': labels_arr[train_indices],
        'test_dynamic': dynamic_arr[test_indices],
        'test_labels': labels_arr[test_indices]
    }


if __name__ == "__main__":
    # 生成改进版真实数据集
    # 最大化使用数据：每类 1870 人（受限于标签3），总计 5610 人
    # 15% 作为独立测试集
    result = generate_realonly_dataset(
        output_path='experiment/model/unified_dataset_realonly.npz',
        test_output_path='experiment/model/test_dataset.npz',
        n_samples_per_class=1870,  # 最大化使用标签3的数据
        test_ratio=0.15,
        random_seed=42
    )
    print(f'\n✅ 改进版数据集已生成!')