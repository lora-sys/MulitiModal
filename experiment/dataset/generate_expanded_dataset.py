"""
生成扩展版数据集
最大化利用静态数据，扩展到7862样本（训练集6683样本）
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


def generate_expanded_dataset(
    output_path='experiment/model/unified_dataset_expanded.npz',
    test_output_path='experiment/model/test_dataset_expanded.npz',
    samples_per_label=None,  # None表示最大化采样
    test_ratio=0.15,
    random_seed=42,
    profile_csv='experiment/rawdata/train_data/train_data.csv',
    enhanced_wave_npz_path='experiment/model/pretrain_10k_enhanced.npz'
):
    """
    生成扩展版数据集

    数据来源：
    - 波形：来自增强版数据集（包含9种噪声类型）
    - 静态特征：来自真实用户画像数据
    - 标签：来自真实用户画像数据
    """

    # 固定随机种子
    np.random.seed(random_seed)

    print("=" * 60)
    print("生成扩展版数据集（最大化利用静态数据）")
    print("=" * 60)
    print(f"随机种子: {random_seed}")
    print(f"测试集比例: {test_ratio*100:.0f}%")

    # [1/5] 加载用户画像
    print(f"\n[1/5] 加载用户画像: {profile_csv}")
    profiles = pd.read_csv(profile_csv)
    print(f"      原始用户数: {len(profiles)}")

    # [2/5] 截断异常值
    print("\n[2/5] 截断异常值...")
    profiles['年龄'] = profiles['年龄'].clip(18, 100)
    profiles['BMI 数值'] = profiles['BMI 数值'].clip(15, 40)
    profiles['心率'] = profiles['心率'].clip(55, 170)
    profiles['血氧'] = profiles['血氧'].clip(95, 100)
    profiles['健康指数'] = profiles['健康指数'].fillna(profiles['健康指数'].median())
    profiles['诊断得分'] = profiles['诊断得分'].fillna(profiles['诊断得分'].median())
    print("      已截断 & 填充完成")

    # [3/5] 加载增强版波形库
    print(f"\n[3/5] 加载增强版波形库: {enhanced_wave_npz_path}")
    enhanced_wave_data = np.load(enhanced_wave_npz_path)
    waves = enhanced_wave_data['dynamic']
    wave_labels = enhanced_wave_data['labels']

    print(f"      波形数: {len(waves)}")

    # 按标签分组波形
    label_to_wave_idx = {}
    for label in range(4):
        indices = np.where(wave_labels == label)[0]
        label_to_wave_idx[label] = indices
        print(f"      标签 {label}: {len(indices)} 条波形")

    # [4/5] 确定每标签采样数量
    print(f"\n[4/5] 计算采样策略...")

    if samples_per_label is None:
        # 最大化采样策略
        label_1_available = len(profiles[profiles['身体状态'] == 1])
        label_2_available = len(profiles[profiles['身体状态'] == 2])
        label_3_available = len(profiles[profiles['身体状态'] == 3])

        # 标签1和2采样相同数量（保持平衡），受限于较小的标签1
        samples_per_label = {
            1: min(label_1_available, label_2_available),  # 2996
            2: min(label_1_available, label_2_available),  # 2996
            3: label_3_available  # 1870
        }
    else:
        # 使用指定的采样数量
        if isinstance(samples_per_label, int):
            samples_per_label = {
                1: min(samples_per_label, len(profiles[profiles['身体状态'] == 1])),
                2: min(samples_per_label, len(profiles[profiles['身体状态'] == 2])),
                3: min(samples_per_label, len(profiles[profiles['身体状态'] == 3]))
            }

    total_samples = sum(samples_per_label.values())
    print(f"      标签1: 采样 {samples_per_label[1]} 人 (可用: {label_1_available} 人)")
    print(f"      标签2: 采样 {samples_per_label[2]} 人 (可用: {label_2_available} 人)")
    print(f"      标签3: 采样 {samples_per_label[3]} 人 (可用: {label_3_available} 人)")
    print(f"      总采样: {total_samples} 人")
    print(f"      预计训练集: {int(total_samples * (1 - test_ratio))} 人")

    # 采样
    all_dynamic = []
    all_static_basic = []
    all_static_scores = []
    all_constitution = []
    all_labels = []

    for label in [1, 2, 3]:
        label_profiles = profiles[profiles['身体状态'] == label]
        to_sample = samples_per_label[label]

        print(f"\n      采样标签 {label}: {to_sample} 人...")

        # 采样真实用户
        sampled_profiles = label_profiles.sample(n=to_sample, replace=False, random_state=random_seed)

        for _, row in sampled_profiles.iterrows():
            # 静态特征：年龄, BMI, 血氧, 心率
            all_static_basic.append([row['年龄'], row['BMI 数值'], row['血氧'], row['心率']])

            # 静态得分：健康指数, 诊断得分
            all_static_scores.append([row['健康指数'], row['诊断得分']])

            # 体质
            constitution_name = row['体质类型名称']
            all_constitution.append(CONSTITUTION_MAP.get(constitution_name, 0))

            # 标签
            all_labels.append(label)

            # 从同标签波形库随机抽取增强版波形
            possible_indices = label_to_wave_idx.get(label, np.arange(len(waves)))
            chosen_idx = np.random.choice(possible_indices)
            all_dynamic.append(waves[chosen_idx])

    # 转换为数组
    dynamic_arr = np.array(all_dynamic, dtype=np.float32)
    static_basic_arr = np.array(all_static_basic, dtype=np.float32)
    static_scores_arr = np.array(all_static_scores, dtype=np.float32)
    constitution_arr = np.array(all_constitution, dtype=np.int64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # 划分训练集和独立测试集
    print(f"\n[划分] 创建独立测试集 ({test_ratio*100:.0f}%)...")

    train_indices = []
    test_indices = []

    for label in [1, 2, 3]:
        label_indices = np.where(labels_arr == label)[0]
        np.random.shuffle(label_indices)

        n_test = int(len(label_indices) * test_ratio)
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    print(f"      训练集: {len(train_indices)} 人")
    print(f"      测试集: {len(test_indices)} 人")

    # [5/5] 计算归一化参数（仅使用训练集）
    print("\n[5/5] 计算归一化参数（仅使用训练集）...")

    train_static_basic = static_basic_arr[train_indices]
    train_static_scores = static_scores_arr[train_indices]

    # 计算均值和标准差
    age_mean, age_std = train_static_basic[:, 0].mean(), train_static_basic[:, 0].std()
    bmi_mean, bmi_std = train_static_basic[:, 1].mean(), train_static_basic[:, 1].std()
    hr_mean, hr_std = train_static_basic[:, 3].mean(), train_static_basic[:, 3].std()
    spo2_mean, spo2_std = train_static_basic[:, 2].mean(), train_static_basic[:, 2].std()

    print(f"      年龄: mean={age_mean:.1f}, std={age_std:.1f}")
    print(f"      BMI:  mean={bmi_mean:.1f}, std={bmi_std:.1f}")
    print(f"      心率: mean={hr_mean:.1f}, std={hr_std:.1f}")
    print(f"      血氧: mean={spo2_mean:.1f}, std={spo2_std:.1f}")

    # 归一化
    static_basic_arr[:, 0] = (static_basic_arr[:, 0] - age_mean) / (age_std + 1e-8)
    static_basic_arr[:, 1] = (static_basic_arr[:, 1] - bmi_mean) / (bmi_std + 1e-8)
    static_basic_arr[:, 2] = (static_basic_arr[:, 2] - spo2_mean) / (spo2_std + 1e-8)
    static_basic_arr[:, 3] = (static_basic_arr[:, 3] - hr_mean) / (hr_std + 1e-8)

    health_mean, health_std = train_static_scores[:, 0].mean(), train_static_scores[:, 0].std()
    diagnosis_mean, diagnosis_std = train_static_scores[:, 1].mean(), train_static_scores[:, 1].std()

    static_scores_arr[:, 0] = (static_scores_arr[:, 0] - health_mean) / (health_std + 1e-8)
    static_scores_arr[:, 1] = (static_scores_arr[:, 1] - diagnosis_mean) / (diagnosis_std + 1e-8)

    # 保存训练集和测试集
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[保存训练集] 写入: {output_path}")
    np.savez_compressed(output_path,
        dynamic=dynamic_arr[train_indices],
        static_basic=static_basic_arr[train_indices],
        static_scores=static_scores_arr[train_indices],
        constitution=constitution_arr[train_indices],
        labels=labels_arr[train_indices],
    )

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

    print(f"\n📈 相比之前的数据集提升:")
    print(f"   - 之前: 5840 样本 (训练集 ~4964)")
    print(f"   - 现在: {total_samples} 样本 (训练集 {len(train_indices)})")
    print(f"   - 提升: {(len(train_indices) - 4964) / 4964 * 100:.1f}%")

    return {
        'train_dynamic': dynamic_arr[train_indices],
        'train_labels': labels_arr[train_indices],
        'test_dynamic': dynamic_arr[test_indices],
        'test_labels': labels_arr[test_indices],
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'n_total': total_samples
    }


if __name__ == "__main__":
    # 生成扩展版数据集（最大化采样）
    result = generate_expanded_dataset(
        output_path='experiment/model/unified_dataset_expanded.npz',
        test_output_path='experiment/model/test_dataset_expanded.npz',
        samples_per_label=None,  # 最大化采样
        test_ratio=0.15,
        random_seed=42
    )
    print(f'\n✅ 扩展版数据集已生成!')
    print(f'训练集: {result["n_train"]} 样本')
    print(f'测试集: {result["n_test"]} 样本')
    print(f'总计: {result["n_total"]} 样本')
