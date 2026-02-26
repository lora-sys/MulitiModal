import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
import pandas as pd
import numpy as np
from self_healing_processor import SelfHealingPreprocessor



def generate_hardware_dirty_data(filename="dirty_real_data.csv"):
    duration = 20
    fs = 50
    ideal_points = duration * fs
    
    # 1. 理想时间轴
    t_ideal = np.linspace(0, duration, ideal_points)
    p1 = 40 + 20 * np.sin(2 * np.pi * 0.5 * t_ideal)
    p2 = 35 + 20 * np.sin(2 * np.pi * 0.5 * t_ideal + 0.2)
    
    df = pd.DataFrame({'时间戳': t_ideal, '压力传感器1': p1, '压力传感器2': p2})
    
    # 2. 注入“时间戳抖动” (模拟硬件时钟不准)
    jitter = np.random.normal(0, 0.005, ideal_points)
    df['时间戳'] = df['时间戳'] + jitter
    df = df.sort_values('时间戳').reset_index(drop=True)
    
    # 3. 注入“随机丢包” (删除 10% 的行)
    drop_indices = np.random.choice(df.index, size=int(ideal_points * 0.1), replace=False)
    df = df.drop(drop_indices)
    
    # 4. 注入“蓝牙断连” (中间硬生生挖掉 1 秒钟的数据)
    # 删除第 10 秒到 11 秒的数据
    df = df[~((df['时间戳'] > 10.0) & (df['时间戳'] < 11.0))]
    
    df.to_csv(filename, index=False)
    print(f"⚠️ 恶劣硬件数据已生成：{filename}")
    print(f"预期长度 1000，实际长度 {len(df)}。包含时间抖动、丢包、和 1 秒的断连！")

if __name__ == "__main__":
    generate_hardware_dirty_data("dirty_data.csv")
    
    df =pd.read_csv("dirty_data.csv")
    print(f"CSV 加载完成: {len(df)} 行")
    
    from interfaces import Sample
    sample  = Sample(
        sample_id="dirty_data",
        raw_data=df,
        metadata={
            "label" : 0,
            "static" : {"weight": 70, "hr": 75, "spo2": 97, "height": 170},
        }
    )
    import yaml
    config_path = "experiment/dataset/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    
    preprocessor = SelfHealingPreprocessor(dataset_config)
    processed = preprocessor.process(sample)
    
    
    print(f"动态特征维度: {processed.dynamic.shape}")  # 期望: (2, 1000)
    print(f"静态特征维度: {processed.static.shape}")    # 期望: (4,)
    print(f"标签: {processed.label}")                  # 期望: tensor(0)
    assert processed.dynamic.shape == (2, 1000), f"维度错误: {processed.dynamic.shape}"
    print("✅ 测试通过！无论 CSV 多么破，输出始终是 (2, 1000)")
