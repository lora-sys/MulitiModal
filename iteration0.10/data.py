import os
import numpy as np
import pandas as pd

# 基础设置
BASE_PATH = "./massage_physical_data"
CATEGORIES = {
    "身体表征很差": 0,
    "身体表征一般": 1,
    "身体表征正常": 2,
    "身体表征良好": 3
}
PEOPLE_PER_CAT = 50
FS = 50  # 50Hz
DURATION = 20 # 20秒

def generate_data_with_physics():
    if not os.path.exists(BASE_PATH): os.makedirs(BASE_PATH)
    np.random.seed(42) # 保证运行结果可以复现
    global_id = 1
    
    for folder_name, label in CATEGORIES.items():
        path = os.path.join(BASE_PATH, folder_name)
        if not os.path.exists(path): os.makedirs(path)
        
        print(f"正在模拟物理特性: {folder_name}...")
        
        for _ in range(PEOPLE_PER_CAT):
            # --- 1. 物理参数生成 (核心逻辑) ---
            height = np.random.randint(155, 190)
            # 体重与身高有一定正相关 (符合常识)
            weight = max(0,int(height - 105 + np.random.normal(0, 5)))
            
            # 生理指标逻辑偏移
            # 良好(3) -> 心率低, 血氧高 | 很差(0) -> 心率高, 血氧低
            hr_base = 70 + (3 - label) * 10 + np.random.randint(-5, 5)
            spo2_base = 98 - (3 - label) + np.random.randint(-1, 1)
            
            # --- 2. 压力波形物理建模 ---
            t = np.linspace(0, DURATION, DURATION * FS)
            
            # 物理定律 A: 压力基准与体重成正比 (F = mg)
            p_offset = weight * 0.6 
            
            # 物理定律 B: 身体表征好的人接受的按摩力度(振幅)通常更大
            p_amplitude = 15 + label * 8 
            
            # 物理定律 C: 身体表征差的人由于不适可能会有更多微动(噪声)
            p_noise_level = 5.0 - label * 1.2
            
            # 构造波形: 基准 + 机械往复运动 + 随机噪声
            p1 = p_offset + p_amplitude * np.sin(2 * np.pi * 0.5 * t) + \
                 np.random.normal(0, p_noise_level, len(t))
            p2 = (p_offset * 0.9) + p_amplitude * np.sin(2 * np.pi * 0.5 * t + 0.1) + \
                 np.random.normal(0, p_noise_level, len(t))
            
            # --- 3. 存储 ---
            filename = f"{global_id:03d}_{weight}_{hr_base}_{spo2_base}_{height}.csv"
            df = pd.DataFrame({'时间戳': t, '压力传感器1': p1, '压力传感器2': p2})
            df.to_csv(os.path.join(path, filename), index=False)
            
            global_id += 1

if __name__ == "__main__":
    generate_data_with_physics()
    print("\n✅ 物理模拟数据集生成完毕！")