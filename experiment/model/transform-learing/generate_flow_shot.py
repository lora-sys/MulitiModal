import numpy as np
import neurokit2 as nk
import pandas as pd

def generate_50_real_samples():
    print("⏳ 正在模拟 50 条带有偏差的【真实数据】...")
    X_dynamic, X_static, Y_labels = [], [],[]
    t = np.linspace(0, 20, 1000)
    
    # 模拟 50 个人，随机分配 4 个等级
    for i in range(50):
        label = np.random.randint(0, 4)
        
        # 静态数据
        weight = np.random.randint(50, 90)
        hr_base = 70 + (3 - label) * 10 + np.random.randint(-5, 5)
        spo2_base = 98 - (3 - label) + np.random.randint(-1, 1)
        height = np.random.randint(155, 185)
        
        # ⚠️ 引入真实偏差 (Domain Shift)
        # 假设真实按摩椅的底座偏硬，所有人的基准压力都多了 10Pa
        p_offset = weight * 0.6 + 10.0  
        # 真实环境的噪声频率可能跟仿真不一样
        real_noise = np.random.normal(0, 5.0, 1000) + 3 * np.sin(2 * np.pi * 5.0 * t) 
        p_amplitude = 25 + label * 3.0
        
        p1 = p_offset + p_amplitude * np.sin(2 * np.pi * 0.5 * t) + real_noise
        p2 = (p_offset * 0.9) + p_amplitude * np.sin(2 * np.pi * 0.5 * t + 0.2) + real_noise
        
        # 走框架清洗
        p1_clean = nk.signal_filter(p1, sampling_rate=50, highcut=10, method='butterworth')
        p2_clean = nk.signal_filter(p2, sampling_rate=50, highcut=10, method='butterworth')
        p1_norm = (p1_clean - np.mean(p1_clean)) / (np.std(p1_clean) + 1e-6)
        p2_norm = (p2_clean - np.mean(p2_clean)) / (np.std(p2_clean) + 1e-6)
        
        X_dynamic.append(np.vstack([p1_norm, p2_norm]).astype(np.float32))
        X_static.append(np.array([weight/100, hr_base/120, spo2_base/100, height/200], dtype=np.float32))
        Y_labels.append(label)

    # 保存为单独的 npz
    save_path = "real_data_50_samples.npz"
    np.savez_compressed(save_path, dynamic=np.array(X_dynamic), static=np.array(X_static), labels=np.array(Y_labels))
    print(f"✅ 50条模拟真实数据已生成：{save_path}")

if __name__ == "__main__":
    generate_50_real_samples()