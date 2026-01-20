import pandas as pd
import numpy as np

def generate_multimodla_raw():
    duration = 60 # seconds

    # Generate 压力信号 （50hz）这里的时间戳整齐
    t_p = np.linspace(0,duration,duration*50) # 0 到 60 秒，取3000个点
    pressure = 40 + 20 * np.sin(2*np.pi*0.2*t_p) 
    # 步长 = 60 / 3000 = 0.02 秒 （50hz）
    # 40 是基线压力，20 是振幅，0.2hz 是频率（5秒一个周期）
    df_p = pd.DataFrame({'ts': t_p, 'pressure': pressure}) # 理想压力
    
    # 心率 信号 （1hz）
    t_h = np.arange(0,duration,1.0) + np.random.uniform(-0.1,1,duration) # 0 到 60 秒，取60个点，并加入抖动
    hr = 70 + 5 * np.sin(2*np.pi*0.05*t_h)
    # 步长 = 1 秒 （1hz）
    # 70 是基线心率，10 是振幅，0.05hz 是频率（20秒一个周期）
    df_h = pd.DataFrame({'ts': t_h, 'hr': hr}) # 理想心率

    # | 信号 | 采样率 | 步长 | 时间戳特点 |
    # |------|--------|------|-----------|
    # | 压力 | 50Hz | 0.02s | 整齐（0, 0.02, 0.04...） |
    # | 心率 | 1Hz | 1s | 抖动（0.03, 1.07, 2.11...） |

    df_p.to_csv("sensor_pressure_50hz.csv", index=False)
    df_h.to_csv("sensor_heartrate_1hz.csv", index=False)   
    print("✅ 原始异构数据已生成：压力(50Hz) 和 心率(1Hz)")

generate_multimodla_raw()
