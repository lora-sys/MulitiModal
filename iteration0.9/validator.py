import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 加载数据与时序补偿 (Phase 2.1) ---
def apply_temporal_compensation(df, lag_sec=3.06, fs=50):
    lag_samples = int(lag_sec * fs)
    # 使用 shift 将心率向前移动，消除延迟
    # 注意：shift(-N) 是把后面的数据往前挪
    df['hr_synced'] = df['hr_cubic'].shift(-lag_samples)
    
    # 丢弃末尾无法对齐的 NaN 行
    return df.dropna().reset_index(drop=True)

# --- 2. 交叉校验逻辑 (Phase 2.2) ---
def cross_modal_anomaly_check(df, fs=50):
    # 2.2.1 压力检测 (复用 3-Sigma)
    rolling = df['pressure'].rolling(window=fs*2, center=True)
    mu, std = rolling.mean(), rolling.std()
    df['p_spike'] = (df['pressure'] > mu + 3*std)
    
    # 2.2.2 心率响应检测 (看压力尖峰后 1s 内心率是否呈上升趋势)
    # 我们计算心率的一阶差分（即变化率）
    df['hr_trend'] = df['hr_synced'].diff().rolling(window=fs).mean()
    
    # 2.2.3 综合决策
    # 情况 A: 传感器噪声 (压力跳变但心率平稳)
    df['is_sensor_noise'] = df['p_spike'] & (df['hr_trend'].abs() < 0.05)
    
    # 情况 B: 生理应激 (压力跳变且心率上升)
    df['is_pain_response'] = df['p_spike'] & (df['hr_trend'] > 0.05)
    
    return df

# --- 3. 计算 CPI 耦合指标 (Phase 3.3) ---
def calculate_stress_index(df):
    scaler = MinMaxScaler()
    # 归一化压力和对齐后的心率
    df[['p_norm', 'h_norm']] = scaler.fit_transform(df[['pressure', 'hr_synced']])
    # CPI = 压力强度 * 心率反馈
    df['CPI'] = df['p_norm'] * df['h_norm']
    return df

aligned = pd.read_csv("sensor_aligned_filled.csv")
# --- 执行 Pipeline ---
# 假设 aligned 是你上一阶段生成的 DataFrame
df_step2 = apply_temporal_compensation(aligned)
df_step2 = cross_modal_anomaly_check(df_step2)
df_step2 = calculate_stress_index(df_step2)

# --- 4. 可视化报告 ---
plt.figure(figsize=(15, 8))

# 子图 1: 对齐后的双模态
plt.subplot(2, 1, 1)
plt.plot(df_step2['ts'], df_step2['p_norm'], label='Synced Pressure', alpha=0.6, color='blue')
plt.plot(df_step2['ts'], df_step2['h_norm'], label='Synced Heart Rate', alpha=0.6, color='green')
plt.scatter(df_step2[df_step2['is_pain_response']]['ts'], 
            df_step2[df_step2['is_pain_response']]['p_norm'], 
            color='red', marker='v', s=50, label='Pain Event', zorder=5)
plt.title("Phase 2: Temporal Alignment & Stress Detection")
plt.legend()

# 子图 2: 耦合后的 CPI 指标
plt.subplot(2, 1, 2)
plt.fill_between(df_step2['ts'], 0, df_step2['CPI'], color='red', alpha=0.3, label='CPI (Comfort Pressure Index)')
plt.ylabel("Stress Level")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.savefig("validator_visualization.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 交叉校验完成：")
print(f"检测到潜在传感器噪声点: {df_step2['is_sensor_noise'].sum()} 个")
print(f"检测到真实用户应激事件: {df_step2['is_pain_response'].sum()} 个")