import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 加载数据与时序补偿 (Phase 2.1) ---
def apply_temporal_compensation(df, lag_sec=3.06, fs=50, hr_col='hr_cubic'):
    lag_samples = int(lag_sec * fs)
    # 使用 shift 将心率向前移动，消除延迟
    # 注意：shift(-N) 是把后面的数据往前挪
    df['hr_synced'] = df[hr_col].shift(-lag_samples)

    # 丢弃末尾无法对齐的 NaN 行
    return df.dropna().reset_index(drop=True)

# --- 2. 交叉校验逻辑 (Phase 2.2) ---
def cross_modal_anomaly_check(df, fs=50):
    # 2.2.1 压力检测 - 使用绝对阈值
    df['p_spike'] = (df['pressure'] > 65)

    # 2.2.2 心率响应检测
    df['hr_trend'] = df['hr_synced'].diff().rolling(window=fs).mean()

    # 2.2.3 综合决策
    # 情况 A: 生理应激 (压力跳变且心率上升 = 疼痛/紧张)
    df['is_pain_response'] = df['p_spike'] & (df['hr_trend'] > 0.05)

    # 情况 B: 放松反应 (压力跳变且心率下降 = 正常放松)
    df['is_relaxation'] = df['p_spike'] & (df['hr_trend'] < -0.05)

    # 情况 C: 心率平稳 (压力跳变但心率变化小 = 可能是传感器噪声或正常波动)
    df['is_stable'] = df['p_spike'] & (df['hr_trend'].abs() <= 0.05)

    return df
    
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

aligned = pd.read_csv("temp.csv")
# 在读取数据时，检查列名
if 'hr_cubic' in aligned.columns:
    hr_col = 'hr_cubic'
elif 'hr' in aligned.columns:
    hr_col = 'hr'

# --- 执行 Pipeline ---
# 假设 aligned 是你上一阶段生成的 DataFrame
df_step2 = apply_temporal_compensation(aligned, hr_col=hr_col)
df_step2 = cross_modal_anomaly_check(df_step2)
df_step2 = calculate_stress_index(df_step2)

# --- 4. 可视化报告 ---
plt.figure(figsize=(15, 10))

# 子图 1: 对齐后的双模态
plt.subplot(3, 1, 1)
plt.plot(df_step2['ts'], df_step2['p_norm'], label='Synced Pressure', alpha=0.6, color='blue')
plt.plot(df_step2['ts'], df_step2['h_norm'], label='Synced Heart Rate', alpha=0.6, color='green')
plt.scatter(df_step2[df_step2['is_pain_response']]['ts'], 
            df_step2[df_step2['is_pain_response']]['p_norm'], 
            color='red', marker='v', s=50, label='Pain Event (Stress)', zorder=5)
plt.scatter(df_step2[df_step2['is_relaxation']]['ts'], 
            df_step2[df_step2['is_relaxation']]['p_norm'], 
            color='green', marker='^', s=30, label='Relaxation Response', zorder=5)
plt.title("Phase 2: Temporal Alignment & Cross-Modal Validation")
plt.legend(loc='upper right')
plt.ylabel("Normalized Value")
plt.grid(True, alpha=0.3)

# 子图 2: 心率趋势
plt.subplot(3, 1, 2)
plt.plot(df_step2['ts'], df_step2['hr_trend'], color='purple', alpha=0.7, label='Heart Rate Trend')
plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Stress Threshold (+0.05)')
plt.axhline(y=-0.05, color='green', linestyle='--', alpha=0.5, label='Relaxation Threshold (-0.05)')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.title("Heart Rate Trend (Change Rate)")
plt.xlabel("Time (s)")
plt.ylabel("HR Trend (bpm/frame)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 子图 3: 耦合后的 CPI 指标
plt.subplot(3, 1, 3)
plt.fill_between(df_step2['ts'], 0, df_step2['CPI'], color='red', alpha=0.3, label='CPI (Stress Index)')
plt.ylabel("Stress Level")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("validator_visualization1.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 交叉校验完成：")
print(f"  检测到应激事件 (压力↑ + 心率↑): {df_step2['is_pain_response'].sum()} 个")
print(f"  检测到放松反应 (压力↑ + 心率↓): {df_step2['is_relaxation'].sum()} 个")
print(f"  心率平稳 (无法判断): {df_step2['is_stable'].sum()} 个")