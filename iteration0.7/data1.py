import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def generate_causal_data():
    fs = 50
    duration = 100
    t = np.linspace(0, duration, duration * fs)
    # 压力信号：正弦波 + 两个脉冲
    pressure = 30 + 10 * np.sin(2 * np.pi * 0.1 * t)
    pressure[20*fs:30*fs] += 30
    pressure[60*fs:70*fs] += 35
    # 心率：压力的延迟响应 (3秒延迟，正相关)
    delay_samples = int(3.0 * fs)
    pressure_delayed = np.roll(pressure, delay_samples)
    pressure_delayed[-delay_samples:] = pressure[0]
    hr = 70 + 0.5 * pressure_delayed + np.random.normal(0, 0.3, len(t))
    # 下采样到1Hz，加抖动
    t_h = np.arange(0, duration, 1.0) + np.random.uniform(-0.1, 0.1, duration)
    hr_hz = hr[::fs]
    # 保存
    df_p = pd.DataFrame({'ts': t, 'pressure': pressure})
    df_h = pd.DataFrame({'ts': t_h, 'hr': hr_hz})
    df_p.to_csv("sensor_pressure_causal.csv", index=False)
    df_h.to_csv("sensor_heartrate_causal.csv", index=False)
    
    return t, pressure, t_h, hr_hz, hr

def align_and_analyze():
    # 1. 读取数据
    df_p = pd.read_csv("sensor_pressure_causal.csv").sort_values('ts')
    df_h = pd.read_csv("sensor_heartrate_causal.csv").sort_values('ts')
    # 2. merge_asof 对齐
    aligned = pd.merge_asof(df_p, df_h, on='ts', direction='backward')
    # 3. cubic插值
    aligned['hr'] = aligned['hr'].interpolate(method='cubic').bfill().ffill()
    # 4. 互相关分析
    p = aligned['pressure'].values
    h = aligned['hr'].values
    p_norm = (p - np.mean(p)) / np.std(p)
    h_norm = (h - np.mean(h)) / np.std(h)
    corr = signal.correlate(p_norm, h_norm, mode='full')
    lags = signal.correlation_lags(len(p_norm), len(h_norm))
    
    best_lag = lags[np.argmax(corr)]
    delay_sec = best_lag / 50.0
    
    print(f"检测到延迟: {delay_sec:.2f} 秒 (设定: 3.0秒)")
    return delay_sec, p, h, corr, lags

def visualize(t, pressure, hr, corr, lags, delay_sec):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1：压力信号
    axes[0].plot(t, pressure, 'b-', linewidth=0.8)
    axes[0].axvline(20, color='r', linestyle='--', alpha=0.5, label='Pulse 1')
    axes[0].axvline(60, color='r', linestyle='--', alpha=0.5, label='Pulse 2')
    axes[0].axvspan(20, 30, alpha=0.1, color='red')
    axes[0].axvspan(60, 70, alpha=0.1, color='red')
    axes[0].set_ylabel('Pressure')
    axes[0].set_title('Pressure Signal (50Hz) with Two Pulses', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 图2：心率信号
    axes[1].plot(t, hr, 'g-', linewidth=0.8)
    axes[1].axvline(23, color='r', linestyle='--', alpha=0.5, label='HR Response 1 (+3s)')
    axes[1].axvline(63, color='r', linestyle='--', alpha=0.5, label='HR Response 2 (+3s)')
    axes[1].axvspan(23, 33, alpha=0.1, color='green')
    axes[1].axvspan(63, 73, alpha=0.1, color='green')
    axes[1].set_ylabel('Heart Rate (bpm)')
    axes[1].set_title('Heart Rate Signal (Delayed Response to Pressure)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 图3：互相关结果
    lags_sec = lags / 50.0
    axes[2].plot(lags_sec, corr, 'purple', linewidth=0.8)
    axes[2].axvline(delay_sec, color='r', linestyle='--', linewidth=2, 
                    label=f'Detected Lag: {delay_sec:.2f}s')
    axes[2].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[2].scatter([delay_sec], [np.max(corr)], color='red', s=100, zorder=5)
    axes[2].set_xlabel('Lag (seconds)')
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Cross-Correlation Analysis', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].set_xlim(-20, 20)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("causal_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n可视化已保存: causal_visualization.png")

if __name__ == "__main__":
    t, pressure, t_h, hr_hz, hr_50hz = generate_causal_data()
    delay_sec, p, h, corr, lags = align_and_analyze()
    visualize(t, pressure, hr_50hz, corr, lags, delay_sec)