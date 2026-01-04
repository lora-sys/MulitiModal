from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def calculate_metrics(clean, noisy):
    return np.mean((clean - noisy) ** 2)

def load_signal_from_csv(filename="pressure_sim.csv"):
    if not os.path.exists(filename): return None
    df = pd.read_csv(filename)
    return df

# 1. 加载并投毒
df = load_signal_from_csv("pressure_sim.csv")
if df is not None:
    # 投毒 10 个脉冲
    np.random.seed(42)
    poison_indices = np.random.choice(df.index[50:-50], size=10, replace=False)
    for idx in poison_indices:
        df.loc[idx, 'noisy_signal'] += np.random.uniform(20, 50) * np.random.choice([-1, 1])

    # 2. 核心算法：检测
    window_size = 15
    # 计算受污染的滑动平均 (作为直接滤波的对比组)
    df['filter_ma'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['rolling_std'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).std()

    # 3-Sigma 边界
    df['upper_bound'] = df['filter_ma'] + 3 * df['rolling_std']
    df['lower_bound'] = df['filter_ma'] - 3 * df['rolling_std']
    df['is_anomaly'] = (df['noisy_signal'] > df['upper_bound']) | (df['noisy_signal'] < df['lower_bound'])

    # --- 真正的专业修复策略 ---
    # 1. 识别异常并“挖坑”
    df['signal_clean'] = df['noisy_signal'].copy()
    df.loc[df['is_anomaly'], 'signal_clean'] = np.nan

    # 2. 线性插值 (Interpolation)
    df['signal_interpolated'] = df['signal_clean'].interpolate(method='linear')
    df['signal_interpolated'] = df['signal_interpolated'].ffill().bfill()

    # 3. 最终平滑 (Final Smoothing)
    df['final_recovered'] = df['signal_interpolated'].rolling(window=window_size, center=True, min_periods=1).mean()

    # 4. 跑分评估 (动态计算)
    mse_raw = calculate_metrics(df['clean_signal'], df['noisy_signal'])
    mse_ma = calculate_metrics(df['clean_signal'], df['filter_ma'])
    mse_final = calculate_metrics(df['clean_signal'], df['final_recovered'])

    improvement = (mse_ma - mse_final) / mse_ma * 100

    # --- 终极可视化 ---
    plt.figure(figsize=(16, 12))
    view_slice = slice(200, 700)

    # 子图1：异常点检测
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'][view_slice], df['noisy_signal'][view_slice], color='red', alpha=0.2, label="Poisoned Data")
    plt.fill_between(df['timestamp'][view_slice], df['lower_bound'][view_slice], df['upper_bound'][view_slice], color='gray', alpha=0.2, label="3-Sigma Zone")
    anomalies = df[view_slice][df[view_slice]['is_anomaly']]
    plt.scatter(anomalies['timestamp'], anomalies['noisy_signal'], color='darkred', marker='x', s=80, label="Detected Anomalies")
    plt.title(f"Step 1: Detection (Identified {len(df[df['is_anomaly']])} points)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # 子图2：插值与最终恢复
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'][view_slice], df['clean_signal'][view_slice], color='black', lw=2.5, label="Ground Truth")
    plt.plot(df['timestamp'][view_slice], df['signal_interpolated'][view_slice], color='orange', alpha=0.4, linestyle='--', label="Interpolated Bridge")
    plt.plot(df['timestamp'][view_slice], df['final_recovered'][view_slice], color='green', lw=2, label="Final Recovered")
    plt.title("Step 2 & 3: Interpolation & Clean Smoothing")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # 子图3：残差深度对比
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'][view_slice], df['filter_ma'][view_slice] - df['clean_signal'][view_slice], color='blue', alpha=0.3, label="Direct MA Error")
    plt.plot(df['timestamp'][view_slice], df['final_recovered'][view_slice] - df['clean_signal'][view_slice], color='green', alpha=0.7, label="Professional Repair Error")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Performance: Repair is {improvement:.1f}% Better than Direct MA")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("professional_repair_final.png")
    plt.show()

    print(f"📊 最终实战报告:")
    print(f"原始投毒 MSE: {mse_raw:.4f}")
    print(f"直接滤波 MSE: {mse_ma:.4f}")
    print(f"专业修复 MSE: {mse_final:.4f}")
    print(f"🚀 综合提升比例: {improvement:.2f}%")
    output_filename = "cleaned_pressure_final.csv"

    # 只取这两列进行导出
    df[['timestamp', 'final_recovered']].to_csv(
        output_filename,
        index=False,
        float_format="%.6f"  # 强制保留 6 位小数，确保时间戳不丢失精度
    )
