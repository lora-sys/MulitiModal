#。手动在数据中插入 10 个巨大的脉冲（Spikes），模拟传感器突然失灵或受到强烈干扰的情况。


from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import medfilt

def calculate_metrics(clean, noisy):
    """
    计算跑分指标：MSE (越小越好)
    """
    mse = np.mean((clean - noisy) ** 2)
    return mse

def save_plot(filename="anomaly_detection_v1.png"):
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"图表已保存至: {filename}")

def load_signal_from_csv(filename="pressure_sim.csv"):
    """
    规范化读取函数
    """
    if not os.path.exists(filename):
        print(f"❌ 错误：文件 {filename} 不存在")
        return None
    df = pd.read_csv(filename)
    required_cols = ["timestamp", "clean_signal", "noisy_signal"]
    if all(col in df.columns for col in required_cols):
        print(f"📖 数据读取成功，样本数: {len(df)}")
        return df
    else:
        print("❌ 错误：文件格式与规范不符")
        return None

def export_anomalies(df, filename="anomaly_pointsv2.csv"):
    """导出异常点数据到CSV"""
    anomaly_df = df[df['is_anomaly']][['timestamp', 'noisy_signal', 'upper_bound', 'lower_bound']]
    anomaly_df.to_csv(filename, index=False)
    print(f"✅ 异常点数据已导出至: {filename} (共{len(anomaly_df)}条)")



# 1. 加载数据
df = load_signal_from_csv("pressure_sim.csv")

if df is not None:
    # --- 任务 0.3b 投毒逻辑 (Outlier Injection) ---
    # 随机选择 10 个位置，注入远超正常范围的脉冲噪声
    np.random.seed(42) # 保证实验可复现，固定位置，确保算法可验证
    poison_indices = np.random.choice(df.index[50:-50], size=10, replace=False) # 避开边缘

    # 注入偏离均值 20-50 个单位的巨型跳变
    for idx in poison_indices:
        spike = np.random.uniform(20, 50) * np.random.choice([-1, 1])
        df.loc[idx, 'noisy_signal'] += spike

    print(f"☣️ 投毒成功：已手动注入 {len(poison_indices)} 个异常脉冲")

    # --- 0.2 阶段：平滑滤波 ---
    window_size = 15
    df['filter_ma'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).mean()

    # --- 0.3 阶段：异常检测 (3-Sigma 准则) ---
    # 计算滑动统计量
    df['rolling_std'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).std()

    # 定义 3-Sigma 边界
    df['upper_bound'] = df['filter_ma'] + 3 * df['rolling_std']
    df['lower_bound'] = df['filter_ma'] - 3 * df['rolling_std']

    # 判定异常点
    df['is_anomaly'] = (df['noisy_signal'] > df['upper_bound']) | (df['noisy_signal'] < df['lower_bound'])
    anomaly_count = df['is_anomaly'].sum()
    # 跑分评价
    mse_raw = calculate_metrics(df['clean_signal'], df['noisy_signal'])
    mse_ma = calculate_metrics(df['clean_signal'], df['filter_ma'])

    # --- 结果可视化 ---
    plt.figure(figsize=(15, 10))

    # 子图1 ：原始信号、滤波信号与 3-Sigma 边界
    plt.subplot(2, 1, 1)
    view_slice = slice(0, 400) # 查看前 400 个点

    plt.plot(df['timestamp'][view_slice], df['noisy_signal'][view_slice], color='red', alpha=0.15, label="Raw Noisy")
    plt.plot(df['timestamp'][view_slice], df['clean_signal'][view_slice], color='black', lw=2, label="Ground Truth")
    plt.plot(df['timestamp'][view_slice], df['filter_ma'][view_slice], color='blue', alpha=0.8, label=f"MA baseLine (window={window_size})")

    # 绘制 3-Sigma 置信区间（灰色阴影）
    plt.fill_between(df['timestamp'][view_slice],
                     df['lower_bound'][view_slice],
                     df['upper_bound'][view_slice],
                     color='gray', alpha=0.2, label="3-Sigma Confidence Range")

    # 标记检测到的异常点 (红色的 X)
    anomalies_in_slice = df[view_slice][df[view_slice]['is_anomaly']]
    plt.scatter(anomalies_in_slice['timestamp'], anomalies_in_slice['noisy_signal'],
                color='darkred', marker='x', s=60, label=f"Detected Anomalies ({anomaly_count} total)")

    plt.title(f"Stage 0.3: Anomaly Detection (3-Sigma Rule)\nTotal Anomalies: {anomaly_count} ({(anomaly_count/len(df))*100:.2f}%)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # 子图2 ：残差分析
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'][view_slice], df['filter_ma'][view_slice] - df['clean_signal'][view_slice], color="blue", alpha=0.6, label="MA Error")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Residual Analysis (MA - Clean)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_plot("anomaly_detection_v2.png")
    export_anomalies(df)
    plt.show()

    print(f"📊 异常检测报告:")
    print(f"样本总数: {len(df)}")
    print(f"检测到异常点: {anomaly_count}")
    print(f"异常比例: {(anomaly_count/len(df))*100:.2f}% (理论值约为 0.27%)")
    print(f"当前 MSE: {mse_ma:.4f}")
    # 计算投毒点的捕捉率 （REVCALL 效率）
    detected_indices = df[df['is_anomaly']].index
    hit_count = len(set(poison_indices) & set(detected_indices))
    hit_rate = (hit_count / len(poison_indices)) * 100

    print(f"🎯 投毒捕捉率 (Recall): {hit_rate:.2f}%")
