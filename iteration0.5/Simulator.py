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

def export_anomalies(df, filename="anomaly_points.csv"):
    """导出异常点数据到CSV"""
    anomaly_df = df[df['is_anomaly']][['timestamp', 'noisy_signal', 'upper_bound', 'lower_bound']]
    anomaly_df.to_csv(filename, index=False)
    print(f"✅ 异常点数据已导出至: {filename} (共{len(anomaly_df)}条)")

# 1. 加载数据
df = load_signal_from_csv("pressure_sim.csv")

if df is not None:
    # --- 0.2 阶段：平滑滤波 ---
    window_size = 15
    # 使用滑动平均作为基准线
    df['filter_ma'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).mean()

    # --- 0.3 阶段：异常检测 (3-Sigma 准则) ---
    # 1. 计算滑动标准差 (Rolling Standard Deviation)
    # 标准差反映了局部噪声的剧烈程度
    df['rolling_std'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).std()

    # 2. 计算动态上下限
    # 均值 +/- 3倍标准差：覆盖了 99.73% 的正常数据波动
    df['upper_bound'] = df['filter_ma'] + 3 * df['rolling_std']
    df['lower_bound'] = df['filter_ma'] - 3 * df['rolling_std']

    # 3. 判定异常点
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
    save_plot("anomaly_detection_v1.png")
    export_anomalies(df)
    plt.show()

    print(f"📊 异常检测报告:")
    print(f"样本总数: {len(df)}")
    print(f"检测到异常点: {anomaly_count}")
    print(f"异常比例: {(anomaly_count/len(df))*100:.2f}% (理论值约为 0.27%)")
    print(f"当前 MSE: {mse_ma:.4f}")

# 引入动态阈值**：使用了 `rolling().std()` 计算滑动标准差。这意味着阈值会随着信号的波动而自动调整。
# 2.  **可视化增强**：
#     *   **灰色阴影区**：直观展示了算法认为的“安全范围”。
#     *   **红色 X 标记**：精准定位了那些跳出安全范围的“坏分子”。
# 3.  **统计输出**：增加了异常比例的计算，方便验证算法的严苛程度
