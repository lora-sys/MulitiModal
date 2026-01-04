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
def save_plot(self, filename="simulation_plot2.png"):
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"图表已保存至: {filename}")

def save_signal_to_csv(df, filename="pressure_sim.csv"):
    """
    规范化写入函数：确保列名和精度统一
    """
    # 强制保留 6 位小数，确保 50Hz 或更高频率下的时间戳不丢失精度
    df.to_csv(filename, index=False, float_format="%.6f")
    print(f"✅ 数据已写入磁盘: {os.path.abspath(filename)}")
def export_to_csv(self, df: pd.DataFrame, filename: str):
    """导出数据为标准CSV格式"""
    df.to_csv(filename, index=False)
    print(f"数据成功导出至: {filename}")

def load_signal_from_csv(filename="pressure_sim.csv"):
    """
    规范化读取函数：后续算法处理的起点
    """
    if not os.path.exists(filename):
        print(f"❌ 错误：文件 {filename} 不存在")
        return None

    # 读取数据
    df = pd.read_csv(filename)

    # 验证关键列是否存在 (这是工程鲁棒性的体现)
    required_cols = ["timestamp", "clean_signal", "noisy_signal"]
    if all(col in df.columns for col in required_cols):
        print(f"📖 数据读取成功，样本数: {len(df)}")
        return df
    else:
        print("❌ 错误：文件格式与规范不符")
        return None

# 验证清洗算法，加载数据
df = load_signal_from_csv("pressure_sim.csv")

if df is None:
    print("文件不存在")
# 滑动平均，以当前数据点，前后左右一共5个取平均值作为当前点的值
# 窗口设置为5，min_periods=1 保证边缘数据不丢失
# 始终以当前点作为中心点，不足，有多少计算多少
df['filter_ma']=df['noisy_signal'].rolling(window=5,center=True,min_periods=1).mean()

#中值滤波
# 以当前点为中心点。左右一共5个排序后取中值作为当前数据点，以kernei_size 滤波核为当前窗口大小。必需设置为奇数，才会有中值出现
df['filter_median']= medfilt(df['noisy_signal'],kernel_size=5)


# 跑分评价
mse_raw=calculate_metrics(df['clean_signal'],df['noisy_signal'])
mse_ma = calculate_metrics(df['clean_signal'],df['filter_ma'])
mse_median =calculate_metrics(df['clean_signal'],df['filter_median'])

# 结果可视化
plt.figure(figsize=(15,8))
# 子图1 ：整体效果对比
plt.subplot(2,1,1)
plt.plot(df['timestamp'][:200],df['noisy_signal'][:200],color='red',alpha=0.3,label="Raw Noisy")
plt.plot(df['timestamp'][:200],df['clean_signal'][:200],color='black',lw=2,label="GRound Truth ")
plt.plot(df['timestamp'][:200],df['filter_ma'][:200],color='blue',alpha=0.3,label="Moving Average(windows=5)")
plt.plot(df['timestamp'][:200],df['filter_median'][:200],color='green',label="Median Filter (kernel=5)")

plt.title(f"Comparison of Cleaning Algorithms (Snippet)\nMSE: Raw={mse_raw:.4f} | MA={mse_ma:.4f} | Median={mse_median:.4f}")
plt.legend()
plt.grid(alpha=0.3)


# 子图2 ：误差残差图 （Residuals）
# 理想情况下，残差越接近0越好
plt.subplot(2,1,2)
plt.plot(df['timestamp'][:200],df['filter_ma'][:200]-df['clean_signal'][:200],color="blue",alpha=0.6,label="MA Error")
plt.plot(df['timestamp'][:200],df['filter_median'][:200]-df['clean_signal'][:200],color="green",alpha=0.6,label="Median Error")
plt.axhline(0,color="black",linestyle="--")
plt.title("Residual Analysis (Filtered - Clean)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
save_plot("result1.png")
plt.show()
print(f"📊 最终跑分对比:")
print(f"原始噪声 MSE: {mse_raw:.4f}")
print(f"滑动平均 MSE: {mse_ma:.4f}")
print(f"中值滤波 MSE: {mse_median:.4f}")
