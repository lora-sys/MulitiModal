from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MassageChairSimulator:
    """
    按摩椅多模态信号仿真器
    用于生成理想的传感器基准信号 (Ground Truth)
    """

    def __init__(self, sampling_rate: int = 50, duration: int = 60):
        """
        :param sampling_rate: 采样频率 (Hz), 默认50Hz
        :param duration: 仿真时长 (秒)
        """
        self.fs = sampling_rate
        self.duration = duration
        self.t = np.linspace(
            0, duration, duration * sampling_rate
        )  # 时间向量，用于生成信号的时间序列，不用循环在与性能区别

    def generate_ideal_pressure(
        self, cycle_freq: float = 0.2, base_value: float = 40.0, amplitude: float = 20.0
    ) -> np.ndarray:
        """
        生成理想的压力传感器信号 (正弦复合波)
        模拟机械手上下往复运动对传感器的挤压

        :param cycle_freq: 机械手往复频率 (Hz), 0.2Hz 代表 5秒一个来回
        :param base_value: 基础压力值 (N)
        :param amplitude: 压力变化振幅 (N)
        :return: 理想压力信号序列
        """
        # 主波形：正弦波模拟往复挤压
        #  在 40N 基础压力之上，机械手「额外多压了多少」或「额外松开了多少
        primary_wave = amplitude * np.sin(2 * np.pi * cycle_freq * self.t)
        # 振幅 × sin( 2π × 循环频率 × 时间 )

        # 谐波成分：模拟机械传动中的次级微动特征（使信号更真实）
        harmonic_wave = (amplitude * 0.1) * np.sin(
            2 * np.pi * (cycle_freq * 3) * self.t
        )

        ideal_signal = base_value + primary_wave + harmonic_wave
        # 放在你的代码末尾，执行后直接输出结果
        print("数据格式验证：")
        print(
            f"primary_wave 类型 → {type(primary_wave)}"
        )  # 输出：numpy.ndarray（数组）
        print(f"primary_wave 形状 → {primary_wave.shape}")  # 输出：(1000,)（一维）
        print(
            f"ideal_signal 数据类型 → {ideal_signal.dtype}"
        )  # 输出：float64（浮点型）
        return ideal_signal

    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """导出数据为标准CSV格式"""
        df.to_csv(filename, index=False)
        print(f"数据成功导出至: {filename}")

    def save_plot(self, filename="simulation_plot.png"):
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"图表已保存至: {filename}")


# --- 执行仿真 ---

# 1. 初始化仿真器
simulator = MassageChairSimulator(sampling_rate=50, duration=60)

# 2. 生成理想压力数据 (Ground Truth)
# 设定机械手 5秒一个往复 (0.2Hz)，压力在 20N 到 60N 之间波动
ideal_p = simulator.generate_ideal_pressure(
    cycle_freq=0.2, base_value=40.0, amplitude=20.0
)

# 3. 封装为 DataFrame
df_ideal = pd.DataFrame({"timestamp": simulator.t, "pressure_ideal": ideal_p})

# 4. 可视化 (专业绘图规范)
plt.figure(figsize=(12, 4), dpi=100)
plt.plot(
    df_ideal["timestamp"][:500],
    df_ideal["pressure_ideal"][:500],
    color="#2E7D32",
    label="Ideal Pressure (Ground Truth)",
    linewidth=2,
)

plt.title(
    "Simulated Ideal Pressure Signal (Mechanical Hand Reciprocation)", fontsize=12
)
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Pressure (Newton)", fontsize=10)
plt.grid(
    True, linestyle="--", alpha=0.6
)  # 添加背景网格。`linestyle="--"` 使用虚线，`alpha=0.6` 设置透明度，这样网格就不会抢了主曲线的风头
plt.legend(loc="upper right")
plt.tight_layout()  # 自动调整边缘，防止标题或坐标轴标签被窗口遮挡或重叠。
simulator.save_plot("pressure_ground_truth.png")
plt.show()

# 5.导出
simulator.export_to_csv(df_ideal, "pressure_ground_truth.csv")
