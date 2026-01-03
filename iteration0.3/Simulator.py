from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_metrics(clean, noisy):
    """
    计算跑分指标：MSE (越小越好)
    """
    mse = np.mean((clean - noisy) ** 2)
    return mse


def save_signal_to_csv(df, filename="pressure_sim.csv"):
    """
    规范化写入函数：确保列名和精度统一
    """
    # 强制保留 6 位小数，确保 50Hz 或更高频率下的时间戳不丢失精度
    df.to_csv(filename, index=False, float_format="%.6f")
    print(f"✅ 数据已写入磁盘: {os.path.abspath(filename)}")


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
        # 标量+一维矩阵   基准变量+ 正波+谐波
        ideal_signal = base_value + primary_wave + harmonic_wave

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


class NoiseEngine:
    """ "
    噪声生成引擎，模拟真实世界的硬件干扰
    """

    @staticmethod
    def add_gaussian_noise(data: np.ndarray, std: float = 2.0) -> np.ndarray:
        """
        添加高斯白噪声（模拟电路底噪和热噪声）
        ：param std: 噪声的标准差，数值越大，波形越抖
        """
        noise = np.random.normal(0, std, size=data.shape)
        return data + noise

    @staticmethod
    def add_impulse_noise(
        data: np.ndarray, prob: float = 0.2, magnitude: float = 50.0
    ) -> np.ndarray:
        """
        模拟脉冲噪声 (模拟电磁干扰产生的尖峰/噪声)
        :param prob : 噪声出现的概率，在每个采样点 （0-1）
        ：param magnitude : 噪声的幅度，数值越大，波形越抖
        """
        noisy_data = data.copy()
        # 随机选择位置注入跳点
        mask = np.random.random(size=data.shape) < prob
        # 布尔数组

        # 产生正向或者反向的随机冲击
        impulse = np.random.uniform(
            magnitude * 0.8, magnitude * 1.2, size=data[mask].shape
        )
        signs = np.random.choice([-1, 1], size=impulse.shape)
        noisy_data[mask] += (
            impulse * signs
        )  # ：只在 `mask` 为 `True` 的那些位置加上噪声，其他位置保持原样
        return noisy_data


# --- 执行仿真 ---

# 1. 初始化仿真器
simulator = MassageChairSimulator(sampling_rate=50, duration=60)

# 2. 生成理想压力数据 (Ground Truth)
# 设定机械手 5秒一个往复 (0.2Hz)，压力在 20N 到 60N 之间波动
ideal_p = simulator.generate_ideal_pressure(
    cycle_freq=0.2, base_value=40.0, amplitude=20.0
)

# 进行噪声注入
# 加入基础底噪
noisy_p = NoiseEngine.add_gaussian_noise(ideal_p, std=1.5)
# 第二步， 加入瞬时电磁干扰
noisy_p = NoiseEngine.add_impulse_noise(noisy_p, prob=0.01, magnitude=10)


# 结果持久化
df_sim = pd.DataFrame(
    {
        "timestamp": simulator.t,
        "clean_signal": ideal_p,
        "noisy_signal": noisy_p,
    }
)

# # 写入文件
# save_signal_to_csv(df_sim, "pressure_sim.csv")

# # 验证读取 （写和读闭环）
# df_loaded = load_signal_from_csv("pressure_sim.csv")

# 4. 可视化对比
plt.figure(figsize=(15, 6))
plt.plot(
    df_sim["timestamp"][:500],
    df_sim["noisy_signal"][:500],
    color="red",
    alpha=0.4,
    label="Measured Signal (Noisy)",
)
plt.plot(
    df_sim["timestamp"][:500],
    df_sim["clean_signal"][:500],
    color="black",
    label="Ground Truth (Clean)",
    linewidth=2,
)

plt.fill_between(
    df_sim["timestamp"][:500],
    df_sim["clean_signal"][:500],
    df_sim["noisy_signal"][:500],
    color="gray",
    alpha=0.2,
    label="Noise/Error",
)

plt.title("Sub-task 3: Signal Contamination Experiment", fontsize=14)

plt.xlabel("Time (s)")
plt.ylabel("Pressure (N)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("signal_contamination.png")
plt.show()

# 计算mse 初始误差
# 计算当前的初始误差
initial_error = calculate_metrics(df_sim["clean_signal"], df_sim["noisy_signal"])

print(f"inital MSE score: {initial_error:.4f}")
