"""
增强版噪声生成器
模拟真实按摩椅传感器的噪声环境

包含以下噪声类型：
1. 高斯底噪 - 基础电子噪声
2. 高频电磁干扰 - 50Hz/60Hz电源噪声及其谐波
3. 接触不良跳变 - 随机尖峰
4. 传感器故障 - 死区/饱和区
5. 量化噪声 - ADC台阶效应
6. 温度漂移 - 慢变偏移
7. 机械振动 - 高频振动噪声
8. 非高斯噪声 - 重尾分布（拉普拉斯）
9. 数据丢失 - 随机采样点缺失
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RealWorldNoiseGenerator:
    """
    真实世界噪声生成器

    模拟按摩椅传感器的实际噪声环境
    """

    def __init__(
        self,
        # 基础参数
        sampling_rate: float = 50.0,  # Hz
        duration: float = 20.0,  # 秒

        # 1. 高斯底噪
        gaussian_std: float = 5.0,

        # 2. 电磁干扰
        power_freq: float = 50.0,  # 50Hz电源干扰
        emi_amp: float = 2.0,  # 干扰幅度
        emi_harmonics: bool = True,  # 包含谐波

        # 3. 接触不良
        spike_prob: float = 0.003,  # 0.3%概率
        spike_amp: Tuple[float, float] = (30, 80),  # 跳变幅度

        # 4. 传感器故障
        dead_zone_prob: float = 0.001,  # 0.1%概率进入死区
        dead_zone_duration: Tuple[float, float] = (0.1, 0.5),  # 死区持续时长（秒）
        saturate_prob: float = 0.0005,  # 0.05%概率饱和
        saturate_threshold: float = 200.0,  # 饱和阈值

        # 5. 量化噪声
        quantization_bits: int = 12,  # 12位ADC
        voltage_range: Tuple[float, float] = (0.0, 250.0),  # 电压范围

        # 6. 温度漂移
        temp_drift_amp: float = 3.0,  # 漂移幅度
        temp_drift_freq: float = 0.005,  # 漂移频率（Hz）

        # 7. 机械振动
        vibration_freq: float = 150.0,  # 振动频率（Hz）
        vibration_amp: float = 1.5,  # 振动幅度
        vibration_modulation: bool = True,  # 振动幅度调制

        # 8. 非高斯噪声
        laplace_b: float = 3.0,  # 拉普拉斯分布参数
        laplace_weight: float = 0.3,  # 拉普拉斯噪声权重

        # 9. 数据丢失
        dropout_prob: float = 0.002,  # 0.2%概率丢失采样点
        dropout_max_consecutive: int = 5,  # 最多连续丢失点数

        # 随机种子
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.fs = sampling_rate
        self.duration = duration
        self.n_samples = int(sampling_rate * duration)
        self.t = np.linspace(0, duration, self.n_samples)

        # 噪声参数
        self.gaussian_std = gaussian_std
        self.power_freq = power_freq
        self.emi_amp = emi_amp
        self.emi_harmonics = emi_harmonics
        self.spike_prob = spike_prob
        self.spike_amp = spike_amp
        self.dead_zone_prob = dead_zone_prob
        self.dead_zone_duration = dead_zone_duration
        self.saturate_prob = saturate_prob
        self.saturate_threshold = saturate_threshold
        self.quantization_bits = quantization_bits
        self.voltage_range = voltage_range
        self.temp_drift_amp = temp_drift_amp
        self.temp_drift_freq = temp_drift_freq
        self.vibration_freq = vibration_freq
        self.vibration_amp = vibration_amp
        self.vibration_modulation = vibration_modulation
        self.laplace_b = laplace_b
        self.laplace_weight = laplace_weight
        self.dropout_prob = dropout_prob
        self.dropout_max_consecutive = dropout_max_consecutive

        print(f"[噪声生成器] 初始化完成")
        print(f"  采样率: {sampling_rate} Hz, 时长: {duration} 秒")
        print(f"  高斯噪声: σ={gaussian_std}")
        print(f"  电磁干扰: {power_freq} Hz, 幅度={emi_amp}")
        print(f"  接触不良: {spike_prob*100:.1f}%")
        print(f"  传感器故障: 死区{dead_zone_prob*100:.1f}%, 饱和{saturate_prob*100:.2f}%")
        print(f"  量化位数: {quantization_bits} bit")
        print(f"  温度漂移: {temp_drift_amp} Pa")
        print(f"  机械振动: {vibration_freq} Hz")
        print(f"  数据丢失: {dropout_prob*100:.1f}%")

    def add_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加高斯底噪"""
        noise = np.random.normal(0, self.gaussian_std, len(signal))
        return signal + noise

    def add_emi_interference(self, signal: np.ndarray) -> np.ndarray:
        """添加电磁干扰（电源噪声）"""
        # 基波
        emi = self.emi_amp * np.sin(2 * np.pi * self.power_freq * self.t)

        # 谐波（2次、3次谐波）
        if self.emi_harmonics:
            emi += (self.emi_amp * 0.5) * np.sin(2 * np.pi * 2 * self.power_freq * self.t)
            emi += (self.emi_amp * 0.25) * np.sin(2 * np.pi * 3 * self.power_freq * self.t)

        return signal + emi

    def add_contact_spikes(self, signal: np.ndarray) -> np.ndarray:
        """添加接触不良跳变"""
        noisy = signal.copy()

        # 随机选择跳变点
        n_spikes = int(len(signal) * self.spike_prob)
        spike_indices = np.random.choice(len(signal), n_spikes, replace=False)

        # 添加跳变（随机正负）
        for idx in spike_indices:
            direction = np.random.choice([-1, 1])
            amplitude = np.random.uniform(*self.spike_amp)
            noisy[idx] += direction * amplitude

        return noisy

    def add_sensor_faults(self, signal: np.ndarray) -> np.ndarray:
        """添加传感器故障（死区和饱和）"""
        noisy = signal.copy()

        # 1. 死区（输出0或固定值）
        if np.random.random() < self.dead_zone_prob:
            dead_duration = np.random.uniform(*self.dead_zone_duration)
            dead_samples = int(dead_duration * self.fs)
            dead_start = np.random.randint(0, len(signal) - dead_samples)
            dead_end = dead_start + dead_samples

            # 死区类型：输出0或固定噪声
            dead_type = np.random.choice(['zero', 'fixed', 'noise'])
            if dead_type == 'zero':
                noisy[dead_start:dead_end] = 0
            elif dead_type == 'fixed':
                noisy[dead_start:dead_end] = 10.0
            else:  # noise
                noisy[dead_start:dead_end] = np.random.normal(0, 2, dead_samples)

        # 2. 饱和
        n_saturate = int(len(signal) * self.saturate_prob)
        saturate_indices = np.random.choice(len(signal), n_saturate, replace=False)

        for idx in saturate_indices:
            direction = np.random.choice([-1, 1])
            noisy[idx] = direction * self.saturate_threshold

        return noisy

    def add_quantization_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加量化噪声（模拟ADC）"""
        v_min, v_max = self.voltage_range
        n_levels = 2 ** self.quantization_bits

        # 量化
        signal_clipped = np.clip(signal, v_min, v_max)
        signal_normalized = (signal_clipped - v_min) / (v_max - v_min)
        signal_quantized = np.round(signal_normalized * (n_levels - 1)) / (n_levels - 1)
        signal_dequantized = signal_quantized * (v_max - v_min) + v_min

        return signal_dequantized

    def add_temperature_drift(self, signal: np.ndarray) -> np.ndarray:
        """添加温度漂移（慢变偏移）"""
        # 慢变正弦漂移
        drift = self.temp_drift_amp * np.sin(2 * np.pi * self.temp_drift_freq * self.t)

        # 添加随机游走
        random_walk = np.cumsum(np.random.normal(0, 0.05, len(signal)))
        random_walk = np.convolve(random_walk, np.ones(200) / 200, mode='same')
        drift += random_walk * 0.3

        return signal + drift

    def add_mechanical_vibration(self, signal: np.ndarray) -> np.ndarray:
        """添加机械振动（高频噪声）"""
        # 基础振动
        vibration = self.vibration_amp * np.sin(2 * np.pi * self.vibration_freq * self.t)

        # 幅度调制（模拟转速变化）
        if self.vibration_modulation:
            modulator = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * self.t)  # 0.5Hz调制
            vibration *= modulator

        # 添加宽带噪声（模拟复杂振动）
        broadband = np.random.normal(0, self.vibration_amp * 0.5, len(signal))
        broadband = np.convolve(broadband, np.ones(10) / 10, mode='same')

        return signal + vibration + broadband

    def add_non_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加非高斯噪声（拉普拉斯分布，模拟重尾噪声）"""
        # 拉普拉斯噪声
        laplace_noise = np.random.laplace(0, self.laplace_b, len(signal))

        # 混合高斯和拉普拉斯
        gaussian_noise = np.random.normal(0, self.gaussian_std * 0.5, len(signal))
        mixed_noise = (1 - self.laplace_weight) * gaussian_noise + self.laplace_weight * laplace_noise

        return signal + mixed_noise

    def add_data_dropout(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """添加数据丢失（返回噪声信号和掩码）"""
        noisy = signal.copy()
        mask = np.ones_like(signal)  # 1=有效, 0=丢失

        n_dropouts = int(len(signal) * self.dropout_prob)

        for _ in range(n_dropouts):
            # 随机起始点
            start = np.random.randint(0, len(signal))
            # 随机持续时间
            duration = np.random.randint(1, self.dropout_max_consecutive + 1)
            end = min(start + duration, len(signal))

            # 标记为丢失
            mask[start:end] = 0
            # 丢失的值设为NaN（或者可以用插值）

        return noisy, mask

    def apply_all_noise(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用所有噪声类型

        返回：
        - noisy_signal: 带噪声的信号
        - dropout_mask: 数据丢失掩码（1=有效, 0=丢失）
        """
        print(f"\n[噪声生成器] 应用噪声到信号（长度={len(signal)}）...")

        # 1. 高斯底噪
        signal = self.add_gaussian_noise(signal)
        print(f"  ✓ 高斯噪声")

        # 2. 电磁干扰
        signal = self.add_emi_interference(signal)
        print(f"  ✓ 电磁干扰")

        # 3. 机械振动
        signal = self.add_mechanical_vibration(signal)
        print(f"  ✓ 机械振动")

        # 4. 非高斯噪声
        signal = self.add_non_gaussian_noise(signal)
        print(f"  ✓ 非高斯噪声")

        # 5. 温度漂移
        signal = self.add_temperature_drift(signal)
        print(f"  ✓ 温度漂移")

        # 6. 接触不良
        signal = self.add_contact_spikes(signal)
        print(f"  ✓ 接触不良")

        # 7. 传感器故障
        signal = self.add_sensor_faults(signal)
        print(f"  ✓ 传感器故障")

        # 8. 量化噪声
        signal = self.add_quantization_noise(signal)
        print(f"  ✓ 量化噪声")

        # 9. 数据丢失
        signal, dropout_mask = self.add_data_dropout(signal)
        print(f"  ✓ 数据丢失")

        # 物理约束（压力不能为负）
        signal = np.maximum(signal, 0.0)

        print(f"✓ 所有噪声应用完成")

        return signal, dropout_mask

    def visualize_noise_components(self, clean_signal: np.ndarray, noisy_signal: np.ndarray, save_path: str = None):
        """可视化各个噪声组件"""
        import os
        if save_path is None:
            save_path = 'experiment/generate/noise_visualization.png'

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 选择前2秒数据（100个采样点）
        n_show = min(100, len(clean_signal))
        t_show = self.t[:n_show]

        # 1. 原始信号 vs 噪声信号
        axes[0, 0].plot(t_show, clean_signal[:n_show], 'b-', label='原始信号', linewidth=2)
        axes[0, 0].plot(t_show, noisy_signal[:n_show], 'r-', label='噪声信号', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('原始信号 vs 噪声信号')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('压力 (Pa)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 噪声分量
        noise = noisy_signal[:n_show] - clean_signal[:n_show]
        axes[0, 1].plot(t_show, noise, 'g-', linewidth=1)
        axes[0, 1].set_title('噪声分量')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('噪声 (Pa)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 频谱分析 - 原始信号
        from scipy import signal as scipy_signal
        f_clean, Pxx_clean = scipy_signal.welch(clean_signal, self.fs, nperseg=256)
        axes[1, 0].semilogy(f_clean, Pxx_clean, 'b-', linewidth=1.5)
        axes[1, 0].set_title('原始信号频谱')
        axes[1, 0].set_xlabel('频率 (Hz)')
        axes[1, 0].set_ylabel('功率谱密度')
        axes[1, 0].set_xlim([0, 100])
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 频谱分析 - 噪声信号
        f_noisy, Pxx_noisy = scipy_signal.welch(noisy_signal, self.fs, nperseg=256)
        axes[1, 1].semilogy(f_noisy, Pxx_noisy, 'r-', linewidth=1.5)
        axes[1, 1].set_title('噪声信号频谱')
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('功率谱密度')
        axes[1, 1].set_xlim([0, 100])
        axes[1, 1].grid(True, alpha=0.3)

        # 5. 统计分布 - 原始信号
        axes[2, 0].hist(clean_signal, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[2, 0].set_title('原始信号分布')
        axes[2, 0].set_xlabel('压力 (Pa)')
        axes[2, 0].set_ylabel('频次')
        axes[2, 0].grid(True, alpha=0.3)

        # 6. 统计分布 - 噪声信号
        axes[2, 1].hist(noisy_signal, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[2, 1].set_title('噪声信号分布')
        axes[2, 1].set_xlabel('压力 (Pa)')
        axes[2, 1].set_ylabel('频次')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 噪声可视化已保存: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("增强版噪声生成器测试")
    print("=" * 60)

    # 生成测试信号（模拟压力波形）
    fs = 50.0
    duration = 20.0
    t = np.linspace(0, duration, int(fs * duration))

    # 模拟一个基础波形（类似按摩椅的周期性压力）
    base_pressure = 50.0
    amplitude = 30.0
    clean_signal = base_pressure + amplitude * np.sin(2 * np.pi * 0.5 * t) ** 2

    # 创建噪声生成器
    noise_gen = RealWorldNoiseGenerator(
        sampling_rate=fs,
        duration=duration,
        seed=42
    )

    # 应用噪声
    noisy_signal, dropout_mask = noise_gen.apply_all_noise(clean_signal)

    # 统计信息
    print(f"\n{'='*60}")
    print("统计信息")
    print(f"{'='*60}")
    print(f"原始信号:")
    print(f"  均值: {np.mean(clean_signal):.2f}")
    print(f"  标准差: {np.std(clean_signal):.2f}")
    print(f"  最小值: {np.min(clean_signal):.2f}")
    print(f"  最大值: {np.max(clean_signal):.2f}")
    print(f"\n噪声信号:")
    print(f"  均值: {np.mean(noisy_signal):.2f}")
    print(f"  标准差: {np.std(noisy_signal):.2f}")
    print(f"  最小值: {np.min(noisy_signal):.2f}")
    print(f"  最大值: {np.max(noisy_signal):.2f}")
    print(f"  数据丢失: {np.sum(dropout_mask == 0)} / {len(dropout_mask)} ({np.sum(dropout_mask == 0)/len(dropout_mask)*100:.2f}%)")
    print(f"\n噪声:")
    print(f"  SNR: {20 * np.log10(np.std(clean_signal) / np.std(noisy_signal - clean_signal)):.2f} dB")

    # 处理NaN值（数据丢失）
    noisy_signal_valid = noisy_signal[~np.isnan(noisy_signal)]
    clean_signal_valid = clean_signal[~np.isnan(noisy_signal)]

    print(f"\n噪声:")
    print(f"  SNR: {20 * np.log10(np.std(clean_signal_valid) / np.std(noisy_signal_valid - clean_signal_valid)):.2f} dB")

    # 可视化
    noise_gen.visualize_noise_components(clean_signal, noisy_signal)

    print(f"\n✓ 测试完成！")