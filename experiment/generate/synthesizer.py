import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import matplotlib.pyplot as plt
from enhanced_noise import RealWorldNoiseGenerator

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WaveformGenerator:
    """
    机械波形生成器
    模拟真实按摩椅机械手的往复运动：
    - 推压期(35%):sin²曲线,模拟滚轮下压
    - 保持期(25%）：平顶，模拟停顿施压
    - 回弹期(30%）：指数衰减，模拟快速回弹
    - 复位期(10%）：基线，模拟机械复位
    """

    def __init__(self, base_freq: float = 0.5):
        self.base_freq = base_freq
        self.period = 1.0 / base_freq
        self.duty_cycle = {"push": 0.35, "hold": 0.25, "release": 0.30, "reset": 0.10}
        # 预计算边界，避免重复计算
        self._bound_push_end = self.duty_cycle["push"]
        self._bound_hold_end = self.duty_cycle["push"] + self.duty_cycle["hold"]
        self._bound_release_end = 1.0 - self.duty_cycle["reset"]

    def generate(
        self,
        t: np.ndarray,
        base_pressure: float,
        amplitude: float,
        phase_shift: float = 0.0,
    ) -> np.ndarray:
        """
        生成机械波形
        """

        # phase_shift / (2*np.pi) 将弧度转换为归一化周期比例
        phase = ((t / self.period) + (phase_shift / (2 * np.pi))) % 1.0

        # 初始化全0数组
        waveform = np.zeros_like(phase)

        # --- 1. 推压期 ---
        mask_push = phase < self._bound_push_end
        if self.duty_cycle["push"] > 0:
            # 归一化到 [0, 1]
            p_norm = phase[mask_push] / self.duty_cycle["push"]
            # sin² 曲线
            waveform[mask_push] = np.sin(np.pi * p_norm / 2) ** 2

        # --- 2. 保持期 ---
        mask_hold = (phase >= self._bound_push_end) & (phase < self._bound_hold_end)
        waveform[mask_hold] = 1.0

        # --- 3. 回弹期 ---
        mask_release = (phase >= self._bound_hold_end) & (
            phase < self._bound_release_end
        )
        if self.duty_cycle["release"] > 0:
            # 归一化到 [0, 1]
            p_norm = (phase[mask_release] - self._bound_hold_end) / self.duty_cycle[
                "release"
            ]
            # 指数衰减: exp(-5x), x从0到1，值从1衰减到exp(-5)≈0.006
            waveform[mask_release] = np.exp(-5 * p_norm)

        # --- 4. 复位期 ---
        # mask_reset = phase >= self._bound_release_end
        # waveform 已经初始化为 0，所以这一步可以省略

        # 映射实际压力值
        pressure = base_pressure + amplitude * waveform
        return pressure


class TemporalJitter:
    """
    时序抖动引擎 - 让机械运动更真实

    模拟三种电机不稳定性：
    1. 长期漂移:电机发热,频率缓慢变化(±10%,周期100秒)
    2. 短期抖动:齿轮间隙,相位随机偏移(±3%，平滑）
    3. 瞬时事件：阻力变化，随机加速/减速(每30秒,持续4-6秒）

    保守参数，确保波形仍可识别但不再"太规律"
    """

    def __init__(
        self,
        long_term_amp: float = 0.10,  # 10%
        short_term_amp: float = 0.03,  # 3%
        transient_freq: float = 0.033,
    ):  # 每30秒1次
        self.long_term_amp = long_term_amp
        self.short_term_amp = short_term_amp
        self.transient_freq = transient_freq

    def apply(self, t: np.ndarray, base_freq: float) -> np.ndarray:
        """
        对时间轴应用抖动，返回调整后的时间轴

        Args:
            t: 原始时间数组（秒）
            base_freq: 基础频率（Hz）

        Returns:
            t_jittered: 抖动后的时间轴（用于波形采样）
        """
        dt = t[1] - t[0]
        # 采样间隔（50Hz = 0.02s）

        # 长期偏移： 慢变正弦调制
        # 模拟电机发热导致的转速周期性变化
        drift = 1.0 + self.long_term_amp * np.sin(2 * np.pi * t / 100.0)

        # --- 2. 短期抖动：平滑随机游走 ---
        # 模拟齿轮间隙导致的微小相位抖动
        # 生成随机噪声，然后用移动平均平滑（50点 = 1秒）
        raw_noise = np.random.normal(0, self.short_term_amp, len(t))
        jitter = np.convolve(raw_noise, np.ones(50) / 50, mode="same")

        # --- 3. 瞬时事件：随机变速 ---
        # 模拟阻力变化导致的瞬时加速或减速
        transient = np.ones_like(t)
        if self.transient_freq > 0:
            # 每30秒左右发生一次
            event_interval = int(30.0 / dt)  # 30秒对应的采样点数
            num_events = len(t) // event_interval

            for _ in range(num_events):
                # 随机起始点
                start = np.random.randint(0, len(t) - 300)
                duration = np.random.randint(200, 300)  # 持续4-6秒
                end = min(start + duration, len(t))

                # 随机选择加速或减速（1.1-1.3倍 或 0.7-0.9倍）随机因子
                factor = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                transient[start:end] = factor
        # --- 组合所有抖动 ---
        # 瞬时频率 = 基础频率 × 长期漂移 × 瞬时事件
        instantaneous_freq = base_freq * drift * transient

        # 从频率重建时间轴：累积积分
        # 频率是相位的变化率，所以相位 = 积分(频率) × 2π
        # instantaneous_freq：决定这一刻跑多快。
        # np.cumsum：算出总共跑了多远（真实相位）。
        # `adjusted_t：算出“这段距离如果按正常跑，需要多少时间”。
        phase = np.cumsum(instantaneous_freq) * dt * 2 * np.pi
        t_jittered = phase / (2 * np.pi * base_freq)

        return t_jittered


# ========== Part 3: 软边界参数生成器 ==========
class FuzzyBoundaryGenerator:
    """
    软边界参数生成器

    解决iteration1.1的问题：类别边界过于清晰（RandomForest一眼看穿）

    策略：
    - 85%样本：清晰类别特征（硬边界）
    - 15%样本：跨类别混合（软边界）

    混合方式：使用Beta分布生成混合比例，偏向中间但不完全平均
    """

    def __init__(self, overlap_ratio: float = 0.15):  # 15% 保守
        self.overlap_ratio = overlap_ratio

        # 类别参数定义
        self.category_profiles = {
            0: {
                # 身体表征很差
                "hr_base": 95,  # 心率高
                "hr_var": 10.0,  # 波动大
                "pressure_factor": 0.8,  # 承受力低
                "amplitude_var": 0.2,  # 振幅变化大
            },
            1: {
                # 身体表征一般
                "hr_base": 85,
                "hr_var": 8,
                "pressure_factor": 1.0,
                "amplitude_var": 0.15,
            },
            2: {
                ## 身体表征正常
                "hr_base": 75,
                "hr_var": 6,
                "pressure_factor": 1.2,
                "amplitude_var": 0.1,
            },
            3: {
                # 身体表征很好
                "hr_base": 65,  # 心率低
                "hr_var": 5,  # 波动小
                "pressure_factor": 1.5,  # 承受力高
                "amplitude_var": 0.08,  # 振幅稳定
            },
        }

    def generate_params(self, category: int) -> dict:
        """
        生成带软边界的物理参数

        Args:
            category: 目标类别（0-3）

        Returns:
            params: 包含所有物理参数的字典
                  如果是软边界样本，会包含'mixed_with'和'mix_ratio'
        """
        # 决定是否生成为软边界样本（15%概率）
        is_fuzzy = np.random.random() < self.overlap_ratio

        if is_fuzzy and category in [1, 2]:  # 只有中间类别才有软边界
            # 选择相邻类别进行混合
            neighbor = category - 1 if np.random.random() < 0.5 else category + 1
            neighbor = np.clip(neighbor, 0, 3)

            # 使用Beta分布生成混合比例（α=2, β=2，中间偏向）
            mix_ratio = np.random.beta(2, 2)

            # 混合参数
            params = self._mix_params(category, neighbor, mix_ratio)
            params["is_fuzzy"] = True
            params["mixed_with"] = neighbor
            params["mix_ratio"] = mix_ratio

        else:
            # 硬边界：从目标类别直接采样
            params = self._sample_from_category(category)
            params["is_fuzzy"] = False

        return params

    def _sample_from_category(self, cat: int) -> dict:
        """从单个类别采样参数"""
        profile = self.category_profiles[cat]

        # 身高体重
        height = np.random.randint(155, 198)
        weight = int(height - 105 + np.random.normal(0, 5))

        # 心率
        hr = profile["hr_base"] + np.random.randint(
            -profile["hr_var"], profile["hr_var"]
        )
        hr = np.clip(hr, 50, 120)

        # 血氧（与类别相关）
        spo2 = 98 - (3 - cat) + np.random.randint(-1, 1)
        spo2 = np.clip(spo2, 90, 100)

        pressure_factor = profile["pressure_factor"] + np.random.normal(0, 0.1)
        base_pressure = weight * 0.6 * pressure_factor
        # 非线性振幅梯度
        amplitude_base = [12, 30, 60, 100]  # Cat0,1,2,3的基础振幅
        amplitude = amplitude_base[cat] + np.random.normal(
            0, profile["amplitude_var"] * 15
        )

        return {
            "category": cat,
            "height": height,
            "weight": weight,
            "hr": hr,
            "spo2": spo2,
            "base_pressure": base_pressure,
            "amplitude": amplitude,
        }

    def _mix_params(self, cat1: int, cat2: int, ratio: float) -> dict:
        """混合两个类别的参数"""
        param1 = self._sample_from_category(cat1)
        param2 = self._sample_from_category(cat2)

        mixed = {}
        for key in ["height", "weight", "hr", "spo2", "base_pressure", "amplitude"]:
            # 线性插值混合
            mixed[key] = ratio * param1[key] + (1 - ratio) * param2[key]

        # 类别标签使用混合比列最大的
        mixed["category"] = cat1 if ratio > 0.5 else cat2
        mixed["time_category"] = cat1  # 记录原始目标类别

        return mixed


class IndustalNoiseGenerator:
    """
    工业级噪声生成器

    在iteration1.1基础上增强：
    - 强力底噪：高斯噪声（保守σ=6）
    - 接触不良跳变：0.2%概率瞬间爆表（50-100）
    - 基线漂移：正弦+随机游走混合
    """

    def __init__(
        self,
        gaussian_std: float = 4.5,
        spike_prob: float = 0.002,
        spike_range: Tuple[float, float] = (50, 100),
        drift_amp: float = 5.0,
        drift_freq: float = 0.01,
    ):
        self.gaussian_std = gaussian_std
        self.spike_prob = spike_prob
        self.spike_range = spike_range
        self.drift_amp = drift_amp
        self.drift_freq = drift_freq

    def add_noise(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """添加工业噪声"""
        # 1. 高斯底噪
        noise = np.random.normal(0, self.gaussian_std, len(signal))

        # 2. 接触不良跳变
        spikes = np.zeros_like(signal)  # 0 数组
        spike_indices = np.random.choice(
            len(signal), int(len(signal) * self.spike_prob), replace=False
        )
        for idx in spike_indices:
            spikes[idx] = np.random.uniform(*self.spike_range) * np.random.choice(
                [-1, 1]
            )

        # 3. 基线漂移（正弦+随机游走）
        drift = self.drift_amp * np.sin(2 * np.pi * self.drift_freq * t)
        random_walk = np.cumsum(np.random.normal(0, 0.1, len(t)))
        random_walk = np.convolve(random_walk, np.ones(100) / 100, mode="same")
        drift += random_walk * 0.5

        return signal + noise + spikes + drift


class MassageDataSynthesizer:
    """
    主合成器 - 端到端数据生成

    整合所有组件，生成符合iteration1.1格式的数据
    """

    def __init__(
        self,
        seed: int = 42,
        overlap_ratio: float = 0.15,
        output_dir: str = "experiment/data",
    ):
        np.random.seed(seed)

        self.output_dir = Path(output_dir)
        self.wave_gen = WaveformGenerator(base_freq=0.5)
        self.jitter = TemporalJitter()
        self.boundart_gen = FuzzyBoundaryGenerator(overlap_ratio=overlap_ratio)
        self.noise_gen = RealWorldNoiseGenerator(
            sampling_rate=50.0,
            duration=20.0,
            seed=seed
        )
        # 类别配置
        self.categories = {
            "身体表征很差": 0,
            "身体表征一般": 1,
            "身体表征正常": 2,
            "身体表征良好": 3,
        }

    def generate_person(self, category_name: str, global_id: int) -> pd.DataFrame:
        """
        生成单个人数据

        Args:
            category_name: 类别名称（中文）
            global_id: 全局ID（001-1000）

        Returns:
            DataFrame: 符合iteration1.1格式的数据
        """

        category_id = self.categories[category_name]

        # 1 . 生成物理参数，边界
        params = self.boundart_gen.generate_params(category_id)

        # 2. 生成时间轴
        duration = 20
        fs = 50
        t = np.linspace(0, duration, duration * fs)

        # 3.时序抖动
        t_jittered = self.jitter.apply(t, 0.5)

        # 4. 生成双传感器波形 (相位偏移， 弧度区分)
        wave_s1 = self.wave_gen.generate(
            t_jittered, params["base_pressure"], params["amplitude"], phase_shift=0.0
        )
        wave_s2 = self.wave_gen.generate(
            t_jittered,
            params["base_pressure"] * 0.9,
            params["amplitude"],
            phase_shift=0.1,
        )

        # 5. 添加增强噪声（独立噪声）
        noisy_s1, _ = self.noise_gen.apply_all_noise(wave_s1)
        noisy_s2, _ = self.noise_gen.apply_all_noise(wave_s2)

        # 物理约束
        noisy_s1 = np.maximum(noisy_s1, 5.0)
        noisy_s2 = np.maximum(noisy_s2, 5.0)

        # 6. 构建DataFrame
        df = pd.DataFrame(
            {"时间戳": t, "压力传感器1": noisy_s1, "压力传感器2": noisy_s2}
        )

        return df, params, category_name

    def generate_batch(self, num_people: int = 1000):
        """
        批量生成数据

        生成1000人数据，按类别分文件夹存储
        """
        people_per_cat = num_people // len(self.categories)
        global_id = 1

        logging.info(f"开始生成{num_people} 人数据")

        for cat_name, cat_id in self.categories.items():
            # 创建类别目录
            cat_dir = self.output_dir / cat_name
            cat_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"生成类别: '{cat_name}' ({people_per_cat})人")

            for i in range(people_per_cat):
                # 生成数据

                df, params, _ = self.generate_person(cat_name, global_id)

                # 构建文件名：{id:03d}_{weight}_{hr}_{spo2}_{height}.csv
                filename = f"{global_id:03d}_{int(params['weight'])}_{int(params['hr'])}_{int(params['spo2'])}_{int(params['height'])}.csv"
                # 保存
                df.to_csv(cat_dir / filename, index=False)

                # 进度显示
                if (i + 1) % 50 == 0:
                    logging.info(f" {cat_name} : {i - 1}/ {people_per_cat} 完成")

                global_id += 1
            logging.info(f"✓ 数据生成完成！总计 {num_people} 人")
            logging.info(f"  输出目录: {self.output_dir.absolute()}")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 创建合成器
    synthesizer = MassageDataSynthesizer(
        seed=42, overlap_ratio=0.15, output_dir="experiment/data"
    )

    # 生成完整数据集（1000人）
    print("=" * 60)
    print("Simulation 4.0: 高保真多模态数据合成")
    print("=" * 60)
    print("配置：")
    print("  - 总人数: 1000")
    print("  - 每类人数: 250")
    print("  - 软边界比例: 15%")
    print("  - 采样率: 50Hz")
    print("  - 时长: 20秒")
    print("=" * 60)

    synthesizer.generate_batch(1000)

    # 生成示例可视化
    print("\n生成示例可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 使用英文类别名称
    categories_en = [
        "Category_0_Poor",
        "Category_1_Fair",
        "Category_2_Normal",
        "Category_3_Good",
    ]
    categories_map = {
        "身体表征很差": "Category_0_Poor",
        "身体表征一般": "Category_1_Fair",
        "身体表征正常": "Category_2_Normal",
        "身体表征良好": "Category_3_Good",
    }

    categories = ["身体表征很差", "身体表征一般", "身体表征正常", "身体表征良好"]
    for idx, cat_name in enumerate(categories):
        ax = axes[idx // 2, idx % 2]

        # 读取第一个样本
        cat_dir = Path("experiment/data") / cat_name
        sample_file = list(cat_dir.glob("*.csv"))[0]
        df = pd.read_csv(sample_file)

        # 绘制
        ax.plot(
            df["时间戳"][:500], df["压力传感器1"][:500], label="Sensor 1", alpha=0.8
        )
        ax.plot(
            df["时间戳"][:500], df["压力传感器2"][:500], label="Sensor 2", alpha=0.8
        )

        cat_name_en = categories_map[cat_name]
        ax.set_title(f"{cat_name_en} (Sample: {sample_file.stem})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (Pa)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment/test/generate/final_samples.png", dpi=150)
    plt.show()

    print("\n✓ 所有阶段完成！")
    print("  数据位置: /root/repos/MulitiModal/experiment/data/industrial")
    print("  可视化: final_samples.png")
