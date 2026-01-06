import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt


class SignalMaster:
    def __init__(self, window_size=15, sigma=3):
        self.window_size = window_size
        self.sigma = sigma
        self.df = None
        self.metrics = {}

    def load_data(self, filename):
        """加载数据"""
        if not os.path.exists(filename):
            print(f"❌ 错误：文件 {filename} 不存在")
            return False
        self.df = pd.read_csv(filename)
        print(f"📖 数据加载成功: {len(self.df)} 样本")
        return True

    def inject_poison(self, n_spikes=10, seed=42):
        """模拟传感器故障：投毒注入异常脉冲"""
        if self.df is None:
            return
        np.random.seed(seed)
        indices = np.random.choice(self.df.index[50:-50], size=n_spikes, replace=False)
        for idx in indices:
            spike = np.random.uniform(20, 50) * np.random.choice([-1, 1])
            self.df.loc[idx, "noisy_signal"] += spike
        print(f"☣️ 投毒成功: 已注入 {n_spikes} 个异常点")

    def process(self):
        """核心处理流水线：检测 + 修复"""
        if self.df is None:
            return

        # 1. 基础平滑 (用于对比和检测基准)
        self.df["base_ma"] = (
            self.df["noisy_signal"]
            .rolling(window=self.window_size, center=True, min_periods=1)
            .mean()
        )

        # 2. 异常检测 (3-Sigma)
        rolling_std = (
            self.df["noisy_signal"]
            .rolling(window=self.window_size, center=True, min_periods=1)
            .std()
        )

        upper_bound = self.df["base_ma"] + self.sigma * rolling_std
        lower_bound = self.df["base_ma"] - self.sigma * rolling_std

        self.df["is_anomaly"] = (self.df["noisy_signal"] > upper_bound) | (
            self.df["noisy_signal"] < lower_bound
        )

        # 3. 专业修复 (插值 + 最终平滑)
        clean_tmp = self.df["noisy_signal"].copy()
        clean_tmp[self.df["is_anomaly"]] = np.nan

        # 线性插值修复“弹坑”
        self.df["repaired"] = clean_tmp.interpolate(method="linear").ffill().bfill()

        # 最终平滑处理
        self.df["final_signal"] = (
            self.df["repaired"]
            .rolling(window=self.window_size, center=True, min_periods=1)
            .mean()
        )

        self._calculate_all_metrics()
        return self.metrics

    def _calculate_all_metrics(self):
        """内部评估函数"""

        def mse(a, b):
            return np.mean((a - b) ** 2)

        clean = self.df["clean_signal"]
        self.metrics["mse_raw"] = mse(clean, self.df["noisy_signal"])
        self.metrics["mse_ma"] = mse(clean, self.df["base_ma"])
        self.metrics["mse_final"] = mse(clean, self.df["final_signal"])
        self.metrics["improvement"] = (
            (self.metrics["mse_ma"] - self.metrics["mse_final"])
            / self.metrics["mse_ma"]
            * 100
        )
        self.metrics["anomaly_count"] = self.df["is_anomaly"].sum()

    def visualize(self, save_path="final_report.png"):
        """可视化报告"""
        if self.df is None:
            return

        plt.figure(figsize=(15, 12))
        view = slice(200, 700)  # 观察核心区域
        t = self.df["timestamp"][view]

        # 子图1：原始与检测
        plt.subplot(3, 1, 1)
        plt.plot(
            t,
            self.df["noisy_signal"][view],
            color="red",
            alpha=0.2,
            label="Dirty Signal",
        )
        anomalies = self.df[view][self.df[view]["is_anomaly"]]
        plt.scatter(
            anomalies["timestamp"],
            anomalies["noisy_signal"],
            color="red",
            marker="x",
            label="Detected",
        )
        plt.title(f"Detection Phase: {self.metrics['anomaly_count']} Anomalies Found")
        plt.legend()

        # 子图2：修复对比
        plt.subplot(3, 1, 2)
        plt.plot(t, self.df["clean_signal"][view], color="black", lw=2, label="Truth")
        plt.plot(
            t,
            self.df["final_signal"][view],
            color="green",
            lw=1.5,
            label="SignalMaster Recovered",
        )
        plt.title("Repair Phase: Truth vs Recovered")
        plt.legend()

        # 子图3：残差改进
        plt.subplot(3, 1, 3)
        plt.plot(
            t,
            self.df["base_ma"][view] - self.df["clean_signal"][view],
            alpha=0.3,
            label="Simple MA Error",
        )
        plt.plot(
            t,
            self.df["final_signal"][view] - self.df["clean_signal"][view],
            color="green",
            label="Smart Repair Error",
        )
        plt.axhline(0, color="black", ls="--")
        plt.title(
            f"Performance: {self.metrics['improvement']:.2f}% Improvement over Simple Filter"
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def export_results(self, folder="output_results"):
        """一键导出所有中间与最终结果"""
        if self.df is None:
            print("❌ 导出失败：没有可导出的数据")
            return

        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1. 导出完整处理链路表 (用于算法回溯)
        full_path = os.path.join(folder, "full_process_log.csv")
        # 仅选择关键列，保持文件整洁
        cols = [
            "timestamp",
            "clean_signal",
            "noisy_signal",
            "is_anomaly",
            "upper_bound",
            "lower_bound",
            "final_signal",
        ]
        # 过滤掉不存在的列（防止报错）
        valid_cols = [c for c in cols if c in self.df.columns]
        self.df[valid_cols].to_csv(full_path, index=False, float_format="%.6f")

        # 2. 专项导出异常点日志 (用于硬件排查)
        anomaly_path = os.path.join(folder, "anomaly_report.csv")
        anomaly_df = self.df[self.df["is_anomaly"]]
        anomaly_df[valid_cols].to_csv(anomaly_path, index=False, float_format="%.6f")

        # 3. 导出精简后的最终清洗数据 (交付给下游使用)
        final_path = os.path.join(folder, "cleaned_pressure_data.csv")
        self.df[["timestamp", "final_signal"]].to_csv(
            final_path, index=False, float_format="%.6f"
        )

        print(f"📂 数据已成功导出至文件夹: {os.path.abspath(folder)}")
        print(f"   - 完整日志: full_process_log.csv")
        print(f"   - 异常清单: anomaly_report.csv (共 {len(anomaly_df)} 条)")
        print(f"   - 交付数据: cleaned_pressure_data.csv")


# --- 修改示例用法 ---
if __name__ == "__main__":
    master = SignalMaster(window_size=15, sigma=3)

    if master.load_data("pressure_sim.csv"):
        master.inject_poison(n_spikes=12)
        master.process()

        # 新增：导出结果
        master.export_results("simulation_outputs_v1")

        master.visualize()
