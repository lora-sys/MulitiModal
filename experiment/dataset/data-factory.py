import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import torch
import yaml
import os
import sys
import re

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# add path
sys.path.append("experiment/dataset")
from csv_source import CSVDataSource, NPZDataSource
from nk2_processor import NK2Preprocessor
from massage_dataset import MassageDataset
from interfaces import Sample


# load npz raw data
npz_path = "experiment/model/processed_data.npz"
with open("experiment/dataset/config.yaml", "r") as f:
    config = yaml.safe_load(f)

preprocessor_params = config["preprocessor"]["params"]
print(f"sampling: {preprocessor_params['sampling_rate']} hz")
print(f"target length: {preprocessor_params['target_length']}")
print(
    f"filter_length: {preprocessor_params['filter']['lowcut']}-{preprocessor_params['filter']['highcut']} hz "
)


# ============================
# 创建数据源 + 预处理器 + Dataset
# ============================
print("\n" + "=" * 50)
print("创建数据 pipeline")
print("=" * 50)

# 1. 创建 npz 数据源
npz_source = NPZDataSource(npz_path)
npz_source.initialize()

# 2. 创建预处理器
preprocessor = NK2Preprocessor(config)

# 3. 创建 Dataset
dataset = MassageDataset(npz_source, preprocessor)

print(f"Dataset 大小: {len(dataset)}")

# ============================
# 获取一个样本测试
# ============================
print("\n" + "=" * 50)
print("获取样本测试")
print("=" * 50)

idx = 0
sample = dataset[idx]

print(f"样本 {idx}:")
print(f"  dynamic shape: {sample['dynamic'].shape}")  # [2, 1000]
print(f"  static shape: {sample['static'].shape}")  # [4]
print(f"  label: {sample['label']}")  # scalar

# ============================
# 可视化处理前后对比 (改进版)
# ============================
print("\n" + "=" * 50)
print("可视化对比 (改进版)")
print("=" * 50)

# 重新加载原始数据用于对比
raw_data = np.load(npz_path)
s1_raw = raw_data["X_dynamic"][idx, 0, :]  # 原始传感器1
s2_raw = raw_data["X_dynamic"][idx, 1, :]  # 原始传感器2
s1_processed = sample["dynamic"][0, :].numpy()  # 处理后传感器1
s2_processed = sample["dynamic"][1, :].numpy()  # 处理后传感器2

# 计算固定Y轴范围（用于对比）
# 方法：将处理后数据反向归一化到原始范围，进行对比
s1_mean, s1_std = np.mean(s1_raw), np.std(s1_raw)
s2_mean, s2_std = np.mean(s2_raw), np.std(s2_raw)

# 将处理后数据转换到原始尺度进行对比
s1_processed_scaled = s1_processed * s1_std + s1_mean
s2_processed_scaled = s2_processed * s2_std + s2_mean

# 固定Y轴范围（取两者的并集）
y1_min = min(s1_raw.min(), s1_processed_scaled.min()) * 1.1
y1_max = max(s1_raw.max(), s1_processed_scaled.max()) * 1.1
y2_min = min(s2_raw.min(), s2_processed_scaled.min()) * 1.1
y2_max = max(s2_raw.max(), s2_processed_scaled.max()) * 1.1

print(
    f"传感器1 原始范围: [{s1_raw.min():.1f}, {s1_raw.max():.1f}], 均值: {s1_mean:.1f}, 标准差: {s1_std:.1f}"
)
print(f"传感器1 处理后范围: [{s1_processed.min():.2f}, {s1_processed.max():.2f}]")
print(
    f"传感器1 处理后(反归一化): [{s1_processed_scaled.min():.1f}, {s1_processed_scaled.max():.1f}]"
)

time = np.linspace(0, 20, len(s1_raw))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== 传感器1: 叠加对比 (固定Y轴) =====
axes[0, 0].plot(time, s1_raw, "r-", alpha=0.7, linewidth=1, label="Raw")
axes[0, 0].plot(
    time, s1_processed_scaled, "b-", alpha=0.7, linewidth=1, label="Processed (scaled)"
)
axes[0, 0].set_title(
    f"Sensor 1 - Overlay Comparison\nRaw: [{s1_raw.min():.0f}, {s1_raw.max():.0f}] | Processed: [{s1_processed.min():.2f}, {s1_processed.max():.2f}]"
)
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Pressure")
axes[0, 0].set_ylim(y1_min, y1_max)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ===== 传感器1: 分离显示 =====
axes[0, 1].plot(time, s1_raw, "r-", alpha=0.6, linewidth=0.8, label="Raw")
ax1_twin = axes[0, 1].twinx()
ax1_twin.plot(time, s1_processed, "b-", linewidth=0.8, label="Processed")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Raw Pressure", color="r")
ax1_twin.set_ylabel("Normalized Pressure", color="b")
axes[0, 1].set_title(
    f"Sensor 1 - Separate Display\nRaw mean={s1_mean:.1f}, std={s1_std:.1f}"
)
axes[0, 1].grid(True, alpha=0.3)
lines1, labels1 = axes[0, 1].get_legend_handles_labels()
lines2, labels1 = ax1_twin.get_legend_handles_labels()
axes[0, 1].legend(lines1 + lines2, ["Raw", "Processed"], loc="upper right")

# ===== 传感器2: 叠加对比 (固定Y轴) =====
axes[1, 0].plot(time, s2_raw, "r-", alpha=0.7, linewidth=1, label="Raw")
axes[1, 0].plot(
    time, s2_processed_scaled, "b-", alpha=0.7, linewidth=1, label="Processed (scaled)"
)
axes[1, 0].set_title(
    f"Sensor 2 - Overlay Comparison\nRaw: [{s2_raw.min():.0f}, {s2_raw.max():.0f}] | Processed: [{s2_processed.min():.2f}, {s2_processed.max():.2f}]"
)
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Pressure")
axes[1, 0].set_ylim(y2_min, y2_max)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ===== 传感器2: 分离显示 =====
axes[1, 1].plot(time, s2_raw, "r-", alpha=0.6, linewidth=0.8, label="Raw")
ax2_twin = axes[1, 1].twinx()
ax2_twin.plot(time, s2_processed, "b-", linewidth=0.8, label="Processed")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Raw Pressure", color="r")
ax2_twin.set_ylabel("Normalized Pressure", color="b")
axes[1, 1].set_title(
    f"Sensor 2 - Separate Display\nRaw mean={s2_mean:.1f}, std={s2_std:.1f}"
)
axes[1, 1].grid(True, alpha=0.3)
lines3, labels3 = axes[1, 1].get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
axes[1, 1].legend(lines3 + lines4, ["Raw", "Processed"], loc="upper right")

label_names = {0: "很差", 1: "一般", 2: "正常", 3: "良好"}
label_val = int(sample["label"].item())
plt.suptitle(
    f"Sample idx={idx} | Label={label_val}({label_names[label_val]}) | Static: {sample['static'].numpy()}",
    fontsize=12,
)
plt.tight_layout()

# 保存图片
output_img = "experiment/test/npz_visualization.png"
plt.savefig(output_img, dpi=150, bbox_inches="tight")
print(f"✅ 可视化图片已保存: {output_img}")

# ============================
# 批量处理所有数据并保存
# ============================
print("\n" + "=" * 50)
print("批量处理并保存")
print("=" * 50)

# 从 config 获取参数
num_channels = config["preprocessor"]["params"].get("num_channels", 2)
num_static_features = config["preprocessor"]["params"].get("num_static_features", 4)
target_length = config["preprocessor"]["params"].get("target_length", 1000)

print(
    f"配置: channels={num_channels}, static_features={num_static_features}, length={target_length}"
)

n_samples = len(dataset)
processed_dynamic = np.zeros((n_samples, num_channels, target_length))
processed_static = np.zeros((n_samples, num_static_features))
processed_labels = np.zeros(n_samples)

for i in range(n_samples):
    item = dataset[i]
    processed_dynamic[i] = item["dynamic"].numpy()
    processed_static[i] = item["static"].numpy()
    processed_labels[i] = item["label"].item()

    if (i + 1) % 200 == 0:
        print(f"已处理: {i + 1}/{n_samples}")

# 保存处理后的数据
output_npz = "experiment/model/processed_data_nk2.npz"
np.savez(
    output_npz,
    X_dynamic=processed_dynamic,
    X_static=processed_static,
    Y=processed_labels,
)

print(f"✅ 处理后数据已保存: {output_npz}")
print(
    f"   shape: X_dynamic={processed_dynamic.shape}, X_static={processed_static.shape}, Y={processed_labels.shape}"
)
