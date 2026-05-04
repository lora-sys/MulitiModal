#!/usr/bin/env python3
"""
LLM 标签质量验证实验
从 FT-Transformer 训练数据中随机抽取 500 条样本，
比较 LLM 生成的标签与 FT-Transformer 预测的一致性。

运行方式：
  cd /path/to/MulitiModal
  python scripts/validate_llm_labels.py

输出：
  - Top-1 体质类型一致率
  - 9 维概率分布的平均 Pearson 相关系数
  - 每个体质维度的单独 Pearson r
  - 混淆矩阵（Top-1）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

# 添加项目路径
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tcm_ft_transformer"))

from ft_transformer import FTTransformer
from config import DATA_CONFIG

# ============================================================
# 配置
# ============================================================
N_SAMPLE = 500           # 抽样数量
RANDOM_SEED = 42         # 随机种子
CHECKPOINT_PATH = PROJ / "tcm_ft_transformer" / "checkpoints" / "best_model.pth"
SCALER_PATH = PROJ / "tcm_ft_transformer" / "scaler_params.npz"
DATA_PATH = PROJ / "vital_signs_dataset_final.csv"

CONSTITUTION_NAMES = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 只用 20 条样本快速验证")
    args = parser.parse_args()

    n_sample = 20 if args.smoke else N_SAMPLE

    print("=" * 60)
    print(f"LLM 标签质量验证实验{' (SMOKE)' if args.smoke else ''}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------------
    print(f"\n[1] 加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"    总样本数: {len(df)}")

    n_features = DATA_CONFIG["n_features"]
    n_classes = DATA_CONFIG["n_classes"]

    X = df.iloc[:, :n_features].values.astype(np.float32)
    y_llm = df.iloc[:, -n_classes:].values.astype(np.float32)  # LLM 生成的标签

    # 标签归一化（与训练时一致：epsilon 平滑 + 行归一化）
    epsilon = 0.01
    y_llm_smooth = y_llm + epsilon
    y_llm_norm = y_llm_smooth / y_llm_smooth.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------
    # 2. 加载标准化参数
    # ----------------------------------------------------------
    print(f"\n[2] 加载标准化参数: {SCALER_PATH}")
    scaler = np.load(SCALER_PATH)
    X_scaled = (X - scaler["mean"]) / scaler["std"]

    # ----------------------------------------------------------
    # 3. 抽样
    # ----------------------------------------------------------
    print(f"\n[3] 随机抽样 {n_sample} 条 (seed={RANDOM_SEED})")
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X_scaled[sample_idx]
    y_llm_sample = y_llm_norm[sample_idx]

    # ----------------------------------------------------------
    # 4. 加载 FT-Transformer 并推理
    # ----------------------------------------------------------
    print(f"\n[4] 加载 FT-Transformer: {CHECKPOINT_PATH}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.3
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"    模型加载完成, 设备: {device}")

    # 推理
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).to(device)
        y_pred = model(X_tensor).cpu().numpy()  # (N_SAMPLE, 9)

    # ----------------------------------------------------------
    # 5. 计算指标
    # ----------------------------------------------------------
    print(f"\n[5] 计算一致性指标...")

    # Top-1 一致率
    top1_llm = np.argmax(y_llm_sample, axis=1)
    top1_pred = np.argmax(y_pred, axis=1)
    top1_acc = np.mean(top1_llm == top1_pred)
    print(f"\n    Top-1 体质类型一致率: {top1_acc:.4f} ({top1_acc*100:.1f}%)")

    # 逐维度 Pearson r
    print(f"\n    逐维度 Pearson 相关系数:")
    pearson_per_dim = []
    for i in range(n_classes):
        r, p = pearsonr(y_llm_sample[:, i], y_pred[:, i])
        pearson_per_dim.append(r)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"      {CONSTITUTION_NAMES[i]:>4s}: r = {r:.4f} {sig}")

    # 整体平均 Pearson r
    mean_pearson = np.mean(pearson_per_dim)
    print(f"\n    平均 Pearson r (9维): {mean_pearson:.4f}")

    # 整体分布的 KL 散度（可选）
    from scipy.stats import entropy
    kl_divs = []
    for i in range(n_sample):
        kl = entropy(y_llm_sample[i], y_pred[i])
        kl_divs.append(kl)
    mean_kl = np.mean(kl_divs)
    print(f"    平均 KL(LLM || Pred): {mean_kl:.4f}")

    # ----------------------------------------------------------
    # 6. 混淆矩阵（Top-1）
    # ----------------------------------------------------------
    print(f"\n    Top-1 混淆矩阵:")
    cm = confusion_matrix(top1_llm, top1_pred, labels=list(range(n_classes)))
    print(f"    {'':>6s}", end="")
    for i in range(n_classes):
        print(f" {CONSTITUTION_NAMES[i]:>4s}", end="")
    print()
    for i in range(n_classes):
        print(f"    {CONSTITUTION_NAMES[i]:>4s}", end="")
        for j in range(n_classes):
            print(f" {cm[i,j]:>4d}", end="")
        print()

    # ----------------------------------------------------------
    # 7. 保存结果
    # ----------------------------------------------------------
    out_dir = PROJ / "paper" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_label_validation.json"

    import json
    result = {
        "n_sample": n_sample,
        "random_seed": RANDOM_SEED,
        "top1_accuracy": float(top1_acc),
        "mean_pearson_r": float(mean_pearson),
        "pearson_per_dim": {
            CONSTITUTION_NAMES[i]: float(pearson_per_dim[i])
            for i in range(n_classes)
        },
        "mean_kl_divergence": float(mean_kl),
        "confusion_matrix": cm.tolist()
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n    结果已保存: {out_path}")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
