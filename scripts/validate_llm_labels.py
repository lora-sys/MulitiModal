#!/usr/bin/env python3
"""
LLM 标签质量验证实验
从 FT-Transformer 训练数据中随机抽取 500 条样本，
比较 LLM 生成的标签与 FT-Transformer 预测的一致性。

运行方式：
  cd /path/to/MulitiModal
  python scripts/validate_llm_labels.py
  python scripts/validate_llm_labels.py --smoke   # 冒烟测试 (20 条)

输出：
  - Top-1 体质类型一致率
  - 9 维概率分布的平均 Pearson 相关系数
  - 每个体质维度的单独 Pearson r
  - 混淆矩阵（Top-1）
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr, entropy
from sklearn.metrics import confusion_matrix

# 添加项目路径
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tcm_ft_transformer"))

from ft_transformer import FTTransformer
from config import DATA_CONFIG

# ============================================================
# 配置
# ============================================================
N_SAMPLE = 500
RANDOM_SEED = 42

CONSTITUTION_NAMES = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"
]


def _find_file(filename: str, search_dirs: list[Path]) -> Path | None:
    """在多个目录中搜索文件，返回第一个找到的路径。"""
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
        # 也搜索子目录 (最多 2 层)
        for sub in d.iterdir():
            if sub.is_dir():
                candidate = sub / filename
                if candidate.exists():
                    return candidate
                for sub2 in sub.iterdir():
                    if sub2.is_dir():
                        candidate = sub2 / filename
                        if candidate.exists():
                            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 只用 20 条样本快速验证")
    parser.add_argument("--data-path", type=str, default="", help="vital_signs_dataset_final.csv 路径")
    parser.add_argument("--checkpoint", type=str, default="", help="FT-Transformer checkpoint 路径")
    parser.add_argument("--scaler-path", type=str, default="", help="scaler_params.npz 路径")
    args = parser.parse_args()

    n_sample = 20 if args.smoke else N_SAMPLE

    # --- 自动搜索文件 ---
    search_dirs = [PROJ, PROJ / "tcm_ft_transformer", PROJ / "tcm_ft_transformer" / "data",
                   PROJ / "checkpoints", PROJ / "tcm_ft_transformer" / "checkpoints"]

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = _find_file("vital_signs_dataset_final.csv", search_dirs)
    if data_path is None or not data_path.exists():
        print("ERROR: 找不到 vital_signs_dataset_final.csv")
        print("请用 --data-path 指定路径")
        sys.exit(1)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # 优先找 4 维模型 (best_tcm_model.pth)，再 fallback 到 best_model.pth
        checkpoint_path = _find_file("best_tcm_model.pth", search_dirs)
        if checkpoint_path is None:
            checkpoint_path = _find_file("best_model.pth", search_dirs)
    if checkpoint_path is None or not checkpoint_path.exists():
        print("ERROR: 找不到 FT-Transformer checkpoint")
        print("请用 --checkpoint 指定路径")
        sys.exit(1)

    if args.scaler_path:
        scaler_path = Path(args.scaler_path)
    else:
        scaler_path = _find_file("scaler_params.npz", search_dirs)
    if scaler_path is None or not scaler_path.exists():
        print("ERROR: 找不到 scaler_params.npz")
        print("请用 --scaler-path 指定路径")
        sys.exit(1)

    print("=" * 60)
    print(f"LLM 标签质量验证实验{' (SMOKE)' if args.smoke else ''}")
    print(f"  数据: {data_path}")
    print(f"  模型: {checkpoint_path}")
    print(f"  Scaler: {scaler_path}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------------
    print(f"\n[1] 加载数据...")
    df = pd.read_csv(data_path)
    print(f"    总样本数: {len(df)}")

    n_features = DATA_CONFIG["n_features"]  # 4: Age, Gender, BMI, HR
    n_classes = DATA_CONFIG["n_classes"]    # 9

    # Encode string columns (e.g. Gender: 'Male'→1, 'Female'→0)
    df_feat = df.iloc[:, :n_features].copy()
    for col in df_feat.columns:
        if df_feat[col].dtype == object:
            # 与训练代码一致: Male→0, Female→1
            df_feat[col] = df_feat[col].astype(str).str.strip().str.lower().map(
                lambda v: 0.0 if v.startswith("m") else (1.0 if v.startswith("f") else np.nan)
            )
            df_feat[col] = df_feat[col].fillna(0.0)

    X = df_feat.values.astype(np.float32)
    y_llm = df.iloc[:, -n_classes:].values.astype(np.float32)

    # 标签归一化（与训练时一致：epsilon 平滑 + 行归一化）
    epsilon = 0.01
    y_llm_smooth = y_llm + epsilon
    y_llm_norm = y_llm_smooth / y_llm_smooth.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------
    # 2. 加载标准化参数
    # ----------------------------------------------------------
    print(f"\n[2] 加载标准化参数...")
    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)[:n_features]
    std = scaler["std"].astype(np.float32)[:n_features]
    std = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std

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
    print(f"\n[4] 加载 FT-Transformer...")
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        y_pred = model(X_tensor).cpu().numpy()  # (n_sample, 9)

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

    # KL 散度
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
