"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-011 假设（由受试者级异质性驱动）：线性/H 稳健模型都假设全局函数形式；
改用 k 近邻局部回归，对 9 维标准化特征取 k 个最近训练受试者的**逐维中位数**
（中位数直接对齐 MAE）。k 由训练折内 5 折 CV 挑选，所有距离统计量、填补值与
选择都只使用训练折数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INNER_FOLDS = 5
K_GRID = (5, 10, 20, 30, 40, 60)


def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [[np.nan if value is None else float(value) for value in row["features"]]
         for row in records], dtype=float,
    )
    mask = np.array([row["feature_mask"] for row in records], dtype=bool)
    values[~mask] = np.nan
    return values, mask


def build_features(x: np.ndarray, mask: np.ndarray, stats) -> np.ndarray:
    medians, mean, scale = stats
    filled = np.where(mask, x, medians)
    return np.concatenate([(filled - mean) / scale, (~mask).astype(float)], axis=1)


def fit_stats(x: np.ndarray, mask: np.ndarray):
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(mask, x, medians)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    scale = np.where(std > 1e-9, std, 1.0)
    return medians, mean, scale


def knn_predict(train_f, train_y, test_f, k: int) -> np.ndarray:
    # 欧氏距离，按可用特征维度归一（特征数固定）
    diff = test_f[:, None, :] - train_f[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    neighbours = np.argsort(distances, axis=1)[:, :k]
    return np.median(train_y[neighbours], axis=1)


def choose_k(train_f, y, seed: int) -> int:
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, INNER_FOLDS)
    losses = {k: 0.0 for k in K_GRID}
    for part in parts:
        keep = np.ones(n, dtype=bool)
        keep[part] = False
        inner_f, inner_y = train_f[keep], y[keep]
        val_f, val_y = train_f[part], y[part]
        diff = val_f[:, None, :] - inner_f[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        order_nn = np.argsort(distances, axis=1)
        for k in K_GRID:
            pred = np.median(inner_y[order_nn[:, :k]], axis=1)
            losses[k] += float(np.abs(pred - val_y).sum())
    return min(K_GRID, key=lambda k: (losses[k], K_GRID.index(k)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    rows = training["rows"]
    n_dims = len(training["score_dims"])
    if n_dims != 8 or any(len(row["scores"]) != n_dims for row in rows):
        raise ValueError("candidate expects exactly 8 observed constitution scores")

    train_x, train_mask = matrix(rows)
    predict_x, predict_mask = matrix(inputs["rows"])
    keep = slice(0, 9)  # 冠军特征子集：人口学 6 + 人体测量 3
    train_x, train_mask = train_x[:, keep], train_mask[:, keep]
    predict_x, predict_mask = predict_x[:, keep], predict_mask[:, keep]
    y = np.array([row["scores"] for row in rows], dtype=float)

    stats = fit_stats(train_x, train_mask)
    train_f = build_features(train_x, train_mask, stats)
    test_f = build_features(predict_x, predict_mask, stats)
    k = choose_k(train_f, y, args.seed)
    output = knn_predict(train_f, y, test_f, k)
    Path(args.output).write_text(json.dumps(output.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
