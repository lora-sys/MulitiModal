"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-012 假设（由模型对比驱动）：Huber 全局线性（14.1773）与 kNN 局部中位数
（14.2081）都优于基线，且归纳偏置互补。在**训练折内部**做 5 折 CV，为每一维
在两个模型的逐维 MAE 之间挑选混合权重 alpha；所有统计量、k、权重与拟合都
只使用训练折数据，验证折标签从不参与。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGE = 100.0
IRLS_STEPS = 6
INNER_FOLDS = 5
K_GRID = (5, 10, 20, 30, 40, 60)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [[np.nan if value is None else float(value) for value in row["features"]]
         for row in records], dtype=float,
    )
    mask = np.array([row["feature_mask"] for row in records], dtype=bool)
    values[~mask] = np.nan
    return values, mask


def fit_stats(x: np.ndarray, mask: np.ndarray):
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(mask, x, medians)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    scale = np.where(std > 1e-9, std, 1.0)
    return medians, mean, scale


def design(x: np.ndarray, mask: np.ndarray, stats) -> np.ndarray:
    medians, mean, scale = stats
    filled = np.where(mask, x, medians)
    return np.concatenate([(filled - mean) / scale, (~mask).astype(float)], axis=1)


def huber_fit(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eye = RIDGE * np.eye(features.shape[1])
    center = np.median(y, axis=0)
    target = y - center
    coefficients = np.linalg.solve(features.T @ features + eye, features.T @ target)
    residual = np.abs(target - features @ coefficients)
    delta = max(float(np.median(residual)), 1.0)
    for _ in range(IRLS_STEPS):
        weight = np.where(residual <= delta, 1.0,
                          delta / np.maximum(residual, 1e-9))
        for j in range(y.shape[1]):
            w = weight[:, j]
            weighted = features * np.sqrt(w)[:, None]
            coefficients[:, j] = np.linalg.solve(
                weighted.T @ weighted + eye,
                features.T @ (w * target[:, j]),
            )
        residual = np.abs(target - features @ coefficients)
    return coefficients, center


def knn_predict(train_f, train_y, test_f, k: int) -> np.ndarray:
    diff = test_f[:, None, :] - train_f[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    neighbours = np.argsort(distances, axis=1)[:, :k]
    return np.median(train_y[neighbours], axis=1)


def fit_both_with_y(x, mask, y, seed: int):
    stats = fit_stats(x, mask)
    features = design(x, mask, stats)
    coefficients, center = huber_fit(features, y)
    k = choose_k(features, y, seed)
    return {"stats": stats, "coefficients": coefficients, "center": center,
            "train_features": features, "train_y": y, "k": k}


def predict_both(state, x, mask):
    features = design(x, mask, state["stats"])
    huber = features @ state["coefficients"] + state["center"]
    knn = knn_predict(state["train_features"], state["train_y"], features, state["k"])
    return huber, knn


def choose_k(features, y, seed: int) -> int:
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, INNER_FOLDS)
    losses = {k: 0.0 for k in K_GRID}
    for part in parts:
        keep = np.ones(n, dtype=bool)
        keep[part] = False
        inner_f, inner_y = features[keep], y[keep]
        val_f, val_y = features[part], y[part]
        diff = val_f[:, None, :] - inner_f[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        order_nn = np.argsort(distances, axis=1)
        for k in K_GRID:
            pred = np.median(inner_y[order_nn[:, :k]], axis=1)
            losses[k] += float(np.abs(pred - val_y).sum())
    return min(K_GRID, key=lambda k: (losses[k], K_GRID.index(k)))


def choose_alphas(x, mask, y, seed: int) -> np.ndarray:
    """训练折内 5 折 CV：逐维挑选 huber<-knn 混合权重。"""
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, INNER_FOLDS)
    losses = np.zeros((len(ALPHA_GRID), y.shape[1]))
    for part in parts:
        keep = np.ones(n, dtype=bool)
        keep[part] = False
        state = fit_both_with_y(x[keep], mask[keep], y[keep], seed)
        huber, knn = predict_both(state, x[part], mask[part])
        for index, alpha in enumerate(ALPHA_GRID):
            blended = alpha * huber + (1.0 - alpha) * knn
            losses[index] += np.abs(blended - y[part]).sum(axis=0)
    return np.array([ALPHA_GRID[int(np.argmin(losses[:, j]))] for j in range(y.shape[1])])


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

    alphas = choose_alphas(train_x, train_mask, y, args.seed)
    state = fit_both_with_y(train_x, train_mask, y, args.seed)
    huber, knn = predict_both(state, predict_x, predict_mask)
    output = alphas * huber + (1.0 - alphas) * knn
    Path(args.output).write_text(json.dumps(output.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
