"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-007 假设（由逐折/逐维/受试者级分析驱动）：冠军相对基线的改进高度异质
（151/274 改善，平均 |delta|=1.82 远大于平均 delta=-0.216），且逐维差异明显
（阴虚/气郁改善、阳虚略退）。因此不再用单一全局模型，而是在**训练折内部**
再做 5 折 CV，为每个维度单独在「训练折中位数基线」与「岭回归预测」之间挑选
收缩权重 alpha；所有预处理与权重选择都只使用训练折数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGE = 100.0
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
INNER_FOLDS = 5


def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [[np.nan if value is None else float(value) for value in row["features"]]
         for row in records], dtype=float,
    )
    mask = np.array([row["feature_mask"] for row in records], dtype=bool)
    values[~mask] = np.nan
    return values, mask


def fit_preprocess(x: np.ndarray, mask: np.ndarray):
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(mask, x, medians)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    scale = np.where(std > 1e-9, std, 1.0)
    return medians, mean, scale


def transform(x: np.ndarray, mask: np.ndarray, preprocess) -> np.ndarray:
    medians, mean, scale = preprocess
    filled = np.where(mask, x, medians)
    return np.concatenate([(filled - mean) / scale, (~mask).astype(float)], axis=1)


def fit_ridge(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (系数, 每维中位数基线)，全部由传入的训练行拟合。"""
    center = np.median(y, axis=0)
    gram = features.T @ features + RIDGE * np.eye(features.shape[1])
    coefficients = np.linalg.solve(gram, features.T @ (y - center))
    return coefficients, center


def predict(features: np.ndarray, coefficients: np.ndarray,
            center: np.ndarray) -> np.ndarray:
    return features @ coefficients + center


def select_alphas(train_x, train_mask, y, seed: int) -> np.ndarray:
    """训练折内部 5 折 CV：逐维挑选 median→ridge 的收缩权重。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, INNER_FOLDS)
    losses = np.zeros((len(ALPHA_GRID), y.shape[1]))
    for part in parts:
        mask = np.ones(n, dtype=bool)
        mask[part] = False
        inner_x, inner_m, inner_y = train_x[mask], train_mask[mask], y[mask]
        val_x, val_m, val_y = train_x[part], train_mask[part], y[part]
        preprocess = fit_preprocess(inner_x, inner_m)
        coefficients, center = fit_ridge(transform(inner_x, inner_m, preprocess), inner_y)
        ridge_pred = predict(transform(val_x, val_m, preprocess), coefficients, center)
        median_pred = np.broadcast_to(np.median(inner_y, axis=0), val_y.shape)
        for index, alpha in enumerate(ALPHA_GRID):
            blended = (1.0 - alpha) * median_pred + alpha * ridge_pred
            losses[index] += np.abs(blended - val_y).sum(axis=0)
    # 平手时偏好更小的 alpha（更稳健的中位数端）
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

    preprocess = fit_preprocess(train_x, train_mask)
    coefficients, center = fit_ridge(transform(train_x, train_mask, preprocess), y)
    ridge_pred = predict(transform(predict_x, predict_mask, preprocess), coefficients, center)
    median_pred = np.broadcast_to(np.median(y, axis=0), (len(predict_x), n_dims))
    alphas = select_alphas(train_x, train_mask, y, args.seed)
    output = (1.0 - alphas) * median_pred + alphas * ridge_pred
    Path(args.output).write_text(json.dumps(output.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
