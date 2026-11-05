"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-010 假设（由逐维分析驱动）：冠军相对基线的改善逐维差异极大（阴虚 -0.72、
气郁 -0.49、阳虚 +0.11）；round-009 又证明全局加特征组有害。因此不再为所有
维度使用同一特征集，而是在**训练折内部**做 5 折 CV，为每个维度独立地从
「中位数基线」与多个 Huber 特征组中挑选 MAE 最小的候选。所有统计量、权重与
选择都只用训练折数据，验证折标签从不参与。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGE = 100.0
IRLS_STEPS = 6
INNER_FOLDS = 5

# 候选特征组：None 表示纯中位数基线
GROUPS: dict[str, list[int] | None] = {
    "median": None,
    "base9": list(range(0, 9)),
    "demo6": list(range(0, 6)),
    "anthro3": list(range(6, 9)),
    "meridian12": list(range(9, 21)),
    "pulse15": list(range(36, 51)),
    "tongue12": list(range(51, 63)),
}


def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [[np.nan if value is None else float(value) for value in row["features"]]
         for row in records], dtype=float,
    )
    mask = np.array([row["feature_mask"] for row in records], dtype=bool)
    values[~mask] = np.nan
    return values, mask


def preprocess(x: np.ndarray, mask: np.ndarray):
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
    columns = features.shape[1]
    center = np.median(y, axis=0)
    target = y - center
    eye = RIDGE * np.eye(columns)
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


def fit_group(name: str, x, mask, y):
    """在给定训练行上拟合一个候选，返回可用于预测的状态。"""
    columns = GROUPS[name]
    if columns is None:
        return {"median": np.median(y, axis=0)}
    sub_x, sub_mask = x[:, columns], mask[:, columns]
    stats = preprocess(sub_x, sub_mask)
    coefficients, center = huber_fit(design(sub_x, sub_mask, stats), y)
    return {"stats": stats, "columns": columns,
            "coefficients": coefficients, "center": center}


def predict_group(state, x, mask, n_rows: int) -> np.ndarray:
    if "median" in state:
        return np.broadcast_to(state["median"], (n_rows, len(state["median"])))
    columns = state["columns"]
    features = design(x[:, columns], mask[:, columns], state["stats"])
    return features @ state["coefficients"] + state["center"]


def choose_per_dimension(x, mask, y, seed: int) -> list[str]:
    """训练折内 5 折 CV：逐维挑选 MAE 最小的候选组。"""
    names = list(GROUPS)
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, INNER_FOLDS)
    losses = {name: np.zeros(y.shape[1]) for name in names}
    for part in parts:
        keep = np.ones(n, dtype=bool)
        keep[part] = False
        inner_x, inner_m, inner_y = x[keep], mask[keep], y[keep]
        val_x, val_m, val_y = x[part], mask[part], y[part]
        for name in names:
            state = fit_group(name, inner_x, inner_m, inner_y)
            pred = predict_group(state, val_x, val_m, len(part))
            losses[name] += np.abs(pred - val_y).sum(axis=0)
    chosen = []
    for j in range(y.shape[1]):
        # 平手时按 GROUPS 顺序（median 优先），保持确定性
        best = min(names, key=lambda name: (losses[name][j], names.index(name)))
        chosen.append(best)
    return chosen


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
    y = np.array([row["scores"] for row in rows], dtype=float)

    chosen = choose_per_dimension(train_x, train_mask, y, args.seed)
    states = {name: fit_group(name, train_x, train_mask, y)
              for name in set(chosen)}
    n_rows = len(predict_x)
    output = np.empty((n_rows, n_dims), dtype=float)
    for j, name in enumerate(chosen):
        output[:, j] = predict_group(states[name], predict_x, predict_mask, n_rows)[:, j]
    Path(args.output).write_text(json.dumps(output.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
