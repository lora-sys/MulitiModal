"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-008 假设（由受试者级分析驱动）：评估指标是 MAE，而 Ridge 拟合的是 MSE；
受试者误差高度异质（top10% 占约 20% 总误差），高误差受试者会拉动最小二乘。
改用 Huber 稳健损失（IRLS 对大残差降权）拟合同一 9 维特征子集，delta 只由
训练折残差决定；所有预处理与拟合都只使用训练折数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGE = 100.0
IRLS_STEPS = 6


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


def huber_fit(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Huber IRLS，多输出；delta 由训练折残差中位数决定。"""
    columns = features.shape[1]
    center = np.median(y, axis=0)
    target = y - center
    eye = RIDGE * np.eye(columns)
    gram = features.T @ features + eye
    coefficients = np.linalg.solve(gram, features.T @ target)
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
    coefficients, center = huber_fit(transform(train_x, train_mask, preprocess), y)
    output = transform(predict_x, predict_mask, preprocess) @ coefficients + center
    Path(args.output).write_text(json.dumps(output.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
