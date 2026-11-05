"""第二阶段可编辑候选：八维体质评分回归。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。平和质完全未观测，
输出必须是 (n, 8)，绝不预测第 9 维。

round-002 假设：全部 63 维特征在 274 人上过拟合；只保留人口学与人体测量
9 维（mendeley.FEATURE_GROUPS 的前两组），岭系数与折内预处理保持不变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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

    def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        values = np.array(
            [[np.nan if value is None else float(value) for value in row["features"]]
             for row in records], dtype=float,
        )
        mask = np.array([row["feature_mask"] for row in records], dtype=bool)
        values[~mask] = np.nan
        return values, mask

    train_x, train_mask = matrix(rows)
    predict_x, predict_mask = matrix(inputs["rows"])
    # 只用前 9 维（人口学 6 + 人体测量 3），降低容量
    selected = slice(0, 9)
    train_x, train_mask = train_x[:, selected], train_mask[:, selected]
    predict_x, predict_mask = predict_x[:, selected], predict_mask[:, selected]
    y = np.array([row["scores"] for row in rows], dtype=float)

    # 折内拟合：中位数填补 + 标准化 + 缺失指示（统计量只来自训练折）
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled_train = np.where(train_mask, train_x, medians)
    mean = filled_train.mean(axis=0)
    scale = np.where(filled_train.std(axis=0) > 1e-9, filled_train.std(axis=0), 1.0)

    def transform(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = np.where(mask, x, medians)
        return np.concatenate([(filled - mean) / scale, (~mask).astype(float)], axis=1)

    features = transform(train_x, train_mask)
    test_features = transform(predict_x, predict_mask)
    ridge = 30.0
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    coefficients = np.linalg.solve(gram, features.T @ (y - y.mean(axis=0)))
    predictions = test_features @ coefficients + y.mean(axis=0)
    Path(args.output).write_text(json.dumps(predictions.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
