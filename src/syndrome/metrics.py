"""评估指标（#53 Scope 3）：macro-F1 主口径、逐类 P/R/F1、Top-k、bootstrap 置信区间。

全部指标自实现（numpy），避免 sklearn zero_division 语义差异；
bootstrap 对样本有放回重采样，适配 42~5486 规模的测试集，
配对差值 bootstrap 用于"模型 vs 朴素基线"的差异报告。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

PerClass = dict[str, float | int | str]


def _confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每类的 tp / fp / fn 计数（向量化，供 bootstrap 反复调用）。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("metric input must not be empty")
    pair = y_true * n_classes + y_pred
    counts = np.bincount(pair, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(counts).astype(np.float64)
    fp = counts.sum(axis=0) - tp
    fn = counts.sum(axis=1) - tp
    return tp, fp, fn


def _prf1(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # zero_division=0 语义：分母为 0 的类记 0（支撑不足如实为 0，不虚构）
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return precision, recall, f1


def per_class_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> list[PerClass]:
    """逐类 support / precision / recall / f1；labels 为冻结词表（决定 macro 口径的分母）。"""
    n = len(labels)
    tp, fp, fn = _confusion_counts(y_true, y_pred, n)
    precision, recall, f1 = _prf1(tp, fp, fn)
    support = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=n)
    return [
        {
            "label": labels[i],
            "support": int(support[i]),
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
        }
        for i in range(n)
    ]


def macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> float:
    """macro-F1：对全部 n_classes 逐类 F1 取算术平均（缺席类 F1=0，zero_division=0）。"""
    _, _, f1 = _prf1(*_confusion_counts(y_true, y_pred, n_classes))
    return float(f1.mean())


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def topk_accuracy(y_true: Sequence[int], scores: np.ndarray, k: int) -> float:
    """Top-k 命中率；scores 形状 (n, C)，值越大越优（logits 或概率均可）。"""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] != len(y_true):
        raise ValueError(f"scores shape {scores.shape} incompatible with {len(y_true)} labels")
    k = min(k, scores.shape[1])
    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def _metric_with_classes(
    metric_fn: Callable[[Sequence[int], Sequence[int]], float], n_classes: int
) -> Callable[[Sequence[int], Sequence[int]], float]:
    """bootstrap 重采样内调用 2 参数 metric_fn；默认 macro_f1 需要绑定 n_classes
    （重采样可能丢失类别，macro 分母必须保持冻结词表大小）。"""
    if metric_fn is macro_f1:
        return lambda a, b: macro_f1(a, b, n_classes)
    return metric_fn


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_classes: int,
    metric_fn: Callable[[Sequence[int], Sequence[int]], float] = macro_f1,
    n_boot: int = 1000,
    seed: int = 20261101,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """对样本有放回重采样的指标置信区间（百分位法）。

    小支持度类（如 support<30）的逐类指标本身不稳定，区间只会如实变宽，不做修正。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric = _metric_with_classes(metric_fn, n_classes)
    point = float(metric(y_true, y_pred))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    alpha = (1 - confidence) / 2
    return {
        "point": round(point, 6),
        "low": round(float(np.quantile(stats, alpha)), 6),
        "high": round(float(np.quantile(stats, 1 - alpha)), 6),
        "confidence": confidence,
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_bootstrap_diff(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    n_classes: int,
    metric_fn: Callable[[Sequence[int], Sequence[int]], float] = macro_f1,
    n_boot: int = 1000,
    seed: int = 20261101,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """模型 A 相对基线 B 的配对差值（同一重采样索引下计算，保留样本配对性）。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric = _metric_with_classes(metric_fn, n_classes)
    point = float(metric(y_true, pred_a) - metric(y_true, pred_b))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    alpha = (1 - confidence) / 2
    low, high = float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))
    # 双侧经验 p 值：差值分布穿越 0 的比例
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "point_diff": round(point, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "p_value": round(float(min(1.0, p_value)), 6),
        "confidence": confidence,
        "n_boot": n_boot,
        "seed": seed,
    }


def summarize_evaluation(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: np.ndarray,
    labels: Sequence[str],
    baseline_pred: Sequence[int] | None = None,
    n_boot: int = 1000,
    seed: int = 20261101,
) -> dict:
    """一次评估的完整报告：macro-F1（含 CI）、Top-3/5、逐类结果、与朴素基线差异。"""
    n_classes = len(labels)
    report: dict = {
        "n_samples": len(y_true),
        "n_classes_total": n_classes,
        "macro_f1": round(macro_f1(y_true, y_pred, n_classes), 6),
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "top3_accuracy": round(topk_accuracy(y_true, scores, 3), 6),
        "top5_accuracy": round(topk_accuracy(y_true, scores, 5), 6),
        "macro_f1_ci": bootstrap_ci(y_true, y_pred, n_classes, n_boot=n_boot, seed=seed),
        "per_class": per_class_metrics(y_true, y_pred, labels),
    }
    covered = {labels[i] for i in y_true}
    report["n_classes_observed"] = len(covered)
    if baseline_pred is not None:
        report["vs_baseline"] = paired_bootstrap_diff(
            y_true, y_pred, baseline_pred, n_classes, n_boot=n_boot, seed=seed
        )
    return report


def majority_class_predictions(y_train: Sequence[int], n_eval: int, n_classes: int) -> list[int]:
    """朴素多数类 Dummy 的预测序列（全体预测训练集最频类）。"""
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=n_classes)
    top = int(counts.argmax())
    return [top] * n_eval
