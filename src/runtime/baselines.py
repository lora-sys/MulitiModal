"""Fair baselines for the #50 real-data TCM evaluation.

Dummy majority, BMI-only logistic and full-feature logistic regression —
all fit on the frozen TRAIN partition only (fold-only preprocessing, #29),
evaluated on the same partitions and label contract as the neural model so
the comparison is fair. Complex-network scores are never accepted as
evidence of clinical validity; these baselines exist to show whether the
network learned anything beyond body-size proxies (#50).
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from data.mendeley import SubjectRecord
from data.real_split import derive_classification_targets
from labels.standards import CANONICAL_ORDER as CANONICAL_DIMS


class _LogisticPerDim(BaseEstimator, ClassifierMixin):
    """LogisticRegression that falls back to the constant class when a
    dimension has a single class in the training fold (rare dimensions)."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y) -> _LogisticPerDim:
        from sklearn.linear_model import LogisticRegression

        self._lr = None
        self.classes_ = np.unique(y)
        if self.classes_.size > 1:
            self._lr = LogisticRegression(
                C=self.C, max_iter=self.max_iter
            ).fit(X, y)
        return self

    def predict_proba(self, X):
        if self._lr is None:
            out = np.zeros((len(X), 2))
            out[:, int(self.classes_[0])] = 1.0
            return out
        return self._lr.predict_proba(X)

    def predict(self, X):
        if self._lr is None:
            return np.full(len(X), self.classes_[0])
        return self._lr.predict(X)


def _scaled_vector(
    rec: SubjectRecord, manifest: dict, modality: str
) -> tuple[list[float], bool]:
    groups = manifest["modalities"][modality]["feature_groups"]
    values = [v for g in groups for v in rec.features[g]]
    params = manifest["scaler"][modality]
    mean, std, degenerate = params["mean"], params["std"], params["degenerate"]
    ok = not any(math.isnan(v) for v in values)
    return (
        [
            0.0 if d or not ok else (v - m) / s
            for v, m, s, d in zip(values, mean, std, degenerate, strict=True)
        ],
        ok,
    )


def _design_matrix(
    records_by_row: dict, manifest: dict, subject_ids: tuple[str, ...], features: str
) -> tuple[np.ndarray, list[int]]:
    """Standardized feature matrix using the manifest's train-fold scaler.

    features: 'full' (all manifest modalities + one availability bit per
    modality — the same missing-modality information the neural model
    receives, so the comparison is fair) or 'bmi_only' (the standardized
    BMI column + the body_features availability bit).
    """
    rows: list[list[float]] = []
    rows_index: list[int] = []
    body_dim = manifest["modalities"]["body_features"]["dim"]
    for sid in subject_ids:
        row = manifest["subject_row_by_id"][sid]
        rec: SubjectRecord = records_by_row[row]
        if features == "bmi_only":
            body, ok = _scaled_vector(rec, manifest, "body_features")
            vector = [body[body_dim - 1], 1.0 if ok else 0.0]  # BH, BW, BMI -> BMI last
        else:
            vector = []
            bits = []
            for name in manifest["modalities"]:
                scaled, ok = _scaled_vector(rec, manifest, name)
                vector.extend(scaled)
                bits.append(1.0 if ok else 0.0)
            vector.extend(bits)
        rows.append(vector)
        rows_index.append(row)
    return np.asarray(rows, dtype=np.float64), rows_index


def _labels(records_by_row: dict, rows_index: list[int]) -> tuple[np.ndarray, np.ndarray]:
    y: list[list[float]] = []
    mask: list[list[bool]] = []
    for row in rows_index:
        rec = records_by_row[row]
        labels, m = derive_classification_targets(rec.scores, rec.score_mask)
        y.append(labels)
        mask.append(m)
    return np.asarray(y), np.asarray(mask)


def _per_dim_f1(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    """Binary F1 per dim over observed entries only; None when undefined."""
    out = {}
    for d, dim in enumerate(CANONICAL_DIMS):
        obs = mask[:, d]
        t, p = y_true[obs, d], y_pred[obs, d]
        tp = float(((t == 1) & (p == 1)).sum())
        fp = float(((t == 0) & (p == 1)).sum())
        fn = float(((t == 1) & (p == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        out[dim] = {
            "f1": f1,
            "support_positive": int((t == 1).sum()),
            "n_observed": int(obs.sum()),
            "positive_rate": float(t.mean()) if obs.any() else None,
        }
    return out


def macro_f1(per_dim: dict) -> float | None:
    """Macro-F1 over the FIXED set of observed dimensions: an undefined F1
    (no positive and no predicted positive) counts as 0.0, so the averaging
    denominator cannot drift between models and reward models that simply
    never predict a rare dimension (#eval E02)."""
    observed = [v for v in per_dim.values() if v["n_observed"] > 0]
    if not observed:
        return None
    return sum(v["f1"] or 0.0 for v in observed) / len(observed)


def count_undefined_f1(per_dim: dict) -> int:
    """Observed dimensions whose F1 is undefined (reported alongside macro)."""
    return sum(
        1 for v in per_dim.values() if v["n_observed"] > 0 and v["f1"] is None
    )


def subject_mae_from_proba(
    proba: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> float | None:
    """Subject-level MAE of probability outputs against 0/1 labels, same
    aggregation as TCMTrustedEvaluator (per subject over observed dims)."""
    obs = mask.astype(bool)
    rows = obs.any(axis=1)
    if not rows.any():
        return None
    diffs = np.abs(proba - y) * obs
    per_subject = diffs.sum(axis=1) / np.maximum(obs.sum(axis=1), 1)
    return float(per_subject[rows].mean())


def subject_rmse_from_proba(
    proba: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> float | None:
    """Subject-level RMSE (= Brier score up to a monotone root), the same
    aggregation as TCMTrustedEvaluator. This is the AutoResearch selection
    metric; it is a proper scoring rule, so a base-rate constant does not
    automatically beat a real model the way it does under L1."""
    obs = mask.astype(bool)
    rows = obs.any(axis=1)
    if not rows.any():
        return None
    sq = ((proba - y) ** 2) * obs
    per_subject = sq.sum(axis=1) / np.maximum(obs.sum(axis=1), 1)
    return float(np.sqrt(per_subject[rows].mean()))


def _proba_matrix(model, X: np.ndarray, n_dims: int) -> np.ndarray:
    """Column of positive-class probabilities per dim from a
    MultiOutputClassifier (its predict_proba returns per-output arrays)."""
    cols = []
    for d in range(n_dims):
        est = model.estimators_[d]
        if len(est.classes_) == 1:
            cols.append(np.full(len(X), float(est.classes_[0])))
            continue
        col = 1 if est.classes_[1] == 1 else 0
        cols.append(est.predict_proba(X)[:, col])
    return np.stack(cols, axis=1)


def evaluate_baselines(manifest: dict, records: list) -> dict:
    """Fit the three baselines on TRAIN, evaluate F1 on dev and holdout."""
    from sklearn.dummy import DummyClassifier
    from sklearn.multioutput import MultiOutputClassifier

    records_by_row = {r.subject_index: r for r in records}
    result: dict = {"label_version": manifest["label_version"]}
    for features in ("bmi_only", "full"):
        X_train, rows_train = _design_matrix(
            records_by_row, manifest, manifest["split"]["train"], features
        )
        y_train, _ = _labels(records_by_row, rows_train)
        models = {}
        if features == "bmi_only":
            models["dummy_majority"] = MultiOutputClassifier(
                DummyClassifier(strategy="most_frequent")
            )
            models["bmi_only_logistic"] = MultiOutputClassifier(_LogisticPerDim())
        else:
            models["feature_logistic"] = MultiOutputClassifier(
                _LogisticPerDim(max_iter=5000)
            )
        for name, model in models.items():
            model.fit(X_train, y_train)
            entry: dict = {"features": features}
            for partition in ("dev", "holdout"):
                X, rows = _design_matrix(
                    records_by_row, manifest, manifest["split"][partition], features
                )
                y, mask = _labels(records_by_row, rows)
                mask = mask.astype(bool)
                proba = _proba_matrix(model, X, y.shape[1])
                per_dim = _per_dim_f1(y, (proba >= 0.5).astype(float), mask)
                entry[partition] = {
                    "per_dim": per_dim,
                    "macro_f1": macro_f1(per_dim),
                    "n_undefined_f1": count_undefined_f1(per_dim),
                    "subject_mae": subject_mae_from_proba(proba, y, mask),
                    "subject_rmse": subject_rmse_from_proba(proba, y, mask),
                }
            result[name] = entry
    return result


def evaluate_neural(model, manifest: dict, records: list, batch_size: int = 64) -> dict:
    """Per-dim F1/positive-rate for the neural model on dev and holdout."""
    import torch

    from .real_data import RealBatchProvider

    out: dict = {}
    for partition in ("dev", "holdout"):
        provider = RealBatchProvider(manifest, records, partition, batch_size=batch_size)
        probs, ys, ms = [], [], []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for item in provider:
                out_model = model(item.batch)
                probs.append(out_model.predictions["category_probs"])
                ys.append(item.batch.targets["category_labels"])
                ms.append(item.batch.target_mask["category_labels"])
        if was_training:
            model.train()
        proba = torch.cat(probs).numpy()
        y = torch.cat(ys).numpy()
        mask = torch.cat(ms).numpy().astype(bool)
        per_dim = _per_dim_f1(y, (proba >= 0.5).astype(float), mask)
        out[partition] = {
            "per_dim": per_dim,
            "macro_f1": macro_f1(per_dim),
            "n_undefined_f1": count_undefined_f1(per_dim),
            "subject_mae": subject_mae_from_proba(proba, y, mask),
            "subject_rmse": subject_rmse_from_proba(proba, y, mask),
        }
    return out
