"""中医体质冠军模型交付与验收（与证候线交付同构，数据/协议/工件独立）。

冻结冠军 commit 16c26582：Huber 全局线性 + kNN 局部中位数的逐维软集成，
特征为人口学 6 维 + 人体测量 3 维（冻结选择集的前 9 列）。

交付内容：
- 最终训练：在冻结的 274 名受试者选择集上拟合一次（不再调参、不使用任何
  holdout）；所有预处理统计量只在训练集内拟合。
- 工件：model.joblib / labels.json / registry_entry.json，带 sha256 清单与
  provenance（数据哈希、标签版本、配置、seed、依赖、代码 commit）。
- 重载：哈希校验、taxonomy 校验、候选源码哈希绑定，损坏即拒载。
- 八维推理：只输出 8 个已观测维度；**第 9 维平和质永远 ABSTAIN**，
  不补零、不猜测、不伪造；缺失/越界输入 ABSTAIN 或 OUT_OF_DOMAIN。
- 逐维误差分析：全部基于冻结 4 折 × 2 重复的折外预测，不是训练内自评。
- 鲁棒性与工程验收；verify 动作只核对、不训练。

九维来源 CANONICAL_ORDER，平和质在源数据中完全未观测（0/274），
标签层级 PUBLISHED，标准 ZYYXH/T157-2009；与证候线 148 类分类不可互换。
所有数字均为 LOCAL_EXPERIMENTAL，评估器隔离 AGENT_ISOLATION_UNVERIFIED。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from data.mendeley import EXPECTED_SHA256
from evidence.registry import EvidenceRegistry, evaluate_gates
from research.constitution import _fold_payload
from research.constitution_data import (
    BUDGET_SECONDS,
    EVIDENCE,
    FOLDS,
    N_SUBJECTS,
    N_TARGET_DIMS,
    OBSERVED_DIMS,
    REPEATS,
    SEED,
    STANDARD_VERSION,
    UNOBSERVED_DIMS,
    load_selection,
    selection_fold_ids,
)
from research.constitution_train import (
    ALPHA_GRID,
    INNER_FOLDS,
    IRLS_STEPS,
    K_GRID,
    RIDGE,
    choose_alphas,
    fit_both_with_y,
    matrix,
    predict_both,
)
from research.prepare import committed_source, run_candidate

CANDIDATE_PATH = "src/research/constitution_train.py"
CHAMPION_COMMIT = "16c26582de2f6e76afe4b73669e4a16e146071fe"
CHAMPION_SOURCE_SHA256 = "53240a285314e1b38b05c9a78711c5e875092c7237572750bfd5e7fe553a71ac"
CHAMPION_MASKED_MAE = 14.14368356353204
MODEL_ID = "constitution-huber-knn-cv1"
MODEL_VERSION = "cv1-champion-16c26582"
TAXONOMY_VERSION = "tcm-constitution-9d-zYYXH157-v1"
ARTIFACT_SCHEMA = "tcm-constitution-artifact-v1"
LABELS_SCHEMA = "tcm-constitution-labels-v1"
REPORT_SCHEMA = "tcm-constitution-delivery-report-v1"
VERIFY_SCHEMA = "tcm-constitution-delivery-verification-v1"
DATA_LICENSE = "Mendeley cxtph3tjsg v1, CC BY 4.0"
FEATURE_KEEP = slice(0, 9)  # 冻结冠军的特征子集
ARTIFACT_NAMES = ("model.joblib", "labels.json", "registry_entry.json")

REQUIRED_REPORT_KEYS = (
    "schema_version", "status", "champion", "inputs", "final_training",
    "acceptance", "artifact", "inference", "per_dimension", "robustness",
    "unverified_generalization",
)


def _balanced_abstain() -> dict:
    return {"dim": "balanced", "status": "ABSTAIN", "value": None,
            "reason": "dimension unobserved in source data; never fabricated"}


class ConstitutionPredictor:
    """可序列化的最终模型：复用冻结冠军候选的拟合/预测函数。

    只输出 (n, 8) 的 PUBLISHED 维度分数；第 9 维平和质不在模型输出中，
    由 ConstitutionInference 显式以 ABSTAIN 表达。
    """

    def __init__(self, records: list[dict], seed: int = SEED) -> None:
        self.seed = seed
        self.dims = list(OBSERVED_DIMS)
        self.feature_keep = (0, 9)
        self.fit(records)

    def fit(self, records: list[dict]) -> None:
        x, mask = matrix(records)
        keep = slice(*self.feature_keep)
        x, mask = x[:, keep], mask[:, keep]
        y = np.array([row["scores"] for row in records], dtype=float)
        if y.shape[1] != N_TARGET_DIMS:
            raise ValueError("final model expects exactly 8 observed constitution scores")
        self.alphas = choose_alphas(x, mask, y, self.seed)
        self.state = fit_both_with_y(x, mask, y, self.seed)
        self.n_training = len(records)

    def predict(self, records: list[dict]) -> np.ndarray:
        if not records:
            return np.empty((0, N_TARGET_DIMS), dtype=float)
        x, mask = matrix(records)
        keep = slice(*self.feature_keep)
        x, mask = x[:, keep], mask[:, keep]
        huber, knn = predict_both(self.state, x, mask)
        output = self.alphas * huber + (1.0 - self.alphas) * knn
        if output.shape[1] != N_TARGET_DIMS or not np.isfinite(output).all():
            raise ValueError("final model produced invalid 8-dim predictions")
        return output

    def provenance(self) -> dict:
        import research.constitution_train as candidate_module

        return {
            "model_id": MODEL_ID,
            "champion_commit": CHAMPION_COMMIT,
            "champion_source_sha256": CHAMPION_SOURCE_SHA256,
            "candidate_module_sha256": hashlib.sha256(
                Path(candidate_module.__file__).read_bytes()).hexdigest(),
            "config": {
                "RIDGE": RIDGE, "IRLS_STEPS": IRLS_STEPS, "INNER_FOLDS": INNER_FOLDS,
                "K_GRID": list(K_GRID), "ALPHA_GRID": list(ALPHA_GRID),
                "feature_keep": list(self.feature_keep), "seed": self.seed,
            },
            "score_semantics": (
                "point estimates on 0-100 PUBLISHED scores; uncalibrated, "
                "not probabilities, not a diagnosis"
            ),
            "n_training": self.n_training,
        }


class ConstitutionInference:
    """八维推理接口；第 9 维平和质 ABSTAIN，缺项不补零、不猜测。"""

    def __init__(self, predictor: ConstitutionPredictor, provenance: dict) -> None:
        self.predictor = predictor
        self.provenance = dict(provenance)

    def predict_one(self, row: dict | None) -> dict:
        base = {
            "label_taxonomy_version": TAXONOMY_VERSION,
            "model_version": MODEL_VERSION,
            "provenance": self.provenance,
            "calibrated": False,
            "evidence": EVIDENCE,
        }
        status, reason = self._validate(row)
        if status != "OK":
            return self._reject(base, status, reason)
        try:
            # Overflow in the Huber/kNN arithmetic must become a rejection, not
            # a crash or a fabricated out-of-scale score.
            with np.errstate(all="ignore"):
                scores = self.predictor.predict([row])[0].tolist()
        except (ValueError, FloatingPointError, OverflowError) as exc:
            return self._reject(
                base, "OUT_OF_DOMAIN",
                f"模型未能产生有限输出（{type(exc).__name__}），输入超出可判定域",
            )
        if not all(math.isfinite(v) and 0.0 <= v <= 100.0 for v in scores):
            return self._reject(
                base, "OUT_OF_DOMAIN",
                "预测分数超出 0–100 已发布量表范围，拒绝输出以免误导",
            )
        return {
            **base, "status": "OK",
            "scores": [{"dim": dim, "value": float(value), "calibrated": False}
                       for dim, value in zip(self.predictor.dims, scores, strict=True)],
            "balanced": _balanced_abstain(),
            "missing_information": [
                f"feature {i}" for i, observed in enumerate(row["feature_mask"])
                if not observed
            ],
        }

    @staticmethod
    def _reject(base: dict, status: str, reason: str) -> dict:
        return {**base, "status": status, "scores": [], "reason": reason,
                "balanced": _balanced_abstain()}

    @staticmethod
    def _validate(row: dict | None) -> tuple[str, str]:
        if row is None:
            return "ABSTAIN", "未提供身体信息"
        if not isinstance(row, Mapping):
            return "OUT_OF_DOMAIN", f"输入类型 {type(row).__name__} 不是身体信息记录"
        features = row.get("features")
        mask = row.get("feature_mask")
        if features is None and mask is None:
            return "ABSTAIN", "未提供身体特征与观测掩码"
        if not isinstance(features, list) or not isinstance(mask, list) \
                or len(features) != 63 or len(mask) != 63:
            return "OUT_OF_DOMAIN", "身体特征必须是 63 维列表"
        if not all(isinstance(observed, bool) for observed in mask):
            return "OUT_OF_DOMAIN", "feature_mask 必须是布尔值"
        if not any(mask):
            return "ABSTAIN", "所有身体特征均缺失"
        if not any(mask[FEATURE_KEEP]):
            return "ABSTAIN", "模型所需的 9 项基础特征全部缺失，拒绝凭空推断"
        for value, observed in zip(features, mask, strict=True):
            if observed and (value is None or isinstance(value, bool)
                             or not isinstance(value, (int, float))
                             or not math.isfinite(value)):
                return "OUT_OF_DOMAIN", "已观测特征必须是有限数值"
            if not observed and value is not None:
                return "OUT_OF_DOMAIN", "缺失特征不得携带数值（禁止补零）"
        return "OK", ""

    def predict_batch(self, rows: list[dict | None]) -> list[dict]:
        if rows is None:
            return [self.predict_one(None)]
        if not isinstance(rows, (list, tuple)):
            return [self.predict_one(rows)]
        return [self.predict_one(row) for row in rows]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_champion_source(source: bytes) -> str:
    digest = hashlib.sha256(source).hexdigest()
    if digest != CHAMPION_SOURCE_SHA256:
        raise ValueError("champion candidate source sha256 mismatch")
    text = source.decode("utf-8")
    for token in ("def huber_fit", "def knn_predict", "def choose_alphas",
                  "def fit_both_with_y", "slice(0, 9)", "RIDGE = 100.0"):
        if token not in text:
            raise ValueError(f"champion source is missing frozen token: {token}")
    return digest


def save_artifact(predictor, artifact_dir: Path, provenance: dict) -> dict:
    """保存工件 + registry（hash 清单）；返回 registry entry。"""
    import joblib

    # `python -m research.constitution_delivery deliver` used to define this
    # class in __main__, so joblib wrote an unloadable __main__ reference.
    # Refuse to emit such an artifact instead of silently breaking every
    # independent process (the CLI entry point now imports this module).
    if type(predictor).__module__ == "__main__":
        raise ValueError(
            "artifact must be saved from the importable "
            "research.constitution_delivery module"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    labels_path = artifact_dir / "labels.json"
    joblib.dump(predictor, model_path)
    labels = {
        "schema_version": LABELS_SCHEMA,
        "taxonomy_version": TAXONOMY_VERSION,
        "canonical_order": ["balanced", *OBSERVED_DIMS],
        "score_dims": list(OBSERVED_DIMS),
        "unobserved_dimensions": list(UNOBSERVED_DIMS),
        "abstain_policy": "balanced is never predicted, never zero-filled",
        "label_tier": "PUBLISHED",
        "standard_version": STANDARD_VERSION,
    }
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    entry = {
        "schema_version": ARTIFACT_SCHEMA,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "kind": "huber-knn-ensemble-score-regression",
        "taxonomy_version": TAXONOMY_VERSION,
        "artifacts": {"model.joblib": _sha256(model_path),
                      "labels.json": _sha256(labels_path)},
        "data_license": DATA_LICENSE,
        "provenance": dict(provenance),
    }
    (artifact_dir / "registry_entry.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def load_artifact(artifact_dir: Path):
    """重载工件：schema/taxonomy/哈希/候选源码绑定全部校验，损坏即拒载。"""
    import joblib

    import research.constitution_train as candidate_module

    entry_path = artifact_dir / "registry_entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    if entry.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported constitution artifact schema")
    if entry.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError("artifact taxonomy version mismatch")
    for name, expected in entry["artifacts"].items():
        if _sha256(artifact_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    labels = json.loads((artifact_dir / "labels.json").read_text(encoding="utf-8"))
    if labels.get("schema_version") != LABELS_SCHEMA \
            or tuple(labels.get("score_dims", ())) != OBSERVED_DIMS \
            or tuple(labels.get("unobserved_dimensions", ())) != UNOBSERVED_DIMS:
        raise ValueError("labels.json does not declare the frozen 8+1 taxonomy")
    if hashlib.sha256(Path(candidate_module.__file__).read_bytes()).hexdigest() \
            != CHAMPION_SOURCE_SHA256:
        raise ValueError("champion candidate module drifted after the artifact was saved")
    predictor = joblib.load(artifact_dir / "model.joblib")
    if tuple(predictor.dims) != OBSERVED_DIMS:
        raise ValueError("reloaded predictor has the wrong score dimensions")
    return predictor, entry, labels


def validate_report_schema(report: dict) -> None:
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"delivery report missing keys: {missing}")
    if report["schema_version"] != REPORT_SCHEMA:
        raise ValueError("unsupported delivery report schema")


def _fold_replay(manifest: dict, rows: list[dict], candidate: bytes) -> dict:
    """用冻结协议折外重放候选。只读，不改评估器。"""
    oof: dict[str, dict] = {}
    fold_values = []
    for repeat in range(REPEATS):
        for fold in range(FOLDS):
            train_ids, dev_ids = selection_fold_ids(manifest, fold, FOLDS, repeat)
            training, inputs, dev_rows = _fold_payload(manifest, rows, train_ids, dev_ids)
            started = time.monotonic()
            predictions = run_candidate(candidate, training, inputs,
                                        budget_seconds=BUDGET_SECONDS, seed=SEED)
            elapsed = time.monotonic() - started
            errors = []
            for row, pred in zip(dev_rows, predictions, strict=True):
                if len(pred) != N_TARGET_DIMS:
                    raise ValueError("champion replay returned a 9th dimension")
                errors.append([abs(float(p) - float(t))
                               for p, t in zip(pred, row["scores"], strict=True)])
                oof.setdefault(row["subject_id"], []).append(sum(errors[-1]) / N_TARGET_DIMS)
            fold_values.append({
                "fold": fold, "repeat": repeat,
                "value": statistics.mean(sum(e) / N_TARGET_DIMS for e in errors),
                "train_seconds": round(elapsed, 3), "n_validation": len(dev_rows),
                "per_dim_mae": [statistics.mean(e[j] for e in errors)
                                for j in range(N_TARGET_DIMS)],
            })
    return {"oof": oof, "folds": fold_values}


def deliver(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).resolve()
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)

    committed = committed_source(repo, CHAMPION_COMMIT, CANDIDATE_PATH)
    champion_sha = verify_champion_source(committed)
    working = (repo / CANDIDATE_PATH).read_bytes()
    if hashlib.sha256(working).hexdigest() != champion_sha:
        raise ValueError("working tree candidate differs from the frozen champion")

    manifest, records = load_selection(Path(args.manifest), Path(args.dataset), Path(args.xls))
    if manifest["split"]["holdout"]:
        raise ValueError("no holdout evaluation is permitted")
    if len(records) != N_SUBJECTS:
        raise ValueError("frozen selection no longer has 274 subjects")
    if any("balanced" in row["score_dims"] for row in records):
        raise ValueError("balanced label must never be fabricated")

    # 1. 最终训练：在冻结选择集上拟合一次，不再调参
    started = time.monotonic()
    predictor = ConstitutionPredictor(records, seed=SEED)
    fit_seconds = round(time.monotonic() - started, 3)

    # 2. 冻结协议的折外重放（与冠军评估一致，不是训练内自评）
    champion_replay = _fold_replay(manifest, records, committed)
    baseline_source = committed_source(
        repo, "97ee268fd81d0fa466ca324dda5254f7c947d689", CANDIDATE_PATH)
    baseline_replay = _fold_replay(manifest, records, baseline_source)
    champion_fold_mae = statistics.mean(f["value"] for f in champion_replay["folds"])
    baseline_fold_mae = statistics.mean(f["value"] for f in baseline_replay["folds"])
    if abs(champion_fold_mae - CHAMPION_MASKED_MAE) > 1e-6:
        raise ValueError("out-of-fold replay disagrees with the frozen champion score")

    # 逐维误差分析
    import random

    rng = random.Random(SEED)
    per_dimension = []
    for j, dim in enumerate(OBSERVED_DIMS):
        champ = [f["per_dim_mae"][j] for f in champion_replay["folds"]]
        base = [f["per_dim_mae"][j] for f in baseline_replay["folds"]]
        paired = [c - b for c, b in zip(champ, base, strict=True)]
        boot = sorted(statistics.mean(rng.choices(paired, k=len(paired)))
                      for _ in range(2000))
        per_dimension.append({
            "dim": dim,
            "champion_mae": round(statistics.mean(champ), 4),
            "baseline_mae": round(statistics.mean(base), 4),
            "delta": round(statistics.mean(paired), 4),
            "paired_ci95": [round(boot[50], 4), round(boot[1950], 4)],
            "improved_folds": sum(d < 0 for d in paired),
        })
    paired_folds = [c["value"] - b["value"] for c, b in zip(
        champion_replay["folds"], baseline_replay["folds"], strict=True)]

    # 受试者级误差与异常影响
    subjects = sorted(champion_replay["oof"])
    champ_mean = {s: statistics.mean(champion_replay["oof"][s]) for s in subjects}
    base_mean = {s: statistics.mean(baseline_replay["oof"][s]) for s in subjects}
    champ_sorted = sorted(champ_mean.values(), reverse=True)
    total_error = sum(champ_sorted)
    trimmed = {}
    for k in (0, 7, 14, 27):
        c = sorted(champ_mean.values(), reverse=True)[k:]
        b = sorted(base_mean.values(), reverse=True)[k:]
        trimmed[str(k)] = round(statistics.mean(c) - statistics.mean(b), 4)
    subject_analysis = {
        "n_subjects": len(subjects),
        "paired_improved_subjects": sum(champ_mean[s] < base_mean[s] for s in subjects),
        "mean_paired_delta": round(statistics.mean(
            champ_mean[s] - base_mean[s] for s in subjects), 4),
        "top1_error_share": round(champ_sorted[0] / total_error, 4),
        "top10pct_error_share": round(
            sum(champ_sorted[:27]) / total_error, 4),
        "trimmed_champion_minus_baseline": trimmed,
        "p95_subject_error": round(sorted(champ_mean.values())[int(.95 * len(subjects))], 3),
        "max_subject_error": round(max(champ_mean.values()), 3),
    }

    # 3. 保存 / 重载 / 往返一致性
    provenance = {
        "champion_commit": CHAMPION_COMMIT,
        "champion_source_sha256": champion_sha,
        "champion_selection_masked_mae": CHAMPION_MASKED_MAE,
        "dataset_sha256": EXPECTED_SHA256,
        "manifest_sha256": _sha256(Path(args.manifest)),
        "records_sha256": _sha256(Path(args.dataset)),
        "label_taxonomy_version": TAXONOMY_VERSION,
        "label_tier": "PUBLISHED",
        "label_version": manifest["label_version"],
        "standard_version": manifest["standard_version"],
        "observed_dimensions": list(OBSERVED_DIMS),
        "unobserved_dimensions": list(UNOBSERVED_DIMS),
        "config": predictor.provenance()["config"],
        "seed": SEED,
        "n_training_subjects": len(records),
        "split": "frozen selection-only 4-fold x 2-repeat CV; holdout empty",
        "holdout_used": False,
        "hyperparameter_optimization": "NONE_AFTER_FREEZE",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "sklearn": "NOT_USED", "joblib": "USED"},
        "evidence": EVIDENCE,
        "attested_evaluator": "AGENT_ISOLATION_UNVERIFIED",
        "data_license": DATA_LICENSE,
    }
    entry = save_artifact(predictor, artifact_dir, provenance)
    loaded, reloaded_entry, labels = load_artifact(artifact_dir)
    sample = records[:16]
    roundtrip = bool(np.allclose(loaded.predict(sample), predictor.predict(sample), atol=1e-12))

    tamper_rejected = False
    with tempfile.TemporaryDirectory(prefix="cv1-tamper-") as tmp:
        copy_dir = Path(tmp) / "artifact"
        shutil.copytree(artifact_dir, copy_dir)
        model_path = copy_dir / "model.joblib"
        payload = bytearray(model_path.read_bytes())
        payload[0] ^= 0x01
        model_path.write_bytes(bytes(payload))
        try:
            load_artifact(copy_dir)
        except ValueError:
            tamper_rejected = True
    if not (roundtrip and tamper_rejected):
        raise ValueError("artifact round-trip or hash-tamper guard failed")

    # 4. 八维推理（含第 9 维 ABSTAIN）与鲁棒性
    inference = ConstitutionInference(loaded, provenance)
    missing_probe = dict(records[10])
    missing_probe["features"] = [None if i == 12 else v
                                for i, v in enumerate(records[10]["features"])]
    missing_probe["feature_mask"] = [False if i == 12 else m
                                     for i, m in enumerate(records[10]["feature_mask"])]
    probes = [
        ("dev_subject", records[0]),
        ("dev_subject_with_missing", missing_probe),
        ("empty_input", None),
        ("all_missing", {"features": [None] * 63, "feature_mask": [False] * 63}),
        ("wrong_width", {"features": [1.0] * 3, "feature_mask": [True] * 3}),
    ]
    examples = [{"name": name, "response": inference.predict_one(row)}
                for name, row in probes]
    batch = inference.predict_batch([records[0], None, records[1]])
    single = [inference.predict_one(records[0]), inference.predict_one(None),
              inference.predict_one(records[1])]
    batch_matches_single = all(
        b["status"] == s["status"] and len(b["scores"]) == len(s["scores"])
        for b, s in zip(batch, single, strict=True))
    balanced_abstain = all(
        e["response"]["balanced"]["status"] == "ABSTAIN"
        and e["response"]["balanced"]["value"] is None for e in examples)
    no_calibrated = all(not c["calibrated"] for e in examples
                        for c in e["response"]["scores"])
    robustness = {
        "out_of_fold_replay_matches_champion": True,
        "save_load_roundtrip": roundtrip,
        "artifact_hash_tamper_rejected": tamper_rejected,
        "batch_matches_single": bool(batch_matches_single),
        "balanced_dimension_abstain_everywhere": bool(balanced_abstain),
        "scores_never_marked_calibrated": no_calibrated,
        "missing_feature_never_zero_filled": True,
        "ninth_dimension_never_predicted": True,
        "probe_statuses": [e["response"]["status"] for e in examples],
        "holdout_subjects": len(manifest["split"]["holdout"]),
    }
    if not all((batch_matches_single, balanced_abstain, no_calibrated)):
        raise ValueError("inference acceptance checks failed")

    # 5. GO 状态（机械判定，不由 dev/CV 分数推导）
    evidence_registry = EvidenceRegistry.load(Path(args.evidence_registry))
    gates = {}
    for model in ("tcm_constitution", "physio_bpsi_efri"):
        gates[model] = {
            gate: {"state": d.state, "missing_claims": list(d.missing_claims)}
            for gate, d in evaluate_gates(evidence_registry, model).items()
        }
    gates["tcm_sd_syndrome"] = {
        "state": "NOT_ASSESSED",
        "reason": "no GO gate requirements are registered for the syndrome line",
    }

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": {
            "evidence": EVIDENCE, "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NOT_VALIDATED", "holdout_used": False,
            "hyperparameter_optimization": "NONE_AFTER_FREEZE",
            "go_engineering": "NOT_ASSESSED",
            "go_real_data_training": "LOCAL_EXPERIMENTAL_ONLY",
            "go_research_claims": "NOT_APPROVED", "clinical_efficacy_claim": False,
        },
        "champion": {
            "commit": CHAMPION_COMMIT, "candidate_sha256": champion_sha,
            "selection_masked_mae": CHAMPION_MASKED_MAE,
            "working_tree_matches_commit": True,
            "note": "4-fold x 2-repeat grouped CV mean; not an independent test error",
        },
        "inputs": {
            "manifest": str(args.manifest), "dataset": str(args.dataset),
            "xls": str(args.xls), "dataset_sha256": EXPECTED_SHA256,
            "manifest_sha256": _sha256(Path(args.manifest)),
            "records_sha256": _sha256(Path(args.dataset)),
            "n_subjects": len(records), "n_observed_dims": N_TARGET_DIMS,
            "unobserved_dims": list(UNOBSERVED_DIMS),
            "holdout_subjects": len(manifest["split"]["holdout"]),
            "label_tier": "PUBLISHED", "label_version": manifest["label_version"],
            "standard_version": manifest["standard_version"],
            "text_policy": "no raw subject rows are written to this report",
        },
        "final_training": {
            "model_id": MODEL_ID, "model_version": MODEL_VERSION,
            "config": predictor.provenance()["config"], "seed": SEED,
            "fit_seconds": fit_seconds, "n_training_subjects": len(records),
            "environment": provenance["environment"],
        },
        "acceptance": {
            "out_of_fold_champion_mae": champion_fold_mae,
            "out_of_fold_baseline_mae": baseline_fold_mae,
            "paired_fold_delta": round(statistics.mean(paired_folds), 4),
            "improved_folds": sum(d < 0 for d in paired_folds),
            "fold_count": len(champion_replay["folds"]),
            "scope": "frozen selection-only 4-fold x 2-repeat out-of-fold; no holdout",
        },
        "per_dimension": {
            "dimensions": per_dimension,
            "subject_level": subject_analysis,
        },
        "artifact": {
            "directory": str(artifact_dir),
            "files": {name: {"sha256": _sha256(artifact_dir / name),
                             "size_bytes": (artifact_dir / name).stat().st_size}
                      for name in ARTIFACT_NAMES},
            "model_id": entry["model_id"], "model_version": entry["model_version"],
            "taxonomy_version": entry["taxonomy_version"],
            "registry_entry": entry, "labels": labels,
            "save_load_roundtrip": roundtrip,
            "hash_tamper_rejected": tamper_rejected,
        },
        "inference": {
            "interface": "research.constitution_delivery.ConstitutionInference",
            "score_semantics": predictor.provenance()["score_semantics"],
            "balanced_dimension": "ABSTAIN_NEVER_FABRICATED",
            "examples": examples,
        },
        "robustness": robustness,
        "unverified_generalization": [
            "无独立外部数据源：跨来源/设备/时间泛化未验证",
            "历史体质 holdout 解锁预算已耗尽，本次未使用任何 holdout",
            "评估器隔离 AGENT_ISOLATION_UNVERIFIED，无独立防篡改认证",
            "最终模型在全部 274 人上拟合，报告误差来自折外预测，二者非同一模型对象",
            "平和质（第 9 维）完全未观测，ABSTAIN，不伪造、不补零、不猜测",
            "分数未校准，不得解释为概率、诊断或临床疗效",
            "不构成诊断、临床疗效或有效性结论；GO_RESEARCH_CLAIMS 未放行",
            "数据许可 Mendeley CC BY 4.0",
        ],
    }
    validate_report_schema(report)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "delivery-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "per-dimension.json").write_text(
        json.dumps(report["per_dimension"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    report["go_status"] = gates
    return report


def verify_only(args: argparse.Namespace) -> dict:
    """只核对已交付工件与冻结契约，不重新训练、不写模型、不修改冠军。"""
    repo = Path(args.repo).resolve()
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    committed = committed_source(repo, CHAMPION_COMMIT, CANDIDATE_PATH)
    champion_sha = verify_champion_source(committed)
    if hashlib.sha256((repo / CANDIDATE_PATH).read_bytes()).hexdigest() != champion_sha:
        raise ValueError("working tree candidate differs from frozen champion")

    manifest, _ = load_selection(Path(args.manifest), Path(args.dataset), Path(args.xls))
    if manifest["split"]["holdout"]:
        raise ValueError("selection manifest must have an empty holdout")
    predictor, entry, labels = load_artifact(artifact_dir)
    provenance = entry["provenance"]
    if provenance.get("dataset_sha256") != EXPECTED_SHA256 \
            or provenance.get("manifest_sha256") != _sha256(Path(args.manifest)) \
            or provenance.get("records_sha256") != _sha256(Path(args.dataset)):
        raise ValueError("registry provenance data hashes do not match the frozen inputs")
    if provenance.get("champion_commit") != CHAMPION_COMMIT \
            or provenance.get("champion_source_sha256") != champion_sha:
        raise ValueError("registry champion binding does not match")
    if provenance.get("holdout_used") is not False:
        raise ValueError("registry claims a holdout was used")

    inference = ConstitutionInference(predictor, provenance)
    statuses = [inference.predict_one(None)["status"],
                inference.predict_one({"features": [None] * 63,
                                        "feature_mask": [False] * 63})["status"]]
    example = inference.predict_one({
        "features": [1.0] * 63, "feature_mask": [True] * 63})
    report = json.loads((report_dir / "delivery-report.json").read_text(encoding="utf-8"))
    validate_report_schema(report)
    for name, meta in report["artifact"]["files"].items():
        if _sha256(artifact_dir / name) != meta["sha256"]:
            raise ValueError(f"report hash for {name} does not match artifact")

    evidence_registry = EvidenceRegistry.load(Path(args.evidence_registry))
    gates = {}
    for model in ("tcm_constitution", "physio_bpsi_efri"):
        gates[model] = {gate: d.state
                        for gate, d in evaluate_gates(evidence_registry, model).items()}
    gates["tcm_sd_syndrome"] = "NOT_ASSESSED"

    result = {
        "verification_schema": VERIFY_SCHEMA,
        "model_version": entry["model_version"],
        "champion_commit": CHAMPION_COMMIT,
        "champion_source_sha256": champion_sha,
        "dataset_sha256": EXPECTED_SHA256,
        "manifest_sha256": _sha256(Path(args.manifest)),
        "records_sha256": _sha256(Path(args.dataset)),
        "holdout_subjects": len(manifest["split"]["holdout"]),
        "registry_schema": entry["schema_version"],
        "registry_artifact_hashes_verified": True,
        "labels_score_dims": labels["score_dims"],
        "labels_unobserved_dims": labels["unobserved_dimensions"],
        "balanced_status": example["balanced"]["status"],
        "n_scores_returned": len(example["scores"]),
        "abstain_statuses": statuses,
        "retraining_performed": False,
        "evidence": report["status"]["evidence"],
        "go_status": gates,
        "go_note": "GO gates are independent; no GO state is derived from dev/CV scores",
    }
    (report_dir / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="deliver",
                        choices=("deliver", "verify"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--manifest", default=("data/local/mendeley-cxtph3tjsg-v1/"
                                                "protocol-v1/selection_manifest.json"))
    parser.add_argument("--dataset", default=("data/local/mendeley-cxtph3tjsg-v1/"
                                               "protocol-v1/selection_records.json"))
    parser.add_argument("--xls", default="data/local/mendeley-cxtph3tjsg-v1/Share data.xls")
    parser.add_argument("--artifact-dir", default="artifacts/constitution-v1-champion")
    parser.add_argument("--report-dir", default="runs/constitution-delivery/v1")
    parser.add_argument("--evidence-registry", default="docs/evidence/registry.json")
    args = parser.parse_args(argv)
    if args.action == "verify":
        result = verify_only(args)
        print(json.dumps({"action": "verify", "model_version": result["model_version"],
                          "registry_hashes_verified": result["registry_artifact_hashes_verified"],
                          "balanced_status": result["balanced_status"],
                          "retraining_performed": result["retraining_performed"]},
                         ensure_ascii=False))
        return 0
    report = deliver(args)
    summary = {
        "action": "deliver",
        "model_version": report["final_training"]["model_version"],
        "model_files": sorted(report["artifact"]["files"]),
        "out_of_fold_champion_mae": report["acceptance"]["out_of_fold_champion_mae"],
        "balanced": report["inference"]["balanced_dimension"],
        "evidence": report["status"]["evidence"],
        "report": str(Path(args.report_dir) / "delivery-report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    # Delegate through the importable module so the pickled class resolves to
    # research.constitution_delivery.ConstitutionPredictor in any process.
    import research.constitution_delivery as _entry  # noqa: PLC0415

    raise SystemExit(_entry.main())
