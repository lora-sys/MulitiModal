"""任务 03：真实用户外部验证的冻结与评估脚本（当前不含任何真实患者数据）。

本模块只定义「将来拿到独立数据后如何评估」的可执行流程：
- freeze：冻结模型工件、148 类标签词表、拒判阈值、评估代码与数据划分；
- predict：仅推理，用冻结工件产出分数与状态，不训练、不调阈值；
- evaluate：以独立中医师标注为参考，计算 Top-1/Top-3/Macro-F1/逐类 Recall
  （含有效样本量与 Wilson 95% CI）、拒判率，并单独统计分歧/未标注/无法确定样本；
- smoke-test：完全合成数据的端到端流程演练（SMOKE_ONLY），绝非模型能力结论。

边界：无新受试者数据与独立专家标注时，任务 04 仍为 BLOCKED；
本脚本不产生任何临床或泛化结论，GO_RESEARCH_CLAIMS 未放行。

运行（acam-lab，PYTHONPATH=src）：
    python -m syndrome.external_validation smoke-test --out runs/validation/smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

VALIDATION_SCHEMA = "tcm-sd-external-validation-v1"
FROZEN_MODEL_ID = "syndrome-tfidf-charwb-1-4-c10-ar1"
FROZEN_THRESHOLD = 0.27808433274565425
FROZEN_TAXONOMY = "tcm-sd-syndrome-vocab-v1"
FROZEN_ARTIFACT_SHA256 = {
    "model.joblib": "944b325d273d79f7b11b3d4a5e5e07340f99aa973a86e15ead53ca798ae5e74f",
    "labels.json": "65d797c4c43a6977453aab98e36c2dab4fb0596df79eb05f6def7aaf2009285d",
    "registry_entry.json": "a73830d8e577195a66bf7d0e0b3751aac816533bd804fce36e2904332cbf246f",
    "syndrome_vocab.txt": "63ee108754d90914a3833328abda7e04130f200617ca79033a66d22e47c45c83",
}
N_CLASSES = 148
LOW_SAMPLE_THRESHOLD = 30
MIN_EFFECTIVE_SAMPLES_PER_CLASS = 30
RECRUITMENT_STRATA = ("fatigue", "sleep_problem", "shoulder_neck_discomfort", "other")
EXPERT_STATUSES = ("DETERMINED", "UNDETERMINABLE", "DISAGREEMENT")
REFERENCE_SOURCES = ("TCM_PROFESSIONAL",)
DECISION_STATUSES = ("OK", "UNCERTAIN", "ABSTAIN", "OUT_OF_DOMAIN")
SPLIT_RULES = (
    "all consented external subjects form ONE evaluation set",
    "no training, no threshold tuning, no model selection on this data",
    "subject-level uniqueness; no duplicate or near-duplicate text across splits",
    "expert disagreement / unlabeled / undeterminable counted separately",
    "any post-test retuning invalidates the independent test",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_sha256() -> str:
    return sha256(Path(__file__).resolve())


def build_freeze(artifact_dir: Path, raw_dir: Path,
                 dataset_path: Path | None = None) -> dict:
    """Frozen before execution: model, labels, threshold, eval code, split."""
    paths = {
        "model.joblib": Path(artifact_dir) / "model.joblib",
        "labels.json": Path(artifact_dir) / "labels.json",
        "registry_entry.json": Path(artifact_dir) / "registry_entry.json",
        "syndrome_vocab.txt": Path(raw_dir) / "syndrome_vocab.txt",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    return {
        "schema_version": VALIDATION_SCHEMA,
        "frozen": {
            "model_id": FROZEN_MODEL_ID,
            "taxonomy_version": FROZEN_TAXONOMY,
            "threshold": FROZEN_THRESHOLD,
            "n_classes": N_CLASSES,
            "artifact_sha256": hashes,
            "evaluation_code_sha256": self_sha256(),
            "dataset_sha256": sha256(dataset_path)
            if dataset_path is not None and Path(dataset_path).is_file() else "NOT_AVAILABLE",
        },
        "split": {
            "policy": "ALL_CONSENTED_EXTERNAL_SUBJECTS_FORM_ONE_EVALUATION_SET",
            "train": 0, "dev": 0, "holdout": 0,
            "rules": list(SPLIT_RULES),
        },
        "status": {
            "evidence": "EXTERNAL_VALIDATION_NOT_STARTED",
            "go_research_claims": "NOT_GRANTED",
        },
    }


def verify_freeze(freeze: dict, artifact_dir: Path, raw_dir: Path,
                  dataset_path: Path | None = None) -> dict:
    """Reject any drift in the frozen model, taxonomy, threshold or eval code."""
    if freeze.get("schema_version") != VALIDATION_SCHEMA:
        raise ValueError("unsupported freeze manifest schema")
    frozen = freeze["frozen"]
    for key, value in (("model_id", FROZEN_MODEL_ID),
                       ("taxonomy_version", FROZEN_TAXONOMY),
                       ("threshold", FROZEN_THRESHOLD), ("n_classes", N_CLASSES)):
        if frozen.get(key) != value:
            raise ValueError(f"frozen {key} drifted")
    paths = {
        "model.joblib": Path(artifact_dir) / "model.joblib",
        "labels.json": Path(artifact_dir) / "labels.json",
        "registry_entry.json": Path(artifact_dir) / "registry_entry.json",
        "syndrome_vocab.txt": Path(raw_dir) / "syndrome_vocab.txt",
    }
    for name, path in paths.items():
        expected = FROZEN_ARTIFACT_SHA256[name]
        if frozen.get("artifact_sha256", {}).get(name) != expected:
            raise ValueError(f"frozen {name} does not match the pinned artifact hash")
        if sha256(path) != expected:
            raise ValueError(f"frozen {name} does not match the current artifact")
    if frozen.get("evaluation_code_sha256") != self_sha256():
        raise ValueError("frozen evaluation_code_sha256 does not match the current code")
    if dataset_path is not None:
        if frozen.get("dataset_sha256") == "NOT_AVAILABLE":
            raise ValueError("freeze manifest does not pin the external dataset")
        if frozen.get("dataset_sha256") != sha256(dataset_path):
            raise ValueError("external dataset differs from the frozen dataset")
    return {"verified": True, "model_id": FROZEN_MODEL_ID,
            "threshold": FROZEN_THRESHOLD, "n_classes": N_CLASSES}


def prediction_from_scores(scores_row: list[float], threshold: float,
                           n_classes: int = N_CLASSES) -> dict:
    """Deterministic tie-break: score descending, then frozen label index ascending."""
    if len(scores_row) != n_classes:
        raise ValueError("prediction scores must cover the frozen class taxonomy")
    ordered = sorted(range(len(scores_row)), key=lambda i: (-scores_row[i], i))
    top_score = float(scores_row[ordered[0]])
    return {
        "scores": [float(value) for value in scores_row],
        "top1_index": ordered[0],
        "top3_indices": ordered[:3],
        "max_score": top_score,
        "status": "OK" if top_score >= threshold else "UNCERTAIN",
    }


def predict_scores(predictor, texts: list[str], threshold: float) -> list[dict]:
    """Inference only; the caller supplies an already-frozen predictor."""
    import numpy as np

    from syndrome.inference import validate_probability_matrix

    matrix = validate_probability_matrix(predictor.predict_proba(texts), len(texts), N_CLASSES)
    return [prediction_from_scores(np.asarray(row, dtype=float).tolist(), threshold)
            for row in matrix]


def wilson_interval(successes: int, trials: int,
                    confidence: float = 0.95) -> list[float | None]:
    """Wilson score interval for a per-class recall (binomial, no numpy)."""
    if trials == 0:
        return [None, None]
    z = 1.959963984540054 if confidence == 0.95 else 2.5758293035489004
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
            / denominator)
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def _validate_records(records: list[dict], labels: list[str]) -> dict:
    if not records:
        raise ValueError("no external records provided")
    label_set = set(labels)
    seen: set[str] = set()
    counts = {"no_consent": 0, "undeterminable": 0, "disagreement": 0,
              "unlabeled": 0, "invalid_reference": 0}
    included = []
    for record in records:
        for forbidden in ("syndrome", "syndrome_label", "diagnosis"):
            if forbidden in record:
                raise ValueError("collection form must not pre-map symptoms to a syndrome")
        subject = record.get("subject_id")
        if not isinstance(subject, str) or not subject:
            raise ValueError("every record needs a non-empty subject_id")
        if subject in seen:
            raise ValueError(f"duplicate subject_id: {subject}")
        seen.add(subject)
        if record.get("reference_source") not in REFERENCE_SOURCES:
            counts["invalid_reference"] += 1
            continue
        if record.get("consent") is not True:
            counts["no_consent"] += 1
            continue
        status = record.get("expert_status")
        if status not in EXPERT_STATUSES:
            raise ValueError(f"unsupported expert_status: {status!r}")
        if status == "UNDETERMINABLE":
            counts["undeterminable"] += 1
            continue
        if status == "DISAGREEMENT":
            counts["disagreement"] += 1
            continue
        if record.get("annotator_count", 0) < 2:
            raise ValueError("a determined reference requires >= 2 independent experts")
        primary = record.get("primary_label")
        if not isinstance(primary, str) or primary not in label_set:
            counts["unlabeled"] += 1
            continue
        if not isinstance(record.get("symptom_text"), str) or not record["symptom_text"].strip():
            raise ValueError("every record needs free-text colloquial symptoms")
        if record.get("recruitment_stratum") not in RECRUITMENT_STRATA:
            raise ValueError("unknown recruitment stratum")
        included.append(record)
    return {"counts": counts, "included": included}


def evaluate(records: list[dict], predictions: dict[str, dict], labels: list[str],
             freeze: dict, *, freeze_verified: bool = False,
             min_subjects: int = 100) -> dict:
    """Pure-Python evaluation of frozen predictions against expert reference labels."""
    if freeze.get("schema_version") != VALIDATION_SCHEMA:
        raise ValueError("unsupported freeze manifest schema")
    frozen = freeze.get("frozen", {})
    if frozen.get("evaluation_code_sha256") != self_sha256():
        raise ValueError("evaluation code changed since the freeze manifest was written")
    if len(labels) != frozen.get("n_classes") or len(set(labels)) != len(labels):
        raise ValueError("label vocabulary does not match the frozen taxonomy")
    threshold = float(frozen["threshold"])
    label_index = {name: i for i, name in enumerate(labels)}
    checked = _validate_records(records, labels)
    counts = checked["counts"]
    n = len(labels)

    missing_predictions = 0
    evaluated = []
    status_all: dict[str, int] = {status: 0 for status in DECISION_STATUSES}
    for record in checked["included"]:
        prediction = predictions.get(record["subject_id"])
        if prediction is None:
            missing_predictions += 1
            continue
        if prediction.get("status") not in DECISION_STATUSES:
            raise ValueError("prediction status outside the inference contract")
        top1 = prediction.get("top1_index")
        top3 = prediction.get("top3_indices")
        if (not isinstance(top1, int) or not 0 <= top1 < n
                or not isinstance(top3, list)
                or any(not isinstance(i, int) or not 0 <= i < n for i in top3)):
            raise ValueError("prediction label indices outside the frozen taxonomy")
        status_all[prediction["status"]] += 1
        evaluated.append((record, prediction))

    truth = [label_index[record["primary_label"]] for record, _ in evaluated]
    predicted = [prediction["top1_index"] for _, prediction in evaluated]
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    support = [0] * n
    top3_hits = [0] * n
    rejected = [0] * n
    status_by_class = {status: [0] * n for status in DECISION_STATUSES}
    for (_record, prediction), y_true, y_pred in zip(evaluated, truth, predicted, strict=True):
        support[y_true] += 1
        if y_true == y_pred:
            tp[y_true] += 1
        else:
            fn[y_true] += 1
            fp[y_pred] += 1
        if y_true in prediction["top3_indices"]:
            top3_hits[y_true] += 1
        status_by_class[prediction["status"]][y_true] += 1
        if prediction["status"] != "OK":
            rejected[y_true] += 1

    per_class = []
    for i, name in enumerate(labels):
        recall = tp[i] / support[i] if support[i] else None
        low, high = wilson_interval(tp[i], support[i])
        per_class.append({
            "index": i,
            "syndrome": name,
            "effective_sample_size": support[i],
            "tp": tp[i], "fp": fp[i], "fn": fn[i],
            "recall": round(recall, 6) if recall is not None else None,
            "recall_ci95": [low, high],
            "top3_hits": top3_hits[i],
            "top3_hit_rate": round(top3_hits[i] / support[i], 6) if support[i] else None,
            "rejected": rejected[i],
            "rejection_rate": round(rejected[i] / support[i], 6) if support[i] else None,
            "status_counts": {status: status_by_class[status][i]
                              for status in DECISION_STATUSES},
            "evidence_status": ("NO_SAMPLES" if support[i] == 0 else
                                "LOW_SAMPLE_INSUFFICIENT"
                                if support[i] < LOW_SAMPLE_THRESHOLD else "MEASURABLE"),
            "meets_minimum": support[i] >= MIN_EFFECTIVE_SAMPLES_PER_CLASS,
        })
    f1 = [
        (2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]))
        if (2 * tp[i] + fp[i] + fn[i]) else 0.0
        for i in range(n)
    ]
    n_evaluated = len(evaluated)
    top1_hits = sum(tp)
    top3_total = sum(top3_hits)
    n_rejected = sum(rejected)
    conditions = {
        "freeze_verified": freeze_verified,
        "enough_subjects": n_evaluated >= min_subjects,
        "independent_subjects": True,
        "post_hoc_tuning": False,
        "expert_reference_only": counts["invalid_reference"] == 0,
    }
    use_conditions_met = all(conditions.values())
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": {
            "evidence": "SMOKE_ONLY" if freeze.get("purpose") == "SMOKE_ONLY"
                        else "LOCAL_EXPERIMENTAL",
            "external_validation": "NOT_PERFORMED" if not freeze_verified
                                   else ("INDEPENDENT_TEST" if use_conditions_met
                                         else "RESEARCH_USE_ONLY"),
            "go_research_claims": "NOT_GRANTED",
            "clinical_claim": False,
            "post_hoc_tuning": False,
            "use_conditions_met": use_conditions_met,
        },
        "conditions": conditions,
        "freeze": {"model_id": frozen.get("model_id"),
                   "threshold": threshold, "n_classes": n,
                   "verified": freeze_verified},
        "counts": {
            "n_records": len(records),
            "n_no_consent": counts["no_consent"],
            "n_undeterminable": counts["undeterminable"],
            "n_disagreement": counts["disagreement"],
            "n_unlabeled": counts["unlabeled"],
            "n_invalid_reference": counts["invalid_reference"],
            "n_missing_prediction": missing_predictions,
            "n_excluded_total": (counts["no_consent"] + counts["undeterminable"]
                                 + counts["disagreement"] + counts["unlabeled"]
                                 + counts["invalid_reference"] + missing_predictions),
            "n_evaluated": n_evaluated,
            "status_all_with_predictions": status_all,
        },
        "metrics": {
            "top1_accuracy": round(top1_hits / n_evaluated, 6) if n_evaluated else None,
            "top3_accuracy": round(top3_total / n_evaluated, 6) if n_evaluated else None,
            "macro_f1": round(sum(f1) / n, 6) if n_evaluated else None,
            "rejection_rate": round(n_rejected / n_evaluated, 6) if n_evaluated else None,
            "decision_coverage": round(1 - n_rejected / n_evaluated, 6)
            if n_evaluated else None,
            "n_classes_with_effective_samples": sum(s > 0 for s in support),
            "n_classes_meeting_minimum": sum(
                row["meets_minimum"] for row in per_class),
        },
        "metric_definitions": {
            "top1_accuracy": "冻结预测 Top-1 标签 == 裁決后的 primary_label 的样本占比",
            "top3_accuracy": "primary_label 落在冻结预测 Top-3 标签集合内的样本占比",
            "macro_f1": "全部 148 类逐类 F1=2TP/(2TP+FP+FN) 的算术平均，缺席类记 0",
            "per_class_recall": "TP/(TP+FN)，含有效样本量与 Wilson 95% CI",
            "rejection_rate": "评估集内状态非 OK 的样本占比；决策覆盖率 = 1 - 拒判率",
            "exclusions": "无同意 / 专家分歧 / 无法确定 / 未标注 / 无预测 分开计数，不进入主指标",
        },
        "per_class": per_class,
        "unverified_generalization": [
            "若 use_conditions_met=false，结果只能作为研究用途，不能称为独立泛化测试",
            "任何测试后调参/重训/换阈值都会使独立性失效，必须重新冻结并重新执行",
            "100–200 人仅是初始可行性规模，不足以可靠验证全部 148 类",
            "专家分歧、未标注、无法确定样本必须单独报告，不得静默丢弃",
        ],
    }


def smoke_freeze() -> dict:
    """Synthetic freeze for the flow test only; never used with real subjects."""
    return {
        "schema_version": VALIDATION_SCHEMA,
        "purpose": "SMOKE_ONLY",
        "synthetic_inputs": True,
        "frozen": {
            "model_id": "SYNTHETIC_STUB", "taxonomy_version": "SYNTHETIC_VOCAB",
            "threshold": 0.5, "n_classes": 6,
            "artifact_sha256": {}, "evaluation_code_sha256": self_sha256(),
            "dataset_sha256": "NOT_AVAILABLE",
        },
        "split": {"policy": "SMOKE_ONLY", "train": 0, "dev": 0, "holdout": 0,
                  "rules": list(SPLIT_RULES)},
        "status": {"evidence": "SMOKE_ONLY", "go_research_claims": "NOT_GRANTED"},
    }


def smoke_fixture() -> tuple[list[dict], dict[str, dict], list[str]]:
    """Fully synthetic records + frozen-style predictions (no real patient data)."""
    labels = [f"SYNTHETIC_{i}" for i in range(6)]
    records = []
    predictions = {}
    strata = list(RECRUITMENT_STRATA)
    for i in range(14):
        subject = f"smoke-{i:03d}"
        record = {
            "subject_id": subject,
            "symptom_text": "最近总觉得累、睡不好、肩膀发紧（合成文本）",
            "recruitment_stratum": strata[i % len(strata)],
            "consent": True,
            "expert_status": "DETERMINED",
            "expert_labels": [labels[i % 6]],
            "primary_label": labels[i % 6],
            "annotator_count": 2,
            "reference_source": "TCM_PROFESSIONAL",
            "form_version": "symptom-form-v1-synthetic",
        }
        if i == 10:
            record["consent"] = False
        if i == 11:
            record["expert_status"] = "DISAGREEMENT"
        if i == 12:
            record["expert_status"] = "UNDETERMINABLE"
        if i == 13:
            record["primary_label"] = "NOT_IN_TAXONOMY"
        records.append(record)
        scores = [0.01] * 6
        scores[i % 6] = 0.8 if i % 2 == 0 else 0.3
        predictions[subject] = prediction_from_scores(scores, 0.5, len(labels))
    return records, predictions, labels


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze_cmd = sub.add_parser("freeze", help="write the pre-execution freeze manifest")
    freeze_cmd.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    freeze_cmd.add_argument("--raw-dir", default="data/local/tcm-sd")
    freeze_cmd.add_argument("--dataset")
    freeze_cmd.add_argument("--out", default="runs/validation/freeze-manifest.json")

    predict_cmd = sub.add_parser("predict", help="inference-only frozen predictions")
    predict_cmd.add_argument("--freeze", required=True)
    predict_cmd.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    predict_cmd.add_argument("--raw-dir", default="data/local/tcm-sd")
    predict_cmd.add_argument("--records", required=True)
    predict_cmd.add_argument("--out", required=True)

    eval_cmd = sub.add_parser("evaluate", help="frozen evaluation against expert labels")
    eval_cmd.add_argument("--freeze", required=True)
    eval_cmd.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    eval_cmd.add_argument("--raw-dir", default="data/local/tcm-sd")
    eval_cmd.add_argument("--labels", default="artifacts/sd-v2-ar1-champion/labels.json")
    eval_cmd.add_argument("--dataset")
    eval_cmd.add_argument("--records", required=True)
    eval_cmd.add_argument("--predictions", required=True)
    eval_cmd.add_argument("--min-subjects", type=int, default=100)
    eval_cmd.add_argument("--out", required=True)

    smoke_cmd = sub.add_parser("smoke-test", help="synthetic end-to-end flow test")
    smoke_cmd.add_argument("--out")

    args = parser.parse_args()

    if args.command == "freeze":
        manifest = build_freeze(Path(args.artifact_dir), Path(args.raw_dir),
                                Path(args.dataset) if args.dataset else None)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(json.dumps({"frozen": True, "out": str(out),
                          "dataset_sha256": manifest["frozen"]["dataset_sha256"]},
                         ensure_ascii=False))
        return 0

    if args.command == "predict":
        from syndrome.service import tfidf_logreg_spec

        freeze = _load_json(Path(args.freeze))
        verify_freeze(freeze, Path(args.artifact_dir), Path(args.raw_dir),
                      Path(args.dataset) if args.dataset else None)
        spec = tfidf_logreg_spec(args.artifact_dir)
        if spec is None:
            raise ValueError("frozen syndrome artifact unavailable")
        loaded = spec.loader()
        records = _load_json(Path(args.records))
        predictions = {
            record["subject_id"]: result
            for record, result in zip(
                records,
                predict_scores(loaded.predictor,
                               [record["symptom_text"] for record in records],
                               freeze["frozen"]["threshold"]),
                strict=True)
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(json.dumps({"predicted": len(predictions), "out": str(out),
                          "threshold": freeze["frozen"]["threshold"]}, ensure_ascii=False))
        return 0

    if args.command == "evaluate":
        freeze = _load_json(Path(args.freeze))
        verify_freeze(freeze, Path(args.artifact_dir), Path(args.raw_dir),
                      Path(args.dataset) if args.dataset else None)
        labels = _load_json(Path(args.labels))
        if len(labels) != N_CLASSES or len(set(labels)) != N_CLASSES:
            raise ValueError("labels.json is not the frozen 148-class vocabulary")
        report = evaluate(_load_json(Path(args.records)),
                          _load_json(Path(args.predictions)), labels, freeze,
                          freeze_verified=True, min_subjects=args.min_subjects)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(json.dumps({"n_evaluated": report["counts"]["n_evaluated"],
                          "metrics": report["metrics"],
                          "use_conditions_met": report["status"]["use_conditions_met"],
                          "out": str(out)}, ensure_ascii=False))
        return 0

    freeze = smoke_freeze()
    records, predictions, labels = smoke_fixture()
    report = evaluate(records, predictions, labels, freeze,
                      freeze_verified=False, min_subjects=100)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(json.dumps({"purpose": "SMOKE_ONLY",
                      "synthetic_inputs": True,
                      "n_evaluated": report["counts"]["n_evaluated"],
                      "exclusions": {k: v for k, v in report["counts"].items()
                                     if k.startswith("n_")},
                      "use_conditions_met": report["status"]["use_conditions_met"],
                      "out": args.out or "NOT_WRITTEN"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
