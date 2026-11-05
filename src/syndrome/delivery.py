"""Frozen AutoResearch champion delivery and acceptance (TCM-SD syndrome).

This is the independent delivery task described in skills/autoresearch/program.md
Handoff: fit the frozen champion once, persist and reload it through the existing
service artifact contract, exercise the product inference interface, compute
per-class metrics with the existing metric implementation, and run robustness
checks. It does not run KEEP/REVERT rounds, does not tune hyperparameters, does
not change the frozen evaluator, and never reads a holdout partition.

Only development selection inputs are used (``holdout`` must be empty). Every
number below is ``LOCAL_EXPERIMENTAL``: no attested evaluator, no external
validation, no clinical claim. See ``unverified_generalization`` in the report.

CLI (run on the development host, PYTHONPATH=src):

    python -m syndrome.delivery \
        --repo . --raw-dir data/local/tcm-sd \
        --manifest data/local/tcm-sd/protocol-v2/short/selection_manifest.json \
        --dataset data/local/tcm-sd/protocol-v2/short/selection_records.json \
        --results runs/ar1/results.tsv \
        --artifact-dir artifacts/sd-v2-ar1-champion \
        --report-dir runs/delivery/ar1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from data.tcm_sd import load_syndrome_knowledge, load_syndrome_vocab
from research.prepare import (
    SELECTION_DATASET_SHA256,
    SELECTION_MANIFEST_SHA256,
    committed_source,
    load_selection,
)
from syndrome.baselines import (
    TfidfNgramBaseline,
    TfidfNgramConfig,
    environment_record,
)
from syndrome.experiment import ADVERSARIAL_PROBES
from syndrome.inference import (
    SyndromeInference,
    select_uncertainty_threshold,
    validate_probability_matrix,
)
from syndrome.labels import (
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    LabelSpace,
)
from syndrome.metrics import (
    macro_f1,
    majority_class_predictions,
    summarize_evaluation,
    topk_accuracy,
)
from syndrome.service import (
    SERVICE_SCHEMA_VERSION,
    ModelRegistry,
    SymptomAnalysisRequest,
    SyndromeService,
    save_tfidf_logreg_artifact,
    tfidf_logreg_spec,
)

CHAMPION_COMMIT = "d8371c7b14e3e47a3330cde6234ef1e843eb6136"
CHAMPION_SOURCE_SHA256 = "bd1bed98ab0407141e0059e361ee1872bd068b4680b4061f3f50a00bea844dcb"
CHAMPION_MACRO_F1 = 0.33332767910681194
SEED = 20261104
N_CLASSES = 148
MODEL_ID = "syndrome-tfidf-charwb-1-4-c10-ar1"
MODEL_VERSION = "ar1-champion-d8371c7b"
REPORT_SCHEMA_VERSION = "tcm-sd-delivery-report-v1"

#: Literals that bind this delivery to the frozen champion source. A change to
#: src/research/train.py must fail loudly instead of silently shipping a
#: different model under the champion's name.
CHAMPION_SOURCE_LITERALS = (
    'analyzer="char_wb"',
    "ngram_range=(1, 4)",
    "min_df=1",
    "binary=True",
    "sublinear_tf=True",
    'solver="liblinear"',
    "C=10.0",
    'class_weight="balanced"',
    "max_iter=2000",
    "OneVsRestClassifier",
)

REQUIRED_REPORT_KEYS = (
    "schema_version",
    "status",
    "champion",
    "inputs",
    "final_training",
    "acceptance",
    "threshold",
    "artifact",
    "inference",
    "robustness",
    "unverified_generalization",
)


def champion_config(seed: int = SEED) -> TfidfNgramConfig:
    """The exact configuration of the frozen champion, expressed in the
    existing baseline config (no new hyperparameters, no search)."""
    return TfidfNgramConfig(
        ngram_low=1,
        ngram_high=4,
        min_df=1,
        sublinear_tf=True,
        C=10.0,
        class_weight="balanced",
        max_iter=2000,
        seed=seed,
        solver="ovr-liblinear",
        analyzer="char_wb",
        binary=True,
    )


def verify_champion_source(source: bytes) -> str:
    """Bind the delivery to the frozen champion source bytes."""
    digest = hashlib.sha256(source).hexdigest()
    if digest != CHAMPION_SOURCE_SHA256:
        raise ValueError(
            f"champion source sha256 mismatch: expected {CHAMPION_SOURCE_SHA256}, got {digest}"
        )
    text = source.decode("utf-8")
    missing = [token for token in CHAMPION_SOURCE_LITERALS if token not in text]
    if missing:
        raise ValueError(f"champion source is missing frozen literals: {missing}")
    return digest


def assert_holdout_untouched(manifest: dict) -> None:
    """Delivery never uses a holdout; the frozen selection set must have none."""
    holdout = manifest.get("split", {}).get("holdout")
    if holdout:
        raise ValueError(f"selection manifest has a non-empty holdout ({len(holdout)} subjects)")
    if (manifest.get("manifest_sha256") != SELECTION_MANIFEST_SHA256
            or manifest.get("dataset_sha256") != SELECTION_DATASET_SHA256):
        raise ValueError("selection inputs do not match the frozen protocol-v2 hashes")


def selection_split(manifest: dict, records: dict) -> dict:
    """Frozen train/dev subject partition with one-way exact-text dev dedup.

    Mirrors research.prepare.fold_inputs: dev rows whose text is byte-identical
    to any train row are dropped; train-side duplicates are genuine repeat
    visits and are kept. No fold re-assignment, no holdout access.
    """
    assert_holdout_untouched(manifest)
    split = manifest["split"]
    rows = manifest["subject_row_by_id"]
    users = manifest["subject_user_by_id"]
    train_ids = list(split["train"])
    dev_ids = list(split["dev"])
    train_users = {users[sid] for sid in train_ids}
    dev_users = {users[sid] for sid in dev_ids}
    if train_users & dev_users:
        raise ValueError("train and dev subjects share a user (grouping violated)")
    train_records = [records[rows[sid]] for sid in train_ids]
    train_texts = {record.input_text for record in train_records}
    dev_records = [records[rows[sid]] for sid in dev_ids]
    dev_dropped = sum(record.input_text in train_texts for record in dev_records)
    dev_records = [
        record for record in dev_records
        if record.input_text not in train_texts and record.label is not None
    ]
    train_records = [record for record in train_records if record.label is not None]
    if not train_records or not dev_records:
        raise ValueError("selection split has no labelled train or dev records")
    return {
        "train_texts": [r.input_text for r in train_records],
        "train_labels": [r.label for r in train_records],
        "dev_texts": [r.input_text for r in dev_records],
        "dev_labels": [r.label for r in dev_records],
        "train_rows": len(train_records),
        "dev_rows": len(dev_records),
        "dev_dedup_dropped": dev_dropped,
        "train_users": len(train_users),
        "dev_users": len(dev_users),
        "holdout_subjects": len(split["holdout"]),
        "policy": "one-way exact-text dev dedup against train; subject-grouped",
    }


def read_champion_result(results_path: Path) -> dict:
    """Read the frozen experiment history, not a new score."""
    with results_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t", strict=True))
    matches = [row for row in rows if row["commit"] == CHAMPION_COMMIT]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one champion row in {results_path}")
    row = matches[0]
    if row["decision"] != "KEEP" or row["metric"] != "syndrome/macro_f1":
        raise ValueError("champion history row is not a KEEP of the frozen metric")
    value = float(row["value"])
    if abs(value - CHAMPION_MACRO_F1) > 1e-12:
        raise ValueError("champion history value does not match the frozen constant")
    return {
        "commit": row["commit"],
        "decision": row["decision"],
        "metric": row["metric"],
        "value": value,
        "evidence": row["evidence"],
        "promotion": row["promotion"],
        "protocol_sha256": row["protocol_sha256"],
        "description": row["description"],
        "history_rows": len(rows),
    }


def validate_report_schema(report: dict) -> None:
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"delivery report is missing required keys: {missing}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("delivery report has an unsupported schema_version")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deliver(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).resolve()
    raw_dir = Path(args.raw_dir)
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bind to the frozen champion commit and source (no working-tree drift).
    committed = committed_source(repo, CHAMPION_COMMIT, "src/research/train.py")
    champion_sha = verify_champion_source(committed)
    working = (repo / "src/research/train.py").read_bytes()
    if hashlib.sha256(working).hexdigest() != champion_sha:
        raise ValueError("working tree src/research/train.py differs from frozen champion")
    champion_history = read_champion_result(Path(args.results))

    # 2. Frozen, hash-verified selection inputs (holdout must be empty).
    manifest, records = load_selection(Path(args.manifest), Path(args.dataset))
    split = selection_split(manifest, records)
    labels = load_syndrome_vocab(raw_dir)
    if len(labels) != N_CLASSES or len(set(labels)) != N_CLASSES:
        raise ValueError("frozen syndrome vocabulary is not the 148-class taxonomy")
    label_space = LabelSpace(tuple(labels))
    knowledge = load_syndrome_knowledge(raw_dir)

    # 3. Final training: exactly one fit on the frozen train partition.
    config = champion_config(args.seed)
    model = TfidfNgramBaseline(config, n_classes=N_CLASSES)
    started = time.monotonic()
    model.fit(split["train_texts"], split["train_labels"])
    fit_seconds = round(time.monotonic() - started, 3)
    vectorizer = model._require().named_steps["tfidf"]  # fitted pipeline, for feature count
    n_features = len(vectorizer.vocabulary_)

    # 4. Acceptance on the frozen dev partition, using the existing metrics.
    dev_pred = model.predict(split["dev_texts"])
    dev_scores = validate_probability_matrix(
        model.predict_proba(split["dev_texts"]), split["dev_rows"], N_CLASSES
    )
    baseline_pred = majority_class_predictions(
        split["train_labels"], split["dev_rows"], N_CLASSES
    )
    evaluation = summarize_evaluation(
        split["dev_labels"], dev_pred, dev_scores, labels,
        baseline_pred=baseline_pred, n_boot=args.n_boot, seed=args.seed,
    )
    per_class = evaluation.pop("per_class")
    f1_values = [row["f1"] for row in per_class]
    per_class_summary = {
        "n_classes": len(per_class),
        "n_classes_with_support": sum(row["support"] > 0 for row in per_class),
        "n_classes_zero_f1": sum(row["f1"] == 0.0 for row in per_class),
        "min_f1": min(f1_values),
        "max_f1": max(f1_values),
        "mean_f1": round(sum(f1_values) / len(f1_values), 6),
        "min_support": min(row["support"] for row in per_class),
        "max_support": max(row["support"] for row in per_class),
    }

    # 5. UNCERTAIN threshold selected on dev (existing selection helper).
    max_probs = dev_scores.max(axis=1)
    correct = [pred == truth for pred, truth in zip(dev_pred, split["dev_labels"], strict=True)]
    threshold_info = select_uncertainty_threshold(max_probs, correct)

    # 6. Save the fitted predictor through the existing service artifact path.
    provenance = {
        "champion_commit": CHAMPION_COMMIT,
        "champion_source_sha256": champion_sha,
        "champion_selection_cv_macro_f1": champion_history["value"],
        "champion_history_evidence": champion_history["evidence"],
        "config": model.provenance()["params"],
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "label_taxonomy_version": manifest["label_taxonomy_version"],
        "label_version": manifest.get("label_version", "NOT_AVAILABLE"),
        "standard_version": manifest.get("standard_version", "NOT_AVAILABLE"),
        "seed": args.seed,
        "environment": environment_record(seed=args.seed),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "split": "frozen protocol-v2 selection train/dev; holdout empty",
        "evidence": "LOCAL_EXPERIMENTAL",
        "attested_evaluator": "AGENT_ISOLATION_UNVERIFIED",
        "data_license": "TCM-SD CC BY-NC-SA 4.0 (non-commercial research only)",
    }
    spec = save_tfidf_logreg_artifact(
        model, labels, str(artifact_dir),
        model_version=MODEL_VERSION,
        threshold=threshold_info["threshold"],
        threshold_scope=(
            "selected on frozen protocol-v2 dev partition; holdout untouched; "
            "UNVERIFIED for external data"
        ),
        provenance=provenance,
        model_id=MODEL_ID,
    )

    # 7. Reload through the registry and verify save/load round-trip + hash guard.
    reloaded_spec = tfidf_logreg_spec(str(artifact_dir))
    loaded = reloaded_spec.loader()
    sample_texts = split["dev_texts"][:16]
    roundtrip_predictions = loaded.predictor.predict(sample_texts) == model.predict(sample_texts)
    roundtrip_probabilities = bool(np.allclose(
        loaded.predictor.predict_proba(sample_texts),
        model.predict_proba(sample_texts),
        atol=1e-12,
    ))
    tamper_rejected = False
    with tempfile.TemporaryDirectory(prefix="delivery-tamper-") as tmp:
        copy_dir = Path(tmp) / "artifact"
        shutil.copytree(artifact_dir, copy_dir)
        predictor_path = copy_dir / "model.joblib"
        payload = bytearray(predictor_path.read_bytes())
        payload[0] ^= 0x01
        predictor_path.write_bytes(bytes(payload))
        try:
            tfidf_logreg_spec(str(copy_dir)).loader()
        except ValueError:
            tamper_rejected = True
    if not (roundtrip_predictions and roundtrip_probabilities and tamper_rejected):
        raise ValueError("artifact round-trip or hash-tamper guard failed")

    artifact_files = {}
    for name in ("model.joblib", "labels.json", "registry_entry.json"):
        path = artifact_dir / name
        artifact_files[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}

    # 8. Product inference interface (service + registry), plus a direct
    #    SyndromeInference instance for the robustness suite.
    registry = ModelRegistry()
    registry.register(reloaded_spec, default=True)
    service = SyndromeService(registry=registry, knowledge=knowledge)
    inference = SyndromeInference(
        loaded.predictor, label_space,
        model_version=MODEL_VERSION,
        provenance={"model_id": MODEL_ID, **loaded.provenance},
        threshold=loaded.threshold,
        knowledge=knowledge,
        weight_path=str(artifact_dir / "model.joblib"),
        threshold_scope=loaded.threshold_scope,
    )

    example_requests = [
        ("dev_chief_complaint", split["dev_texts"][0], None),
        ("dev_chief_complaint_2", split["dev_texts"][1], None),
        ("with_structured_fields", split["dev_texts"][2],
         {"age": 45, "gender": "女", "tongue": "舌红", "pulse": "脉细"}),
        ("empty_input", "", None),
        ("english_input", "I have insomnia and palpitations for two weeks", None),
        ("syndrome_name_as_input", "气虚血瘀证", None),
        ("short_colloquial", "老是怕冷拉肚子", None),
        ("negated_symptom", "无发热，无畏寒，稍有乏力，舌淡", None),
    ]
    examples = []
    for name, text, fields in example_requests:
        response = service.analyze_symptoms(
            SymptomAnalysisRequest(symptom_text=text, fields=fields)
        ).to_dict()
        examples.append({"name": name, "request_text": text,
                         "fields": fields, "response": response})

    # 9. Robustness: adversarial rejections, truncated text, batch/single
    #    agreement, malformed predictor fail-closed.
    adversarial_statuses = [
        result.status for result in inference.predict_batch(list(ADVERSARIAL_PROBES))
    ]
    adversarial_reject_rate = sum(status != STATUS_OK for status in adversarial_statuses) / len(
        adversarial_statuses
    )
    truncated = [text[:20] for text in split["dev_texts"]]
    truncated_f1 = macro_f1(split["dev_labels"], model.predict(truncated), N_CLASSES)
    full_f1 = macro_f1(split["dev_labels"], dev_pred, N_CLASSES)
    probe_texts = [
        "畏寒肢冷",
        "无发热，无畏寒，稍有乏力，舌淡",
        "老是怕冷拉肚子",
        "晚上睡不着心慌得很",
        "尿尿烫得慌还黄",
        "",
        "I cannot sleep",
        "气虚血瘀证",
    ]
    batch_results = inference.predict_batch(probe_texts)
    single_results = [inference.predict_one(text) for text in probe_texts]
    batch_matches_single = all(
        batch.status == single.status
        and [(c.label, round(c.score, 9)) for c in batch.candidates]
        == [(c.label, round(c.score, 9)) for c in single.candidates]
        for batch, single in zip(batch_results, single_results, strict=True)
    )
    rejected_have_no_candidates = all(
        not result.candidates for result in batch_results
        if result.status in (STATUS_ABSTAIN, STATUS_OUT_OF_DOMAIN)
    )
    malformed_rejected = False

    class _Malformed:
        def predict_proba(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), N_CLASSES - 1))

    try:
        SyndromeInference(
            _Malformed(), label_space, model_version="malformed-check",
            provenance={}, threshold=0.0,
        ).predict_one("患者刺痛固定")
    except ValueError:
        malformed_rejected = True

    robustness = {
        "adversarial_probes": list(ADVERSARIAL_PROBES),
        "adversarial_statuses": adversarial_statuses,
        "adversarial_reject_rate": round(adversarial_reject_rate, 6),
        "dev_macro_f1_full_text": round(full_f1, 6),
        "dev_macro_f1_truncated_20": round(truncated_f1, 6),
        "degradation_truncated": round(full_f1 - truncated_f1, 6),
        "top3_accuracy_full": round(topk_accuracy(split["dev_labels"], dev_scores, 3), 6),
        "probe_statuses": [result.status for result in batch_results],
        "batch_matches_single": bool(batch_matches_single),
        "rejected_have_no_candidates": bool(rejected_have_no_candidates),
        "malformed_probability_matrix_rejected": malformed_rejected,
        "negation_robustness": "NOT_AVAILABLE",
        "negation_reason": "真实病例否定改写无标签；接口行为由合成探针覆盖",
    }
    if not (batch_matches_single and rejected_have_no_candidates and malformed_rejected
            and adversarial_reject_rate == 1.0):
        raise ValueError("robustness acceptance checks failed")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": {
            "evidence": "LOCAL_EXPERIMENTAL",
            "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NOT_VALIDATED",
            "holdout_used": False,
            "hyperparameter_optimization": "NONE_AFTER_FREEZE",
            "go_engineering": "NOT_ASSESSED",
            "go_real_data_training": "LOCAL_EXPERIMENTAL_ONLY",
            "go_research_claims": "NOT_APPROVED",
            "clinical_efficacy_claim": False,
        },
        "champion": {
            "commit": CHAMPION_COMMIT,
            "train_py_sha256": champion_sha,
            "working_tree_matches_commit": True,
            "selection_cv_macro_f1": champion_history["value"],
            "selection_cv_note": (
                "4 folds x 2 repeats grouped CV over the whole selection pool; "
                "not directly comparable to the single-split delivery dev score"
            ),
            "history": champion_history,
        },
        "inputs": {
            "manifest": str(args.manifest),
            "dataset": str(args.dataset),
            "results": str(args.results),
            "raw_dir": str(raw_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_sha256": manifest["dataset_sha256"],
            "label_taxonomy_version": manifest["label_taxonomy_version"],
            "label_tier": manifest["label_tier"],
            "n_classes": manifest["n_classes"],
            "split": {
                key: value for key, value in split.items()
                if key not in ("train_texts", "train_labels", "dev_texts", "dev_labels")
            },
            "split_text_policy": "raw development texts are never written to the report",
        },
        "final_training": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "config": model.provenance()["params"],
            "seed": args.seed,
            "fit_seconds": fit_seconds,
            "n_features": n_features,
            "environment": environment_record(seed=args.seed),
        },
        "acceptance": {
            **evaluation,
            "per_class_summary": per_class_summary,
            "per_class": per_class,
            "scope": "frozen protocol-v2 dev partition; not an independent test set",
        },
        "threshold": {
            **threshold_info,
            "scope": loaded.threshold_scope,
            "score_semantics": inference.score_semantics,
            "calibrated": False,
        },
        "artifact": {
            "directory": str(artifact_dir),
            "files": artifact_files,
            "model_id": spec.model_id,
            "model_version": spec.model_version,
            "taxonomy_version": spec.taxonomy_version,
            "save_load_roundtrip_predictions": roundtrip_predictions,
            "save_load_roundtrip_probabilities": roundtrip_probabilities,
            "hash_tamper_rejected": tamper_rejected,
            "registry_entry": json.loads(
                (artifact_dir / "registry_entry.json").read_text(encoding="utf-8")
            ),
        },
        "inference": {
            "interface": "syndrome.service.SyndromeService.analyze_symptoms",
            "schema_version": SERVICE_SCHEMA_VERSION,
            "backend_availability": "AVAILABLE",
            "examples": examples,
        },
        "robustness": robustness,
        "unverified_generalization": [
            "无独立外部数据源：跨来源/跨设备/跨时间泛化未验证",
            "历史 holdout 解锁预算已耗尽，本次未使用任何 holdout",
            "评估器隔离状态 AGENT_ISOLATION_UNVERIFIED，无独立防篡改认证",
            "分数为 OvR sigmoid 归一化分数，未校准，不得解释为概率或置信度",
            "dev 分区参与了原 4x2 CV 的模型选择，交付验收指标非独立测试误差",
            "不构成诊断、临床疗效或有效性结论；GO_RESEARCH_CLAIMS 未放行",
            "数据许可 CC BY-NC-SA 4.0，未经授权不得商业部署",
        ],
    }
    validate_report_schema(report)
    (report_dir / "delivery-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "per-class.json").write_text(
        json.dumps(per_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--manifest",
                        default="data/local/tcm-sd/protocol-v2/short/selection_manifest.json")
    parser.add_argument("--dataset",
                        default="data/local/tcm-sd/protocol-v2/short/selection_records.json")
    parser.add_argument("--results", default="runs/ar1/results.tsv")
    parser.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    parser.add_argument("--report-dir", default="runs/delivery/ar1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    report = deliver(args)
    print(json.dumps({
        "model_version": report["final_training"]["model_version"],
        "model_files": sorted(report["artifact"]["files"]),
        "dev_macro_f1": report["acceptance"]["macro_f1"],
        "threshold": report["threshold"]["threshold"],
        "adversarial_reject_rate": report["robustness"]["adversarial_reject_rate"],
        "evidence": report["status"]["evidence"],
        "report": str(Path(args.report_dir) / "delivery-report.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
