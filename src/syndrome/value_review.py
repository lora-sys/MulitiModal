"""任务 B（证候线）：只读复核「最新协议」的 Top-3、逐类表现、拒判覆盖率。

不训练、不调参、不改冠军、不写 results.tsv、不解锁历史 holdout。
复用冻结 protocol-v2 选择集 + 已交付工件，重新计算：
- Top-1/3/5 与 macro-F1（含 CI）与逐类表现汇总；
- 拒判覆盖率：dev 文本与对抗探针上的 OK / UNCERTAIN / ABSTAIN /
  OUT_OF_DOMAIN 状态分布，以及阈值以上/以下的准确率差异。

口径：选择集 dev 分区（LOCAL_EXPERIMENTAL / AGENT_ISOLATION_UNVERIFIED），
不是独立测试集，不构成临床有效性结论；分数未校准，不是概率或诊断。

运行（acam-lab，PYTHONPATH=src）：
    python -m syndrome.value_review --out runs/value-review/syndrome.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from data.tcm_sd import load_syndrome_vocab
from research.prepare import load_selection
from syndrome.delivery import N_CLASSES, SEED, assert_holdout_untouched, selection_split
from syndrome.experiment import ADVERSARIAL_PROBES
from syndrome.inference import (
    STATUS_OK,
    SyndromeInference,
    select_uncertainty_threshold,
    validate_probability_matrix,
)
from syndrome.labels import LabelSpace
from syndrome.metrics import summarize_evaluation
from syndrome.service import ModelRegistry, SyndromeService, tfidf_logreg_spec

REVIEW_SCHEMA = "tcm-syndrome-value-review-v1"
N_BOOT = 1000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_coverage(statuses: list[str]) -> dict:
    """拒判覆盖率：状态分布 + 覆盖率（OK 占比）+ 拒判率。"""
    counts = Counter(statuses)
    n = len(statuses)
    ok = counts.get(STATUS_OK, 0)
    return {
        "n_inputs": n,
        "counts": {key: counts.get(key, 0) for key in (
            STATUS_OK, "UNCERTAIN", "ABSTAIN", "OUT_OF_DOMAIN")},
        "decision_coverage": round(ok / n, 6) if n else 0.0,
        "abstain_coverage": round((n - ok) / n, 6) if n else 0.0,
        "all_rejected": bool(n and ok == 0),
    }


def review(manifest_path: Path, dataset_path: Path, raw_dir: Path,
           artifact_dir: str, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """只读复核：返回完整报告（调用方负责写 runs/，不提交）。"""
    manifest, records = load_selection(manifest_path, dataset_path)
    assert_holdout_untouched(manifest)
    split = selection_split(manifest, records)
    labels = load_syndrome_vocab(raw_dir)
    if len(labels) != N_CLASSES or len(set(labels)) != N_CLASSES:
        raise ValueError("frozen syndrome vocabulary is not the 148-class taxonomy")
    label_space = LabelSpace(tuple(labels))

    spec = tfidf_logreg_spec(artifact_dir)
    if spec is None:
        raise ValueError(f"syndrome artifact unavailable: {artifact_dir}")
    registry = ModelRegistry()
    registry.register(spec, default=True)
    service = SyndromeService(registry=registry)

    dev_texts = split["dev_texts"]
    dev_labels = split["dev_labels"]
    loaded = spec.loader()
    dev_scores = validate_probability_matrix(
        loaded.predictor.predict_proba(dev_texts), len(dev_texts), N_CLASSES
    )
    dev_pred = np.asarray(dev_scores).argmax(axis=1).tolist()

    evaluation = summarize_evaluation(
        dev_labels, dev_pred, dev_scores, labels,
        baseline_pred=None, n_boot=n_boot, seed=seed,
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

    max_probs = np.asarray(dev_scores).max(axis=1)
    correct = [pred == truth for pred, truth in zip(dev_pred, dev_labels, strict=True)]
    threshold_info = select_uncertainty_threshold(max_probs, correct)

    # 拒判覆盖率：用产品服务接口实际跑一遍 dev 文本，而不是只看阈值数学。
    dev_statuses = [
        service.analyze_symptoms(_request(text)).to_dict()["status"]
        for text in dev_texts
    ]
    probe_statuses = [
        service.analyze_symptoms(_request(text)).to_dict()["status"]
        for text in ADVERSARIAL_PROBES
    ]

    # 交付工件自带的阈值（服务实际使用的），与本次重新选定的阈值并列记录。
    inference = SyndromeInference(
        loaded.predictor, label_space,
        model_version=spec.model_version, provenance={},
        threshold=loaded.threshold, knowledge=None,
        threshold_scope=loaded.threshold_scope,
    )
    below = [c for p, c in zip(max_probs, correct, strict=True)
             if p < threshold_info["threshold"]]
    above = [c for p, c in zip(max_probs, correct, strict=True)
             if p >= threshold_info["threshold"]]

    return {
        "schema_version": REVIEW_SCHEMA,
        "status": {
            "evidence": "LOCAL_EXPERIMENTAL",
            "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NONE",
            "clinical_claim": False,
        },
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "dataset_sha256": _sha256(dataset_path),
            "artifact_dir": artifact_dir,
            "holdout_subjects": split["holdout_subjects"],
            "scope": "frozen protocol-v2 selection dev partition; no holdout, no retraining, no tuning",
        },
        "protocol": {"n_classes": N_CLASSES, "seed": seed, "n_boot": n_boot,
                     "dev_dedup_dropped": split["dev_dedup_dropped"]},
        "acceptance": {
            "top1_accuracy": evaluation["accuracy"],
            "top3_accuracy": evaluation["top3_accuracy"],
            "top5_accuracy": evaluation["top5_accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "macro_f1_ci": evaluation["macro_f1_ci"],
            "per_class_summary": per_class_summary,
            "n_samples": evaluation["n_samples"],
        },
        "threshold": {
            "artifact_threshold": loaded.threshold,
            "artifact_threshold_scope": loaded.threshold_scope,
            "recomputed": threshold_info,
            "score_semantics": inference.score_semantics,
            "calibrated": False,
        },
        "abstention": {
            "definition": "SyndromeService.analyze_symptoms status distribution on allowed dev data",
            "dev": status_coverage(dev_statuses),
            "adversarial_probes": status_coverage(probe_statuses),
            "dev_accuracy_above_threshold": round(sum(above) / len(above), 6) if above else None,
            "dev_accuracy_below_threshold": round(sum(below) / len(below), 6) if below else None,
            "n_probes": len(probe_statuses),
        },
        "per_class": per_class,
        "unverified_generalization": [
            "dev 是选择集内分区，不是独立测试集；跨来源/时间泛化未验证",
            "TCM-SD CC BY-NC-SA 4.0（仅非商业科研）；派生知识同许可",
            "分数未经校准，不是概率、置信度、诊断或临床判断",
            "阈值在 dev 上选定，拒判覆盖率属于同一批数据的内部复核",
            "GO_RESEARCH_CLAIMS 未放行；不构成临床有效性结论",
        ],
    }


def _request(text: str):
    from syndrome.service import SymptomAnalysisRequest
    return SymptomAnalysisRequest(symptom_text=text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/local/tcm-sd")
    parser.add_argument("--manifest", default=("data/local/tcm-sd/protocol-v2/"
                                               "short/selection_manifest.json"))
    parser.add_argument("--dataset", default=("data/local/tcm-sd/protocol-v2/"
                                              "short/selection_records.json"))
    parser.add_argument("--artifact-dir", default="artifacts/sd-v2-ar1-champion")
    parser.add_argument("--out", default="runs/value-review/syndrome.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    report = review(Path(args.manifest), Path(args.dataset), Path(args.raw_dir),
                    args.artifact_dir, args.seed, args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "top3_accuracy": report["acceptance"]["top3_accuracy"],
        "macro_f1": report["acceptance"]["macro_f1"],
        "n_classes_zero_f1": report["acceptance"]["per_class_summary"]["n_classes_zero_f1"],
        "dev_coverage": report["abstention"]["dev"]["decision_coverage"],
        "dev_abstain_coverage": report["abstention"]["dev"]["abstain_coverage"],
        "adversarial_abstain_coverage": report["abstention"]["adversarial_probes"]["abstain_coverage"],
        "report": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
