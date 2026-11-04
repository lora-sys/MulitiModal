"""真实数据训练与评估入口（#53 阶段 1）。

只使用 train / dev 两个 subject partition：
- 基线（多数类 Dummy、TF-IDF char 1–3 + LR）与 BERT 均在 train partition 拟合；
- 模型选择 / 阈值选择只用 dev partition；
- test partition 本入口一概不读——最终解锁测试由主线按 AGENTS.md 硬规则 11 执行。

依赖 #53 冻结契约的 data.tcm_sd（惰性导入，便于在 D1 落位前收集其余模块）。
报告写入 runs/（git-ignored）。CLI：
    python -m syndrome.experiment baselines --raw-dir data/local/tcm-sd \
        --out runs/tcm-sd-syndrome-v1
    python -m syndrome.experiment bert --raw-dir data/local/tcm-sd \
        --weights data/local/models/bert-base-chinese --out runs/tcm-sd-syndrome-v1 --scenario short
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from syndrome.baselines import (
    MostFrequentBaseline,
    TfidfNgramBaseline,
    TfidfNgramConfig,
)
from syndrome.inference import (
    SyndromeInference,
    select_uncertainty_threshold,
    validate_probability_matrix,
)
from syndrome.labels import LabelSpace
from syndrome.metrics import macro_f1, summarize_evaluation, topk_accuracy

SCENARIOS = {"clinical": "build_clinical_text", "short": "build_short_text"}


def _identity_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _development_identity(tcm_sd, partitions: dict[str, frozenset[str]]) -> dict[str, str]:
    development_files = (
        *tcm_sd.DEVELOPMENT_RECORD_FILES,
        tcm_sd.VOCAB_FILE,
    )
    manifest = {
        "taxonomy_id": tcm_sd.LABEL_TAXONOMY_ID,
        "files": {name: tcm_sd.EXPECTED_SHA256[name] for name in development_files},
    }
    manifest_sha256 = _identity_sha256(manifest)
    split_definition = {
        "development_manifest_sha256": manifest_sha256,
        "seed": tcm_sd.SPLIT_SEED,
        "ratios": {
            "train": tcm_sd.TRAIN_RATIO,
            "dev": tcm_sd.DEV_RATIO,
            "test": tcm_sd.TEST_RATIO,
        },
        "partitions": {
            name: sorted(partitions[name]) for name in ("train", "dev")
        },
    }
    return {
        "development_manifest_sha256": manifest_sha256,
        "split_identity_sha256": _identity_sha256(split_definition),
    }


def _verified_preprocessing_binding(model_dir: Path) -> dict[str, str | int]:
    from syndrome.bert_model import _preprocessing_binding

    config_path = model_dir / "train_config.json"
    artifact_meta_path = model_dir / "artifact_meta.json"
    try:
        train_config = json.loads(config_path.read_text(encoding="utf-8"))
        artifact_meta = json.loads(artifact_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("saved model preprocessing metadata is unavailable") from exc
    if not isinstance(train_config, dict) or not isinstance(artifact_meta, dict):
        raise ValueError("saved model preprocessing metadata is malformed")
    max_len = train_config.get("max_len")
    if type(max_len) is not int or max_len < 1:
        raise ValueError("saved model has no valid fitted max_len")
    actual = _preprocessing_binding(model_dir, max_len)
    if type(artifact_meta.get("max_len")) is not int:
        raise ValueError("saved model artifact max_len is UNVERIFIED")
    for key, value in actual.items():
        if artifact_meta.get(key) != value:
            raise ValueError(
                "saved tokenizer assets or preprocessing do not match artifact metadata"
            )
    return actual


def _validated_threshold_metrics(threshold_info: object) -> dict[str, float]:
    required_metrics = {"threshold", "dev_coverage", "dev_accuracy_above"}
    if (
        not isinstance(threshold_info, dict)
        or not required_metrics <= threshold_info.keys()
    ):
        raise ValueError("threshold artifact is missing dev selection metrics")
    try:
        metric_values = {
            name: float(threshold_info[name]) for name in required_metrics
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold artifact has invalid dev selection metrics") from exc
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in metric_values.values()
    ):
        raise ValueError("threshold artifact has invalid dev selection metrics")
    return metric_values


def _threshold_binding(
    model_dir: Path,
    aux: dict,
    metric_values: dict[str, float],
    *,
    selection_identity: dict[str, object],
) -> dict[str, object]:
    from syndrome.bert_model import _artifact_sha256

    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing model weights for threshold binding: {weight_path}")
    required_identity = ("development_manifest_sha256", "split_identity_sha256")
    if any(not isinstance(aux.get(name), str) for name in required_identity):
        raise ValueError("development manifest/split identity is unavailable")
    if not isinstance(selection_identity, dict):
        raise ValueError("threshold selection dataset identity is unavailable")
    selection_scenario = selection_identity.get("scenario")
    selection_builder = selection_identity.get("text_builder")
    selection_record_count = selection_identity.get("record_count")
    selection_dataset_sha256 = selection_identity.get("dataset_sha256")
    if (
        not isinstance(selection_scenario, str)
        or selection_scenario not in SCENARIOS
        or selection_builder != SCENARIOS[selection_scenario]
        or type(selection_record_count) is not int
        or selection_record_count < 0
        or not isinstance(selection_dataset_sha256, str)
        or len(selection_dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_dataset_sha256)
    ):
        raise ValueError("threshold selection dataset identity is unavailable")
    return {
        "weights_sha256": _artifact_sha256(weight_path),
        "development_manifest_sha256": aux["development_manifest_sha256"],
        "split_identity_sha256": aux["split_identity_sha256"],
        "selection_scenario": selection_scenario,
        "selection_text_builder": selection_builder,
        "selection_record_count": selection_record_count,
        "selection_dataset_sha256": selection_dataset_sha256,
        "selection_metrics_sha256": _identity_sha256(metric_values),
        "selection_scope": "dev_partition_only",
        "selection_partition": "dev",
        **_verified_preprocessing_binding(model_dir),
    }


def _validate_threshold_binding(
    threshold_info: object,
    provenance: object,
    model_dir: Path,
    aux: dict,
    *,
    selection_identity: dict[str, object],
) -> dict[str, object]:
    metric_values = _validated_threshold_metrics(threshold_info)
    if not isinstance(provenance, dict):
        raise ValueError("threshold provenance is UNVERIFIED")
    expected = _threshold_binding(
        model_dir,
        aux,
        metric_values,
        selection_identity=selection_identity,
    )
    if any(provenance.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "threshold provenance does not match model weights, preprocessing, "
            "and dev split, threshold selection dataset, or selection metrics"
        )
    return expected

# 鲁棒性探针（合成，无标签）：期望全部被拒判（ABSTAIN / OUT_OF_DOMAIN），不得给出 OK
ADVERSARIAL_PROBES: tuple[str, ...] = (
    "",
    "   ",
    "I have been feeling dizzy and cannot sleep well for two weeks",
    "@#$%^&*!!!????",
    "asdfghjkl qwertyuiop 12345678",
    "气虚血瘀证",
    "脾肾阳虚",
)


def load_partitions(raw_dir: str) -> tuple[list, dict[str, frozenset[str]], LabelSpace, dict]:
    """仅加载 train/dev 开发文件，绝不请求已消耗解锁额度的官方 test 文件。"""
    from importlib import import_module
    tcm_sd = import_module("data.tcm_sd")
    records = tcm_sd.load_tcm_sd_development_records(raw_dir)
    unexpected_files = {
        Path(r.source_file).name for r in records
    } - {"train.json", "dev.json"}
    if unexpected_files:
        raise ValueError(
            f"development loader returned non-development files: {sorted(unexpected_files)}"
        )
    partitions = tcm_sd.subject_partitions(records, seed=tcm_sd.SPLIT_SEED)
    label_space = LabelSpace(tcm_sd.load_syndrome_vocab(raw_dir))
    identity = _development_identity(tcm_sd, partitions)
    development = {
        split: sum(1 for r in records if Path(r.source_file).name == f"{split}.json")
        for split in ("train", "dev")
    }
    return records, partitions, label_space, {
        "development_file_counts": development,
        **identity,
    }
def scenario_data(
    records: list,
    partitions: dict[str, frozenset[str]],
    label_space: LabelSpace,
    scenario: str,
) -> dict:
    """构造一个文本场景的 train/dev 数据；UNKNOWN 标签如实报告并剔除（不补标签）。

    评估侧单向去重（R1 F1，在 partitions 计算后应用）：dev 中与任一 train
    病例在**本场景输入文本**（text_fn）下逐字节相同的病例被剔除，test 对
    train∪dev 同理（test partition 本阶段不构造，函数已支持，最终解锁时
    生效）；train 侧全量保留——重复是真实再诊数据。去重基准必须随场景切换
    （主线仲裁 2026-11-02）：short 场景按主诉判定（真实语料剔 2,065 条，
    LR 0.3835→0.3497），临床全文基准会漏掉主诉相同但现病史不同的可记忆
    副本。见 data.tcm_sd.deduplicate_evaluation_records。
    """
    from importlib import import_module

    tcm_sd = import_module("data.tcm_sd")
    if scenario == "clinical":
        text_fn = tcm_sd.build_clinical_text
    elif scenario == "short":
        text_fn = tcm_sd.build_short_text
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    kept_records, dedup_stats = tcm_sd.deduplicate_evaluation_records(
        records, partitions, text_basis=text_fn
    )
    out: dict = {}
    unknown = {"train": 0, "dev": 0}
    threshold_rows: list[dict[str, object]] = []
    for split in ("train", "dev"):
        texts: list[str] = []
        labels: list[int] = []
        for r in kept_records:
            if r.user_id not in partitions[split]:
                continue
            idx = label_space.resolve(r.norm_syndrome)
            if idx is None:
                unknown[split] += 1
                continue
            input_text = text_fn(r)
            texts.append(input_text)
            labels.append(idx)
            if split == "dev":
                source_file = getattr(r, "source_file", "")
                threshold_rows.append(
                    {
                        "source_file": Path(source_file).name,
                        "subject_index": getattr(r, "subject_index", None),
                        "record_id": getattr(r, "lcd_id", None),
                        "user_id": r.user_id,
                        "label_index": idx,
                        "input_sha256": hashlib.sha256(
                            input_text.encode("utf-8")
                        ).hexdigest(),
                    }
                )
        out[split] = {"texts": texts, "labels": labels}
    out["unknown_labels"] = unknown
    out["dedup"] = dedup_stats
    selection_payload = {
        "scenario": scenario,
        "text_builder": text_fn.__name__,
        "dev_rows": threshold_rows,
        "evaluation_dedup": dedup_stats,
        "unknown_dev_labels_excluded": unknown["dev"],
    }
    out["threshold_selection_identity"] = {
        "scenario": scenario,
        "text_builder": text_fn.__name__,
        "record_count": len(threshold_rows),
        "dataset_sha256": _identity_sha256(selection_payload),
    }
    return out


# --------------------------------------------------------------------------- 基线


def run_baseline_experiment(
    raw_dir: str,
    output_dir: str,
    seed: int = 20261101,
    n_boot: int = 1000,
    solver: str = "ovr-liblinear",
) -> dict:
    records, partitions, label_space, aux = load_partitions(raw_dir)
    config = TfidfNgramConfig(seed=seed, solver=solver)
    report: dict = {
        "taxonomy_id": label_space.taxonomy_id,
        "n_labels": len(label_space),
        "partitions": {k: len(v) for k, v in partitions.items()},
        **aux,
        "scenarios": {},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        data = scenario_data(records, partitions, label_space, scenario)
        train, dev = data["train"], data["dev"]
        dummy = MostFrequentBaseline().fit(train["texts"], train["labels"])
        dummy_pred = dummy.predict(dev["texts"])
        tfidf = TfidfNgramBaseline(config).fit(train["texts"], train["labels"])
        tfidf_pred = tfidf.predict(dev["texts"])
        tfidf_scores = validate_probability_matrix(
            tfidf.predict_proba(dev["texts"]), len(dev["texts"]), len(label_space)
        )
        summary = summarize_evaluation(
            dev["labels"],
            tfidf_pred,
            tfidf_scores,
            label_space.labels,
            baseline_pred=dummy_pred,
            n_boot=n_boot,
            seed=seed,
        )
        summary["dummy_macro_f1"] = round(macro_f1(dev["labels"], dummy_pred, len(label_space)), 6)
        summary["unknown_labels_excluded"] = data["unknown_labels"]
        summary["evaluation_dedup"] = data["dedup"]
        summary["baseline_provenance"] = {
            "dummy": dummy.provenance(),
            "tfidf": tfidf.provenance(),
        }
        report["scenarios"][scenario] = summary
        with open(out / f"baselines-{scenario}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open(out / "baselines-report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def run_robustness_suite(
    raw_dir: str,
    output_dir: str,
    seed: int = 20261101,
    threshold: float | None = None,
    n_boot: int = 1000,
    solver: str = "ovr-liblinear",
) -> dict:
    """短输入 / 对抗输入的 coverage / 拒判率 / 性能下降（dev partition，TF-IDF 模型）。

    否定症状（如「无发热」）的真实数据改写没有可用真值：真实否定鲁棒性为
    NOT_AVAILABLE，仅由 tests/syndrome 的合成否定用例覆盖接口行为。
    """
    records, partitions, label_space, _aux = load_partitions(raw_dir)
    data = scenario_data(records, partitions, label_space, "short")
    train, dev = data["train"], data["dev"]
    tfidf = TfidfNgramBaseline(TfidfNgramConfig(seed=seed, solver=solver)).fit(
        train["texts"], train["labels"]
    )

    dev_scores = validate_probability_matrix(
        tfidf.predict_proba(dev["texts"]), len(dev["texts"]), len(label_space)
    )
    dev_pred = tfidf.predict(dev["texts"])
    full_f1 = macro_f1(dev["labels"], dev_pred, len(label_space))

    threshold_selected_on_dev = threshold is None
    if threshold_selected_on_dev:
        max_probs = dev_scores.max(axis=1)
        correct = [p == y for p, y in zip(dev_pred, dev["labels"], strict=False)]
        threshold = select_uncertainty_threshold(max_probs, correct)["threshold"]

    truncated = 20
    trunc_texts = [t[:truncated] for t in dev["texts"]]
    trunc_pred = tfidf.predict(trunc_texts)
    trunc_f1 = macro_f1(dev["labels"], trunc_pred, len(label_space))

    inference = SyndromeInference(
        tfidf,
        label_space,
        model_version="tfidf-char-1-3-logreg",
        provenance={"scenario": "short"},
        threshold=threshold,
        threshold_scope=(
            "selected on dev partition; holdout untouched"
            if threshold_selected_on_dev
            else "caller-provided; selection scope unverified"
        ),
    )
    statuses = [r.status for r in inference.predict_batch(list(ADVERSARIAL_PROBES))]
    rejected = [s for s in statuses if s != "OK"]

    report = {
        "taxonomy_id": label_space.taxonomy_id,
        "scenario": "short",
        "uncertainty_threshold": threshold,
        "dev_macro_f1_full_text": round(full_f1, 6),
        "dev_macro_f1_truncated_20": round(trunc_f1, 6),
        "degradation_truncated": round(full_f1 - trunc_f1, 6),
        "top3_accuracy_full": round(topk_accuracy(dev["labels"], dev_scores, 3), 6),
        "adversarial_probes": list(ADVERSARIAL_PROBES),
        "adversarial_statuses": statuses,
        "adversarial_reject_rate": round(len(rejected) / len(statuses), 6),
        "negation_robustness": "NOT_AVAILABLE",
        "negation_reason": "真实病例否定改写无真值；接口行为由合成测试覆盖",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness-short.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


# --------------------------------------------------------------------------- BERT


def run_bert_experiment(
    raw_dir: str,
    weights_dir: str,
    output_dir: str,
    scenario: str = "short",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 256,
    seed: int = 20261101,
    n_boot: int = 1000,
) -> dict:
    from syndrome.bert_model import MODEL_VERSION, BertSyndromeClassifier, BertTrainConfig

    records, partitions, label_space, aux = load_partitions(raw_dir)
    data = scenario_data(records, partitions, label_space, scenario)
    train, dev = data["train"], data["dev"]
    config = BertTrainConfig(
        model_path=weights_dir,
        max_len=max_len,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )
    run_dir = Path(output_dir) / f"bert-{scenario}"
    clf = BertSyndromeClassifier(label_space, weights_dir, seed=seed)
    fit_info = clf.fit(
        train["texts"], train["labels"], dev["texts"], dev["labels"], config, str(run_dir)
    )
    dev_scores = validate_probability_matrix(
        clf.predict_proba(dev["texts"]), len(dev["texts"]), len(label_space)
    )
    dev_pred = [int(i) for i in dev_scores.argmax(axis=1)]
    summary = summarize_evaluation(
        dev["labels"],
        dev_pred,
        dev_scores,
        label_space.labels,
        n_boot=n_boot,
        seed=seed,
    )
    summary["model_version"] = MODEL_VERSION
    summary["fit"] = fit_info
    summary["unknown_labels_excluded"] = data["unknown_labels"]
    summary["evaluation_dedup"] = data["dedup"]
    threshold_info = select_uncertainty_threshold(
        dev_scores.max(axis=1), [p == y for p, y in zip(dev_pred, dev["labels"], strict=False)]
    )
    summary["threshold"] = threshold_info
    summary["threshold_provenance"] = _threshold_binding(
        run_dir,
        aux,
        _validated_threshold_metrics(threshold_info),
        selection_identity=data["threshold_selection_identity"],
    )
    with open(run_dir / "dev-evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary
def rescore_saved_bert(
    raw_dir: str,
    output_dir: str,
    scenario: str = "short",
    seed: int = 20261101,
    n_boot: int = 1000,
) -> dict:
    """去重后 dev 上重评分已保存 BERT 权重；校验原阈值绑定后原样继承。"""
    from syndrome.bert_model import MODEL_VERSION, BertSyndromeClassifier

    records, partitions, label_space, aux = load_partitions(raw_dir)
    data = scenario_data(records, partitions, label_space, scenario)
    dev = data["dev"]
    run_dir = Path(output_dir) / f"bert-{scenario}"
    prev_path = run_dir / "dev-evaluation.json"
    if not prev_path.exists():
        raise FileNotFoundError(
            f"no saved evaluation at {prev_path}; rescore needs the weights "
            "and threshold of a completed bert run"
        )
    with open(prev_path, encoding="utf-8") as fh:
        previous = json.load(fh)
    threshold_binding = _validate_threshold_binding(
        previous.get("threshold"),
        previous.get("threshold_provenance"),
        run_dir,
        aux,
        selection_identity=data["threshold_selection_identity"],
    )
    clf = BertSyndromeClassifier.load(str(run_dir), expected_label_space=label_space)
    dev_scores = validate_probability_matrix(
        clf.predict_proba(dev["texts"]), len(dev["texts"]), len(label_space)
    )
    dev_pred = [int(i) for i in dev_scores.argmax(axis=1)]
    summary = summarize_evaluation(
        dev["labels"],
        dev_pred,
        dev_scores,
        label_space.labels,
        n_boot=n_boot,
        seed=seed,
    )
    summary["model_version"] = MODEL_VERSION
    summary["unknown_labels_excluded"] = data["unknown_labels"]
    summary["evaluation_dedup"] = data["dedup"]
    summary["threshold"] = previous["threshold"]
    summary["threshold_provenance"] = threshold_binding
    summary["threshold_note"] = (
        "threshold and best-epoch selection happened on the pre-dedup dev; "
        "inherited unchanged here (selection dev != scoring dev, not re-selected)"
    )
    summary["rescore_of"] = {
        "source_evaluation": str(prev_path),
        "weight_sha256": threshold_binding["weights_sha256"],
    }
    with open(run_dir / "dev-evaluation-deduped.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary
def _inference_scenario(model_dir: str | Path, scenario: str | None) -> str:
    if scenario is None:
        directory_name = Path(model_dir).name
        prefix = "bert-"
        if not directory_name.startswith(prefix):
            raise ValueError(
                "scenario is required unless model_dir is named bert-short or bert-clinical"
            )
        scenario = directory_name.removeprefix(prefix)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown threshold selection scenario: {scenario!r}")
    return scenario


def build_bert_inference(
    model_dir: str,
    model_version: str = "tcm-sd-syndrome-bert-v1",
    knowledge: dict | None = None,
    raw_dir: str | None = None,
    scenario: str | None = None,
) -> SyndromeInference:
    """Load BERT inference only when threshold provenance matches its dev selection."""
    eval_path = Path(model_dir) / "dev-evaluation.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing dev threshold artifact: {eval_path}")
    try:
        with open(eval_path, encoding="utf-8") as fh:
            evaluation = json.load(fh)
        threshold_info = evaluation["threshold"]
        threshold = float(threshold_info["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid dev threshold artifact: {eval_path}") from exc
    if raw_dir is None:
        raise ValueError("raw_dir is required to verify dev threshold provenance")
    scenario = _inference_scenario(model_dir, scenario)
    records, partitions, label_space, aux = load_partitions(raw_dir)
    selection_data = scenario_data(records, partitions, label_space, scenario)
    weight = Path(model_dir) / "model.safetensors"
    binding = _validate_threshold_binding(
        threshold_info,
        evaluation.get("threshold_provenance"),
        Path(model_dir),
        aux,
        selection_identity=selection_data["threshold_selection_identity"],
    )

    from importlib import import_module

    from syndrome.bert_model import BertSyndromeClassifier

    if knowledge is None:
        knowledge = import_module("data.tcm_sd").load_syndrome_knowledge(raw_dir)
    clf = BertSyndromeClassifier.load(model_dir, expected_label_space=label_space)
    return SyndromeInference(
        clf,
        clf.label_space,
        model_version=model_version,
        provenance={"artifact_dir": model_dir, "threshold_binding": binding},
        threshold=threshold,
        knowledge=knowledge,
        weight_path=str(weight),
        threshold_scope="selected on dev partition; holdout untouched",
    )
# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="#53 stage-1 training entry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baselines")
    p_base.add_argument("--raw-dir", default="data/local/tcm-sd")
    p_base.add_argument("--out", default="runs/tcm-sd-syndrome-v1")
    p_base.add_argument("--seed", type=int, default=20261101)
    p_base.add_argument(
        "--solver", choices=["ovr-liblinear", "multinomial-saga"], default="ovr-liblinear"
    )

    p_rob = sub.add_parser("robustness")
    p_rob.add_argument("--raw-dir", default="data/local/tcm-sd")
    p_rob.add_argument("--out", default="runs/tcm-sd-syndrome-v1")
    p_rob.add_argument("--seed", type=int, default=20261101)
    p_rob.add_argument(
        "--solver", choices=["ovr-liblinear", "multinomial-saga"], default="ovr-liblinear"
    )

    p_bert = sub.add_parser("bert")
    p_bert.add_argument("--raw-dir", default="data/local/tcm-sd")
    p_bert.add_argument("--weights", default="data/local/models/bert-base-chinese")
    p_bert.add_argument("--out", default="runs/tcm-sd-syndrome-v1")
    p_bert.add_argument("--scenario", choices=sorted(SCENARIOS), default="short")
    p_bert.add_argument("--epochs", type=int, default=3)
    p_bert.add_argument("--batch-size", type=int, default=32)
    p_bert.add_argument("--lr", type=float, default=2e-5)
    p_bert.add_argument("--max-len", type=int, default=256)
    p_bert.add_argument("--seed", type=int, default=20261101)

    p_rescore = sub.add_parser(
        "bert-rescore",
        help="去重后 dev 上重评分已保存的 BERT 权重（不重训）",
    )
    p_rescore.add_argument("--raw-dir", default="data/local/tcm-sd")
    p_rescore.add_argument("--out", default="runs/tcm-sd-syndrome-v1")
    p_rescore.add_argument("--scenario", choices=sorted(SCENARIOS), default="short")
    p_rescore.add_argument("--seed", type=int, default=20261101)

    args = parser.parse_args(argv)
    if args.command == "baselines":
        report = run_baseline_experiment(args.raw_dir, args.out, seed=args.seed, solver=args.solver)
    elif args.command == "robustness":
        report = run_robustness_suite(args.raw_dir, args.out, seed=args.seed, solver=args.solver)
    elif args.command == "bert-rescore":
        report = rescore_saved_bert(
            args.raw_dir, args.out, scenario=args.scenario, seed=args.seed,
        )
    else:
        report = run_bert_experiment(
            args.raw_dir,
            args.weights,
            args.out,
            scenario=args.scenario,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            seed=args.seed,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
