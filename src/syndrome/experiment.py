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
import json
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
)
from syndrome.labels import LabelSpace
from syndrome.metrics import macro_f1, summarize_evaluation, topk_accuracy

SCENARIOS = {"clinical": "build_clinical_text", "short": "build_short_text"}

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
    """加载记录、subject partitions（官方文件边界仅作对照记录）与冻结词表。

    记录经加载层泄漏剔除（R1 F2，EXCLUDED_LEAK_RECORDS）。
    """
    from importlib import import_module

    tcm_sd = import_module("data.tcm_sd")
    records = tcm_sd.load_tcm_sd_records(raw_dir)
    partitions = tcm_sd.subject_partitions(records)
    label_space = LabelSpace(tcm_sd.load_syndrome_vocab(raw_dir))
    official = {
        split: sum(1 for r in records if r.source_file.endswith(f"{split}.json"))
        for split in ("train", "dev", "test")
    }
    return records, partitions, label_space, {"official_file_counts": official}


def scenario_data(
    records: list,
    partitions: dict[str, frozenset[str]],
    label_space: LabelSpace,
    scenario: str,
) -> dict:
    """构造一个文本场景的 train/dev 数据；UNKNOWN 标签如实报告并剔除（不补标签）。

    评估侧单向去重（R1 F1，在 partitions 计算后应用）：dev 中与任一 train
    病例临床全文（build_clinical_text）逐字节相同的病例被剔除，test 对
    train∪dev 同理（test partition 本阶段不构造，函数已支持，最终解锁时
    生效）；train 侧全量保留——重复是真实再诊数据。去重按临床全文判定，
    不随文本场景切换。见 data.tcm_sd.deduplicate_evaluation_records。
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
        records, partitions
    )
    out: dict = {}
    unknown = {"train": 0, "dev": 0}
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
            texts.append(text_fn(r))
            labels.append(idx)
        out[split] = {"texts": texts, "labels": labels}
    out["unknown_labels"] = unknown
    out["dedup"] = dedup_stats
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
        tfidf_scores = tfidf.predict_proba(dev["texts"])
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

    dev_scores = tfidf.predict_proba(dev["texts"])
    dev_pred = tfidf.predict(dev["texts"])
    full_f1 = macro_f1(dev["labels"], dev_pred, len(label_space))

    if threshold is None:
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

    dev_scores = clf.predict_proba(dev["texts"])
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
    """去重后 dev 上重评分已保存的 BERT 权重（R1 F1）：不重训、不重选阈值。

    模型选择（最佳 epoch）与 UNCERTAIN 阈值都发生在去重前的 dev 上，这里
    原样继承——选择用 dev 与评分用 dev 的差异如实入档（threshold_note），
    去重后 dev 只用于评分。结果写入 <out>/bert-<scenario>/dev-evaluation-deduped.json。
    """
    from syndrome.bert_model import MODEL_VERSION, BertSyndromeClassifier

    records, partitions, label_space, _aux = load_partitions(raw_dir)
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

    clf = BertSyndromeClassifier.load(str(run_dir))
    dev_scores = clf.predict_proba(dev["texts"])
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
    summary["threshold_note"] = (
        "threshold and best-epoch selection happened on the pre-dedup dev; "
        "inherited unchanged here (selection dev != scoring dev, not re-selected)"
    )
    summary["rescore_of"] = {
        "source_evaluation": str(prev_path),
        "weight_sha256": previous.get("weight_sha256", "NOT_AVAILABLE"),
    }
    with open(run_dir / "dev-evaluation-deduped.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def build_bert_inference(
    model_dir: str,
    model_version: str = "tcm-sd-syndrome-bert-v1",
    knowledge: dict | None = None,
    raw_dir: str | None = None,
) -> SyndromeInference:
    """从 runs/ 的已保存模型目录构建标准推理对象（阈值从 dev-evaluation.json 读）。

    knowledge 可直接传入 syndrome_knowledge 映射，或给 raw_dir 让本函数加载
    （依赖 data.tcm_sd 契约）。
    """
    from importlib import import_module

    from syndrome.bert_model import BertSyndromeClassifier

    if knowledge is None and raw_dir is not None:
        knowledge = import_module("data.tcm_sd").load_syndrome_knowledge(raw_dir)
    clf = BertSyndromeClassifier.load(model_dir)
    eval_path = Path(model_dir) / "dev-evaluation.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as fh:
            threshold = float(json.load(fh)["threshold"]["threshold"])
    else:
        threshold = 0.0
    weight = Path(model_dir) / "model.safetensors"
    return SyndromeInference(
        clf,
        clf.label_space,
        model_version=model_version,
        provenance={"artifact_dir": model_dir},
        threshold=threshold,
        knowledge=knowledge,
        weight_path=str(weight) if weight.exists() else None,
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
