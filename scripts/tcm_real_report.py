"""#50 final deliverable report generator (dev host only, real data).

Consumes the AutoResearch run directory + frozen manifest and produces:
- final dev/holdout evaluation of the champion (per-dim F1 + macro F1)
- Dummy / BMI-only / logistic baselines on the same partitions
- row-273 QC sensitivity (retrain baseline config excluding subject 273)
- the separate 8-score PUBLISHED regression task (fixed config, per-dim MAE)
- charts (round metrics, per-dim F1, class support, regression MAE)
- report.md and model-card.md

Everything lands under --out (git-ignored artifacts/); the committed report
references artifact hashes, never raw data.

Usage:
  PYTHONPATH=src python scripts/tcm_real_report.py \
    --runs-dir runs/real-tcm --manifest <manifest.json> --dataset <data.xls> \
    --out artifacts/tcm-50
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from data.mendeley import load_mendeley_scores  # noqa: E402
from data.real_split import build_real_manifest, load_real_manifest  # noqa: E402
from runtime.baselines import CANONICAL_DIMS, evaluate_baselines, evaluate_neural  # noqa: E402
from runtime.checkpoint import load_training_checkpoint  # noqa: E402
from runtime.manifest import MANIFEST_FILE, RunManifestStore  # noqa: E402
from runtime.real_data import RealBatchProvider, build_real_model, real_eval_manifest  # noqa: E402
from runtime.tcm_eval import TCMTrustedEvaluator  # noqa: E402
from runtime.trainer import Trainer, TrainerConfig  # noqa: E402

KEEP_COLOR, REVERT_COLOR = "#2a9d8f", "#e76f51"
BASELINE_STYLE = {
    "dummy_majority": ("Dummy (majority)", "#adb5bd"),
    "bmi_only_logistic": ("BMI-only logistic", "#e9c46a"),
    "feature_logistic": ("Logistic (all features)", "#4361ee"),
    "neural_champion": ("Neural champion (AutoResearch)", "#2a9d8f"),
}


def dev_primary(model, manifest, records) -> float:
    evaluator = TCMTrustedEvaluator(real_eval_manifest(manifest))
    provider = RealBatchProvider(manifest, records, "dev")
    report = evaluator.evaluate(model, provider, partition="dev")
    return report.subject_mae


def _fixed_training(manifest, records, model_cfg: dict, epochs: int, out_dir: Path):
    model = build_real_model(manifest, model_cfg)
    trainer = Trainer(model, TrainerConfig(seed=0, epochs=epochs, learning_rate=3e-3))
    trainer.fit(
        RealBatchProvider(manifest, records, "train", batch_size=32),
        RealBatchProvider(manifest, records, "dev", batch_size=64),
    )
    trainer.save_checkpoint(
        out_dir, smoke_only=False,
        extra_state={"manifest_sha256": manifest["manifest_sha256"]},
    )
    model, _, _ = load_training_checkpoint(out_dir)
    return model


def regression_task(manifest_cls, records, out_dir: Path) -> dict:
    """Separate 8-score PUBLISHED regression training (fixed config)."""
    manifest = build_real_manifest(
        records,
        dataset_sha256=manifest_cls["dataset_sha256"],
        task_type="score_regression",
        modality_set="full",
        split_seed=manifest_cls["split"]["seed"],
        notes="regression deliverable: same split seed as the classification manifest",
    ).data
    model = _fixed_training(
        manifest, records,
        {"include_category_head": False, "include_score_head": True}, 60,
        out_dir / "regression",
    )
    per_dim: dict = {}
    model.eval()
    with torch.no_grad():
        for partition in ("dev", "holdout"):
            provider = RealBatchProvider(manifest, records, partition, batch_size=64)
            sq: dict[int, list] = {d: [] for d in range(9)}
            absd: dict[int, list] = {d: [] for d in range(9)}
            for item in provider:
                out = model(item.batch)
                pred = out.predictions["constitution_scores"]
                target = item.batch.targets["constitution_scores"]
                mask = item.batch.target_mask["constitution_scores"]
                for d in range(9):
                    obs = mask[:, d]
                    if obs.any():
                        diffs = (pred[:, d] - target[:, d])[obs]
                        absd[d].extend(diffs.abs().tolist())
                        sq[d].extend((diffs**2).tolist())
            per_dim[partition] = {
                CANONICAL_DIMS[d]: {
                    "mae": sum(absd[d]) / len(absd[d]) if absd[d] else None,
                    "rmse": (sum(sq[d]) / len(sq[d])) ** 0.5 if sq[d] else None,
                }
                for d in range(9)
            }
    return {"manifest_sha256": manifest["manifest_sha256"], "per_dim": per_dim}


def row273_sensitivity(manifest, records, out_dir: Path) -> dict:
    """Retrain the champion-free baseline config excluding the QC subject."""
    manifest_qc = build_real_manifest(
        records,
        dataset_sha256=manifest["dataset_sha256"],
        task_type="derived_multilabel_classification",
        modality_set="full",
        split_seed=manifest["split"]["seed"],
        exclude_subject_rows=(273,),
        notes="row-273 QC sensitivity: exclude the fingerprint-anomalous subject",
    ).data
    model_qc = _fixed_training(
        manifest_qc, records, {"hidden_dim": 32, "fusion_dim": 32}, 40,
        out_dir / "qc-exclude-273",
    )
    records_qc = [r for r in records if r.subject_index != 273]
    return {
        "manifest_sha256": manifest_qc["manifest_sha256"],
        "dev_subject_mae_without_273": dev_primary(model_qc, manifest_qc, records_qc),
    }


def deployable_ablation(manifest, records, out_dir: Path) -> dict:
    """Body/Vitals-only training (#50 deployability): same split seed."""
    manifest_dep = build_real_manifest(
        records,
        dataset_sha256=manifest["dataset_sha256"],
        task_type="derived_multilabel_classification",
        modality_set="deployable",
        split_seed=manifest["split"]["seed"],
        notes="deployable Body/Vitals-only ablation (no exam hardware needed)",
    ).data
    model_dep = _fixed_training(
        manifest_dep, records, {"hidden_dim": 32, "fusion_dim": 32}, 40,
        out_dir / "deployable-only",
    )
    return {
        "manifest_sha256": manifest_dep["manifest_sha256"],
        "dev_subject_mae": dev_primary(model_dep, manifest_dep, records),
        "f1": evaluate_neural(model_dep, manifest_dep, records),
    }


def chart_rounds(entries: list[dict], out: Path) -> None:
    rounds = [e for e in entries if e.get("metrics")]
    xs = list(range(1, len(rounds) + 1))
    ys = [e["metrics"]["primary_value"] for e in rounds]
    colors = [KEEP_COLOR if e["decision"] == "KEEP" else REVERT_COLOR for e in rounds]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(xs, ys, color=colors)
    best = min(ys) if ys else 0
    ax.axhline(best, color="#264653", linestyle="--", linewidth=1,
               label=f"best {best:.4f}")
    ax.set_xlabel("AutoResearch round")
    ax.set_ylabel("dev subject MAE\n(lower is better)")
    ax.set_title("AutoResearch rounds — primary metric")
    ax.legend()
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "chart-rounds.png", dpi=150)
    plt.close(fig)


def chart_per_dim_f1(baselines: dict, neural: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    series = []
    for key, (label, color) in BASELINE_STYLE.items():
        if key == "neural_champion":
            vals = [neural["holdout"]["per_dim"][d]["f1"] for d in dims]
        else:
            entry = baselines.get(key)
            if not entry:
                continue
            vals = [entry["holdout"]["per_dim"][d]["f1"] for d in dims]
        series.append((label, color, vals))
    width = 0.8 / len(series)
    xs = range(len(dims))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (label, color, vals) in enumerate(series):
        pos = [x + i * width for x in xs]
        ax.bar(pos, [v or 0.0 for v in vals], width, label=label, color=color)
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("holdout F1 (>=40 = YES)")
    ax.set_title("Per-dimension holdout F1 — neural champion vs baselines")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "chart-per-dim-f1.png", dpi=150)
    plt.close(fig)


def chart_support(neural: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    support = [neural["dev"]["per_dim"][d]["support_positive"] for d in dims]
    total = [neural["dev"]["per_dim"][d]["n_observed"] for d in dims]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims, total, color="#dee2e6", label="observed")
    ax.bar(dims, support, color="#2a9d8f", label="positive (YES)")
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("dev subjects")
    ax.set_title("Per-dimension label support (dev partition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart-support.png", dpi=150)
    plt.close(fig)


def chart_regression(regression: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    for partition, fname in (("dev", "chart-regression-dev.png"),
                             ("holdout", "chart-regression-holdout.png")):
        maes = [regression["per_dim"][partition][d]["mae"] or 0.0 for d in dims]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dims, maes, color="#4361ee")
        ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"{partition} MAE (0-100 score points)")
        ax.set_title("8-score PUBLISHED regression — per-dimension MAE")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def write_reports(out: Path, summary: dict, manifest: dict, records_count: int) -> None:
    ar = summary["autoresearch"]
    neural = summary["neural_evaluation"]
    base = summary["baselines"]
    regression = summary["regression_task"]
    qc = summary["row273_sensitivity"]
    dep = summary["deployable_ablation"]

    keep = [e["run_id"] for e in ar["per_round"] if e["decision"] == "KEEP"]
    revert = [e["run_id"] for e in ar["per_round"] if e["decision"] == "REVERT"]

    def holdout_macro_f1(entry: dict) -> str:
        value = entry["holdout"]["macro_f1"]
        return f"{value:.3f}" if value is not None else "N/A"

    lines = [
        "# TCM 真实训练报告（#50，AutoResearch V1）",
        "",
        f"生成时间：{summary['generated_at']}  ",
        f"数据：`mendeley-cxtph3tjsg-v1`（SHA-256 `{summary['dataset_sha256'][:16]}…`，"
        f"{records_count} 名受试者，row-273 QC 挂起）  ",
        f"标准：{summary['standard_version']}；标签契约：`{summary['label_version']}`  ",
        f"冻结 manifest：`{summary['manifest_sha256'][:16]}…`；划分："
        f"train {summary['split_sizes']['train']} / dev {summary['split_sizes']['dev']} / "
        f"holdout {summary['split_sizes']['holdout']}（subject-level，#27）",
        "",
        "## AutoResearch 轮次",
        "",
        f"共 {ar['rounds_completed']} 轮：KEEP {keep}，REVERT {revert}。",
        "冠军：`" + ar["champion"]["run_id"] + "`（dev subject MAE "
        f"{ar['champion']['best_so_far']:.4f}，权重 SHA-256 "
        f"`{(ar['champion']['artifact_sha256'] or '')[:16]}…`）。",
        "",
        "| 轮 | 意图 | 决定 | dev MAE |",
        "| --- | --- | --- | --- |",
    ]
    for e in ar["per_round"]:
        value = e.get("primary_value")
        cell = f"{value:.4f}" if value else "—"
        lines.append(f"| {e['run_id']} | {e['intent']} | {e['decision']} | {cell} |")
    lines += [
        "",
        "## 最终评估（holdout 仅在研究结束后解锁一次）",
        "",
        f"- 神经冠军：dev macro-F1 {neural['dev']['macro_f1']:.3f}，"
        f"holdout macro-F1 {neural['holdout']['macro_f1']:.3f}",
        f"- Dummy（多数类）：holdout macro-F1 {holdout_macro_f1(base['dummy_majority'])}",
        f"- BMI-only 逻辑回归：holdout macro-F1 {holdout_macro_f1(base['bmi_only_logistic'])}",
        f"- 全特征逻辑回归：holdout macro-F1 {holdout_macro_f1(base['feature_logistic'])}",
        "",
        "图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、"
        "`chart-regression-dev.png`、`chart-regression-holdout.png`。",
        "",
        "## 独立交付任务",
        "",
        f"- **8-score PUBLISHED 回归**（独立 manifest，固定配置）：dev 逐维 MAE 见 "
        f"`summary.json`（regression_task），manifest `{regression['manifest_sha256'][:16]}…`。",
        f"- **row-273 敏感性**：排除后 dev subject MAE "
        f"{qc['dev_subject_mae_without_273']:.4f}（含 273 的冠军为 "
        f"{qc['dev_subject_mae_with_273']:.4f}）。",
        f"- **Deployable（Body/Vitals-only）消融**：dev subject MAE "
        f"{dep['dev_subject_mae']:.4f}，dev macro-F1 {dep['f1']['dev']['macro_f1']:.3f}",
        "  （经络/舌/脉特征在现场不可得时的可部署口径，#50）。",
        "",
        "## 证据状态与限制",
        "",
        "- 训练模式：REAL_DATA_TRAINING（agent provider=确定性计划，AGENT_NOT_VERIFIED；"
        "OS/container 隔离仍未验证，AGENT_ISOLATION_UNVERIFIED）。",
        "- 标签为 RULE 派生类别（>=40=YES；TENDENCY/NO=0；平和维 ABSTAIN）——不是作者发布的"
        "分类标签，也不是临床诊断真值。",
        "- 列映射为 FINGERPRINT_CROSS_CHECKED（论文定义表付费墙内）；row 273 四列指纹异常。",
        "- n=274、类别支持不均衡（见 chart-support.png）；F1/AUROC 置信区间未做重采样；"
        "结果仅代表研究口径，不可临床使用。",
        "- 负类为主的维度上 Dummy/逻辑回归可能超过网络——逐维对比见 chart-per-dim-f1.png。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    card = [
        "# TCM Constitution Model Card（#50 交付）",
        "",
        f"- model_type：`tcm_constitution_v1`；champion run：`{ar['champion']['run_id']}`",
        f"- 权重：`{ar['champion']['artifact_dir']}/weights.pt`（SHA-256 "
        f"`{(ar['champion']['artifact_sha256'] or '')}`）",
        "- 输入契约：`tcm-input-real-v1`（"
        + ", ".join(f"{k}({v['dim']})" for k, v in manifest["modalities"].items())
        + "；train-fold 标准化，缺失模态行 availability-masked）",
        "- 输出：9 维独立 sigmoid（`category_probs`）；平和维永远 ABSTAIN",
        f"- 训练数据：{records_count} 名受试者（Mendeley cxtph3tjsg/1，CC BY 4.0），"
        f"RULE 派生二值标签（`{summary['label_version']}`）",
        "- 适用人群：与研究样本同源的成年体检人群；不适用于临床诊断、无按摩场景外的任何干预决策",
        "- 已知偏差：类别支持不均衡；体型代理变量（BMI）基线较强；row 273 异常未混入敏感性对照",
        "- 禁止：临床诊断、GB/T 46939—2025 语义混用、LLM_SILVER 当真值",
    ]
    (out / "model-card.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="artifacts/tcm-50")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_real_manifest(args.manifest)
    records = load_mendeley_scores(args.dataset)

    store = RunManifestStore(Path(args.runs_dir) / MANIFEST_FILE)
    champion = store.champion()
    if not champion:
        raise SystemExit("no AutoResearch champion found; run the research loop first")
    model, state, _ = load_training_checkpoint(champion["artifact_dir"])
    assert not state["smoke_only"]
    entries = store.load()

    neural_eval = evaluate_neural(model, manifest, records)
    baselines = evaluate_baselines(manifest, records)
    regression = regression_task(manifest, records, out)
    qc = row273_sensitivity(manifest, records, out)
    qc["dev_subject_mae_with_273"] = dev_primary(model, manifest, records)
    deployable = deployable_ablation(manifest, records, out)

    chart_rounds(entries, out)
    chart_per_dim_f1(baselines, neural_eval, out)
    chart_support(neural_eval, out)
    chart_regression(regression, out)

    per_round = [
        {**{k: e.get(k) for k in ("run_id", "intent", "decision", "status",
                                  "artifact_sha256")},
         "primary_value": (e.get("metrics") or {}).get("primary_value")}
        for e in entries
    ]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_version": manifest["label_version"],
        "standard_version": manifest["standard_version"],
        "split_sizes": {p: len(manifest["split"][p]) for p in ("train", "dev", "holdout")},
        "autoresearch": {
            "rounds_completed": len(entries),
            "champion": {
                "run_id": champion["run_id"],
                "artifact_dir": champion["artifact_dir"],
                "artifact_sha256": champion.get("artifact_sha256"),
                "best_so_far": champion.get("best_so_far"),
            },
            "per_round": per_round,
        },
        "neural_evaluation": neural_eval,
        "baselines": baselines,
        "regression_task": regression,
        "row273_sensitivity": qc,
        "deployable_ablation": deployable,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_reports(out, summary, manifest, len(records))
    print(json.dumps({
        "out": str(out),
        "champion_run": champion["run_id"],
        "dev_subject_mae": champion.get("best_so_far"),
        "dev_macro_f1": neural_eval["dev"]["macro_f1"],
        "holdout_macro_f1": neural_eval["holdout"]["macro_f1"],
        "baseline_holdout_macro_f1": {
            k: (v["holdout"]["macro_f1"] if isinstance(v, dict) and "holdout" in v else None)
            for k, v in baselines.items() if k != "label_version"
        },
        "row273_without": qc["dev_subject_mae_without_273"],
        "deployable_dev_mae": deployable["dev_subject_mae"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
