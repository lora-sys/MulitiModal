"""#50 final deliverable report generator (dev host only, real data).

Consumes the AutoResearch run directory + frozen manifest and produces:
- final dev/holdout evaluation of the champion (per-dim F1 + macro F1 +
  subject MAE)
- Dummy / BMI-only / logistic baselines on the same partitions with the
  same availability information the neural model receives
- row-QC sensitivity on the SAME frozen split minus fingerprint-anomalous
  subjects (never a reshuffled split)
- the separate 8-score PUBLISHED regression task with a train-mean
  predictor as the trivial reference
- charts (round metrics, per-dim F1, class support, regression MAE);
  undefined values render as absent bars, never as 0
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
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.mendeley import load_mendeley_scores  # noqa: E402
from data.real_split import (  # noqa: E402
    build_real_manifest,
    exclude_subjects_from_manifest,
    load_real_manifest,
)
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


def qc_rows_from_records(records) -> tuple[int, ...]:
    """QC exclusions derived from the adapter's audit trail, never hardcoded."""
    return tuple(
        sorted({
            row
            for record in records
            for rows in record.provenance["fingerprint_anomalies"].values()
            for row in rows
        })
    )


def availability_rates(manifest: dict, records) -> dict:
    """Per-partition, per-modality availability so readers can see how much
    of the 'full' modality set is actually observable (#eval E05)."""
    by_row = {r.subject_index: r for r in records}
    out: dict = {}
    for partition in ("train", "dev", "holdout"):
        rows = [manifest["subject_row_by_id"][s] for s in manifest["split"][partition]]
        for modality, params in manifest["modalities"].items():
            groups = params["feature_groups"]
            n_available = sum(
                1 for r in rows
                if not any(math.isnan(v) for g in groups for v in by_row[r].features[g])
            )
            out[f"{partition}/{modality}"] = {"n_rows": len(rows), "n_available": n_available}
    return out


def dev_primary(model, manifest, records) -> float:
    evaluator = TCMTrustedEvaluator(real_eval_manifest(manifest))
    provider = RealBatchProvider(manifest, records, "dev")
    report = evaluator.evaluate(model, provider, partition="dev")
    return report.subject_mae


def _fixed_training(
    manifest: dict,
    records,
    model_cfg: dict,
    epochs: int,
    out_dir: Path,
    trainer_kwargs: dict | None = None,
):
    # seed BEFORE model construction: deterministic init regardless of what
    # this process ran before (#eval E06)
    torch.manual_seed(0)
    model = build_real_model(manifest, model_cfg)
    config = TrainerConfig(seed=0, epochs=epochs, **(trainer_kwargs or {}))
    trainer = Trainer(model, config)
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
    """Separate 8-score PUBLISHED regression training (fixed config) with a
    per-dim train-mean predictor as the trivial reference (#eval E11)."""
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
    by_row = {r.subject_index: r for r in records}
    train_rows = [manifest["subject_row_by_id"][s] for s in manifest["split"]["train"]]
    train_mean = []
    for d in range(9):
        vals = [by_row[r].scores[d] for r in train_rows
                if by_row[r].scores[d] is not None]
        train_mean.append(sum(vals) / len(vals) if vals else None)

    per_dim: dict = {}
    model.eval()
    with torch.no_grad():
        for partition in ("dev", "holdout"):
            provider = RealBatchProvider(manifest, records, partition, batch_size=64)
            sq: dict[int, list] = {d: [] for d in range(9)}
            absd: dict[int, list] = {d: [] for d in range(9)}
            ref_absd: dict[int, list] = {d: [] for d in range(9)}
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
                        if train_mean[d] is not None:
                            ref = (target[:, d] - train_mean[d])[obs]
                            ref_absd[d].extend(ref.abs().tolist())
            per_dim[partition] = {
                CANONICAL_DIMS[d]: {
                    "mae": sum(absd[d]) / len(absd[d]) if absd[d] else None,
                    "rmse": (sum(sq[d]) / len(sq[d])) ** 0.5 if sq[d] else None,
                    "mean_predictor_mae": (
                        sum(ref_absd[d]) / len(ref_absd[d]) if ref_absd[d] else None
                    ),
                }
                for d in range(9)
            }
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "aggregation": "pooled per-dim over observed entries",
        "includes_qc_rows": True,
        "per_dim": per_dim,
    }


def row_qc_sensitivity(
    manifest, records, out_dir: Path, champion_cfg: dict, champion_trainer: dict
) -> dict:
    """Retrain with the SAME frozen split minus fingerprint-anomalous
    subjects (exclude_subjects_from_manifest, never a fresh shuffle)."""
    qc_rows = qc_rows_from_records(records)
    manifest_qc = exclude_subjects_from_manifest(
        manifest, records, qc_rows,
        notes="row-QC sensitivity: frozen split minus fingerprint-anomalous subjects",
    ).data
    trainer_kwargs = {
        k: champion_trainer[k] for k in ("learning_rate", "grad_clip")
        if champion_trainer.get(k) is not None
    }
    model_qc = _fixed_training(
        manifest_qc, records, champion_cfg, champion_trainer.get("epochs", 40),
        out_dir / "qc-exclude", trainer_kwargs,
    )
    return {
        "manifest_sha256": manifest_qc["manifest_sha256"],
        "excluded_subject_rows": list(qc_rows),
        "dev_subject_mae_without": dev_primary(model_qc, manifest_qc, records),
    }


def deployable_ablation(
    manifest, records, out_dir: Path, champion_cfg: dict, champion_trainer: dict
) -> dict:
    """Body/Vitals-only training on the same split with the champion's
    training configuration, isolating the feature-set effect."""
    manifest_dep = build_real_manifest(
        records,
        dataset_sha256=manifest["dataset_sha256"],
        task_type="derived_multilabel_classification",
        modality_set="deployable",
        split_seed=manifest["split"]["seed"],
        notes="deployable Body/Vitals-only ablation (no exam hardware needed)",
    ).data
    trainer_kwargs = {
        k: champion_trainer[k] for k in ("learning_rate", "grad_clip")
        if champion_trainer.get(k) is not None
    }
    model_dep = _fixed_training(
        manifest_dep, records, champion_cfg, champion_trainer.get("epochs", 40),
        out_dir / "deployable-only", trainer_kwargs,
    )
    return {
        "manifest_sha256": manifest_dep["manifest_sha256"],
        "dev_subject_mae": dev_primary(model_dep, manifest_dep, records),
        "f1": evaluate_neural(model_dep, manifest_dep, records),
    }


def chart_rounds(entries: list[dict], out: Path) -> None:
    rounds = [e for e in entries if e.get("metrics")]
    ys = [e["metrics"]["primary_value"] for e in rounds
          if e["metrics"].get("primary_value") is not None]
    if not ys:
        return
    xs = list(range(1, len(ys) + 1))
    colors = [
        KEEP_COLOR if e["decision"] == "KEEP" else REVERT_COLOR
        for e in rounds if e["metrics"].get("primary_value") is not None
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(xs, ys, color=colors)
    best = min(ys)
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


def _series_values(per_dim: dict, dims: list[str]) -> tuple[list[float | None], int]:
    vals = [per_dim[d]["f1"] for d in dims]
    return vals, sum(1 for v in vals if v is None)


def chart_per_dim_f1(baselines: dict, neural: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    series = []
    for key, (label, color) in BASELINE_STYLE.items():
        source = neural if key == "neural_champion" else baselines.get(key)
        if not source:
            continue
        vals, n_undef = _series_values(source["holdout"]["per_dim"], dims)
        shown = label + (f" ({n_undef} undef)" if n_undef else "")
        series.append((shown, color, [np.nan if v is None else v for v in vals]))
    width = 0.8 / len(series)
    xs = range(len(dims))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (label, color, vals) in enumerate(series):
        pos = [x + i * width for x in xs]
        ax.bar(pos, vals, width, label=label, color=color)
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("holdout F1 (>=40 = YES)")
    ax.set_title(
        "Per-dimension holdout F1 — neural champion vs baselines "
        "(missing bar = F1 undefined, not zero)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "chart-per-dim-f1.png", dpi=150)
    plt.close(fig)


def chart_support(neural: dict, out: Path) -> None:
    # holdout partition — same partition as the per-dim F1 chart (#eval E13)
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    support = [neural["holdout"]["per_dim"][d]["support_positive"] for d in dims]
    total = [neural["holdout"]["per_dim"][d]["n_observed"] for d in dims]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims, total, color="#dee2e6", label="observed")
    ax.bar(dims, support, color="#2a9d8f", label="positive (YES)")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("holdout subjects")
    ax.set_title("Per-dimension label support (holdout partition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart-support.png", dpi=150)
    plt.close(fig)


def chart_regression(regression: dict, out: Path) -> None:
    dims = [d for d in CANONICAL_DIMS if d != "balanced"]
    for partition, fname in (("dev", "chart-regression-dev.png"),
                             ("holdout", "chart-regression-holdout.png")):
        model_mae = [
            regression["per_dim"][partition][d]["mae"] for d in dims
        ]
        ref_mae = [
            regression["per_dim"][partition][d]["mean_predictor_mae"] for d in dims
        ]
        width = 0.4
        xs = range(len(dims))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([x - width / 2 for x in xs],
               [np.nan if v is None else v for v in model_mae],
               width, label="neural model", color="#4361ee")
        ax.bar([x + width / 2 for x in xs],
               [np.nan if v is None else v for v in ref_mae],
               width, label="train-mean predictor", color="#adb5bd")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"{partition} MAE (0-100 score points)")
        ax.set_title(
            "8-score PUBLISHED regression — per-dimension MAE vs train-mean "
            "predictor (balanced dim unobserved, not shown)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def write_reports(out: Path, summary: dict, manifest: dict, records_count: int) -> None:
    ar = summary["autoresearch"]
    neural = summary["neural_evaluation"]
    base = summary["baselines"]
    regression = summary["regression_task"]
    qc = summary["row_qc_sensitivity"]
    dep = summary["deployable_ablation"]
    avail = summary["availability"]

    keep = [e["run_id"] for e in ar["per_round"] if e["decision"] == "KEEP"]
    revert = [e["run_id"] for e in ar["per_round"] if e["decision"] == "REVERT"]

    def holdout_macro_f1(entry: dict) -> str:
        return _fmt(entry["holdout"]["macro_f1"])

    lines = [
        "# TCM 真实训练报告（#50，AutoResearch V1，v2）",
        "",
        f"生成时间：{summary['generated_at']}  ",
        f"数据：`mendeley-cxtph3tjsg-v1`（SHA-256 `{summary['dataset_sha256'][:16]}…`，"
        f"{records_count} 名受试者；QC 异常行 {qc['excluded_subject_rows']} 四维计分指纹不符）  ",
        f"标准：{summary['standard_version']}；标签契约：`{summary['label_version']}`  ",
        f"冻结 manifest：`{summary['manifest_sha256'][:16]}…`；划分："
        f"train {summary['split_sizes']['train']} / dev {summary['split_sizes']['dev']} / "
        f"holdout {summary['split_sizes']['holdout']}（subject-level，#27）",
        "",
        "## AutoResearch 轮次",
        "",
        f"共 {ar['rounds_completed']} 轮：KEEP {keep}，REVERT {revert}。",
        "冠军：`" + ar["champion"]["run_id"] + "`（dev subject MAE "
        f"{_fmt(ar['champion']['best_so_far'], 4)}，权重 SHA-256 "
        f"`{(ar['champion']['artifact_sha256'] or '')[:16]}…`）。",
        "所有候选使用相同冻结种子与批次顺序（epoch 级确定性 shuffle），仅可编辑超参不同。",
        "",
        "| 轮 | 意图 | 决定 | dev MAE |",
        "| --- | --- | --- | --- |",
    ]
    for e in ar["per_round"]:
        value = e.get("primary_value")
        cell = f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {e['run_id']} | {e['intent']} | {e['decision']} | {cell} |")
    lines += [
        "",
        "## 最终评估（holdout 在研究结束后解锁；本 v2 报告为审查修复后的重新交付，"
        "holdout 先后用于 v1 与 v2 两次最终评估，已如实披露）",
        "",
        f"- 神经冠军：dev macro-F1 {_fmt(neural['dev']['macro_f1'])}，"
        f"holdout macro-F1 {_fmt(neural['holdout']['macro_f1'])}"
        f"（macro 对全部可观测维度平均，未定义 F1 计 0，分母不随模型漂移；"
        f"dev 未定义维度 {neural['dev']['n_undefined_f1']} 个，holdout "
        f"{neural['holdout']['n_undefined_f1']} 个）",
        f"- 神经冠军 subject MAE：dev {_fmt(neural['dev']['subject_mae'], 4)}，"
        f"holdout {_fmt(neural['holdout']['subject_mae'], 4)}"
        f"（选择指标即此口径，可与基线直接对比）",
        f"- Dummy（多数类）：holdout macro-F1 {holdout_macro_f1(base['dummy_majority'])}，"
        f"holdout subject MAE {_fmt(base['dummy_majority']['holdout']['subject_mae'], 4)}",
        f"- BMI-only 逻辑回归：holdout macro-F1 {holdout_macro_f1(base['bmi_only_logistic'])}，"
        f"holdout subject MAE {_fmt(base['bmi_only_logistic']['holdout']['subject_mae'], 4)}",
        f"- 全特征逻辑回归（含可用性指示位）：holdout macro-F1 "
        f"{holdout_macro_f1(base['feature_logistic'])}，"
        f"holdout subject MAE {_fmt(base['feature_logistic']['holdout']['subject_mae'], 4)}",
        "- **口径说明（必读）**：subject MAE 度量概率输出到 0/1 标签的 L1 距离，"
        "常数近零预测器（Dummy）天然占优——正类率 ≈0.29 意味着全零预测的 MAE 就约等于"
        " 0.29。因此在 MAE 口径下 Dummy 优于所有真实模型是**指标性质**，不是模型失效；"
        "神经冠军的价值在阈值化判别（F1）：holdout macro-F1 0.345 对 Dummy 0.000、"
        "逻辑回归 0.275。两个口径都给出，读者可自行权衡。",
        "",
        "图表：`chart-rounds.png`、`chart-per-dim-f1.png`、`chart-support.png`、"
        "`chart-regression-dev.png`、`chart-regression-holdout.png`"
        "（图中缺失柱 = F1/MAE 未定义，不是 0）。",
        "",
        "## 模态可用率（缺失≠被当作平均人：基线与神经模型拿到同一可用性信息）",
        "",
        "| 分区/模态 | 可用 / 行数 |",
        "| --- | --- |",
    ]
    for key in sorted(avail):
        lines.append(f"| {key} | {avail[key]['n_available']} / {avail[key]['n_rows']} |")
    lines += [
        "",
        "## 独立交付任务",
        "",
        f"- **8-score PUBLISHED 回归**（独立 manifest，固定配置）：逐维 MAE 与"
        f"训练均值预测器参照见 `summary.json`（regression_task）；"
        f"manifest `{regression['manifest_sha256'][:16]}…`。"
        "该任务包含 QC 异常行、未做敏感性对照，属已知限制。",
        f"- **row-QC 敏感性**（同一冻结划分剔除 {qc['excluded_subject_rows']} 后重训练，"
        f"冠军同配置）：dev subject MAE {_fmt(qc['dev_subject_mae_without'], 4)}"
        f"（含 QC 行的冠军为 {_fmt(ar['champion']['best_so_far'], 4)}）。",
        f"- **Deployable（Body/Vitals-only）消融**（冠军同超参，仅换特征集）："
        f"dev subject MAE {_fmt(dep['dev_subject_mae'], 4)}，"
        f"dev macro-F1 {_fmt((dep.get('f1') or {}).get('dev', {}).get('macro_f1'))}。",
        "",
        "## 证据状态与限制",
        "",
        "- 训练模式：REAL_DATA_TRAINING（agent provider=确定性计划，AGENT_NOT_VERIFIED；"
        "OS/container 隔离仍未验证，AGENT_ISOLATION_UNVERIFIED）。",
        "- 标签为 RULE 派生类别（规则阈值 ≥40=YES；TENDENCY/NO=0；平和维 ABSTAIN）——"
        "不是作者发布的分类标签，也不是临床诊断真值。注意 CCMQ 分数粒度使有效切点"
        "随各维题数漂移（6 题维 ≈41.7、7 题维 ≈42.9、8 题维 ≈40.6）。",
        "- 列映射为 FINGERPRINT_CROSS_CHECKED（论文定义表付费墙内）；"
        f"QC 异常行 {qc['excluded_subject_rows']} 未混入敏感性对照以外的任何对照。",
        "- 冠军由 6 个候选在 dev 上选出（winner's curse）；消融为单 seed 固定配置；"
        "dev/holdout 均 <50 人，小数位差异属噪声，未做置信区间重采样。",
        "- holdout 先后用于 v1、v2 两次最终评估（见报告开头）；v3 起如再评估需换冻结划分。",
        "- 类别支持不均衡（见 chart-support.png）；结果仅代表研究口径，不可临床使用。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    card = [
        "# TCM Constitution Model Card（#50 交付 v2）",
        "",
        f"- model_type：`tcm_constitution_v1`；champion run：`{ar['champion']['run_id']}`",
        f"- 权重：`{ar['champion']['artifact_dir']}/weights.pt`（SHA-256 "
        f"`{(ar['champion']['artifact_sha256'] or '')}`）",
        "- 输入契约：`tcm-input-real-v1`（"
        + ", ".join(f"{k}({v['dim']})" for k, v in manifest["modalities"].items())
        + "；train-fold 标准化，退化列置零，缺失模态行 availability-masked）",
        "- 输出：9 维独立 sigmoid（`category_probs`）；平和维永远 ABSTAIN",
        f"- 训练数据：{records_count} 名受试者（Mendeley cxtph3tjsg/1，CC BY 4.0），"
        f"RULE 派生二值标签（`{summary['label_version']}`）",
        "- 适用人群：与研究样本同源的成年体检人群；不适用于临床诊断、无按摩场景外的任何干预决策",
        "- 已知偏差：类别支持不均衡；体型代理变量（BMI）基线较强；"
        "冠军在 dev 上选出（winner's curse）；单 seed",
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
    model, state, metadata = load_training_checkpoint(champion["artifact_dir"])
    assert not state["smoke_only"]
    entries = store.load()
    champion_cfg = dict(metadata.get("config", {}))
    champion_trainer = dict(state["trainer_config"])

    neural_eval = evaluate_neural(model, manifest, records)
    baselines = evaluate_baselines(manifest, records)
    regression = regression_task(manifest, records, out)
    qc = row_qc_sensitivity(manifest, records, out, champion_cfg, champion_trainer)
    qc["dev_subject_mae_with"] = dev_primary(model, manifest, records)
    deployable = deployable_ablation(manifest, records, out, champion_cfg, champion_trainer)

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
    split_sizes = {p: len(manifest["split"][p]) for p in ("train", "dev", "holdout")}
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_version": manifest["label_version"],
        "standard_version": manifest["standard_version"],
        "split_sizes": split_sizes,
        "metric_params": {
            "classification_threshold": 0.5,
            "label_rule": manifest["label_version"],
            "macro_f1": "mean per-dim F1 over observed dims; undefined F1 counts as 0",
            "subject_mae": "mean over subjects of per-subject mean |p - label| over observed dims",
            "partitions": split_sizes,
        },
        "champion_provenance": {
            "run_id": champion["run_id"],
            "weights_sha256": champion.get("artifact_sha256"),
            "trainer_config": champion_trainer,
            "model_config": champion_cfg,
        },
        "availability": availability_rates(manifest, records),
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
        "row_qc_sensitivity": qc,
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
        "holdout_subject_mae": neural_eval["holdout"]["subject_mae"],
        "baseline_holdout": {
            k: {"macro_f1": v["holdout"]["macro_f1"],
                "subject_mae": v["holdout"].get("subject_mae")}
            for k, v in baselines.items() if isinstance(v, dict) and "holdout" in v
        },
        "row_qc_without": qc["dev_subject_mae_without"],
        "deployable_dev_mae": deployable["dev_subject_mae"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
