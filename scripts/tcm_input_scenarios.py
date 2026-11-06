"""任务 C：实际输入场景测试（普通人日常表达 / 缺失身体信息 / 极端合理输入 /
重复或矛盾症状），记录两个模型的真实输出、错误类型和拒判行为。

边界（AGENTS.md / 任务书）：
- 场景输入是合成文本与合成身体记录，显式标注 SMOKE_ONLY；
- 没有专家标签，只做鲁棒性检查，**不报告预测准确率**（report 无 accuracy 字段）；
- 输出是 LOCAL_EXPERIMENTAL 研究接口，未校准，不是诊断或临床判断；
- 不训练、不调参、不改冠军、不读历史 holdout；不改冻结评价协议。

运行（acam-lab，PYTHONPATH=src）：
    python -m scripts.tcm_input_scenarios --out runs/value-review/input-scenarios.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

REPORT_SCHEMA = "tcm-input-scenario-report-v1"
FEATURE_DIM = 63
REQUIRED_FEATURES = 9  # 冻结冠军实际使用的前 9 项基础特征

CATEGORIES = (
    "everyday_expression",
    "missing_body_info",
    "extreme_reasonable_input",
    "repeated_contradictory_symptom",
)

EVERYDAY_EXPRESSIONS = (
    "最近老是头晕，睡不好，吃饭也没胃口",
    "一吹风就咳嗽，嗓子干",
    "吃点凉的就拉肚子，肚子咕噜叫",
    "胸口闷，爬两层楼就喘不上气",
    "手脚冰凉，怕冷，容易累",
)

REPEATED_CONTRADICTORY = (
    "头痛头痛头痛头痛",
    "又怕冷又怕热，一会儿冷一会儿热",
    "发热同时又恶寒，两个都有",
    "既腹泻又便秘，交替出现",
    "胸闷，不胸闷，还是胸闷",
)

RESEARCH_NOTE = (
    "LOCAL_EXPERIMENTAL 研究估计；未校准，不是概率/诊断/临床判断；"
    "场景输入 SMOKE_ONLY 合成，无专家标签，仅用于鲁棒性检查"
)


def _body(features, mask):
    return {"features": list(features), "feature_mask": list(mask)}


def _full_body(base):
    return _body(base, [True] * FEATURE_DIM)


def _all_missing_body():
    return _body([None] * FEATURE_DIM, [False] * FEATURE_DIM)


def _required_missing_body(base):
    """9 项必需特征全缺、54 项无关特征齐全：不得给出误导性分数。"""
    features = [None] * REQUIRED_FEATURES + list(base[REQUIRED_FEATURES:])
    mask = [False] * REQUIRED_FEATURES + [True] * (FEATURE_DIM - REQUIRED_FEATURES)
    return _body(features, mask)


def _partial_missing_body(base):
    features = list(base)
    mask = [True] * FEATURE_DIM
    for index in (2, 5):
        features[index] = None
        mask[index] = False
    return _body(features, mask)


def _extreme_reasonable_bodies(bounds):
    """极端但合理：每个基础特征取真实观测到的最小/最大值（不是凭空外推）。"""
    low = [pair[0] for pair in bounds] + [0.0] * (FEATURE_DIM - REQUIRED_FEATURES)
    high = [pair[1] for pair in bounds] + [0.0] * (FEATURE_DIM - REQUIRED_FEATURES)
    return {"observed_min": _full_body(low), "observed_max": _full_body(high)}


def observed_bounds(records_path: Path) -> list[tuple[float, float]]:
    """从冻结选择集计算前 9 项特征的观测范围；只用于构造合理极端输入。"""
    records = json.loads(records_path.read_text(encoding="utf-8"))
    per_feature: list[list[float]] = [[] for _ in range(REQUIRED_FEATURES)]
    for row in records:
        for index, (value, observed) in enumerate(
                zip(row["features"][:REQUIRED_FEATURES],
                    row["feature_mask"][:REQUIRED_FEATURES], strict=True)):
            if observed and value is not None:
                per_feature[index].append(float(value))
    bounds = [(min(values), max(values)) for values in per_feature]
    if len(bounds) != REQUIRED_FEATURES or any(lo > hi for lo, hi in bounds):
        raise ValueError("cannot derive observed feature bounds")
    return bounds


def build_scenarios(base: list[float] | None = None,
                    bounds: list[tuple[float, float]] | None = None) -> list[dict]:
    """构造固定场景表；base/bounds 可由调用方注入以便本地合成测试。"""
    base = list(base) if base is not None else [float(i + 1) for i in range(FEATURE_DIM)]
    if len(base) != FEATURE_DIM:
        raise ValueError("base must have 63 features")
    bounds = bounds or [(0.0, 2.0 * base[i]) for i in range(REQUIRED_FEATURES)]
    extremes = _extreme_reasonable_bodies(bounds)

    scenarios: list[dict] = []
    for index, text in enumerate(EVERYDAY_EXPRESSIONS):
        scenarios.append({
            "id": f"everyday_{index + 1}", "category": "everyday_expression",
            "text": text, "body": None,
            "note": "普通人日常口语症状描述，无专家标签",
        })
    for index, text in enumerate(REPEATED_CONTRADICTORY):
        scenarios.append({
            "id": f"repeated_{index + 1}",
            "category": "repeated_contradictory_symptom",
            "text": text, "body": None,
            "note": "重复或自相矛盾的症状自述，无专家标签",
        })

    scenarios.extend([
        {"id": "missing_all", "category": "missing_body_info",
         "text": None, "body": _all_missing_body(),
         "note": "身体信息全部缺失（mask 全 False）"},
        {"id": "missing_required_9", "category": "missing_body_info",
         "text": None, "body": _required_missing_body(base),
         "note": "模型需要的 9 项基础特征全缺，其余 54 项齐全"},
        {"id": "missing_partial", "category": "missing_body_info",
         "text": None, "body": _partial_missing_body(base),
         "note": "基础特征部分缺失"},
        {"id": "missing_none", "category": "missing_body_info",
         "text": None, "body": None, "note": "未提供身体记录"},
        {"id": "missing_wrong_width", "category": "missing_body_info",
         "text": None, "body": _body([1.0] * 3, [True] * 3),
         "note": "特征维度错误"},
        {"id": "extreme_min", "category": "extreme_reasonable_input",
         "text": None, "body": extremes["observed_min"],
         "note": "前 9 项取真实观测最小值（极端但仍在观测范围内）"},
        {"id": "extreme_max", "category": "extreme_reasonable_input",
         "text": None, "body": extremes["observed_max"],
         "note": "前 9 项取真实观测最大值（极端但仍在观测范围内）"},
    ])
    return scenarios


def _summarise(model: str, applicable: bool, payload, fn: Callable) -> dict:
    if not applicable:
        return {"model": model, "applicable": False, "status": "NOT_EVALUATED",
                "error_type": None, "abstain": None, "note": "该场景不含此模型的输入"}
    try:
        result = fn(payload)
    except Exception as exc:  # 记录真实错误类型，绝不吞掉
        return {
            "model": model, "applicable": True, "status": "ERROR",
            "error_type": type(exc).__name__, "error_message": str(exc)[:200],
            "abstain": None, "n_items": 0, "reason": None,
            "backend_availability": None, "note": RESEARCH_NOTE,
        }
    status = result.get("status")
    reason = result.get("reason")
    n_items = len(result.get("candidates", result.get("scores", [])))
    return {
        "model": model, "applicable": True, "status": status,
        "error_type": None,
        "abstain": bool(status in ("ABSTAIN", "OUT_OF_DOMAIN", "UNCERTAIN")),
        "n_items": n_items,
        "reason": reason,
        "backend_availability": result.get("backend_availability"),
        "calibrated": all(not item.get("calibrated", False)
                          for item in result.get("candidates", result.get("scores", []))),
        "note": RESEARCH_NOTE,
    }


def run_scenarios(syndrome_fn: Callable, constitution_fn: Callable,
                  scenarios: list[dict]) -> dict:
    """逐场景记录两个模型的真实输出 / 错误类型 / 拒判行为。"""
    records = []
    for scenario in scenarios:
        records.append({
            "id": scenario["id"], "category": scenario["category"],
            "input": {"text": scenario.get("text"),
                      "body_provided": scenario.get("body") is not None},
            "note": scenario["note"],
            "models": {
                "syndrome": _summarise("syndrome", scenario.get("text") is not None,
                                       scenario.get("text"), syndrome_fn),
                "constitution": _summarise("constitution", scenario.get("body") is not None,
                                           scenario.get("body"), constitution_fn),
            },
        })
    return {
        "schema_version": REPORT_SCHEMA,
        "status": {
            "evidence": "LOCAL_EXPERIMENTAL",
            "trusted_evaluator": "AGENT_ISOLATION_UNVERIFIED",
            "external_validation": "NONE",
        },
        "purpose": "ROBUSTNESS_ONLY",
        "expert_labels": False,
        "accuracy_reported": False,
        "synthetic_inputs": "SMOKE_ONLY",
        "categories": list(CATEGORIES),
        "n_scenarios": len(records),
        "by_category": _category_summary(records),
        "scenarios": records,
        "disclaimer": (
            "无专家标签，场景只用于鲁棒性与拒判检查；不报告预测准确率，"
            "不构成诊断、临床疗效或泛化有效性结论"
        ),
    }


def _category_summary(records: list[dict]) -> dict:
    summary = {}
    for category in CATEGORIES:
        subset = [r for r in records if r["category"] == category]
        summary[category] = {
            "n_scenarios": len(subset),
            "syndrome_abstain": sum(r["models"]["syndrome"]["abstain"] is True
                                    for r in subset),
            "constitution_abstain": sum(r["models"]["constitution"]["abstain"] is True
                                        for r in subset),
            "error_types": sorted({
                r["models"][model]["error_type"]
                for r in subset for model in ("syndrome", "constitution")
                if r["models"][model]["error_type"] is not None
            }),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=("data/local/mendeley-cxtph3tjsg-v1/"
                                              "protocol-v1/selection_records.json"))
    parser.add_argument("--out", default="runs/value-review/input-scenarios.json")
    args = parser.parse_args()

    from scripts import tcm_minimal_demo as demo

    records_path = Path(args.records)
    bounds = observed_bounds(records_path) if records_path.exists() else None
    sample = json.loads(records_path.read_text(encoding="utf-8"))[0]["features"] \
        if records_path.exists() else None
    scenarios = build_scenarios(sample, bounds)
    report = run_scenarios(demo.syndrome_candidates, demo.constitution_estimate, scenarios)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_scenarios": report["n_scenarios"],
        "by_category": report["by_category"],
        "accuracy_reported": report["accuracy_reported"],
        "report": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
