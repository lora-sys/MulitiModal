"""最小中医模型演示：真实工件 + 冻结真实数据样本，全部标记为科研估计。

运行（acam-lab，PYTHONPATH=src）：

    python -m scripts.tcm_minimal_demo
    python -m scripts.tcm_minimal_demo --symptom "反复胸闷隐痛3年余，加重伴憋喘1周"
    python -m scripts.tcm_minimal_demo --physiology-only

边界（AGENTS.md / 模型卡）：
- 八维体质分数是 LOCAL_EXPERIMENTAL 折外 CV 选择出的研究接口，8 个已观测维度；
  第 9 维平和质在源数据中完全未观测，ABSTAIN，不伪造、不补零。
- 证候候选是 TCM-SD（CC BY-NC-SA 4.0，仅非商业科研）；候选分数未校准，
  不是概率、置信度、诊断或临床判断；
  生理→证候配对验证仍为 BLOCKED（缺乏配对真实数据，严禁伪造关联）。
- 实时生理数值不能推断体质或证候：没有配对训练证据时明确 ABSTAIN。
- 本演示不训练、不调参、不改冠军、不读历史 holdout；不改冻结评价协议。
- 体质工件由 `research.constitution_delivery` 的 importable 类重存，
  独立进程可直接 `load_artifact`，无需 `__main__` 别名 shim。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from research.constitution_delivery import ConstitutionInference, load_artifact
from syndrome.service import (
    ModelRegistry,
    SymptomAnalysisRequest,
    SyndromeService,
    tfidf_logreg_spec,
)

DEMO_SCHEMA = "tcm-minimal-demo-v1"
RECORDS = Path("data/local/mendeley-cxtph3tjsg-v1/protocol-v1/selection_records.json")
CONSTITUTION_ARTIFACT = Path("artifacts/constitution-v1-champion")
SYNDROME_ARTIFACT = "artifacts/sd-v2-ar1-champion"
RESEARCH_NOTE = "LOCAL_EXPERIMENTAL research estimate; not calibrated, not a diagnosis"

DEMO_PHYSIOLOGY = {"heart_rate": 72, "spo2": 98, "skin_temp": 36.5, "hrv": 45}


def _body_record(index: int) -> dict:
    """从冻结真实选择集取一名受试者的 63 维身体特征；不读其标签分数。"""
    rows = json.loads(RECORDS.read_text(encoding="utf-8"))
    row = rows[index]
    return {
        "features": row["features"],
        "feature_mask": row["feature_mask"],
        "source": f"frozen subject {row['subject_id']}（真实身体特征，未使用其标签分数）",
    }


def constitution_estimate(body: Mapping | None,
                          artifact_dir: Path = CONSTITUTION_ARTIFACT) -> dict:
    """身体信息 → 八维体质研究性估计；无效输入 / 工件不可用明确拒判。"""
    if body is None:
        return {"status": "ABSTAIN", "scores": [], "balanced": "ABSTAIN",
                "reason": "未提供身体信息", "backend_availability": "NOT_EVALUATED"}
    if not isinstance(body, Mapping):
        return {"status": "OUT_OF_DOMAIN", "scores": [], "balanced": "ABSTAIN",
                "reason": f"输入类型 {type(body).__name__} 不是身体信息记录",
                "backend_availability": "NOT_EVALUATED"}
    try:
        predictor, entry, labels = load_artifact(artifact_dir)
    except (OSError, ValueError, AttributeError, KeyError) as exc:
        return {"status": "ABSTAIN", "scores": [], "balanced": "ABSTAIN",
                "reason": f"体质工件不可用（{type(exc).__name__}）",
                "backend_availability": "NOT_AVAILABLE"}
    response = ConstitutionInference(predictor, entry["provenance"]).predict_one(dict(body))
    scores = [
        {"dim": s["dim"], "value": round(s["value"], 2), "calibrated": False}
        for s in response["scores"]
    ]
    return {
        "status": response["status"],
        "scores": scores,
        "balanced": {"dim": "balanced", "status": "ABSTAIN", "value": None},
        "score_dims": labels["score_dims"],
        "unobserved_dimensions": labels["unobserved_dimensions"],
        "evidence": response["evidence"],
        "label_tier": "PUBLISHED",
        "champion_cv_masked_mae": entry["provenance"]["champion_selection_masked_mae"],
        "reason": response.get("reason", ""),
        "backend_availability": "AVAILABLE",
        "note": RESEARCH_NOTE,
    }


def syndrome_candidates(text: str | None, artifact_dir: str = SYNDROME_ARTIFACT) -> dict:
    """中文症状文字 → 证候候选（未校准，非诊断）。空/越界/工件不可用均明确拒判。"""
    if not text:
        return {"status": "ABSTAIN", "candidates": [],
                "backend_availability": "NOT_EVALUATED", "note": RESEARCH_NOTE}
    if not isinstance(text, str):
        return {"status": "OUT_OF_DOMAIN", "candidates": [],
                "reason": f"症状文本类型 {type(text).__name__} 非法",
                "backend_availability": "NOT_EVALUATED", "note": RESEARCH_NOTE}
    try:
        spec = tfidf_logreg_spec(artifact_dir)
    except (ValueError, OSError) as exc:
        return {"status": "ABSTAIN", "candidates": [],
                "reason": f"证候工件损坏（{type(exc).__name__}）",
                "backend_availability": "NOT_AVAILABLE", "note": RESEARCH_NOTE}
    if spec is None:
        return {"status": "ABSTAIN", "candidates": [],
                "reason": "证候工件目录缺失，后端未注册",
                "backend_availability": "NOT_AVAILABLE", "note": RESEARCH_NOTE}
    registry = ModelRegistry()
    registry.register(spec, default=True)
    response = SyndromeService(registry=registry).analyze_symptoms(
        SymptomAnalysisRequest(symptom_text=text)
    ).to_dict()
    candidates = [
        {"label": c["label"], "score": round(c["score"], 4), "calibrated": False}
        for c in response["candidates"]
    ]
    return {"status": response["status"], "candidates": candidates,
            "backend_availability": response["backend_availability"],
            "note": RESEARCH_NOTE}


def physiology_only(measurements: Mapping | None) -> dict:
    """只有实时生理数值、没有配对训练证据 → 明确无法判断体质或证候。"""
    return {
        "status": "ABSTAIN",
        "measurements": dict(measurements or {}),
        "constitution": "UNAVAILABLE",
        "syndrome": "UNAVAILABLE",
        "reason": (
            "实时生理数值没有配对的体质/证候训练证据；#54 生理→证候仍为 BLOCKED；"
            "不从生理数值推断体质或证候"
        ),
        "evidence": "NOT_AVAILABLE（无配对训练证据）",
        "note": "SMOKE_ONLY 输入样例；不构成任何判断",
    }


def demo(body_index: int = 0, symptom_text: str | None = None) -> dict:
    """身体信息 → 八维体质估计 + 可选文字 → 证候候选。"""
    body = _body_record(body_index)
    return {
        "schema_version": DEMO_SCHEMA,
        "mode": "BODY_FEATURES",
        "body_input": {
            "n_features": len(body["features"]),
            "n_observed": sum(body["feature_mask"]),
            "source": body["source"],
        },
        "constitution": constitution_estimate(body),
        "syndrome": syndrome_candidates(symptom_text) if symptom_text else None,
        "disclaimer": RESEARCH_NOTE,
    }


def physiology_demo() -> dict:
    """只有实时生理数值的演示入口。"""
    return {
        "schema_version": DEMO_SCHEMA,
        "mode": "PHYSIOLOGY_ONLY",
        "physiology": physiology_only(DEMO_PHYSIOLOGY),
        "disclaimer": RESEARCH_NOTE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-index", type=int, default=0)
    parser.add_argument("--symptom", default=None)
    parser.add_argument("--physiology-only", action="store_true")
    args = parser.parse_args()
    result = physiology_demo() if args.physiology_only else demo(args.body_index, args.symptom)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
