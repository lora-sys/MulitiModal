"""#58 双模式中医健康接口演示（无真实设备、无真实数据）。

运行（acam-lab，PYTHONPATH=src）：

    python -m scripts.syndrome_service_demo \
        [--artifact-dir runs/syndrome-service/tfidf-logreg-default]

- 症状分析模式：仅在宿主机存在真实训练保存的 TF-IDF+LR 后端工件
  （save_tfidf_logreg_artifact 产物）时运行；缺失时打印 NOT_AVAILABLE，
  不崩溃、不回退到任何合成模型——合成结果不得冒充真实模型输出。
- 实时知识模式：全部测量流为 SYNTHETIC_MOCK，覆盖持续更新、越界值、
  设备掉线重连、时间戳回退与空流/未知指标 fail-closed 路径。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

from syndrome.feedback import MeasurementSample, TopicKnowledge
from syndrome.service import (
    SERVICE_SCHEMA_VERSION,
    RealtimeKnowledgeRequest,
    SymptomAnalysisRequest,
    SyndromeService,
    tfidf_logreg_spec,
)

T0 = datetime.fromisoformat("2026-11-03T10:00:00+08:00")

LICENSE_BANNER = """
许可与边界（AGENTS.md / 模型卡 tcm-sd-syndrome-v1）：
- 本演示为科研原型；输出不是诊断，不构成任何临床/疗效声明；
- 症状模式默认后端与派生知识基于 TCM-SD（CC BY-NC-SA 4.0，仅限非商业科研）；
  商业使用需独立许可与医疗内容审核门槛，#54 实际配对验证仍为 BLOCKED；
- 实时知识模式不调用证候分类器，不从生理数值推断个人证候（#53/#58 契约）。
"""

# 演示主题知识：显式 SYNTHETIC_MOCK 占位，非真实医学内容，无许可授予。
DEMO_TOPIC_KNOWLEDGE: dict[str, TopicKnowledge] = {
    topic.topic_id: topic
    for topic in (
        TopicKnowledge(
            topic_id="cardiovascular-general",
            title="心血管一般健康主题（SYNTHETIC_MOCK 占位）",
            text=(
                "SYNTHETIC_MOCK：这是一般性心血管健康主题的演示占位文本，"
                "与本次测量数值和个人状态无关，不构成医学建议。"
            ),
            source="SYNTHETIC_DEMO",
            license="SYNTHETIC_MOCK（非真实知识，无许可授予）",
            version="demo-0.0.1",
        ),
        TopicKnowledge(
            topic_id="respiratory-general",
            title="呼吸一般健康主题（SYNTHETIC_MOCK 占位）",
            text=(
                "SYNTHETIC_MOCK：这是一般性呼吸健康主题的演示占位文本，"
                "与本次测量数值和个人状态无关，不构成医学建议。"
            ),
            source="SYNTHETIC_DEMO",
            license="SYNTHETIC_MOCK（非真实知识，无许可授予）",
            version="demo-0.0.1",
        ),
        TopicKnowledge(
            topic_id="stress-general",
            title="压力一般健康主题（SYNTHETIC_MOCK 占位）",
            text=(
                "SYNTHETIC_MOCK：这是一般性压力健康主题的演示占位文本，"
                "与本次测量数值和个人状态无关，不构成医学建议。"
            ),
            source="SYNTHETIC_DEMO",
            license="SYNTHETIC_MOCK（非真实知识，无许可授予）",
            version="demo-0.0.1",
        ),
    )
}


def _ts(offset_seconds: float) -> str:
    return (T0 + timedelta(seconds=offset_seconds)).isoformat()


def _measurement(
    metric: str, value: float | None, offset_seconds: float, **overrides
) -> MeasurementSample:
    base = dict(
        source="wrist-band",
        metric=metric,
        value=value,
        unit="bpm" if metric == "heart_rate" else "%",
        timestamp=_ts(offset_seconds),
        quality="OK",
    )
    base.update(overrides)
    return MeasurementSample(**base)


def symptom_section(service: SyndromeService, artifact_dir: str) -> None:
    print("=" * 72)
    print("模式 1：症状分析（SYMPTOM_ANALYSIS）——仅在存在真实训练工件时运行")
    print("=" * 72)
    try:
        spec = tfidf_logreg_spec(artifact_dir)
    except ValueError as exc:
        print(f"后端工件损坏：{exc}")
        spec = None
    if spec is None:
        print(f"后端工件不存在（{artifact_dir}）：backend_availability=NOT_AVAILABLE")
    else:
        service.registry.register(spec, default=True)
        print(
            f"已注册默认后端：model_id={spec.model_id} "
            f"model_version={spec.model_version} artifacts={len(spec.artifacts)} 个"
        )
    for label, text in (
        ("真实症状输入", "最近总是失眠心悸，食欲差"),
        ("空症状（fail-closed）", ""),
        ("英文输入（fail-closed）", "I feel dizzy and cannot sleep"),
    ):
        request = SymptomAnalysisRequest(symptom_text=text)
        try:
            response = service.analyze_symptoms(request)
        except ValueError as exc:
            print(f"[{label}] 请求被拒绝：{exc}")
            continue
        print(f"[{label}]")
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


def realtime_section(service: SyndromeService) -> None:
    print("=" * 72)
    print("模式 2：实时知识（REALTIME_KNOWLEDGE）——全部测量流为 SYNTHETIC_MOCK")
    print("=" * 72)
    scenarios = {
        "A 持续更新": [
            _measurement("heart_rate", 72.0, 0),
            _measurement("heart_rate", 75.0, 5),
            _measurement("heart_rate", 78.0, 10),
        ],
        "B 越界值（heart_rate=320，fail-closed）": [
            _measurement("heart_rate", 320.0, 0),
        ],
        "C 设备掉线重连（15 分钟间隔）": [
            _measurement("spo2", 97.0, 0),
            _measurement("spo2", 96.0, 900),
        ],
        "D 时间戳回退（fail-closed）": [
            _measurement("heart_rate", 70.0, 20),
            _measurement("heart_rate", 71.0, 10),
        ],
        "E 未知指标 + 低质量（fail-closed）": [
            _measurement("quantum_flux", 1.0, 0),
            _measurement("heart_rate", 80.0, 1, quality="DEGRADED"),
        ],
        "F 空测量流（fail-closed）": [],
    }
    for name, measurements in scenarios.items():
        request = RealtimeKnowledgeRequest(
            session_id="demo-session-0001",
            timestamp=_ts(0),
            measurements=measurements,
            physio_summary={"n_samples": len(measurements), "stream": "SYNTHETIC_MOCK"},
        )
        response = service.realtime_knowledge(request)
        print(f"[{name}]")
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir",
        default="runs/syndrome-service/tfidf-logreg-default",
        help="TF-IDF+LR 默认后端工件目录（save_tfidf_logreg_artifact 产物）",
    )
    args = parser.parse_args(argv)

    print(LICENSE_BANNER)
    service = SyndromeService(
        topic_knowledge=DEMO_TOPIC_KNOWLEDGE,
        provenance={"demo": "SYNTHETIC_MOCK", "schema_version": SERVICE_SCHEMA_VERSION},
    )
    symptom_section(service, args.artifact_dir)
    realtime_section(service)


if __name__ == "__main__":
    main()
