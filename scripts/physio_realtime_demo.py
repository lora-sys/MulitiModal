"""Simulated realtime physiology demo (#95): SMOKE_ONLY, no real device.

Run with the source tree on the path, like the other repo scripts:

    PYTHONPATH=src python -m scripts.physio_realtime_demo --pretty

The stream is generated locally: device events arrive out of order, the link
drops and reconnects, the window is built strictly at the cutoff, unvalidated
inference returns NOT_AVAILABLE, recommendation only offers reviewed static
candidates, and TCM device labels map to approved knowledge or abstain.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physio.acceptance import run_acceptance
from physio.adapters import DeviceEventAdapter
from physio.inference import InferenceSession
from physio.programs import ProgramCapabilityCatalog
from physio.recommendation import MassageRecommendationService, UserPreference
from physio.stream import OnlineIngestor
from physio.tcm_exam import KnowledgeRegistry, build_tcm_sidecar, parse_tcm_exam
from physio.windows import PhysioWindowBuilder, WindowConfig

BASE = datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
SESSION_ID = "sess-smoke-demo"
SUBJECT_REF = "pseudonym-smoke"
DEVICE_REF = "chair-mock-01"


def _ts(seconds: int) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _payload(sequence: int, seconds: int, **measurements) -> dict:
    values = measurements or {
        "hr": 70 + sequence,
        "spo2": 0.97,
        "skin_temp": 33.2,
        "rr": 16,
        "sbp": 118,
        "dbp": 76,
    }
    return {
        "session_id": SESSION_ID,
        "subject_ref": SUBJECT_REF,
        "device_ref": DEVICE_REF,
        "phase": "during",
        "measured_at": _ts(seconds),
        "received_at": _ts(seconds + 1),
        "sequence": sequence,
        "measurements": values,
        "units": {
            "hr": "bpm",
            "spo2": "fraction",
            "skin_temp": "°C",
            "rr": "breaths/min",
            "sbp": "mmHg",
            "dbp": "mmHg",
        },
    }


def _catalog() -> ProgramCapabilityCatalog:
    return ProgramCapabilityCatalog.from_dict(
        {
            "catalog_version": "demo-catalog-v1",
            "reviewed": True,
            "programs": [
                {
                    "program_id": "M01", "name": "舒缓", "intensity_min": 1,
                    "intensity_max": 4, "duration_min_s": 300,
                    "duration_max_s": 1800, "actions": ["knead"],
                    "selectable": True, "reviewed": True,
                },
                {
                    "program_id": "M02", "name": "放松", "intensity_min": 2,
                    "intensity_max": 5, "duration_min_s": 600,
                    "duration_max_s": 1800, "actions": ["vibrate"],
                    "selectable": True, "reviewed": True,
                },
            ],
        }
    )


def _knowledge() -> KnowledgeRegistry:
    return KnowledgeRegistry.from_json(
        {
            "entries": [
                {
                    "topic_id": "demo-rest",
                    "title": "一般作息提示（演示）",
                    "text": "演示用一般性作息提示，不是诊断或处方。",
                    "source": "synthetic-demo-fixture",
                    "license": "SYNTHETIC_SMOKE_ONLY",
                    "version": "v1",
                    "review_status": "APPROVED",
                    "reviewed_by": "synthetic-review",
                    "constitution_tags": ["气虚"],
                }
            ]
        }
    )


def _tcm(payload: dict) -> dict:
    exam = parse_tcm_exam(payload)
    sidecar = build_tcm_sidecar(exam, _knowledge())
    return {
        "exam_status": sidecar.status,
        "constitution_findings": sidecar.constitution_findings,
        "knowledge": sidecar.knowledge,
        "provenance": sidecar.provenance,
        "reason": sidecar.reason,
    }


def _timeline(
    session,
    builder: PhysioWindowBuilder,
    catalog: ProgramCapabilityCatalog,
    cutoffs: list,
) -> list[dict]:
    """Replay each cutoff in measured time; future events never enter the window."""
    items: list[dict] = []
    for cutoff in cutoffs:
        inference = InferenceSession(
            session,
            builder,
        ).update(cutoff)
        service = MassageRecommendationService(catalog)
        recommendation = service.update(
            inference,
            now=cutoff,
            user_preference=UserPreference(max_intensity=4, max_duration_s=900),
        )
        items.append(
            {
                "cutoff": cutoff.isoformat(),
                "inference_status": inference.status,
                "reason": inference.reason,
                "quality": inference.quality,
                "coverage": inference.coverage,
                "candidate_ids": [candidate.program_id for candidate in recommendation.candidates],
                "recommendation_status": recommendation.status,
                "state": recommendation.state,
                "control_command": recommendation.control_command,
            }
        )
    return items


def run() -> dict:
    adapter = DeviceEventAdapter(stale_after_s=300)
    payloads = [_payload(i, i * 10) for i in range(7)]
    events = [adapter.parse_event(payload) for payload in payloads]

    ingestor = OnlineIngestor(SESSION_ID, SUBJECT_REF, DEVICE_REF, max_gap_s=30)
    for event in events[:5]:
        ingestor.push(event)
    ingestor.mark_disconnected(BASE + timedelta(seconds=45))
    ingestor.reconnect(BASE + timedelta(seconds=75))
    for event in reversed(events[5:]):
        ingestor.push(event)
    session = ingestor.session()

    cutoff = BASE + timedelta(seconds=60)
    builder = PhysioWindowBuilder(WindowConfig(window_s=60, min_events=2))
    window = builder.build(session, cutoff)
    inference = InferenceSession(
        session,
        builder,
    ).update(cutoff)

    service = MassageRecommendationService(_catalog())
    recommendation = service.update(
        inference,
        now=cutoff,
        user_preference=UserPreference(max_intensity=4, max_duration_s=900),
    )

    tcm = _tcm(
        {
            "session_id": SESSION_ID,
            "subject_ref": SUBJECT_REF,
            "device_ref": DEVICE_REF,
            "measured_at": _ts(0),
            "received_at": _ts(2),
            "tongue_body_color": "淡红",
            "tongue_shape": "胖大、齿痕",
            "coating_color": "白",
            "coating_texture": "腻",
            "face_color": "萎黄",
            "complexion": "少华",
            "constitution_names": "气虚兼痰湿",
            "device_algorithm": "vendor-demo-fixture",
            "vocabulary_version": "v1",
        }
    )

    acceptance = run_acceptance()
    timeline = _timeline(
        session,
        builder,
        _catalog(),
        [BASE + timedelta(seconds=seconds) for seconds in (0, 20, 40, 60)],
    )
    return {
        "demo": "physio_realtime_demo",
        "evidence_level": "LOCAL_EXPERIMENTAL",
        "isolation": "AGENT_ISOLATION_UNVERIFIED",
        "smoke_only": "SMOKE_ONLY",
        "device_connection": adapter.capabilities().to_dict(),
        "stream": {
            "events": len(session),
            "gaps": [round(gap.seconds, 1) for gap in ingestor.gaps],
            "duplicates": ingestor.duplicates,
            "source": session.source,
        },
        "window": window.to_dict(),
        "inference": inference.to_dict(),
        "recommendation": recommendation.to_dict(),
        "timeline": timeline,
        "tcm_sidecar": tcm,
        "acceptance": {
            "status": acceptance["status"],
            "passed": acceptance["passed"],
            "total": acceptance["total"],
            "checks": acceptance["checks"],
        },
        "limitations": acceptance["limitations"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args(argv)
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if report["acceptance"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
