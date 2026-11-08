"""#100 tests: direct device TCM exam mapping and fail-closed knowledge sidecar."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from physio.schema import QUALITY_DEGRADED, QUALITY_OK, QUALITY_REJECTED, SchemaError
from physio.tcm_exam import (
    CONSTITUTION_SOURCE_DEVICE,
    CONSTITUTION_SOURCE_PROJECT_MODEL,
    STATUS_ABSTAIN,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNKNOWN,
    TONGUE_BODY_COLOR_SYNONYMS,
    TONGUE_SHAPE_SYNONYMS,
    ConstitutionFinding,
    KnowledgeRegistry,
    TCMExamV1,
    build_tcm_sidecar,
    parse_constitution_names,
    parse_multi_value,
    parse_tcm_exam,
)

pytestmark = pytest.mark.synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_TCM_EXAM = REPO_ROOT / "src/physio/tcm_exam.py"


def exam_payload(**overrides):
    payload = {
        "exam_id": "exam-1",
        "session_id": "sess-1",
        "subject_ref": "subj-pseudo-1",
        "device_ref": "tongue-cam-1",
        "source": "device",
        "measured_at": "2026-11-07T10:00:00+00:00",
        "received_at": "2026-11-07T10:00:02+00:00",
        "tongue": {
            "body_color": "淡红色，淡红",
            "shape": "胖大、齿痕",
            "coating_color": "白",
            "coating_texture": "薄",
        },
        "face": {
            "color": "面色萎黄",
            "complexion": "少华",
            "eye_description": "左眼干涩；右眼目赤",
        },
        "constitution_names": ["气虚兼痰湿", "平和质"],
        "device_algorithm": "vendor-tongue-v1",
        "vocabulary_version": "tcm-vocab-v1",
        "provenance": {
            "adapter": "device",
            "adapter_version": "v1",
            "source_record": "msg-1",
        },
    }
    payload.update(overrides)
    return payload


def knowledge_entry(**overrides):
    entry = {
        "topic_id": "qi-deficiency-daily",
        "title": "气虚体质的日常作息参考",
        "text": "保持规律作息与适度活动；本条目不是诊断或处方。",
        "source": "reviewed-wellness-guide",
        "license": "CC-BY-4.0",
        "version": "2026-11",
        "review_status": "APPROVED",
        "reviewed_by": "tcm-reviewer-a",
        "constitution_tags": ["气虚"],
    }
    entry.update(overrides)
    return entry


def approved_registry(*entries):
    return KnowledgeRegistry.from_json({"entries": list(entries or (knowledge_entry(),))})


def test_complete_synthetic_exam_parses_separators_and_synonyms():
    exam = parse_tcm_exam(exam_payload())

    assert exam.quality == QUALITY_OK
    parsed = exam.parsed_fields()
    assert parsed["tongue_body_color"].status == STATUS_OK
    assert parsed["tongue_body_color"].values == ("淡红",)
    assert parsed["tongue_shape"].values == ("胖大", "齿痕")
    assert parsed["coating_color"].values == ("白",)
    assert parsed["coating_texture"].values == ("薄",)
    assert parsed["face_color"].values == ("萎黄",)
    assert parsed["complexion"].values == ("少华",)
    assert exam.eye_description == "左眼干涩；右眼目赤"

    assert [finding.name for finding in exam.constitution_names] == [
        "气虚",
        "痰湿",
        "平和",
    ]
    assert all(
        finding.source == CONSTITUTION_SOURCE_DEVICE
        for finding in exam.constitution_names
    )
    assert exam.device_algorithm == "vendor-tongue-v1"
    assert exam.vocabulary_version == "tcm-vocab-v1"


def test_direct_event_payload_roundtrip_without_csv():
    exam = parse_tcm_exam(exam_payload(source="service", provenance=None))

    assert exam.source == "service"
    assert exam.provenance is not None
    assert exam.provenance.adapter == "tcm-exam"
    restored = TCMExamV1.from_dict(exam.to_dict())
    assert restored == exam


def test_flat_device_event_fields_are_accepted():
    payload = exam_payload(
        tongue=None,
        face=None,
        tongue_body_color="淡红舌",
        tongue_shape="齿痕",
        coating_color="白苔",
        coating_texture="薄苔",
        face_color="面色萎黄",
        complexion="少华",
        eye_description="左眼干涩/右眼目赤",
    )
    exam = parse_tcm_exam(payload)

    assert exam.tongue_body_color == "淡红舌"
    assert exam.face_color == "面色萎黄"
    assert exam.parsed_fields()["tongue_body_color"].values == ("淡红",)


def test_unknown_tokens_stay_unknown():
    parsed = parse_multi_value("淡红、异色", TONGUE_BODY_COLOR_SYNONYMS)

    assert parsed.status == STATUS_PARTIAL
    assert parsed.values == ("淡红",)
    assert parsed.unknown == ("异色",)
    assert "异色" in parsed.reason


@pytest.mark.parametrize("text", [None, "", "   ", "\n"])
def test_blank_input_is_unknown(text):
    parsed = parse_multi_value(text, TONGUE_BODY_COLOR_SYNONYMS)

    assert parsed.status == STATUS_UNKNOWN
    assert parsed.values == ()


def test_conflicting_values_are_reported_not_resolved():
    parsed = parse_multi_value("淡红/红", TONGUE_BODY_COLOR_SYNONYMS)

    assert parsed.status == STATUS_PARTIAL
    assert parsed.values == ("淡红", "红")
    assert parsed.conflicts == ("淡红", "红")


def test_multi_valued_field_does_not_treat_values_as_conflict():
    parsed = parse_multi_value(
        "胖大、齿痕", TONGUE_SHAPE_SYNONYMS, multi_valued=True
    )

    assert parsed.status == STATUS_OK
    assert parsed.values == ("胖大", "齿痕")
    assert parsed.conflicts == ()


def test_compound_constitution_labels_become_multilabels():
    parsed = parse_constitution_names("气虚兼痰湿、湿热质")

    assert parsed.status == STATUS_OK
    assert parsed.values == ("气虚", "痰湿", "湿热")

    partial = parse_constitution_names("气虚兼未知质")
    assert partial.status == STATUS_PARTIAL
    assert partial.values == ("气虚",)
    assert partial.unknown == ("未知质",)


def test_timestamps_must_be_aware_and_ordered():
    with pytest.raises(SchemaError, match="explicit timezone"):
        parse_tcm_exam(
            exam_payload(
                measured_at="2026-11-07T10:00:00",
                received_at="2026-11-07T10:00:02",
            )
        )
    with pytest.raises(SchemaError, match="precedes"):
        parse_tcm_exam(
            exam_payload(
                measured_at="2026-11-07T10:00:05+00:00",
                received_at="2026-11-07T10:00:00+00:00",
            )
        )


def test_vendor_finding_is_never_clinical_truth():
    finding = ConstitutionFinding(
        name="气虚",
        source=CONSTITUTION_SOURCE_DEVICE,
        confidence=0.8,
        algorithm="vendor-v1",
        version="v1",
        is_clinical_truth=True,
    )

    assert finding.is_clinical_truth is False


def test_quality_is_derived_from_observed_fields():
    complete = parse_tcm_exam(exam_payload())
    assert complete.quality == QUALITY_OK

    device_only = parse_tcm_exam(
        exam_payload(tongue={}, face={}, constitution_names=["气虚"])
    )
    assert device_only.quality == QUALITY_DEGRADED

    empty = parse_tcm_exam(
        exam_payload(tongue={}, face={}, constitution_names=[])
    )
    assert empty.quality == QUALITY_REJECTED


@pytest.mark.parametrize(
    "missing",
    ["topic_id", "title", "text", "source", "license", "version", "review_status", "reviewed_by"],
)
def test_registry_excludes_entries_missing_required_metadata(missing):
    raw = knowledge_entry()
    raw.pop(missing)
    registry = KnowledgeRegistry.from_json({"entries": [raw]})

    assert len(registry) == 0
    assert registry.entries == ()
    assert registry.excluded


def test_registry_excludes_non_approved_entries():
    registry = approved_registry(knowledge_entry(review_status="DRAFT"))

    assert len(registry) == 0
    assert "not APPROVED" in registry.excluded[0]


def test_registry_loads_from_json_path(tmp_path):
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps({"entries": [knowledge_entry()]}), encoding="utf-8")

    assert len(KnowledgeRegistry.from_json(path)) == 1


def test_approved_entry_matched_by_device_constitution_is_displayed():
    registry = approved_registry()
    exam = parse_tcm_exam(exam_payload(constitution_names=["气虚"]))

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_OK
    assert len(result.knowledge) == 1
    item = result.knowledge[0]
    assert item["topic_id"] == "qi-deficiency-daily"
    assert item["source"] == "reviewed-wellness-guide"
    assert item["license"] == "CC-BY-4.0"
    assert item["version"] == "2026-11"
    assert item["reviewed_by"] == "tcm-reviewer-a"
    assert item["matched_constitution"] == ["气虚"]
    assert item["source_tag"] == "KNOWLEDGE"

    assert result.provenance["not_a_diagnosis"] is True
    assert result.provenance["knowledge_registry_entries"] == 1
    assert result.provenance["knowledge_entries_matched"] == 1
    assert result.provenance["exam_provenance"]["source_record"] == "msg-1"
    assert result.constitution_findings[0]["source"] == CONSTITUTION_SOURCE_DEVICE
    assert result.constitution_findings[0]["is_clinical_truth"] is False


def test_no_device_constitution_does_not_infer_from_tongue_signs():
    registry = approved_registry()
    exam = parse_tcm_exam(exam_payload(constitution_names=[]))

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_UNKNOWN
    assert result.knowledge == []
    assert "not used to infer" in result.reason
    assert result.provenance["device_constitutions"] == []
    assert exam.tongue_body_color is not None


def test_project_model_finding_is_not_used_for_knowledge():
    registry = approved_registry()
    exam = parse_tcm_exam(
        exam_payload(
            constitution_names=[
                {"name": "气虚", "source": "PROJECT_MODEL", "confidence": 0.4}
            ]
        )
    )

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_UNKNOWN
    assert result.knowledge == []
    assert result.constitution_findings[0]["source"] == CONSTITUTION_SOURCE_PROJECT_MODEL
    assert result.constitution_findings[0]["is_clinical_truth"] is False


def test_no_approved_knowledge_abstains_without_displaying_anything():
    registry = approved_registry(knowledge_entry(review_status="DRAFT"))
    exam = parse_tcm_exam(exam_payload(constitution_names=["气虚"]))

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_ABSTAIN
    assert result.knowledge == []
    assert "no approved knowledge" in result.reason


def test_empty_default_registry_abstains():
    exam = parse_tcm_exam(exam_payload(constitution_names=["气虚"]))

    result = build_tcm_sidecar(exam)

    assert result.status == STATUS_ABSTAIN
    assert result.knowledge == []
    assert result.provenance["knowledge_registry_entries"] == 0


def test_unknown_device_constitution_does_not_fabricate_knowledge():
    registry = approved_registry()
    exam = parse_tcm_exam(exam_payload(constitution_names=["未知体质"]))

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_ABSTAIN
    assert result.knowledge == []
    assert result.constitution_findings[0]["name"] == "未知体质"
    assert result.constitution_findings[0]["source"] == CONSTITUTION_SOURCE_DEVICE


def test_partial_match_when_one_device_label_has_no_knowledge():
    registry = approved_registry()
    exam = parse_tcm_exam(exam_payload(constitution_names=["气虚", "阴虚"]))

    result = build_tcm_sidecar(exam, registry)

    assert result.status == STATUS_PARTIAL
    assert len(result.knowledge) == 1
    assert result.provenance["matched_constitutions"] == ["气虚"]
    assert set(result.provenance["device_constitutions"]) == {"气虚", "阴虚"}


def test_module_has_no_mendeley_or_research_training_imports():
    tree = ast.parse(SRC_TCM_EXAM.read_text(encoding="utf-8"))
    banned_prefixes = ("research", "data.mendeley", "mendeley")
    banned_names = {"research", "mendeley", "importlib", "torch", "sklearn"}

    def banned(module_name: str) -> bool:
        return module_name in banned_names or module_name.startswith(banned_prefixes)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not banned(alias.name), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not banned(module), module
            if node.level > 0:
                for alias in node.names:
                    assert alias.name not in banned_names, alias.name


def test_module_imports_without_research_or_mendeley_models():
    code = (
        "import sys; sys.path.insert(0, 'src'); import physio.tcm_exam; "
        "leaked = [m for m in sys.modules if m.startswith("
        "('research', 'data.mendeley', 'mendeley'))]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
