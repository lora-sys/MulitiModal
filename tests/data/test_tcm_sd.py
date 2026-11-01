"""TCM-SD data access layer tests (#53 stage 1).

Everything runs on explicit SYNTHETIC fixtures built in tmp_path; the
expected-hash table is monkeypatched to the fixture hashes. The real-data
tests run only where data/local/tcm-sd exists (git-ignored, dev machine)
and skip everywhere else — they never read or write anything outside the
ignored data directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data import tcm_sd
from data.tcm_sd import (
    EXCLUDED_LEAK_RECORDS,
    LABEL_TAXONOMY_ID,
    SPLIT_SEED,
    TcmSdRecord,
    build_clinical_text,
    build_short_text,
    build_tcm_sd_manifest,
    deduplicate_evaluation_records,
    load_syndrome_knowledge,
    load_syndrome_vocab,
    load_tcm_sd_records,
    subject_partitions,
    verify_tcm_sd_hashes,
)

pytestmark = pytest.mark.synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / tcm_sd.RAW_DIR_NAME
has_real_data = RAW_DIR.is_dir()

SYN_VOCAB = ["气虚证", "血瘀证", "湿热证"]

# 10 synthetic cases over 6 users; user u1 appears in two files (the
# subject-overlap the real data forces us to repartition around).
SYN_RECORDS = [
    {"user_id": "u1", "lcd_id": "A01", "lcd_name": "咳嗽病", "syndrome": "肺气虚",
     "norm_syndrome": "气虚证", "chief_complaint": "咳嗽2月。",
     "description": "患者2月前受凉后咳嗽，干咳少痰。", "detection": "舌淡红，苔薄白，脉细。"},
    {"user_id": "u2", "lcd_id": "A02", "lcd_name": "胸痹心痛病", "syndrome": "心血瘀阻",
     "norm_syndrome": "血瘀证", "chief_complaint": "胸闷痛3天。",
     "description": "患者3天前出现胸闷痛，痛处固定。", "detection": "舌暗，有瘀斑，脉涩。"},
    {"user_id": "u3", "lcd_id": "A03", "lcd_name": "胃脘痛", "syndrome": "湿热中阻",
     "norm_syndrome": "湿热证", "chief_complaint": "胃脘痞满1周。",
     "description": "患者1周来胃脘痞满，口苦。", "detection": "舌红，苔黄腻，脉滑数。"},
    {"user_id": "u1", "lcd_id": "A04", "lcd_name": "眩晕病", "syndrome": "气血亏虚",
     "norm_syndrome": "气虚证", "chief_complaint": "头晕反复半年。",
     "description": "患者半年来头晕时作，劳累加重。", "detection": "舌淡，苔薄白，脉弱。"},
    {"user_id": "u4", "lcd_id": "A01", "lcd_name": "咳嗽病", "syndrome": "痰热郁肺",
     "norm_syndrome": "湿热证", "chief_complaint": "咳嗽咳痰1周。",
     "description": "患者1周前感冒后咳嗽，痰黄稠。", "detection": ""},
    {"user_id": "u5", "lcd_id": "A05", "lcd_name": "水肿病", "syndrome": "脾虚湿困",
     "norm_syndrome": "气虚证", "chief_complaint": "",
     "description": "患者双下肢浮肿1月，按之凹陷。", "detection": "舌淡胖，苔白滑，脉沉缓。"},
    {"user_id": "u6", "lcd_id": "A06", "lcd_name": "不寐病", "syndrome": "心脾两虚",
     "norm_syndrome": "气虚证", "chief_complaint": "失眠多梦2月。",
     "description": "", "detection": "舌淡，苔薄，脉细弱。"},
    {"user_id": "u2", "lcd_id": "A07", "lcd_name": "腰痛病", "syndrome": "肾虚血瘀",
     "norm_syndrome": "血瘀证", "chief_complaint": "腰痛刺痛半年。",
     "description": "患者半年来腰部刺痛，夜间加重。", "detection": "舌紫暗，脉涩。"},
]

SYN_KNOWLEDGE = [
    {
        "Name": "气虚证",
        "Definition": "元气不足之证。",
        "Typical_performance": "神疲乏力，少气懒言。",
        "Common_isease": "1、咳嗽\n症见咳嗽声低。",
    },
    {
        "Name": "血瘀证",
        "Definition": "瘀血内阻之证。",
        "Typical_performance": "刺痛固定，舌暗有瘀斑。",
        "Common_isease": "",
    },
    {
        "Name": "湿热证",
        "Definition": "湿热蕴结之证。",
        "Typical_performance": "苔黄腻，脉滑数。",
        "Common_isease": "1、胃脘痛",
    },
]


def _write_jsonl(path: Path, rows) -> str:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_raw_dir(tmp_path: Path) -> Path:
    """Write a complete synthetic raw dir and point the module's hash table
    at the fixture hashes."""
    raw = tmp_path / "tcm-sd"
    raw.mkdir()
    hashes = {
        "train.json": _write_jsonl(raw / "train.json", SYN_RECORDS[:6]),
        "dev.json": _write_jsonl(raw / "dev.json", SYN_RECORDS[6:]),
        "test.json": _write_jsonl(raw / "test.json", []),
        "syndrome_vocab.txt": _write_text(raw / "syndrome_vocab.txt", "\n".join(SYN_VOCAB) + "\n"),
        "syndrome_knowledge.json": _write_jsonl(raw / "syndrome_knowledge.json", SYN_KNOWLEDGE),
    }
    return raw, hashes


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    raw, hashes = make_raw_dir(tmp_path)
    monkeypatch.setattr(tcm_sd, "EXPECTED_SHA256", hashes)
    return raw


def _record(**overrides) -> TcmSdRecord:
    base = {
        "user_id": "u1", "lcd_id": "A01", "lcd_name": "咳嗽病", "syndrome": "肺气虚",
        "norm_syndrome": "气虚证", "chief_complaint": "咳嗽2月。",
        "description": "干咳少痰。", "detection": "脉细。", "source_file": "train.json",
    }
    base.update(overrides)
    return TcmSdRecord(**base)


# --- hash gate -----------------------------------------------------------

def test_hash_mismatch_refuses_to_load(raw_dir):
    (raw_dir / "train.json").write_text(
        (raw_dir / "train.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tcm_sd_records(raw_dir)


def test_missing_file_refuses_to_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_tcm_sd_hashes(tmp_path)


def test_tampered_vocab_refuses_to_load(raw_dir):
    (raw_dir / "syndrome_vocab.txt").write_text("气虚证\n伪造证\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_syndrome_vocab(raw_dir)


# --- loaders -------------------------------------------------------------

def test_load_records_parses_all_fields(raw_dir):
    records = load_tcm_sd_records(raw_dir)
    assert len(records) == len(SYN_RECORDS)
    first = records[0]
    assert first.user_id == "u1" and first.norm_syndrome == "气虚证"
    assert first.source_file == "train.json"
    # empty detection survives as "" instead of a missing-key error
    empty_detection = [r for r in records if r.user_id == "u4"][0]
    assert empty_detection.detection == ""


def test_load_vocab_preserves_order_and_rejects_duplicates(raw_dir, monkeypatch):
    assert load_syndrome_vocab(raw_dir) == SYN_VOCAB
    # rebuild a duplicate-label vocab with a matching hash table
    text = "气虚证\n气虚证\n"
    (raw_dir / "syndrome_vocab.txt").write_text(text, encoding="utf-8")
    monkeypatch.setitem(
        tcm_sd.EXPECTED_SHA256,
        "syndrome_vocab.txt",
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ValueError, match="duplicate labels"):
        load_syndrome_vocab(raw_dir)


def test_load_knowledge_keyed_by_name(raw_dir):
    knowledge = load_syndrome_knowledge(raw_dir)
    assert set(knowledge) == {"气虚证", "血瘀证", "湿热证"}
    assert knowledge["气虚证"].typical_performance == "神疲乏力，少气懒言。"
    assert knowledge["气虚证"].common_disease.startswith("1、咳嗽")


def test_load_knowledge_rejects_duplicate_names(raw_dir, monkeypatch):
    rows = SYN_KNOWLEDGE + [dict(SYN_KNOWLEDGE[0])]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (raw_dir / "syndrome_knowledge.json").write_text(payload, encoding="utf-8")
    monkeypatch.setitem(
        tcm_sd.EXPECTED_SHA256,
        "syndrome_knowledge.json",
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ValueError, match="duplicate knowledge entry"):
        load_syndrome_knowledge(raw_dir)


# --- subject partitioning ------------------------------------------------

def _all_users(parts):
    return parts["train"] | parts["dev"] | parts["test"]


def test_partitions_cover_every_user_exactly_once(raw_dir):
    records = load_tcm_sd_records(raw_dir)
    parts = subject_partitions(records)
    users = {r.user_id for r in records}
    assert _all_users(parts) == users
    assert len(parts["train"]) + len(parts["dev"]) + len(parts["test"]) == len(users)
    assert not (parts["train"] & parts["dev"])
    assert not (parts["train"] & parts["test"])
    assert not (parts["dev"] & parts["test"])


def test_partitions_are_deterministic(raw_dir):
    records = load_tcm_sd_records(raw_dir)
    assert subject_partitions(records) == subject_partitions(records, SPLIT_SEED)


def test_different_seed_changes_assignment():
    # 100 users per label: each stratum is big enough that a different seed
    # actually reshuffles members across partitions (the 6-user fixture is
    # too small — every stratum rounds to 0 dev/test users).
    records = [
        _record(user_id=f"a{i:04d}", norm_syndrome="气虚证") for i in range(100)
    ] + [_record(user_id=f"b{i:04d}", norm_syndrome="血瘀证") for i in range(100)]
    first = subject_partitions(records, seed=SPLIT_SEED)
    second = subject_partitions(records, seed=SPLIT_SEED + 1)
    assert first != second
    assert _all_users(first) == _all_users(second)


def test_partitions_hit_target_ratios():
    # 1,000 synthetic users, one record each, one label: ratio assertions
    # need a population large enough for round() to be meaningful.
    records = [
        _record(user_id=f"u{i:05d}", norm_syndrome="气虚证", source_file="train.json")
        for i in range(1000)
    ]
    parts = subject_partitions(records)
    n = len(_all_users(parts))
    assert len(parts["train"]) / n == pytest.approx(0.82, abs=0.02)
    assert len(parts["dev"]) / n == pytest.approx(0.10, abs=0.02)
    assert len(parts["test"]) / n == pytest.approx(0.08, abs=0.02)


def test_partitions_stratify_by_majority_label():
    # u1 has 2 气虚证 vs 1 血瘀证 -> must land in the 气虚证 stratum pool
    records = [
        _record(user_id="u1", norm_syndrome="气虚证"),
        _record(user_id="u1", norm_syndrome="气虚证", source_file="dev.json"),
        _record(user_id="u1", norm_syndrome="血瘀证", source_file="test.json"),
        *[_record(user_id=f"x{i:04d}", norm_syndrome="血瘀证") for i in range(50)],
        *[_record(user_id=f"y{i:04d}", norm_syndrome="气虚证") for i in range(50)],
    ]
    parts = subject_partitions(records, seed=SPLIT_SEED)
    # not assertable which partition u1 gets, only that it gets exactly one
    owners = [name for name in ("train", "dev", "test") if "u1" in parts[name]]
    assert owners == [owners[0]]
    assert _all_users(parts) == {r.user_id for r in records}


def test_split_seed_and_taxonomy_are_frozen():
    assert SPLIT_SEED == 20261101
    assert LABEL_TAXONOMY_ID == "tcm-sd-syndrome-vocab-v1"


# --- text construction ---------------------------------------------------

def test_build_clinical_text_skips_empty_segments():
    r = _record(chief_complaint="", description="  ", detection="脉细。")
    assert build_clinical_text(r) == "脉细。"
    full = _record()
    assert build_clinical_text(full) == "咳嗽2月。\n干咳少痰。\n脉细。"


def test_build_short_text_is_chief_only():
    assert build_short_text(_record()) == "咳嗽2月。"
    assert build_short_text(_record(chief_complaint="")) == ""


# --- manifest ------------------------------------------------------------

def test_manifest_is_replayable(raw_dir):
    m1 = build_tcm_sd_manifest(raw_dir)
    m2 = build_tcm_sd_manifest(raw_dir)
    assert m1 == m2
    assert m1["n_records"] == len(SYN_RECORDS)
    assert m1["n_labels"] == 3
    assert m1["label_taxonomy_id"] == LABEL_TAXONOMY_ID
    by_name = {f["name"]: f for f in m1["files"]}
    assert by_name["train.json"]["lines"] == 6
    assert by_name["test.json"]["lines"] == 0
    assert m1["labels_out_of_vocab"] == []
    assert m1["audit_conclusions"]["report"].endswith("tcm-sd-syndrome-audit.md")


def test_manifest_reports_out_of_vocab_labels(raw_dir, monkeypatch):
    # u7 carries a label outside the vocab -> manifest must surface it,
    # never silently fold it into the taxonomy
    extra = dict(SYN_RECORDS[0], norm_syndrome="编造证")
    rows = [*SYN_RECORDS[:6], extra]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (raw_dir / "train.json").write_text(payload, encoding="utf-8")
    monkeypatch.setitem(
        tcm_sd.EXPECTED_SHA256,
        "train.json",
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
    manifest = build_tcm_sd_manifest(raw_dir)
    assert manifest["labels_out_of_vocab"] == ["编造证"]


# --- evaluation-set dedup (R1 F1: one-way, train kept in full) -------------

PARTS_3 = {
    "train": frozenset({"t1", "t2"}),
    "dev": frozenset({"d1", "d2"}),
    "test": frozenset({"x1"}),
}


def _dup_text_records() -> list[TcmSdRecord]:
    """t1/t2 train; d1 duplicates t1 byte-for-byte (label conflict), d2 unique;
    x1 (test) duplicates d2 (dev-side text, absent from train)."""
    return [
        _record(user_id="t1", norm_syndrome="气虚证", chief_complaint="咳嗽2月。",
                description="干咳少痰。", detection="脉细。"),
        _record(user_id="t2", norm_syndrome="血瘀证", chief_complaint="胸闷3天。",
                description="刺痛固定。", detection="舌暗。"),
        _record(user_id="d1", norm_syndrome="湿热证", chief_complaint="咳嗽2月。",
                description="干咳少痰。", detection="脉细。"),
        _record(user_id="d2", norm_syndrome="气虚证", chief_complaint="失眠2月。",
                description="多梦易醒。", detection="舌淡。"),
        _record(user_id="x1", norm_syndrome="气虚证", chief_complaint="失眠2月。",
                description="多梦易醒。", detection="舌淡。"),
    ]


def test_dedup_drops_dev_cases_duplicating_train_text():
    records = _dup_text_records()
    kept, stats = deduplicate_evaluation_records(records, PARTS_3)
    kept_users = [r.user_id for r in kept]
    # d1 duplicates t1 byte-for-byte -> dropped from dev; train side kept full
    assert "d1" not in kept_users and "t1" in kept_users
    assert "d2" in kept_users
    assert [r.user_id for r in kept if r.user_id in PARTS_3["train"]] == ["t1", "t2"]
    assert stats["dropped"] == {"dev": 1, "test": 1}
    assert stats["kept"] == {"train": 2, "dev": 1, "test": 0}


def test_dedup_drops_test_cases_against_train_and_dev():
    records = _dup_text_records()
    kept, _stats = deduplicate_evaluation_records(records, PARTS_3)
    # x1 duplicates d2 (a kept dev text) but not any train text -> still dropped
    assert "x1" not in [r.user_id for r in kept]


def test_dedup_label_conflict_group_treated_identically():
    # same text with CONFLICTING labels across train/dev is still a duplicate:
    # t1 气虚证 vs d1 湿热证 — the dev copy is dropped regardless of label
    records = _dup_text_records()
    assert records[0].norm_syndrome != records[2].norm_syndrome
    assert build_clinical_text(records[0]) == build_clinical_text(records[2])
    kept, _stats = deduplicate_evaluation_records(records, PARTS_3)
    assert "d1" not in [r.user_id for r in kept]


def test_dedup_is_deterministic_and_replayable():
    records = _dup_text_records()
    first, stats_first = deduplicate_evaluation_records(records, PARTS_3)
    second, stats_second = deduplicate_evaluation_records(records, PARTS_3)
    assert [r.user_id for r in first] == [r.user_id for r in second]
    assert stats_first == stats_second
    # membership is decided by the partition sets alone, not input order
    reversed_kept, reversed_stats = deduplicate_evaluation_records(
        list(reversed(records)), PARTS_3
    )
    assert {r.user_id for r in reversed_kept} == {r.user_id for r in first}
    assert reversed_stats == stats_second


def test_dedup_refuses_records_outside_every_partition():
    records = _dup_text_records() + [_record(user_id="ghost")]
    with pytest.raises(ValueError, match="outside every partition"):
        deduplicate_evaluation_records(records, PARTS_3)


# --- leak-record exclusion (R1 F2) ------------------------------------------

def test_excluded_leak_records_constant_is_frozen():
    assert EXCLUDED_LEAK_RECORDS == {
        "484293": (
            "norm_syndrome string appears verbatim in detection (audit §5)"
        )
    }


def test_load_skips_excluded_leak_records(raw_dir, monkeypatch):
    leak = dict(SYN_RECORDS[0], user_id="484293",
                detection="辩证当属湿热瘀阻证。")
    rows = [*SYN_RECORDS[:6], leak]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (raw_dir / "train.json").write_text(payload, encoding="utf-8")
    monkeypatch.setitem(
        tcm_sd.EXPECTED_SHA256,
        "train.json",
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
    records = load_tcm_sd_records(raw_dir)
    # 9 fixture rows (6 train + 2 dev + the leak record) minus 1 skipped
    assert len(records) == 8
    assert "484293" not in {r.user_id for r in records}


def test_manifest_counts_load_time_exclusions(raw_dir, monkeypatch):
    leak = dict(SYN_RECORDS[0], user_id="484293",
                detection="辩证当属湿热瘀阻证。")
    rows = [*SYN_RECORDS[:6], leak]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (raw_dir / "train.json").write_text(payload, encoding="utf-8")
    monkeypatch.setitem(
        tcm_sd.EXPECTED_SHA256,
        "train.json",
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
    manifest = build_tcm_sd_manifest(raw_dir)
    assert manifest["n_records"] == 8
    assert manifest["records_excluded_at_load"] == 1
    assert manifest["excluded_leak_user_ids"] == ["484293"]


# --- real data (skipped without data/local/tcm-sd) ------------------------

@pytest.mark.skipif(not has_real_data, reason="real dataset only under data/local/tcm-sd")
def test_real_files_load_and_match_frozen_constants():
    records = load_tcm_sd_records(RAW_DIR)
    # 54,152 raw cases minus the 1 leak record excluded at load (R1 F2)
    assert len(records) == 54151
    assert "484293" not in {r.user_id for r in records}
    vocab = load_syndrome_vocab(RAW_DIR)
    assert len(vocab) == 148
    assert {r.norm_syndrome for r in records} == set(vocab)
    knowledge = load_syndrome_knowledge(RAW_DIR)
    assert len(knowledge) == 1027
    assert set(vocab) <= set(knowledge)


@pytest.mark.skipif(not has_real_data, reason="real dataset only under data/local/tcm-sd")
def test_real_subject_partitions_invariants():
    records = load_tcm_sd_records(RAW_DIR)
    parts = subject_partitions(records)
    users = {r.user_id for r in records}
    assert _all_users(parts) == users
    assert not (parts["train"] & parts["dev"] or parts["train"] & parts["test"]
                or parts["dev"] & parts["test"])
    n = len(users)
    assert len(parts["train"]) / n == pytest.approx(0.82, abs=0.01)
    assert len(parts["dev"]) / n == pytest.approx(0.10, abs=0.01)
    assert len(parts["test"]) / n == pytest.approx(0.08, abs=0.01)
    assert subject_partitions(records) == parts  # reproducible


@pytest.mark.skipif(not has_real_data, reason="real dataset only under data/local/tcm-sd")
def test_real_manifest_hashes_match_registry_record():
    from data.datasets import DATASET_REGISTRY, verify_files

    record = DATASET_REGISTRY["tcm-sd-syndrome-v1"]
    assert all(verify_files(record).values())
    manifest = build_tcm_sd_manifest(RAW_DIR)
    # loaded view: raw 54,152 minus the 1 leak record excluded at load
    assert manifest["n_records"] == 54151
    assert manifest["n_users"] == 52294
    assert manifest["records_excluded_at_load"] == 1
    # the registry documents the RAW dataset (before load-time exclusion)
    assert record.n_subjects == 52295


@pytest.mark.skipif(not has_real_data, reason="real dataset only under data/local/tcm-sd")
def test_real_evaluation_dedup_replays_deterministically():
    records = load_tcm_sd_records(RAW_DIR)
    parts = subject_partitions(records)
    kept1, stats1 = deduplicate_evaluation_records(records, parts)
    kept2, stats2 = deduplicate_evaluation_records(records, parts)
    assert [r.lcd_id for r in kept1] == [r.lcd_id for r in kept2]
    assert stats1 == stats2
    # train side is kept in full — only evaluation partitions shrink
    assert stats1["kept"]["train"] == sum(1 for r in records if r.user_id in parts["train"])
    assert stats1["dropped"]["dev"] > 0  # the R1 finding this fix removes
