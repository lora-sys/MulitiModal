"""TCM-SD syndrome data access layer (#53 stage 1).

Frozen contract from the #53 arbitration: module-level constants, SHA-256
gated loading, subject-level partitioning and text construction. The real
files live read-only under data/local/tcm-sd/ (git-ignored per #26/#37);
this module never writes and never copies data into Git. Expected hashes
were taken from the actual downloaded files on 2026-11-01 (see
docs/data/tcm-sd-syndrome-audit.md).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RAW_DIR_NAME = "data/local/tcm-sd"
SPLIT_SEED = 20261101
LABEL_TAXONOMY_ID = "tcm-sd-syndrome-vocab-v1"
LICENSE = "CC BY-NC-SA 4.0 (non-commercial research use only)"
UPSTREAM_URL = "https://github.com/Borororo/ZY-BERT"

RECORD_FILES = ("train.json", "dev.json", "test.json")
VOCAB_FILE = "syndrome_vocab.txt"
KNOWLEDGE_FILE = "syndrome_knowledge.json"

#: SHA-256 of the exact files audited on 2026-11-01. A mismatch aborts
#: loading — raw files are never silently replaced.
EXPECTED_SHA256 = {
    "train.json": "d29d3be5eccc82a1a2654f06f3aa09f5aff8dbecbb29515980a272b1c39663eb",
    "dev.json": "34e0e85f03d2d3e4fab93856c32dc99c71e8f76b74dc5f9b92123127669b1540",
    "test.json": "cfcd6f20e74c6658e4677cbaed93622a90eceb35a62bad792ba443549af31ea9",
    "syndrome_vocab.txt": "63ee108754d90914a3833328abda7e04130f200617ca79033a66d22e47c45c83",
    "syndrome_knowledge.json": "c98b830f4d13c44b106f3bc2d40571c6070f9da7228e8c264cb41e8f8ec3bd9e",
}

#: Subject-level repartition ratios (#53: official user_id sets overlap).
TRAIN_RATIO = 0.82
DEV_RATIO = 0.10
TEST_RATIO = 0.08

#: Records excluded at load time (R1 F2). The audit (§5) found exactly one
#: case whose norm_syndrome string appears verbatim in its own detection
#: text — a label leak into the input. Keyed by user_id; loading skips these
#: records entirely so no downstream path can train or evaluate on them.
EXCLUDED_LEAK_RECORDS = {
    "484293": "norm_syndrome string appears verbatim in detection (audit §5)",
}

AUDIT_CONCLUSIONS = {
    "audit_date": "2026-11-01",
    "records": 54152,
    "unique_users": 52295,
    "label_taxonomy": LABEL_TAXONOMY_ID,
    "vocab_size": 148,
    "norm_syndrome_out_of_vocab": 0,
    "syndrome_differs_from_norm": 6725,
    "label_leak_into_text_records": 1,
    "differentiation_conclusion_marker_records": 9522,
    "prescription_marker_records": 305,
    "exact_duplicate_text_groups": 2635,
    "duplicate_group_label_conflicts": 334,
    "official_split_user_overlap": {"train∩dev": 19, "train∩test": 18, "dev∩test": 13},
    "label_nature": "single-label per record; 148-way closed set",
    "report": "docs/data/tcm-sd-syndrome-audit.md",
}


@dataclass(frozen=True)
class TcmSdRecord:
    user_id: str
    lcd_id: str
    lcd_name: str
    syndrome: str
    norm_syndrome: str
    chief_complaint: str
    description: str
    detection: str
    source_file: str


@dataclass(frozen=True)
class SyndromeKnowledge:
    name: str
    definition: str
    typical_performance: str
    # upstream key is "Common_isease" (sic) — typo is part of the source file
    common_disease: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tcm_sd_hashes(raw_dir: Path, names: tuple[str, ...] = RECORD_FILES) -> None:
    """Raise ValueError unless every named file matches its expected hash."""
    for name in names:
        path = Path(raw_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing TCM-SD file: {path}")
        actual = _sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise ValueError(
                f"hash mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {actual}"
            )


def load_tcm_sd_records(raw_dir: Path) -> list[TcmSdRecord]:
    """Load all official-split records after hash verification (read-only).

    Records whose user_id is in EXCLUDED_LEAK_RECORDS are skipped at this
    boundary (R1 F2), so every consumer downstream sees the same cleaned
    stream; the raw files themselves stay untouched.
    """
    raw_dir = Path(raw_dir)
    verify_tcm_sd_hashes(raw_dir, RECORD_FILES)
    records: list[TcmSdRecord] = []
    for name in RECORD_FILES:
        with (raw_dir / name).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if str(obj["user_id"]) in EXCLUDED_LEAK_RECORDS:
                    continue
                records.append(
                    TcmSdRecord(
                        user_id=str(obj["user_id"]),
                        lcd_id=str(obj["lcd_id"]),
                        lcd_name=str(obj["lcd_name"]),
                        syndrome=str(obj["syndrome"]),
                        norm_syndrome=str(obj["norm_syndrome"]),
                        chief_complaint=str(obj.get("chief_complaint") or ""),
                        description=str(obj.get("description") or ""),
                        detection=str(obj.get("detection") or ""),
                        source_file=name,
                    )
                )
    return records


def load_syndrome_vocab(raw_dir: Path) -> list[str]:
    """Load the frozen 148-syndrome taxonomy, order preserved."""
    raw_dir = Path(raw_dir)
    verify_tcm_sd_hashes(raw_dir, (VOCAB_FILE,))
    lines = (raw_dir / VOCAB_FILE).read_text(encoding="utf-8").splitlines()
    vocab = [line.strip() for line in lines if line.strip()]
    if len(vocab) != len(set(vocab)):
        raise ValueError("syndrome_vocab.txt contains duplicate labels")
    return vocab


def load_syndrome_knowledge(raw_dir: Path) -> dict[str, SyndromeKnowledge]:
    """Load the 1,027-entry syndrome knowledge base, keyed by Name (JSONL)."""
    raw_dir = Path(raw_dir)
    verify_tcm_sd_hashes(raw_dir, (KNOWLEDGE_FILE,))
    knowledge: dict[str, SyndromeKnowledge] = {}
    with (raw_dir / KNOWLEDGE_FILE).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entry = SyndromeKnowledge(
                name=str(obj["Name"]),
                definition=str(obj.get("Definition") or ""),
                typical_performance=str(obj.get("Typical_performance") or ""),
                common_disease=str(obj.get("Common_isease") or ""),
            )
            if entry.name in knowledge:
                raise ValueError(f"duplicate knowledge entry for {entry.name!r}")
            knowledge[entry.name] = entry
    return knowledge


def subject_partitions(
    records: list[TcmSdRecord], seed: int = SPLIT_SEED
) -> dict[str, frozenset[str]]:
    """Deterministic subject-level train/dev/test repartition of user_ids.

    Every user lands in exactly one partition regardless of which official
    file their records came from (the official user_id sets overlap). Users
    are stratified by their majority norm_syndrome (ties broken by the
    lexicographically smallest label) and assigned per stratum with a seeded
    shuffle, targeting 0.82/0.10/0.08 by user count. Same records + same
    seed => same partitions.
    """
    counts: dict[str, Counter[str]] = {}
    for r in records:
        counts.setdefault(r.user_id, Counter())[r.norm_syndrome] += 1
    strata: dict[str, list[str]] = {}
    for user_id, label_counts in counts.items():
        majority = min((-n, label) for label, n in label_counts.items())[1]
        strata.setdefault(majority, []).append(user_id)

    parts: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for label in sorted(strata):
        users = sorted(strata[label])
        random.Random(f"{seed}|{LABEL_TAXONOMY_ID}|{label}").shuffle(users)
        n_test = round(len(users) * TEST_RATIO)
        n_dev = round(len(users) * DEV_RATIO)
        parts["test"] += users[:n_test]
        parts["dev"] += users[n_test : n_test + n_dev]
        parts["train"] += users[n_test + n_dev :]
    return {
        "train": frozenset(parts["train"]),
        "dev": frozenset(parts["dev"]),
        "test": frozenset(parts["test"]),
    }


def build_clinical_text(record: TcmSdRecord) -> str:
    """Full clinical input: chief complaint, description, detection (non-empty)."""
    return "\n".join(
        part
        for part in (record.chief_complaint, record.description, record.detection)
        if part.strip()
    )


def build_short_text(record: TcmSdRecord) -> str:
    """Short user-side input: chief complaint only."""
    return record.chief_complaint


def deduplicate_evaluation_records(
    records: list[TcmSdRecord],
    partitions: dict[str, frozenset[str]],
) -> tuple[list[TcmSdRecord], dict]:
    """One-way evaluation-set dedup against earlier partitions (R1 F1).

    A dev case whose clinical full text (``build_clinical_text``, non-empty
    segments joined — the audit's duplicate basis, §4) is byte-identical to
    any train case is dropped from the evaluation side; a test case is dropped
    when its text matches any train or dev case. TRAIN IS KEPT IN FULL:
    train-side duplicates are genuine repeat visits in the source corpus and
    removing them would discard real data, so the filter is deliberately
    one-way — it only removes evaluation cases whose score could be inflated
    by memorising a train copy.

    Deterministic and replayable: membership is decided by the partition
    sets alone, never by input order or by which duplicate is seen first.
    Records whose user belongs to no partition are refused (partitions from
    ``subject_partitions`` cover every user by construction, so this only
    fires on a caller/records mismatch). Empty clinical texts would compare
    equal to each other; the audited corpus has none (audit §1), so no
    special case is made. Labels are irrelevant to matching — the 334
    label-conflicting duplicate groups (audit §4) are filtered identically.

    Returns ``(kept_records, stats)``; ``stats`` is provenance-grade:
    ``{"policy", "text_basis", "dropped": {"dev", "test"}, "kept":
    {"train", "dev", "test"}}``.
    """
    train, dev, test = (
        [r for r in records if r.user_id in partitions[name]]
        for name in ("train", "dev", "test")
    )
    known = {r.user_id for r in train} | {r.user_id for r in dev} | {
        r.user_id for r in test
    }
    unknown_users = sorted({r.user_id for r in records} - known)
    if unknown_users:
        raise ValueError(
            f"records with users outside every partition: {unknown_users[:3]}"
        )
    train_texts = {build_clinical_text(r) for r in train}
    dev_texts = {build_clinical_text(r) for r in dev}
    kept_train = list(train)
    kept_dev = [r for r in dev if build_clinical_text(r) not in train_texts]
    kept_test = [
        r for r in test
        if build_clinical_text(r) not in train_texts | dev_texts
    ]
    stats = {
        "policy": (
            "one-way evaluation-set dedup: dev cases byte-identical to a "
            "train clinical text are dropped, test cases byte-identical to "
            "any train or dev clinical text are dropped; train is kept in "
            "full (train-side duplicates are genuine repeat visits)"
        ),
        "text_basis": "build_clinical_text",
        "dropped": {"dev": len(dev) - len(kept_dev), "test": len(test) - len(kept_test)},
        "kept": {"train": len(kept_train), "dev": len(kept_dev), "test": len(kept_test)},
    }
    return kept_train + kept_dev + kept_test, stats


def build_tcm_sd_manifest(raw_dir: Path) -> dict:
    """Replayable dataset manifest: per-file provenance + audit conclusions.

    Hashes, sizes, line counts and label distribution are recomputed from the
    local files on every call; conclusions reference the 2026-11-01 audit in
    docs/data/tcm-sd-syndrome-audit.md.
    """
    raw_dir = Path(raw_dir)
    files = []
    for name in (*RECORD_FILES, VOCAB_FILE, KNOWLEDGE_FILE):
        path = raw_dir / name
        verify_tcm_sd_hashes(raw_dir, (name,))
        files.append(
            {
                "name": name,
                "url": f"{UPSTREAM_URL} ({name}; train/dev from TCM_SD_train_dev.zip)",
                "sha256": EXPECTED_SHA256[name],
                "size_bytes": path.stat().st_size,
                "lines": sum(1 for _ in path.open(encoding="utf-8")),
                "license": LICENSE,
            }
        )
    records = load_tcm_sd_records(raw_dir)
    label_counts = Counter(r.norm_syndrome for r in records)
    vocab = load_syndrome_vocab(raw_dir)
    unknown = set(label_counts) - set(vocab)
    partitions = subject_partitions(records)
    raw_record_lines = sum(
        f["lines"] for f in files if f["name"] in RECORD_FILES
    )
    return {
        "dataset_id": "tcm-sd-syndrome-v1",
        "generated_from": str(raw_dir),
        "label_taxonomy_id": LABEL_TAXONOMY_ID,
        "license": LICENSE,
        "source": UPSTREAM_URL,
        "split_seed": SPLIT_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "dev": DEV_RATIO, "test": TEST_RATIO},
        "files": files,
        "n_records": len(records),
        "records_excluded_at_load": raw_record_lines - len(records),
        "excluded_leak_user_ids": sorted(EXCLUDED_LEAK_RECORDS),
        "n_users": len(partitions["train"] | partitions["dev"] | partitions["test"]),
        "partition_users": {k: len(v) for k, v in partitions.items()},
        "n_labels": len(label_counts),
        "min_class_count": min(label_counts.values()),
        "max_class_count": max(label_counts.values()),
        "labels_out_of_vocab": sorted(unknown),
        "audit_conclusions": AUDIT_CONCLUSIONS,
    }
