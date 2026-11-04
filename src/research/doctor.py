"""Research Doctor: read-only diagnosis before and after a research round.

The Doctor exists because a low score is not a licence to tune hyperparameters.
Before a round it asks whether the protocol is trustworthy at all; after a round
it asks what the numbers actually show and what is simply unknown. It never
writes to the data, the split, the evaluator or the holdout — it reports, and a
`BLOCKED` finding stops the run instead of being worked around.

Severity vocabulary (part of the contract, not a style choice):

* ``BLOCKED``    the protocol cannot produce a trustworthy score; do not train.
* ``WARNING``    the run may proceed, but a reader must weigh this.
* ``INFO``       a fact worth recording.
* ``UNVERIFIED`` the check could not be performed. **Never upgrade this to pass
                 by assumption** — an unverifiable claim is stated as such.

What the Doctor deliberately cannot see: the holdout. Its inputs are the
selection manifest (train + dev) and the labelled records those partitions name.
A check that needs the holdout (e.g. "is the locked test truly disjoint?") is
reported UNVERIFIED with that reason rather than answered by peeking.

Two properties the engine already provides are *verified here* rather than
trusted:

* the candidate only ever receives training rows (``training_fold_data``), so an
  outer dev fold cannot drive early stopping — the Doctor checks the config for
  early-stopping keys and checks that no evaluation provider is attached to
  fitting;
* the trusted metric name must be one the frozen evaluator actually computes.

Everything is derived from the task adapter, so no syndrome-specific rule leaks
into the generic core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

DOCTOR_SCHEMA_VERSION = "autoresearch-doctor-v1"
BLOCKED = "BLOCKED"
WARNING = "WARNING"
INFO = "INFO"
UNVERIFIED = "UNVERIFIED"
SEVERITIES = (BLOCKED, WARNING, INFO, UNVERIFIED)

#: a class with fewer labelled examples than this is reported as a tail class
TAIL_CLASS_MIN_SUPPORT = 5
#: an outer fold spread larger than this fraction of the observed gain means the
#: comparison is not resolved at this resolution
NOISE_RATIO_WARNING = 1.0


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    message: str
    evidence: dict = field(default_factory=dict)
    reproduce: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; expected {SEVERITIES}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DoctorReport:
    mode: str
    task: str
    findings: list[Finding]
    run_id: str | None = None
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()

    @property
    def blocked(self) -> bool:
        return any(f.severity == BLOCKED for f in self.findings)

    def counts(self) -> dict:
        out = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            out[finding.severity] += 1
        return out

    def to_dict(self) -> dict:
        return {
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "mode": self.mode,
            "task": self.task,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "blocked": self.blocked,
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary_line(self) -> str:
        counts = self.counts()
        parts = [f"{counts[s]} {s}" for s in SEVERITIES if counts[s]]
        return f"{self.mode} doctor: " + (", ".join(parts) if parts else "no findings")


# -- adapter hooks (generic core; every task-specific rule lives in the adapter) --


def user_key(adapter, record) -> str | None:
    getter = getattr(adapter, "record_user_key", None)
    if getter is None:
        return None
    value = getter(record)
    return None if value is None else str(value)


def text_key(adapter, record) -> str | None:
    getter = getattr(adapter, "record_text_key", None)
    if getter is None:
        return None
    value = getter(record)
    if value is None:
        return None
    return " ".join(str(value).split())


def label_key(adapter, record):
    getter = getattr(adapter, "record_label_key", None)
    return None if getter is None else getter(record)


def label_space(adapter, manifest: dict):
    getter = getattr(adapter, "label_space", None)
    return None if getter is None else getter(manifest)


def metric_names(adapter):
    getter = getattr(adapter, "metric_names", None)
    return None if getter is None else tuple(getter())


# -- preflight ------------------------------------------------------------------


def preflight(config: dict, manifest: dict, records: list, adapter,
              run_id: str | None = None) -> DoctorReport:
    """Diagnose the protocol using ONLY the train/dev side of a frozen manifest."""
    findings: list[Finding] = []
    task = str(config.get("task") or getattr(adapter, "name", "unknown"))

    split = (manifest or {}).get("split") or {}
    train_ids = list(split.get("train") or [])
    dev_ids = list(split.get("dev") or [])
    if not train_ids or not dev_ids:
        findings.append(Finding(
            "split_present", BLOCKED,
            "the selection manifest must carry non-empty train and dev partitions",
            {"n_train": len(train_ids), "n_dev": len(dev_ids)},
            "rebuild the manifest with the frozen splitter; a run needs both sides",
        ))
        return DoctorReport("preflight", task, findings, run_id=run_id)

    findings.extend(_check_holdout_not_in_selection(manifest))
    findings.extend(_check_user_isolation(train_ids, dev_ids, adapter, records, manifest))
    findings.extend(_check_partition_membership(train_ids, dev_ids, manifest))
    findings.extend(_check_records_labelled(train_ids, dev_ids, adapter, records, manifest))
    findings.extend(_check_exact_text(adapter, records, manifest))
    findings.extend(_check_label_conflict(adapter, records, manifest))
    findings.extend(_check_label_space(adapter, records, manifest))
    findings.extend(_check_class_support(adapter, records, manifest))
    findings.extend(_check_selection_protocol(config, adapter))
    findings.extend(_check_budget_and_binding(config))
    return DoctorReport("preflight", task, findings, run_id=run_id)


def _rows_for(ids: list[str], manifest: dict) -> dict[str, int]:
    """subject id -> dataset row; ids the manifest does not map are ignored."""
    mapping = (manifest or {}).get("subject_row_by_id") or {}
    return {sid: mapping[sid] for sid in ids if sid in mapping}


def _check_holdout_not_in_selection(manifest: dict) -> list[Finding]:
    """The holdout may exist, but it must not be reachable by selection.

    A frozen manifest legitimately carries a holdout partition; the engine's
    fold/pool manifests are what empty it. So the defect is not "a holdout
    exists" but "holdout rows also appear in a selection partition", which is
    what would let a fold train on, or score, the locked test. The Doctor never
    reads holdout labels either way — the check is pure set arithmetic.
    """
    split = (manifest or {}).get("split") or {}
    holdout = set(split.get("holdout") or [])
    train = set(split.get("train") or [])
    dev = set(split.get("dev") or [])
    overlap = sorted(holdout & (train | dev))
    if overlap:
        return [Finding(
            "selection_holdout_closed", BLOCKED,
            f"{len(overlap)} holdout row(s) also appear in a selection partition",
            {"n_overlap": len(overlap), "sample": overlap[:5]},
            "the locked test must be disjoint from everything selection touches; "
            "rebuild the split before spending a holdout unlock",
        )]
    if holdout:
        return [Finding(
            "selection_holdout_closed", INFO,
            "the manifest carries a holdout partition, disjoint from train and dev; "
            "selection inputs are pruned per fold and the Doctor never reads it",
            {"n_holdout": len(holdout)},
        )]
    return [Finding(
        "selection_holdout_closed", INFO,
        "no holdout partition in these selection inputs",
        {"n_holdout": 0},
    )]


def _check_user_isolation(train_ids, dev_ids, adapter, records, manifest) -> list[Finding]:
    """The #1 leak: the same subject on both sides of a fold."""
    keyed = getattr(adapter, "record_user_key", None)
    if keyed is None:
        return [Finding(
            "user_isolation", UNVERIFIED,
            "the task adapter does not expose a subject/user key, so user-level "
            "isolation cannot be checked",
            {"adapter": type(adapter).__name__},
            "implement TaskAdapter.record_user_key for this task",
        )]
    rows = _rows_for(train_ids + dev_ids, manifest)
    by_row = {getattr(r, "subject_index", None): r for r in records}
    train_users, dev_users = set(), set()
    for sid in train_ids:
        record = by_row.get(rows.get(sid))
        if record is not None:
            value = user_key(adapter, record)
            if value is not None:
                train_users.add(value)
    for sid in dev_ids:
        record = by_row.get(rows.get(sid))
        if record is not None:
            value = user_key(adapter, record)
            if value is not None:
                dev_users.add(value)
    if not train_users or not dev_users:
        return [Finding(
            "user_isolation", UNVERIFIED,
            "subject keys were empty after resolution, so isolation is unchecked",
            {"n_train_users": len(train_users), "n_dev_users": len(dev_users)},
        )]
    crossing = sorted(train_users & dev_users)[:10]
    if crossing:
        return [Finding(
            "user_isolation", BLOCKED,
            f"{len(train_users & dev_users)} subject(s) appear in train and dev",
            {"n_crossing": len(train_users & dev_users), "sample": crossing},
            "re-split by subject (the frozen splitter keys on the user id); a "
            "row-level split lets one person's near-duplicate text train and "
            "then be scored",
        )]
    return [Finding(
        "user_isolation", INFO,
        "train and dev share no subject",
        {"n_train_users": len(train_users), "n_dev_users": len(dev_users)},
    )]


def _check_partition_membership(train_ids, dev_ids, manifest) -> list[Finding]:
    mapping = (manifest or {}).get("subject_row_by_id") or {}
    missing = [sid for sid in train_ids + dev_ids if sid not in mapping]
    if missing:
        return [Finding(
            "partition_membership", BLOCKED,
            f"{len(missing)} partition subject(s) have no row mapping",
            {"sample": missing[:5]},
            "the manifest and the record set disagree; rebuild both together",
        )]
    rows_train = [mapping[sid] for sid in train_ids]
    rows_dev = [mapping[sid] for sid in dev_ids]
    overlap = sorted(set(rows_train) & set(rows_dev))
    if overlap or len(set(rows_train)) != len(rows_train) or len(set(rows_dev)) != len(rows_dev):
        return [Finding(
            "partition_membership", BLOCKED,
            "partition rows overlap or repeat",
            {"n_overlap": len(overlap), "sample": overlap[:5]},
        )]
    return [Finding(
        "partition_membership", INFO, "train and dev rows are disjoint",
        {"n_train": len(rows_train), "n_dev": len(rows_dev)},
    )]


def _check_records_labelled(train_ids, dev_ids, adapter, records, manifest) -> list[Finding]:
    rows = _rows_for(train_ids + dev_ids, manifest)
    by_row = {getattr(r, "subject_index", None): r for r in records}
    resolved = [by_row.get(rows[sid]) for sid in train_ids + dev_ids if sid in rows]
    missing = [i for i, r in enumerate(resolved) if r is None]
    if missing:
        return [Finding(
            "record_resolution", BLOCKED,
            f"{len(missing)} manifest rows are absent from the record set",
            {"sample_missing": missing[:5]},
            "the dataset file and the manifest must describe the same records",
        )]
    has_labels = getattr(adapter, "record_has_label", None)
    if has_labels is None:
        return [Finding(
            "label_observability", UNVERIFIED,
            "the adapter does not report label observability",
        )]
    unlabeled = [sid for sid in train_ids + dev_ids
                 if sid in rows and not has_labels(by_row[rows[sid]])]
    if unlabeled:
        return [Finding(
            "label_observability", WARNING,
            f"{len(unlabeled)} selection row(s) carry no observable label and "
            "cannot contribute to training or scoring",
            {"n_unlabeled": len(unlabeled), "sample": unlabeled[:5]},
            "filter them explicitly at load time, or the metric denominator is "
            "unclear",
        )]
    return [Finding(
        "label_observability", INFO, "every selection row has an observable label",
        {"n_rows": len(resolved)},
    )]


def _check_exact_text(adapter, records, manifest) -> list[Finding]:
    """Identical input text on both sides of a fold."""
    if not getattr(adapter, "has_text", False):
        return [Finding(
            "exact_text_leak", INFO,
            "this task has no text modality, so verbatim-text checks do not apply",
            {"adapter": type(adapter).__name__},
        )]
    if getattr(adapter, "record_text_key", None) is None:
        return [Finding(
            "exact_text_leak", UNVERIFIED,
            "the adapter does not expose an input text key for this task",
            {"adapter": type(adapter).__name__},
        )]
    split = (manifest or {})["split"]
    rows = _rows_for(list(split["train"]) + list(split["dev"]), manifest)
    by_row = {getattr(r, "subject_index", None): r for r in records}
    train_texts, dev_texts = {}, {}
    for sid in split["train"]:
        record = by_row.get(rows.get(sid))
        if record is None:
            continue
        key = text_key(adapter, record)
        if key:
            train_texts.setdefault(key, []).append(sid)
    for sid in split["dev"]:
        record = by_row.get(rows.get(sid))
        if record is None:
            continue
        key = text_key(adapter, record)
        if key:
            dev_texts.setdefault(key, []).append(sid)
    crossing = sorted(set(train_texts) & set(dev_texts))
    duplicates = {k: len(v) for k, v in train_texts.items() if len(v) > 1}
    evidence: dict = {
        "n_crossing": len(crossing),
        "n_repeated_train": len(duplicates),
        "max_repeat": max(duplicates.values()) if duplicates else 0,
    }
    if crossing:
        evidence["sample"] = [
            [train_texts[k][0], dev_texts[k][0]] for k in crossing[:3]
        ]
        if getattr(adapter, "fold_dedups_dev_text", False):
            # The frozen fold builder removes these dev rows before scoring,
            # so pool-level overlap is not leakage into a fold. Keep it visible
            # with evidence; dropping rows here would change dataset_sha256 and
            # break comparability with the frozen CV baseline.
            return [Finding(
                "exact_text_leak", WARNING,
                f"{len(crossing)} text(s) appear verbatim in both train and dev "
                "(removed from dev by the frozen fold builder)",
                {**evidence,
                 "fold_dedup_policy": "one-way exact input_text dev-against-train"},
                "do not drop rows; compare only within the same frozen inputs",
            )]
        return [Finding(
            "exact_text_leak", BLOCKED,
            f"{len(crossing)} text(s) appear verbatim in both train and dev",
            evidence,
            "de-duplicate on the input text before splitting; the same complaint "
            "is otherwise present in training and scoring",
        )]
    if duplicates:
        return [Finding(
            "exact_text_leak", WARNING,
            f"{len(duplicates)} text(s) repeat inside train",
            evidence,
        )]
    return [Finding(
        "exact_text_leak", INFO, "no verbatim text appears on both sides of the fold",
        {"n_train_texts": len(train_texts), "n_dev_texts": len(dev_texts)},
    )]


def _check_label_conflict(adapter, records, manifest) -> list[Finding]:
    """The same text carrying two different labels."""
    if not (getattr(adapter, "has_text", False) and getattr(adapter, "single_label", False)):
        return [Finding(
            "label_conflict", INFO,
            "not a single-label text task, so one-text-two-labels does not apply",
            {"has_text": bool(getattr(adapter, "has_text", False)),
             "single_label": bool(getattr(adapter, "single_label", False))},
        )]
    if getattr(adapter, "record_text_key", None) is None or \
            getattr(adapter, "record_label_key", None) is None:
        return [Finding("label_conflict", UNVERIFIED,
                        "adapter lacks text/label keys for this task")]
    by_text: dict[str, set] = {}
    by_row = {getattr(r, "subject_index", None): r for r in records}
    for record in by_row.values():
        key = text_key(adapter, record)
        label = label_key(adapter, record)
        if key is None or label is None:
            continue
        by_text.setdefault(key, set()).add(str(label))
    conflicts = {text: sorted(labels) for text, labels in by_text.items() if len(labels) > 1}
    if conflicts:
        sample = [[text[:60], labels] for text, labels in list(conflicts.items())[:3]]
        if getattr(adapter, "text_label_conflict_is_expected", False):
            # A task whose labels are NOT a function of the input text (short
            # TCM chief complaints, 148 syndromes) legitimately has one text
            # published under several labels. Do not call that a labelling
            # error: keep it visible with evidence, but let the frozen fold
            # protocol (dev-against-train exact dedup) handle leakage. Dropping
            # these rows would change dataset_sha256 and break comparability
            # with the frozen CV baseline.
            return [Finding(
                "label_conflict", WARNING,
                f"{len(conflicts)} text(s) carry more than one label "
                "(expected for this task, not a labelling error)",
                {"n_conflicts": len(conflicts), "sample": sample,
                 "task_property": True},
                "do not drop rows; the frozen fold builder dedups dev against "
                "train. Compare only within the same frozen selection inputs.",
            )]
        return [Finding(
            "label_conflict", BLOCKED,
            f"{len(conflicts)} text(s) carry more than one label",
            {"n_conflicts": len(conflicts), "sample": sample},
            "resolve or drop these rows; a model cannot rank a text that the "
            "labels call two different things, and the loss floor is set by them",
        )]
    return [Finding("label_conflict", INFO, "no conflicting labels on identical text",
                    {"n_texts": len(by_text)})]


def _check_label_space(adapter, records, manifest) -> list[Finding]:
    """The class mapping must match the frozen taxonomy, in order."""
    if not getattr(adapter, "single_label", False):
        return [Finding(
            "label_space", INFO,
            "multi-label task: the single-class taxonomy order does not apply",
            {"adapter": type(adapter).__name__},
        )]
    space = label_space(adapter, manifest)
    if space is None:
        return [Finding("label_space", UNVERIFIED,
                        "the adapter declares no frozen label space for this task")]
    declared = list((manifest or {}).get("label_space") or [])
    if declared and list(declared) != list(space):
        return [Finding(
            "label_space", BLOCKED,
            "the manifest's label space differs from the adapter's frozen taxonomy",
            {"manifest": list(declared)[:8], "adapter": list(space)[:8]},
            "the class column order is part of the contract; rebuild the manifest",
        )]
    observed = set()
    getter = getattr(adapter, "record_label_key", None)
    if getter is not None:
        for record in records:
            label = getter(record)
            if label is not None:
                observed.add(str(label))
        unknown = sorted(observed - {str(x) for x in space})
        if unknown:
            return [Finding(
                "label_space", BLOCKED,
                f"{len(unknown)} label(s) fall outside the frozen taxonomy",
                {"sample": unknown[:8]},
                "map the data through the frozen taxonomy before training",
            )]
    return [Finding(
        "label_space", INFO,
        "label space matches the frozen taxonomy",
        {"n_classes": len(space), "declared_in_manifest": bool(declared),
         "n_observed": len(observed)},
    )]


def _check_class_support(adapter, records, manifest) -> list[Finding]:
    if not getattr(adapter, "single_label", False):
        return [Finding(
            "class_support", INFO,
            "multi-label task: per-class support is reported by the evaluator's "
            "per-dimension breakdown, not by this single-label check",
        )]
    split = (manifest or {})["split"]
    rows = _rows_for(list(split["train"]), manifest)
    by_row = {getattr(r, "subject_index", None): r for r in records}
    getter = getattr(adapter, "record_label_key", None)
    if getter is None:
        return [Finding("class_support", UNVERIFIED, "adapter exposes no label key")]
    counts: dict[str, int] = {}
    for sid in split["train"]:
        record = by_row.get(rows.get(sid))
        if record is None:
            continue
        label = getter(record)
        if label is not None:
            counts[str(label)] = counts.get(str(label), 0) + 1
    tail = {k: v for k, v in counts.items() if v < TAIL_CLASS_MIN_SUPPORT}
    if not counts:
        return [Finding("class_support", UNVERIFIED, "no labels resolved in train")]
    if tail:
        return [Finding(
            "class_support", WARNING,
            f"{len(tail)} of {len(counts)} classes have fewer than "
            f"{TAIL_CLASS_MIN_SUPPORT} training examples",
            {"n_classes": len(counts), "n_tail": len(tail),
             "tail_support": dict(sorted(tail.items(), key=lambda kv: kv[1])[:10])},
            "report per-class support with the score; macro-averaged metrics on "
            "these classes move by whole cases",
        )]
    return [Finding("class_support", INFO,
                    f"{len(counts)} classes, all with >= {TAIL_CLASS_MIN_SUPPORT} train rows",
                    {"n_classes": len(counts)}, )]


def _check_selection_protocol(config: dict, adapter) -> list[Finding]:
    """Is the outer dev fold being used to steer fitting?

    The engine says the candidate only receives training rows. A config that
    re-enables early stopping, best-epoch restore or an evaluation provider puts
    the outer fold back into the loop, and then the reported score is a
    selected maximum rather than a held-out measurement.
    """
    findings: list[Finding] = []
    trainer = config.get("trainer") or {}
    offenders = sorted(
        key for key in ("early_stop_patience", "restore_best", "eval_provider")
        if trainer.get(key)
    )
    if config.get("eval_provider"):
        offenders.append("config.eval_provider")
    if offenders:
        findings.append(Finding(
            "outer_dev_steering", BLOCKED,
            "the outer selection fold can steer fitting",
            {"enabled_keys": offenders},
            "remove early stopping / best-epoch restore / the eval provider from "
            "the training config; the outer fold scores, it does not steer",
        ))
    else:
        findings.append(Finding(
            "outer_dev_steering", INFO,
            "no early stopping, best-epoch restore or eval provider is configured",
            {"checked_keys": ["early_stop_patience", "restore_best", "eval_provider"]},
        ))
    metrics = metric_names(adapter)
    primary = str(config.get("primary_metric") or "")
    if metrics is None:
        findings.append(Finding("metric_definition", UNVERIFIED,
                                "the adapter declares no metric list"))
    elif primary and primary not in metrics:
        findings.append(Finding(
            "metric_definition", BLOCKED,
            f"primary metric {primary!r} is not one the trusted evaluator computes",
            {"primary_metric": primary, "trusted_metrics": list(metrics)[:8]},
            "the selection objective must be a metric the frozen evaluator reports",
        ))
    elif not primary:
        findings.append(Finding(
            "metric_definition", WARNING,
            "config declares no primary metric; the evaluator's default will decide",
            {"trusted_metrics": list(metrics)[:8]},
        ))
    else:
        findings.append(Finding("metric_definition", INFO,
                                f"primary metric {primary!r} is trusted",
                                {"trusted_metrics": list(metrics)[:8]}))
    return findings


def _check_budget_and_binding(config: dict) -> list[Finding]:
    findings: list[Finding] = []
    budget = config.get("time_budget_seconds")
    if not isinstance(budget, (int, float)) or budget <= 0:
        findings.append(Finding(
            "time_budget", BLOCKED,
            "no positive fixed training budget in the config",
            {"time_budget_seconds": budget},
            "every candidate must get the same wall clock, so a config without a "
            "budget cannot be compared with one that has it",
        ))
    else:
        findings.append(Finding("time_budget", INFO,
                                f"fixed budget {budget}s per fold",
                                {"time_budget_seconds": budget}))
    hashes = config.get("frozen_hashes") or {}
    if not hashes:
        findings.append(Finding(
            "frozen_binding", WARNING,
            "the config records no frozen data/manifest/metric hashes",
            {},
            "record the hashes so a later reader can tell which contract a score "
            "belongs to",
        ))
    else:
        findings.append(Finding("frozen_binding", INFO,
                                "frozen hashes recorded",
                                {k: str(v)[:16] for k, v in hashes.items()}))
    return findings


# -- postflight -----------------------------------------------------------------


def postflight(round_view: dict, config: dict, adapter,
               run_id: str | None = None) -> DoctorReport:
    """Interpret one finished round: what the numbers show, and what is unknown.

    `round_view` is what the coordinator knows about the round: the decision, the
    metric values and fold spread, the worker's diagnostics if any, and the
    failure reason if it failed. Nothing here recomputes a score — the trusted
    evaluator already did, and the Doctor must not become a second evaluator.
    """
    findings: list[Finding] = []
    task = str(config.get("task") or getattr(adapter, "name", "unknown"))
    decision = round_view.get("decision")
    status = round_view.get("status")

    findings.append(_check_failure_attribution(decision, status, round_view))
    findings.extend(_check_metric_present(round_view))
    findings.extend(_check_convergence(round_view, config))
    findings.extend(_check_resolution(round_view))
    findings.extend(_check_per_class(round_view))
    findings.extend(_check_resources(round_view, config))
    findings.extend(_check_export(decision, round_view))
    return DoctorReport("postflight", task, findings, run_id=run_id)


def _check_failure_attribution(decision, status, round_view) -> Finding:
    if status in ("crash", "rejected", "timeout", "overran", "unknown_architecture"):
        return Finding(
            "failure_attribution", WARNING,
            f"the round did not produce a model score ({status}); this is an "
            "engineering outcome, not evidence about model quality",
            {"decision": decision, "status": status,
             "reason": str(round_view.get("reason") or "")[:300]},
            "read the reason before interpreting any metric; a CRASH or "
            "UNSUPPORTED_ARCHITECTURE must never be recorded as a model loss",
        )
    if decision in ("KEEP", "REVERT"):
        return Finding("failure_attribution", INFO,
                       f"round completed with decision {decision}",
                       {"decision": decision, "status": status})
    return Finding("failure_attribution", UNVERIFIED,
                   "the round has no terminal decision yet",
                   {"decision": decision, "status": status})


def _check_metric_present(round_view: dict) -> list[Finding]:
    value = round_view.get("primary_value")
    if value is None:
        return [Finding(
            "metric_present", UNVERIFIED,
            "the round carries no primary metric value",
            {"primary_metric": round_view.get("primary_metric")},
            "either the round failed (see failure_attribution) or the trusted "
            "report is missing a required field",
        )]
    return [Finding("metric_present", INFO,
                    f"primary metric {round_view.get('primary_metric')} = {value}",
                    {"primary_value": value})]


def _check_convergence(round_view: dict, config: dict) -> list[Finding]:
    """Did the candidate use its budget, and did the loss come down?

    Both are diagnostics the worker may or may not report; a missing value is
    reported as NOT_AVAILABLE rather than assumed good.
    """
    diagnostics = round_view.get("diagnostics") or {}
    fold_seconds = diagnostics.get("fold_seconds") or round_view.get("fold_seconds")
    budget = config.get("time_budget_seconds")
    findings: list[Finding] = []
    if fold_seconds and budget:
        used = max(float(x) for x in fold_seconds)
        if used < 0.5 * float(budget):
            findings.append(Finding(
                "budget_utilisation", WARNING,
                "the slowest fold used less than half the fixed budget",
                {"max_fold_seconds": used, "budget_seconds": float(budget)},
                "a candidate that stops early is not comparable with one that "
                "uses the whole clock",
            ))
        else:
            findings.append(Finding("budget_utilisation", INFO,
                                    "folds used the fixed budget",
                                    {"max_fold_seconds": used,
                                     "budget_seconds": float(budget)}))
    else:
        findings.append(Finding("budget_utilisation", UNVERIFIED,
                                "fold timings are NOT_AVAILABLE for this round"))
    losses = diagnostics.get("losses") or diagnostics.get("loss_history")
    if isinstance(losses, (list, tuple)) and len(losses) >= 2:
        first, last = float(losses[0]), float(losses[-1])
        if not (first == first and last == last):  # NaN check
            findings.append(Finding(
                "loss_trend", BLOCKED, "the loss history contains NaN",
                {"first": first, "last": last},
                "check for a degenerate input column or an unstable learning rate",
            ))
        elif last >= first:
            findings.append(Finding(
                "loss_trend", WARNING, "training loss did not decrease",
                {"first": first, "last": last, "n_points": len(losses)},
                "either the budget is too small for any learning to happen, or "
                "the change removed the signal",
            ))
        else:
            findings.append(Finding("loss_trend", INFO, "training loss decreased",
                                    {"first": first, "last": last,
                                     "n_points": len(losses)}))
    else:
        findings.append(Finding(
            "loss_trend", UNVERIFIED,
            "no loss history is available (NOT_AVAILABLE)",
            {},
            "the training worker must emit a loss history before this check can pass",
        ))
    return findings


def _check_resolution(round_view: dict) -> list[Finding]:
    """Is the round's measurement capable of resolving the claimed effect?"""
    value = round_view.get("primary_value")
    sem = round_view.get("primary_sem")
    best = round_view.get("previous_best")
    if value is None or sem is None:
        return [Finding("measurement_resolution", UNVERIFIED,
                        "primary value or standard error is NOT_AVAILABLE")]
    if best is None:
        return [Finding("measurement_resolution", INFO,
                        "first scored round; nothing to compare against yet",
                        {"value": value, "sem": sem})]
    gain = float(value) - float(best)
    ratio = abs(gain) / float(sem) if sem else None
    if ratio is not None and ratio < NOISE_RATIO_WARNING:
        return [Finding(
            "measurement_resolution", WARNING,
            "the difference from the champion is smaller than one standard error",
            {"gain": round(gain, 6), "sem": round(float(sem), 6),
             "gain_over_sem": None if ratio is None else round(ratio, 3)},
            "do not read this as a model difference; increase resolution "
            "(more folds/repeats) or change the information available",
        )]
    return [Finding("measurement_resolution", INFO,
                    "the difference exceeds one standard error",
                    {"gain": round(gain, 6), "sem": round(float(sem), 6)})]


def _check_per_class(round_view: dict) -> list[Finding]:
    per_class = round_view.get("per_class") or (round_view.get("report") or {}).get("per_dim")
    if not per_class:
        return [Finding("per_class_errors", UNVERIFIED,
                        "per-class error breakdown is NOT_AVAILABLE for this round",
                        {},
                        "the evaluator must return per-class support and error")]
    weak = []
    for name, stats in per_class.items():
        if not isinstance(stats, dict):
            continue
        support = stats.get("n_observed") or stats.get("support") or 0
        score = stats.get("f1", stats.get("ap"))
        if support and score is not None and float(score) < 0.1:
            weak.append((str(name), support, round(float(score), 3)))
    if weak:
        return [Finding(
            "per_class_errors", WARNING,
            f"{len(weak)} class(es) score below 0.1",
            {"weak_classes": weak[:10]},
            "tail classes dominate a macro average; check support before "
            "attributing this to the model",
        )]
    return [Finding("per_class_errors", INFO, "no class scores below 0.1",
                    {"n_classes": len(per_class)})]


def _check_resources(round_view: dict, config: dict) -> list[Finding]:
    diagnostics = round_view.get("diagnostics") or {}
    elapsed = round_view.get("elapsed_s")
    resources = {
        "elapsed_s": elapsed,
        "gpu_memory_mb": diagnostics.get("gpu_memory_mb"),
        "peak_tokens": diagnostics.get("peak_tokens"),
    }
    unavailable = [k for k, v in resources.items() if v is None]
    return [Finding(
        "resource_telemetry", INFO if not unavailable else UNVERIFIED,
        "resource telemetry recorded" if not unavailable
        else f"NOT_AVAILABLE: {', '.join(unavailable)}",
        {k: v for k, v in resources.items() if v is not None},
        "resources are reported only when the host actually measured them",
    )]


def _check_export(decision, round_view: dict) -> list[Finding]:
    status = round_view.get("status")
    if status == "unknown_architecture" or "ArchitectureExportRefused" in str(
            round_view.get("reason") or "") or decision == "UNSUPPORTED_ARCHITECTURE":
        return [Finding(
            "architecture_export", WARNING,
            "the candidate's architecture is not expressible in the frozen "
            "export whitelist, so it could not be scored",
            {"reason": str(round_view.get("reason") or "")[:300]},
            "this is a boundary, not a result: extend the whitelist only through "
            "a reviewed change, and never by deserialising arbitrary candidate code",
        )]
    return [Finding("architecture_export", INFO,
                    "the architecture exported under the frozen whitelist")]


# -- orchestration --------------------------------------------------------------


def run_preflight_or_stop(config: dict, manifest: dict, records: list, adapter,
                          run_id: str | None = None,
                          session=None, writer=None) -> DoctorReport:
    """Preflight, persist the report, and refuse to continue when BLOCKED.

    A blocked protocol must not be trained on. The caller gets the report and an
    exception it cannot accidentally ignore: silent continuation is exactly how a
    leak ends up in a headline number.
    """
    report = preflight(config, manifest, records, adapter, run_id=run_id)
    if session is not None:
        session.write_doctor(report.to_dict())
    if writer is not None:
        writer.emit(
            "DOCTOR",
            "blocked" if report.blocked else "finished",
            metrics={"blocked": int(report.blocked)},
            message=report.summary_line(),
        )
    if report.blocked:
        blocked = [f.check for f in report.findings if f.severity == BLOCKED]
        raise BlockedProtocol(
            f"preflight found {len(blocked)} blocking problem(s): {blocked}; "
            "fix the protocol before training — a score from an untrustworthy "
            "protocol is not a research result"
        )
    return report


class BlockedProtocol(RuntimeError):
    """Raised when the Doctor refuses the protocol; the run must not train."""