"""Research Doctor: fault injection proves each check bites.

Every test here injects ONE defect into an otherwise clean protocol and asserts
the Doctor names it with the right severity. A check that only ever passes is
indistinguishable from a check that does nothing, which is why the faults are
injected rather than described.

The last test is the one that matters operationally: a BLOCKED protocol must
stop the run before any score can be produced, and the reason must survive in
`doctor.json` rather than living only in a raised exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from research import doctor as doctor_mod
from research.doctor import (
    BLOCKED,
    INFO,
    UNVERIFIED,
    WARNING,
    BlockedProtocol,
    postflight,
    preflight,
    run_preflight_or_stop,
)
from research.session import ResearchSession
from research.tasks import TaskAdapter

pytestmark = pytest.mark.synthetic


@dataclass
class Rec:
    subject_index: int
    user_id: str
    input_text: str
    label: int | None = 0


class ToyAdapter(TaskAdapter):
    """A one-label text task: the checks that apply to it are the text ones."""

    name = "toy"
    has_text = True
    single_label = True

    def record_user_key(self, record):
        return record.user_id

    def record_text_key(self, record):
        return record.input_text

    def record_label_key(self, record):
        return record.label

    def record_has_label(self, record):
        return record.label is not None

    #: the frozen taxonomy this task's evaluator indexes by. The adapter is the
    #: AUTHORITY here and the manifest is the CLAIM being checked, so returning
    #: the manifest's own list would make the comparison vacuous.
    FROZEN_CLASSES = ["0", "1", "2"]

    def label_space(self, manifest):
        return list(self.FROZEN_CLASSES)

    def metric_names(self):
        return ("toy/accuracy",)


class TabularAdapter(TaskAdapter):
    """No text, eight independent labels: the text checks do not apply."""

    name = "tabular"

    def record_user_key(self, record):
        return record.user_id

    def record_has_label(self, record):
        return record.label is not None

    def metric_names(self):
        return ("tabular/macro_ap",)


def build(n_train: int = 12, n_dev: int = 6, *, adapter=None):
    records = []
    train_ids, dev_ids = [], []
    for i in range(n_train + n_dev):
        is_dev = i >= n_train
        sid = f"r{i:03d}"
        records.append(Rec(i, user_id=f"u{i}", input_text=f"case text {i}", label=i % 3))
        (dev_ids if is_dev else train_ids).append(sid)
    # the real manifests key `subject_row_by_id` by the subject id STRING that
    # the partitions hold, and map it to the dataset row index
    manifest = {
        "split": {"train": train_ids, "dev": dev_ids, "holdout": []},
        "subject_row_by_id": {f"r{r.subject_index:03d}": r.subject_index for r in records},
        "label_space": ["0", "1", "2"],
    }
    config = {
        "task": (adapter or ToyAdapter()).name,
        "primary_metric": (adapter.metric_names() if adapter else ("toy/accuracy",))[0],
        "time_budget_seconds": 20.0,
        "frozen_hashes": {"dataset_hash": "deadbeef", "manifest_hash": "cafebabe"},
        "trainer": {},
    }
    return config, manifest, records, records


def _findings(report):
    return {f.check: f for f in report.findings}


def test_clean_protocol_has_no_blocked_finding():
    config, manifest, records, _ = build()
    report = preflight(config, manifest, records, ToyAdapter())
    assert report.blocked is False
    by_check = _findings(report)
    assert by_check["user_isolation"].severity == INFO
    assert by_check["exact_text_leak"].severity == INFO
    assert by_check["label_conflict"].severity == INFO
    assert by_check["outer_dev_steering"].severity == INFO
    assert by_check["metric_definition"].severity == INFO


# -- injected protocol faults ---------------------------------------------------


def test_same_subject_on_both_sides_is_blocked():
    config, manifest, records, _ = build()
    # give a dev record the user id of a train record: the #1 leak
    dev_row = manifest["subject_row_by_id"][manifest["split"]["dev"][0]]
    records[dev_row].user_id = records[0].user_id
    report = preflight(config, manifest, records, ToyAdapter())
    finding = _findings(report)["user_isolation"]
    assert finding.severity == BLOCKED
    assert finding.evidence["n_crossing"] == 1
    assert report.blocked is True


def test_verbatim_text_on_both_sides_is_blocked():
    config, manifest, records, _ = build()
    dev_row = manifest["subject_row_by_id"][manifest["split"]["dev"][0]]
    records[dev_row].input_text = records[0].input_text
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["exact_text_leak"]
    assert finding.severity == BLOCKED
    assert finding.evidence["n_crossing"] == 1


def test_one_text_two_labels_is_blocked():
    config, manifest, records, _ = build()
    records[1].input_text = records[0].input_text
    records[1].label = (records[0].label + 1) % 3
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["label_conflict"]
    assert finding.severity == BLOCKED
    assert finding.evidence["n_conflicts"] == 1


def test_expected_text_label_conflicts_warn_instead_of_blocking():
    """TCM-SD short chief complaints legitimately carry different syndromes
    for the same text. The Doctor must keep that visible as evidence, not call
    it a labelling error and refuse to run (which would freeze all real-data
    AutoResearch campaigns). Dropping the rows is not an option: it changes
    dataset_sha256 and breaks comparability with the frozen CV baseline."""
    config, manifest, records, _ = build()
    records[1].input_text = records[0].input_text
    records[1].label = (records[0].label + 1) % 3

    class SyndromeLikeAdapter(ToyAdapter):
        text_label_conflict_is_expected = True

    report = preflight(config, manifest, records, SyndromeLikeAdapter())
    finding = _findings(report)["label_conflict"]
    assert finding.severity == WARNING
    assert finding.evidence["n_conflicts"] == 1
    assert finding.evidence["task_property"] is True
    assert report.blocked is False


def test_outer_dev_steering_is_blocked():
    config, manifest, records, _ = build()
    config["trainer"] = {"early_stop_patience": 3}
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["outer_dev_steering"]
    assert finding.severity == BLOCKED
    assert "early_stop_patience" in finding.evidence["enabled_keys"]

    # an eval provider attached to fitting is the same defect by another route
    config["trainer"] = {}
    config["eval_provider"] = {"n_batches": 1}
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["outer_dev_steering"]
    assert finding.severity == BLOCKED
    assert "config.eval_provider" in finding.evidence["enabled_keys"]


def test_class_mapping_order_mismatch_is_blocked():
    config, manifest, records, _ = build()
    manifest["label_space"] = ["2", "1", "0"]  # same classes, wrong order
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["label_space"]
    assert finding.severity == BLOCKED
    assert finding.evidence["manifest"][:3] == ["2", "1", "0"]


def test_label_outside_the_frozen_taxonomy_is_blocked():
    config, manifest, records, _ = build()
    records[0].label = 99
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["label_space"]
    assert finding.severity == BLOCKED
    assert "99" in finding.evidence["sample"]


def test_holdout_overlapping_selection_is_blocked():
    config, manifest, records, _ = build()
    # a holdout row that is ALSO a train row is exactly how the locked test
    # leaks into selection; a holdout that merely exists is not a defect
    manifest["split"]["holdout"] = [manifest["split"]["train"][0]]
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))[
        "selection_holdout_closed"]
    assert finding.severity == BLOCKED
    assert finding.evidence["n_overlap"] == 1


def test_disjoint_holdout_is_not_a_blocker():
    config, manifest, records, _ = build()
    manifest["split"]["holdout"] = ["r900"]
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))[
        "selection_holdout_closed"]
    assert finding.severity == INFO
    assert finding.evidence["n_holdout"] == 1


def test_untrusted_primary_metric_is_blocked():
    config, manifest, records, _ = build()
    config["primary_metric"] = "candidate/self_reported_score"
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["metric_definition"]
    assert finding.severity == BLOCKED
    assert finding.evidence["primary_metric"] == "candidate/self_reported_score"


def test_missing_fixed_budget_is_blocked():
    config, manifest, records, _ = build()
    config.pop("time_budget_seconds")
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["time_budget"]
    assert finding.severity == BLOCKED


def test_tail_classes_are_a_warning_not_a_pass():
    config, manifest, records, _ = build(n_train=12)
    for i in range(10):
        records[i].label = 0
    records[10].label, records[11].label = 1, 1
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["class_support"]
    assert finding.severity == WARNING
    assert finding.evidence["n_tail"] >= 1


def test_unlabeled_selection_rows_are_flagged():
    config, manifest, records, _ = build()
    records[0].label = None
    finding = _findings(preflight(config, manifest, records, ToyAdapter()))["label_observability"]
    assert finding.severity == WARNING
    assert finding.evidence["n_unlabeled"] == 1


def test_absent_adapter_hooks_are_unverified_not_passed():
    config, manifest, records, _ = build(adapter=TabularAdapter())
    config["primary_metric"] = "tabular/macro_ap"
    record = records[0]
    report = preflight(config, manifest, records, TabularAdapter())
    by_check = _findings(report)
    # a tabular task genuinely does not have these checks
    assert by_check["exact_text_leak"].severity == INFO
    assert by_check["class_support"].severity == INFO

    class Anonymous(TabularAdapter):
        def record_user_key(self, record):
            return None

    report = preflight(config, manifest, records, Anonymous())
    assert _findings(report)["user_isolation"].severity == UNVERIFIED
    assert record is not None


def test_blocked_protocol_stops_the_run_and_leaves_the_reason_on_disk(tmp_path: Path):
    session = ResearchSession.create(
        tmp_path / "run", task="toy", editable="src/research/candidate_toy.py",
        worktree=str(tmp_path / "wt"), branch="autoresearch/toy",
        folds=2, repeats=1, max_rounds=3, budget_seconds=1.0,
        primary_metric="toy/accuracy",
    )
    config, manifest, records, _ = build()
    dev_row = manifest["subject_row_by_id"][manifest["split"]["dev"][0]]
    records[dev_row].user_id = records[0].user_id

    with session.writer() as writer:
        with pytest.raises(BlockedProtocol, match="user_isolation"):
            run_preflight_or_stop(config, manifest, records, ToyAdapter(),
                                  run_id=session.run_id, session=session, writer=writer)
    saved = session.read_doctor()
    assert saved["blocked"] is True
    assert saved["mode"] == "preflight"
    assert any(f["check"] == "user_isolation" and f["severity"] == BLOCKED
               for f in saved["findings"])
    # and no round was recorded: a blocked protocol never reaches the manifest
    assert session.entries() == []
    assert session.status()["progress"]["stage"] == "DOCTOR"


# -- postflight -----------------------------------------------------------------


def test_failed_round_is_attributed_to_engineering_not_model_quality():
    report = postflight(
        {"decision": "REVERT", "status": "crash",
         "reason": "runner failed: ArchitectureExportRefused: unsupported"},
        {"task": "toy", "time_budget_seconds": 20.0}, ToyAdapter(),
    )
    finding = _findings(report)["failure_attribution"]
    assert finding.severity == WARNING
    assert "not evidence about model quality" in finding.message
    assert _findings(report)["metric_present"].severity == UNVERIFIED


def test_a_gain_inside_one_standard_error_is_flagged():
    report = postflight(
        {"decision": "REVERT", "status": "completed", "primary_value": 0.31,
         "primary_sem": 0.02, "previous_best": 0.30},
        {"task": "toy", "time_budget_seconds": 20.0}, ToyAdapter(),
    )
    finding = _findings(report)["measurement_resolution"]
    assert finding.severity == WARNING
    assert finding.evidence["gain_over_sem"] == pytest.approx(0.5)


def test_nan_loss_is_blocked_and_missing_diagnostics_are_unverified():
    report = postflight(
        {"decision": "REVERT", "status": "completed", "primary_value": 0.3,
         "primary_sem": 0.01, "diagnostics": {"losses": [1.0, float("nan")]}},
        {"task": "toy", "time_budget_seconds": 20.0}, ToyAdapter(),
    )
    assert _findings(report)["loss_trend"].severity == BLOCKED

    report = postflight(
        {"decision": "REVERT", "status": "completed", "primary_value": 0.3},
        {"task": "toy", "time_budget_seconds": 20.0}, ToyAdapter(),
    )
    assert _findings(report)["loss_trend"].severity == UNVERIFIED
    assert doctor_mod.UNVERIFIED in _findings(report)["budget_utilisation"].severity


def test_weak_tail_class_and_underused_budget_are_warnings():
    report = postflight(
        {"decision": "KEEP", "status": "completed", "primary_value": 0.4,
         "primary_sem": 0.01,
         "per_class": {"rare": {"n_observed": 12, "f1": 0.0}},
         "diagnostics": {"fold_seconds": [2.0, 3.0]}},
        {"task": "toy", "time_budget_seconds": 20.0}, ToyAdapter(),
    )
    assert _findings(report)["per_class_errors"].severity == WARNING
    assert _findings(report)["budget_utilisation"].severity == WARNING


def test_doctor_report_is_machine_readable_and_serialisable(tmp_path: Path):
    config, manifest, records, _ = build()
    report = preflight(config, manifest, records, ToyAdapter())
    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == "autoresearch-doctor-v1"
    assert payload["counts"][INFO] > 0 and payload["blocked"] is False