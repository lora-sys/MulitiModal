"""Real 94-field Session data check: import audit, quality and pre-experiment (#102).

The raw export is private and lives under ``data/local/``. This entry point reads
it, rebuilds every endpoint and label from the raw records, compares the result
against the reference training set and quality report, then runs the declared
pipeline regression. Nothing here upgrades the evidence level: with five sessions
the evaluation status stays ``INSUFFICIENT_EVIDENCE`` and no performance or
clinical claim is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW = "data/local/physio-legacy/raw_records_5sessions.csv"
DEFAULT_REFERENCE = "data/local/physio-legacy/reference_train94_5sessions.csv"
DEFAULT_QC = "data/local/physio-legacy/reference_qc_5sessions.csv"
DEFAULT_OUT = "runs/physio-real-data"

EVIDENCE_LEVEL = "LOCAL_EXPERIMENTAL"
ISOLATION_LEVEL = "AGENT_ISOLATION_UNVERIFIED"

NOT_GRANTED_GATES = (
    "GO_REAL_DATA_TRAINING(physio)=NOT_GRANTED",
    "GO_RESEARCH_CLAIMS(physio)=NOT_GRANTED",
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--qc", default=DEFAULT_QC)
    parser.add_argument("--target", default="post_state",
                        choices=["post_state", "delta_vendor_fatigue"])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from physio.real_data_audit import audit_real_export, load_reference_rows
    from physio.real_export import collect_issues, contract_coverage, import_real_export
    from physio.real_training import run_real_pre_experiment

    raw_path = REPO_ROOT / args.raw
    if not raw_path.is_file():
        print(f"raw export not found: {raw_path}")
        print("place the de-identified export under data/local/ (never committed)")
        return 2

    sessions = import_real_export(raw_path)
    reference_rows = load_reference_rows(REPO_ROOT / args.reference)
    qc_rows = load_reference_rows(REPO_ROOT / args.qc)
    audit = audit_real_export(
        sessions,
        reference_rows,
        qc_rows,
        notes=[
            "labels are vendor proxies, not independently verified fatigue measurements",
            "identity is a shared account; no subject-level generalization is claimed",
        ],
    )
    training = run_real_pre_experiment(
        sessions,
        target=args.target,
        out_dir=REPO_ROOT / args.out,
        notes=[
            "five sessions cannot support a performance or clinical claim",
            "numbers are a local pipeline regression only",
        ],
    )

    out_dir = REPO_ROOT / args.out
    _write_json(out_dir / "audit.json", audit.to_dict())
    _write_json(out_dir / "pre_experiment.json", training.to_dict())
    _write_json(
        out_dir / "contract_coverage.json",
        {
            "schema": "physio-real-contract-coverage-v1",
            "sessions": [
                {"session_id": session.session_id, **contract_coverage(session)}
                for session in sessions
            ],
        },
    )
    _write_json(
        out_dir / "issues.json",
        {
            "schema": "physio-real-issues-v1",
            "issues": [issue.to_dict() for issue in collect_issues(sessions)],
        },
    )

    report = {
        "schema": "physio-real-data-check-v1",
        "evidence_level": EVIDENCE_LEVEL,
        "isolation": ISOLATION_LEVEL,
        "gates": list(NOT_GRANTED_GATES),
        "sessions": len(sessions),
        "records": audit.records,
        "audit_status": audit.status,
        "audit_verdicts": audit.verdict_counts(),
        "audit_qc_verdicts": audit.qc_verdict_counts(),
        "open_items": list(audit.open_items),
        "evaluation_status": training.evaluation_status,
        "evaluation_reason": training.reason,
        "model_mae": training.model_mae,
        "neural_status": training.neural_status,
        "passed": audit.passed
        and training.evaluation_status in {"EVALUATED", "INSUFFICIENT_EVIDENCE"},
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    print(f"raw export      : {raw_path}")
    print(f"sessions        : {report['sessions']}  records: {report['records']}")
    print(f"audit status    : {audit.status}  {audit.verdict_counts()}")
    print(f"evaluation      : {training.evaluation_status}  ({training.reason})")
    print("pipeline MAE    : "
          + ", ".join(
              f"{name}={value:.4f}" if value is not None else f"{name}=n/a"
              for name, value in training.model_mae.items()
          )
          + "  <- pipeline regression only, NOT a model comparison")
    print(f"neural path     : {training.neural_status}")
    print("model ranking   : not reported; "
          f"{training.n_sessions} sessions with one test row per fold cannot rank models")
    for item in audit.open_items:
        print(f"OPEN ITEM       : {item}")
    print(f"reports         : {out_dir}")
    print(f"evidence        : {EVIDENCE_LEVEL} / {ISOLATION_LEVEL}")
    print("gates           : " + ", ".join(NOT_GRANTED_GATES))
    print("OVERALL:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
