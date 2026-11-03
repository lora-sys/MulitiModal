"""One-shot training subprocess tests (F-02, F-06, F-08).

The structural rule: candidate code runs ONLY in `research.train_worker`, a
one-shot subprocess whose data inputs are trimmed files (no dev, no holdout,
no full-set path) and whose outputs are a state_dict file + a frozen
architecture description. These tests pin:

* the runner never imports the candidate (mechanical source check);
* the candidate-facing FoldData has dev=None and holds only fold rows;
* the worker refuses code whose SHA is not the recorded one;
* the real subprocess chain (train_worker -> evaluator) produces a scoreable
  state_dict-only model.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from data.mendeley import FEATURE_GROUP_SIZES, SubjectRecord, records_to_json
from data.real_split import _manifest_digest, build_real_manifest, fold_manifest
from research.frozen import selection_fold_ids, training_fold_data

pytestmark = pytest.mark.synthetic

DATASET_SHA = "c" * 64
_FIXED_STEP_TRAIN = """def train(
    model, fold: FoldData, budget_seconds: float = TIME_BUDGET_SECONDS,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for item in fold.train:
        batch = item.batch
        output = model(batch)
        loss = masked_multilabel_bce_loss(
            output.auxiliary["category_logits"],
            batch.targets["category_labels"],
            batch.target_mask["category_labels"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        break
    model.eval()
"""
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def fake_records(n=30, seed=13):
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        scores = [None] + [float(rng.randint(10, 90)) for _ in range(8)]
        records.append(
            SubjectRecord(
                subject_index=i,
                features={
                    name: [rng.random() for _ in range(size)]
                    for name, size in FEATURE_GROUP_SIZES.items()
                },
                scores=scores,
                score_mask=[s is not None for s in scores],
                provenance={"source": "test-fixture"},
            )
        )
    return records


def _fit_inputs(tmp_path, records, fold=0, folds=4):
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(base, ensure_ascii=False))
    train_ids, _dev_ids = selection_fold_ids(base, fold, folds, repeat=0)
    fit_manifest = fold_manifest(base, records, train_ids, [], notes="fit").data
    fit_rows = set(fit_manifest["subject_row_by_id"].values())
    fit_records = [r for r in records if r.subject_index in fit_rows]
    fit_manifest_path = tmp_path / "fit-manifest.json"
    fit_manifest_path.write_text(json.dumps(fit_manifest, ensure_ascii=False))
    fit_records_path = tmp_path / "fit-records.json"
    fit_records_path.write_text(records_to_json(fit_records))
    return fit_manifest, fit_manifest_path, fit_records_path


def test_training_fold_data_has_no_dev_and_only_train_rows(tmp_path):
    records = fake_records()
    base = build_real_manifest(records, dataset_sha256=DATASET_SHA).data
    fit_manifest, _, fit_records_path = _fit_inputs(tmp_path, records)
    from data.mendeley import records_from_json

    fit_records = records_from_json(fit_records_path.read_text())
    config = {"train_batch_size": 8, "seed": 0}
    fold = training_fold_data(fit_manifest, fit_records, config)

    # F-06: the candidate-visible fold has NO dev provider at all
    assert fold.dev is None
    assert fold.n_dev_subjects == 0
    train_rows = set(fold.train.records_by_row)
    manifest_rows = set(fit_manifest["subject_row_by_id"].values())
    assert train_rows == manifest_rows
    holdout_rows = {
        base["subject_row_by_id"][sid] for sid in base["split"]["holdout"]
    }
    pool_rows = {
        base["subject_row_by_id"][sid]
        for sid in base["split"]["train"] + base["split"]["dev"]
    }
    assert holdout_rows
    assert not train_rows & holdout_rows
    # the selection pool is train+dev, so fold-train ⊆ pool is the contract
    assert train_rows <= pool_rows


@pytest.mark.parametrize("preflight", [True, False])
def test_candidate_phase_returns_output_contract_and_zeroes_only_preflight_training(
    monkeypatch, preflight,
):
    from types import SimpleNamespace

    import torch

    from research import model_export, train_worker

    observed = {}

    class Candidate:
        def build_model(self, _fold):
            return torch.nn.Linear(1, 1)

        def train(self, _model, _fold, budget_seconds, _seed):
            observed["train_budget"] = budget_seconds

    def phase_runner(_result_fd):
        def run(operation, budget):
            observed["worker_budget"] = budget
            payload = operation(0.0, lambda: 0.0)
            return 0.01, payload
        return run

    contract = {"output_key": "category_probs", "output_shape": [8, 9]}
    monkeypatch.setattr(train_worker, "import_candidate", lambda *_a, **_k: Candidate())
    monkeypatch.setattr(train_worker, "_validate_model_output", lambda *_a: contract)
    monkeypatch.setattr(train_worker, "_trusted_candidate_phase_runner", phase_runner)
    monkeypatch.setattr(model_export, "save_eval_model", lambda *_a, **_k: {})

    config = {
        "task": "constitution",
        "seed": 7,
        "time_budget_seconds": 7.0,
        "output_dir": "/sandbox/output",
        "output_stem": "preflight",
    }
    fold = SimpleNamespace(input_spec=SimpleNamespace(version="synthetic-v1"))
    result = train_worker._run_candidate_phase(
        config, b"synthetic candidate", "c" * 64, fold,
        preflight=preflight,
    )

    assert observed["worker_budget"] == 7.0
    assert observed["train_budget"] == (0.0 if preflight else 5.0)
    assert result["candidate_output"] == contract
    assert result["output_key"] == contract["output_key"]
    assert result["output_shape"] == contract["output_shape"]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([-0.1, 0.7, 0.4], r"within \[0, 1\]"),
        ([0.2, 0.2, 0.2], "sum to one per row"),
    ],
)
def test_syndrome_candidate_output_requires_normalized_probabilities(
    probabilities, message,
):
    from types import SimpleNamespace

    import torch

    from core.batch import ModelOutput
    from research.train_worker import _validate_model_output

    batch = SimpleNamespace(features={"text": torch.zeros((2, 1))})
    fold = SimpleNamespace(
        input_spec=object(),
        train=[SimpleNamespace(batch=batch)],
        dims={"targets": 3},
    )

    class Model:
        input_spec = fold.input_spec

        def eval(self):
            return self

        def __call__(self, _batch):
            row = torch.tensor(probabilities, dtype=torch.float32)
            return ModelOutput(
                predictions={"syndrome_probs": row.repeat(2, 1)},
            )

    with pytest.raises(RuntimeError, match=message):
        _validate_model_output(Model(), fold, "syndrome")


@pytest.mark.parametrize("partition", ["dev", "holdout"])
def test_worker_rejects_eval_rows_in_fit_manifest(tmp_path, partition):
    records = fake_records()
    fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(tmp_path, records)
    subject_id = fit_manifest["split"]["train"].pop()
    fit_manifest["split"][partition].append(subject_id)
    fit_manifest.pop("manifest_sha256")
    fit_manifest["manifest_sha256"] = _manifest_digest(fit_manifest)
    fit_manifest_path.write_text(json.dumps(fit_manifest, ensure_ascii=False))
    from data.mendeley import records_from_json
    from research.tasks import adapter_for
    from research.train_worker import _load_fold
    fit_records = records_from_json(fit_records_path.read_text())
    with pytest.raises(ValueError, match="fit manifest must have empty dev and holdout"):
        _load_fold(
            {"task": "constitution", "fit_manifest_path": str(fit_manifest_path),
             "train_batch_size": 8, "seed": 0},
            adapter_for("constitution"),
            records=fit_records,
        )
def test_worker_refuses_code_that_is_not_the_recorded_sha(tmp_path):
    records = fake_records()
    fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    (worktree / "src" / "research").mkdir(parents=True)
    candidate = worktree / "src" / "research" / "candidate.py"
    marker = tmp_path / "candidate-imported"
    candidate.write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
    )
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": "0" * 64,
        "seed": 0,
        "time_budget_seconds": 0.05,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path),
        "output_stem": "m0",
    }
    from research.train_worker import fit

    with pytest.raises(RuntimeError, match="refusing to run code"):
        fit(config)
    assert not marker.exists()


def test_subprocess_chain_trains_and_scores_state_dict_only(tmp_path, docker_daemon):
    """The real F-02 chain: runner-spawned train_worker subprocess, then the
    evaluator subprocess, over files only — no candidate import in between."""
    records = fake_records()
    _fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate.py")
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": candidate_sha,
        "seed": 0,
        "time_budget_seconds": 20.0,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path / "out"),
        "output_stem": "m0",
    }
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(config))
    from research.spawn import bootstrap_command
    result = subprocess.run(
        bootstrap_command(str(worktree), "research.train_worker",
                          "--config", str(config_path)),
        cwd=worktree, text=True, capture_output=True, timeout=120,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
        },
    )
    payload = _last_json(result)
    assert payload["ok"], payload
    assert payload["fit_timer"] == "trusted_candidate_phase_wall_v1"
    assert payload["fit_seconds"] >= 0

    import torch

    state = torch.load(payload["weights_path"], weights_only=True)
    assert isinstance(state, dict) and state

    (tmp_path / "dataset.xls").write_bytes(b"fixture-workbook")
    # the evaluator consumes selection-only artifacts: a train+dev pool
    # manifest with the holdout emptied, plus the exact bound records bytes.
    # pool_manifest inherits the parent dataset hash, so recompute it over
    # the pool rows like the syndrome selection artifacts do.
    from data.real_split import load_real_manifest, pool_manifest

    base = load_real_manifest(tmp_path / "frozen.json")
    pool_data = pool_manifest(
        base, records, notes="synthetic eval pool"
    ).data
    pool_rows = set(pool_data["subject_row_by_id"].values())
    pool_payload = records_to_json(
        [r for r in records if r.subject_index in pool_rows]
    )
    pool_data.pop("manifest_sha256")
    pool_data["dataset_sha256"] = hashlib.sha256(
        pool_payload.encode("utf-8")
    ).hexdigest()
    pool_data["manifest_sha256"] = _manifest_digest(pool_data)
    pool_manifest_path = tmp_path / "pool-manifest.json"
    pool_manifest_path.write_text(json.dumps(pool_data, ensure_ascii=False))
    pool_records_path = tmp_path / "pool-records.json"
    pool_records_path.write_text(pool_payload)
    eval_config = {
        "data": {
            "manifest_path": str(pool_manifest_path),
            "dataset_path": str(pool_records_path),
        },
        "train_provider": {"batch_size": 8},
        "trainer": {"seed": 0},
    }
    from research.evaluator import integrity_snapshot

    request = {
        "fold": 0, "folds": 4, "repeat": 0,
        "weights_path": payload["weights_path"],
        "architecture_path": payload["architecture_path"],
        "weights_sha256": payload["weights_sha256"],
        "architecture_sha256": payload["architecture_sha256"],
        "integrity": integrity_snapshot(eval_config),
    }
    eval_config_path = tmp_path / "eval-config.json"
    eval_config_path.write_text(json.dumps(eval_config))
    request_path = tmp_path / "eval-request.json"
    request_path.write_text(json.dumps(request))
    scored = subprocess.run(
        [sys.executable, "-m", "research.evaluator",
         "--config", str(eval_config_path), "--request", str(request_path)],
        cwd=worktree, text=True, capture_output=True, timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "PYTHONPATH": str(worktree / "src")},
    )
    score = _last_json(scored)
    assert score["ok"], score
    assert isinstance(score["primary_value"], float)


def test_poisoned_pythonpath_cannot_run_startup_code(tmp_path, docker_daemon):
    """H-02: a PYTHONPATH directory must be dead to a trusted child. A planted
    sitecustomize.py executes BEFORE any SHA check — with the old `-m` launch
    and inherited PYTHONPATH it ran in every worker and could forge the result
    line. The isolated bootstrap launch must leave it unimported."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "sitecustomize.py").write_text(
        "print('SITECUSTOMIZE-RAN')\n"
    )
    records = fake_records()
    _fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    _link_src(worktree)
    candidate_sha = _file_sha(worktree / "src" / "research" / "candidate.py")
    config = {
        "worktree_path": str(worktree),
        "editable": "src/research/candidate.py",
        "candidate_sha256": candidate_sha,
        "seed": 0,
        "time_budget_seconds": 20.0,
        "train_batch_size": 8,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_records_path": str(fit_records_path),
        "output_dir": str(tmp_path / "out"),
        "output_stem": "m0",
    }
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(config))
    from research.spawn import bootstrap_command

    result = subprocess.run(
        bootstrap_command(str(worktree), "research.train_worker",
                          "--config", str(config_path)),
        cwd=worktree, text=True, capture_output=True, timeout=120,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
            "PYTHONPATH": str(probe_dir),
        },
    )
    assert "SITECUSTOMIZE-RAN" not in result.stdout
    assert "SITECUSTOMIZE-RAN" not in result.stderr
    payload = _last_json(result)
    assert payload["ok"], payload


def test_same_config_spawns_bitwise_identical_weights(tmp_path, docker_daemon):
    """Two isolated workers with the same seed produce equal state tensors.

    The temporary candidate runs one fixed training batch. This tests process
    seed/model reproducibility without asserting a stable step count for the
    production candidate's wall-clock budget loop.
    """
    records = fake_records()
    _fit_manifest, fit_manifest_path, fit_records_path = _fit_inputs(
        tmp_path, records
    )
    worktree = tmp_path / "wt"
    _link_src(worktree)
    source = (SRC_ROOT / "research" / "candidate.py").read_text()
    candidate_source = source[:source.index("def train(")] + _FIXED_STEP_TRAIN
    research_root = worktree / "src" / "research"
    research_root.unlink()
    research_root.mkdir()
    for item in (SRC_ROOT / "research").iterdir():
        target = research_root / item.name
        if item.name == "candidate.py":
            target.write_text(candidate_source)
        else:
            target.symlink_to(item, target_is_directory=item.is_dir())
    candidate_sha = _file_sha(research_root / "candidate.py")
    states = []
    import torch
    for run in range(2):
        config = {
            "worktree_path": str(worktree),
            "editable": "src/research/candidate.py",
            "candidate_sha256": candidate_sha,
            "seed": 0,
            "time_budget_seconds": 5.0,
            "train_batch_size": 8,
            "fit_manifest_path": str(fit_manifest_path),
            "fit_records_path": str(fit_records_path),
            "output_dir": str(tmp_path / f"out-{run}"),
            "output_stem": "m0",
        }
        config_path = tmp_path / f"worker-config-{run}.json"
        config_path.write_text(json.dumps(config))
        from research.spawn import bootstrap_command
        result = subprocess.run(
            bootstrap_command(str(worktree), "research.train_worker",
                              "--config", str(config_path)),
            cwd=worktree, text=True, capture_output=True, timeout=120,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(tmp_path),
                "TMPDIR": str(tmp_path),
            },
        )
        payload = _last_json(result)
        assert payload["ok"], payload
        states.append(torch.load(payload["weights_path"], weights_only=True))
    assert states[0].keys() == states[1].keys()
    assert all(
        torch.equal(states[0][name], states[1][name])
        for name in states[0]
    )

def test_runner_source_never_imports_the_candidate():
    runner_src = (SRC_ROOT / "research" / "runner.py").read_text()
    # F-02: the runner is an orchestrator — candidate code never runs there,
    # so a monkeypatched subprocess.run cannot forge its own score
    assert "import research.candidate" not in runner_src
    assert "research.candidate" not in runner_src
    assert "TCMTrustedEvaluator" not in runner_src
    # The trusted worker hashes source bytes before passing the same bytes
    # to the candidate importer; candidate execution is container-gated.
    worker_src = (SRC_ROOT / "research" / "train_worker.py").read_text()
    assert "source, actual_sha = read_candidate_source(config)" in worker_src
    assert "import_candidate_ref = import_candidate" in worker_src
    assert "candidate = import_candidate_ref(config, source=source)" in worker_src
    assert "_require_candidate_container()" in worker_src
    # finalize.py remains a trusted driver and has no candidate import.
    for module in ("evaluator.py", "frozen.py", "guard.py",
                   "loop.py", "model_export.py"):
        assert "import research.candidate" not in (SRC_ROOT / "research" / module).read_text(), (
            f"{module} must not execute candidate code"
        )


def _link_src(worktree: Path) -> None:
    """A fake worktree whose src is the real package: subprocesses import the
    real frozen machinery while the candidate file is the recorded baseline."""
    src = worktree / "src"
    src.mkdir(parents=True)
    for item in SRC_ROOT.iterdir():
        (src / item.name).symlink_to(item)


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_json(result: subprocess.CompletedProcess) -> dict:
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, result.stdout + result.stderr
    return json.loads(lines[-1])


def test_candidate_source_size_limit_is_checked_before_read(tmp_path, monkeypatch):
    import research.train_worker as worker

    monkeypatch.setattr(worker, "MAX_CANDIDATE_SOURCE_BYTES", 8)
    worktree = tmp_path / "oversized"
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"x" * 9)
    with pytest.raises(RuntimeError, match="8-byte limit"):
        worker.read_candidate_source(
            {
                "worktree_path": str(worktree),
                "editable": "src/research/candidate.py",
                "candidate_sha256": "0" * 64,
            }
        )


def test_candidate_source_staging_does_not_compile_on_host(tmp_path):
    import hashlib

    from research.spawn import stage_candidate_source

    worktree = tmp_path / "invalid-source"
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    source = b"def invalid(:\n"
    candidate.write_bytes(source)
    staged = stage_candidate_source(
        worktree,
        "src/research/candidate.py",
        hashlib.sha256(source).hexdigest(),
        tmp_path / "staged",
    )
    assert (staged / "research" / "candidate.py").read_bytes() == source


def test_worker_requires_container_before_parsing_candidate(tmp_path):
    import hashlib

    from research.train_worker import fit

    worktree = tmp_path / "invalid-source"
    candidate = worktree / "src" / "research" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    source = b"def invalid(:\n"
    candidate.write_bytes(source)
    with pytest.raises(RuntimeError, match="requires the restricted Docker sandbox"):
        fit(
            {
                "worktree_path": str(worktree),
                "editable": "src/research/candidate.py",
                "candidate_sha256": hashlib.sha256(source).hexdigest(),
            }
        )


def test_candidate_path_rejects_in_tree_final_component_symlink(tmp_path):
    from research.train_worker import _candidate_path

    source_root = tmp_path / "src" / "research"
    source_root.mkdir(parents=True)
    target = source_root / "other.py"
    target.write_text("VALUE = 1\n")
    candidate = source_root / "candidate.py"
    candidate.symlink_to(target)
    config = {
        "worktree_path": str(tmp_path),
        "editable": "src/research/candidate.py",
    }
    with pytest.raises(RuntimeError, match="symlink"):
        _candidate_path(config)


def test_candidate_source_open_rejects_symlink_swapped_after_path_check(
    tmp_path, monkeypatch
):
    import hashlib

    import research.train_worker as worker

    source_root = tmp_path / "src" / "research"
    source_root.mkdir(parents=True)
    candidate = source_root / "candidate.py"
    target = source_root / "other.py"
    source = b"VALUE = 1\n"
    candidate.write_bytes(source)
    target.write_bytes(source)
    config = {
        "worktree_path": str(tmp_path),
        "editable": "src/research/candidate.py",
        "candidate_sha256": hashlib.sha256(source).hexdigest(),
    }
    original_is_symlink = Path.is_symlink
    swapped = False

    def swap_after_check(path):
        nonlocal swapped
        result = original_is_symlink(path)
        if path == candidate and not result and not swapped:
            candidate.unlink()
            candidate.symlink_to(target)
            swapped = True
        return result

    monkeypatch.setattr(Path, "is_symlink", swap_after_check)
    with pytest.raises(RuntimeError, match="cannot be opened as a regular file"):
        worker.read_candidate_source(config)
    assert swapped
