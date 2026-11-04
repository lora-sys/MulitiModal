"""Synthetic checks for the replacement research protocol."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.prepare import (
    EVIDENCE_LOCAL,
    EVIDENCE_SMOKE,
    append_history,
    evaluate,
    fold_inputs,
    run_candidate,
    score_predictions,
    selection_fold_ids,
    synthetic_selection,
    write_new_atomic,
)


def _selection():
    ids = [f"s{i:02}" for i in range(24)]
    users = {subject: f"u{i // 2:02}" for i, subject in enumerate(ids)}
    partitions = {user: "train" for user in users.values()}
    strata = {user: str(i % 3) for i, user in enumerate(sorted(partitions))}
    manifest = {
        "task": "syndrome",
        "subject_row_by_id": {subject: i + 1 for i, subject in enumerate(ids)},
        "subject_user_by_id": users,
        "user_partition_by_id": partitions,
        "user_stratum_by_id": strata,
        "split": {"seed": 123, "train": ids, "dev": [], "holdout": []},
    }
    return manifest


def _append(path, n, **changes):
    """Append one history row the way a scored round does."""
    fields = dict(commit=f"{n:040x}", metric="loss", value=float(n), direction="minimize",
                  decision="REVERT", description="trial", protocol="a" * 64,
                  evidence=EVIDENCE_SMOKE)
    fields.update(changes)
    append_history(path, **fields)


def test_grouped_folds_match_the_frozen_derivation():
    import hashlib

    manifest = _selection()
    # Recorded from tcm-sd-stratified-user-fold-v2 before removing the old engine.
    expected = [
        ["4863fd6d273cc562b100ce600e0b21cc31835040c035f9c0fc8678aaeed79f2b",
         "7726639f568301a0640545d8b7a41eafb1107c2c006222e2fe50a835a9a3dc66",
         "6f62e941ef85400c6c19836da56e28b62ba60d5038ec66999e0bb188ec27427f",
         "dabaa3b5d7aef9ae7708674d2e9093212a8ad43068c9ccd08456b2b7c55937e5"],
        ["ff815d1e2b396192daaabd2f2efd6228b4380ace317f173cb48bc27e62ce1955",
         "f37f963935441f7ed418d364ca5d31e83eac8bc3164550112c5633bd2e931250",
         "5f0594df856918b84e49bbb68c779788af722bec0a2ee49daea27b16f03ac943",
         "89bb7fee7b5ee15e13badb933883ce6b8783f2addf2e3830b0d2b3a77386ee81"],
    ]
    for repeat in range(2):
        for fold in range(4):
            train, dev = selection_fold_ids(manifest, fold, 4, repeat)
            digest = hashlib.sha256(json.dumps((train, dev)).encode()).hexdigest()
            assert digest == expected[repeat][fold]
            users = manifest["subject_user_by_id"]
            assert not {users[item] for item in train} & {users[item] for item in dev}


def test_fold_drops_dev_texts_seen_in_training():
    manifest = _selection()
    train, dev = selection_fold_ids(manifest, 0, 4, 0)
    rows = manifest["subject_row_by_id"]
    records = {row: SimpleNamespace(input_text=subject, label=row % 2)
               for subject, row in rows.items()}
    records[rows[dev[0]]].input_text = records[rows[train[0]]].input_text
    _train, validation = fold_inputs(manifest, records, 0, 4, 0)
    assert records[rows[dev[0]]] not in validation
    assert len(validation) == len(dev) - 1


def test_scoring_uses_all_classes_and_refuses_bad_predictions():
    assert score_predictions([0, 1], [0, 1], 148) == pytest.approx(2 / 148)
    for predictions in ([0], [0, 148], [0, True]):
        with pytest.raises(ValueError):
            score_predictions([0, 1], predictions, 148)


def test_evaluation_rejects_excessive_compute_before_starting_candidate():
    with pytest.raises(ValueError, match="run limit"):
        evaluate({"split": {"holdout": []}}, {}, b"candidate",
                 budget_seconds=3600, seed=1, folds=10, repeats=10)


def test_smoke_mode_refuses_external_data_paths(tmp_path):
    from research.prepare import main

    with pytest.raises(SystemExit):
        main(["--synthetic", "--manifest", str(tmp_path / "manifest.json"),
              "--dataset", str(tmp_path / "records.json"),
              "--candidate", str(tmp_path / "train.py"), "--out", str(tmp_path / "result")])


def test_real_evaluation_requires_the_frozen_data_paths(tmp_path):
    """Real runs must name the pinned pair; a run without them is refused."""
    from research.prepare import main

    with pytest.raises(SystemExit):
        main(["--candidate", str(tmp_path / "train.py"),
              "--out", str(tmp_path / "result.json")])


def test_scoring_matches_frozen_evaluator():
    labels = [0, 0, 1, 4, 4, 4]
    predictions = [0, 1, 1, 0, 4, 4]
    assert score_predictions(labels, predictions, 148) == pytest.approx(
        (0.5 + 2 / 3 + 0.8) / 148
    )


@pytest.mark.synthetic
def test_candidate_sees_only_the_files_it_is_handed(tmp_path, monkeypatch):
    """The candidate runs in a private temporary directory: it cannot reach the
    repository, the frozen manifest, or the caller's working tree."""
    seen = {}

    def spy(command, **kwargs):
        seen["cwd"] = str(kwargs.get("cwd"))
        Path(command[command.index("--output") + 1]).write_text("[]")
        return subprocess.CompletedProcess(command, 0, "", "")

    import research.prepare as prepare_module

    monkeypatch.setattr(prepare_module.subprocess, "run", spy)
    train = [{"text": "synthetic train", "label": 0}]
    dev = ["synthetic validation"]
    assert run_candidate(b"pass", train, dev, budget_seconds=10, seed=1) == []
    assert "autoresearch-candidate-" in seen["cwd"]


@pytest.mark.synthetic
def test_candidate_timeout_is_not_a_score():
    train = [{"text": "synthetic training", "label": 0}]
    dev = ["synthetic validation"]
    with pytest.raises(TimeoutError, match="exceeded"):
        run_candidate(b"import time; time.sleep(30)", train, dev,
                      budget_seconds=1, seed=1)


@pytest.mark.synthetic
def test_two_fold_candidate_to_fixed_score():
    manifest = _selection()
    manifest.update(n_classes=2, label_tier="SMOKE_ONLY",
                    dataset_sha256="0" * 64, manifest_sha256="1" * 64)
    rows = manifest["subject_row_by_id"]
    records = {
        row: SimpleNamespace(input_text=("头痛" if row % 2 else "腹痛") + str(row),
                             label=row % 2)
        for row in rows.values()
    }
    candidate = (Path(__file__).resolve().parents[2] /
                 "src/research/train.py").read_bytes()
    result = evaluate(manifest, records, candidate,
                      budget_seconds=20, seed=1, folds=2, repeats=1)
    assert result["evidence"] == "SMOKE_ONLY"
    assert len(result["folds"]) == 2
    assert 0 <= result["macro_f1"] <= 1


def test_monitor_reads_results_without_following_symlinks(tmp_path):
    from research.dashboard import read_run

    (tmp_path / "status.json").write_text(json.dumps({"state": "COMPLETE"}))
    (tmp_path / "results.tsv").write_text(
        "commit\tmetric\tvalue\tdecision\tdescription\nabc1234\tmetric\t0.5\tKEEP\t<test>\n"
    )
    view = read_run(tmp_path)
    assert view["status"] == "COMPLETE"
    assert view["results"][0]["description"] == "<test>"
    (tmp_path / "results.tsv").unlink()
    (tmp_path / "results.tsv").symlink_to(tmp_path / "status.json")
    with pytest.raises(OSError):
        read_run(tmp_path)


def test_monitor_serves_read_only_results(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from research.dashboard import make_server

    (tmp_path / "results.tsv").write_text(
        "commit\tmetric\tvalue\tdecision\tdescription\nabc1234\tmetric\t0.5\tKEEP\tbaseline\n"
    )
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url + "/api", timeout=3) as response:
            payload = json.load(response)
        assert payload["results"][0]["value"] == "0.5"
        bad_host = urllib.request.Request(url + "/api", headers={"Host": "attacker.invalid"})
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(bad_host, timeout=3)
        assert denied.value.code == 403
        request = urllib.request.Request(url + "/api", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_monitor_exposes_per_round_fold_scores_and_device(tmp_path):
    """The dashboard shows per-fold scores, device and protocol hashes, not
    just one number per round."""
    import subprocess

    from research.dashboard import read_run

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"],
                   cwd=repo, check=True)
    target = repo / "src/research/train.py"
    target.parent.mkdir(parents=True)
    target.write_text("# synthetic candidate\n")
    subprocess.run(["git", "add", "src/research/train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic candidate"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    run = tmp_path / "run"
    run.mkdir()
    (run / "round-000.json").write_text(json.dumps({
        "macro_f1": 0.5, "fold_std": 0.01, "fold_count": 2, "repeats": 1,
        "folds": [{"fold": 0, "repeat": 0, "macro_f1": 0.48, "n_validation": 10,
                   "train_seconds": 1.5},
                  {"fold": 1, "repeat": 0, "macro_f1": 0.52, "n_validation": 10,
                   "train_seconds": 1.7}],
        "device": "cuda", "gpu": "TEST GPU", "budget_seconds": 60, "seed": 7,
        "dataset_sha256": "a" * 64, "manifest_sha256": "b" * 64, "evidence": "LOCAL_EXPERIMENTAL",
    }))
    (run / "results.tsv").write_text(
        "commit\tmetric\tvalue\tdirection\tdecision\tdescription\tpeak_vram_mb\t"
        "protocol_sha256\tevidence\tpromotion\n"
        f"{commit}\tsyndrome/macro_f1\t0.5\tmaximize\tKEEP\tbaseline\tNOT_AVAILABLE\t"
        + "c" * 64 + "\tLOCAL_EXPERIMENTAL\tstrict\n"
    )
    view = read_run(run)
    assert view["rounds"][0]["detail"]["folds"][1]["macro_f1"] == 0.52
    assert view["rounds"][0]["detail"]["device"] == "cuda"
    assert view["device"] == "cuda"
    assert view["rounds"][0]["value"] == 0.5


def test_record_binds_score_to_committed_training_file(tmp_path):
    import hashlib

    from research.prepare import fixed_code_hashes, record_main

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"],
                   cwd=repo, check=True)
    target = repo / "src/research/train.py"
    target.parent.mkdir(parents=True)
    target.write_text("# synthetic candidate\n")
    subprocess.run(["git", "add", "src/research/train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic candidate"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    result = tmp_path / "result.json"
    fixture, _ = synthetic_selection()
    payload = {"candidate_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "macro_f1": 0.5, "metric": "syndrome/macro_f1", "direction": "maximize",
               "metric_version": "tcm-sd-eval-v3", "evidence": "SMOKE_ONLY",
               "code_sha256": fixed_code_hashes(), "promotion": "strict",
               "dataset_sha256": fixture["dataset_sha256"],
               "manifest_sha256": fixture["manifest_sha256"],
               "seed": 1, "budget_seconds": 20, "image": "sha256:" + "0" * 64,
               "fold_count": 2, "repeats": 1,
               "folds": [{"fold": i, "repeat": 0, "macro_f1": 0.5,
                          "n_validation": 1} for i in range(2)]}
    result.write_text(json.dumps(payload))
    history = tmp_path / "results.tsv"
    args = ["--repo", str(repo), "--results", str(history), "--commit", commit,
            "--decision", "KEEP", "--description", "synthetic", "--result", str(result)]
    assert record_main(args) == 0
    assert "0.5\tmaximize\tKEEP\tsynthetic" in history.read_text()
    result.write_text(json.dumps({"candidate_sha256": "0" * 64, "macro_f1": 0.5}))
    with pytest.raises(ValueError, match="does not match"):
        record_main(args)
    target.write_text("# second synthetic candidate\n")
    subprocess.run(["git", "add", "src/research/train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "second synthetic candidate"], cwd=repo, check=True)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    payload["candidate_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    payload["budget_seconds"] = 30
    result.write_text(json.dumps(payload))
    next_args = args.copy()
    next_args[next_args.index(commit)] = second
    with pytest.raises(ValueError, match="protocol differs"):
        record_main(next_args)
    payload["budget_seconds"] = 20
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires an improvement"):
        record_main(next_args)


def test_record_accepts_a_real_protocol_v2_result_as_local_experimental(tmp_path):
    """A real-data round records LOCAL_EXPERIMENTAL, never SMOKE_ONLY, and the
    result must be bound to the pinned protocol-v2 hashes."""
    import hashlib

    from research.prepare import (
        SELECTION_DATASET_SHA256,
        SELECTION_MANIFEST_SHA256,
        fixed_code_hashes,
        record_main,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"],
                   cwd=repo, check=True)
    target = repo / "src/research/train.py"
    target.parent.mkdir(parents=True)
    target.write_text("# synthetic candidate\n")
    subprocess.run(["git", "add", "src/research/train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic candidate"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    result = tmp_path / "result.json"
    payload = {
        "candidate_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "macro_f1": 0.3199, "metric": "syndrome/macro_f1", "direction": "maximize",
        "metric_version": "tcm-sd-eval-v3", "evidence": EVIDENCE_LOCAL,
        "code_sha256": fixed_code_hashes(), "promotion": "strict",
        "dataset_sha256": SELECTION_DATASET_SHA256,
        "manifest_sha256": SELECTION_MANIFEST_SHA256,
        "seed": 1, "budget_seconds": 60, "fold_count": 2, "repeats": 1,
        "device": "cpu", "gpu": "NOT_AVAILABLE",
        "folds": [{"fold": i, "repeat": 0, "macro_f1": 0.3199,
                   "n_validation": 1} for i in range(2)],
    }
    result.write_text(json.dumps(payload))
    history = tmp_path / "results.tsv"
    args = ["--repo", str(repo), "--results", str(history), "--commit", commit,
            "--decision", "KEEP", "--description", "real baseline", "--result", str(result)]
    assert record_main(args) == 0
    row = history.read_text().splitlines()[1].split("\t")
    assert EVIDENCE_LOCAL in row
    # a synthetic-labelled result carrying the real hashes is refused, and a
    # real-labelled result carrying the synthetic hashes is refused too
    payload["evidence"] = EVIDENCE_SMOKE
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="built-in synthetic protocol"):
        record_main(args)
    payload["evidence"] = EVIDENCE_LOCAL
    payload["dataset_sha256"] = "0" * 64
    payload["manifest_sha256"] = "1" * 64
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen protocol-v2"):
        record_main(args)


def test_output_files_do_not_follow_symlinks(tmp_path):
    from research.prepare import open_regular

    target = tmp_path / "target"
    target.write_text("unchanged")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        open_regular(link)
    assert target.read_text() == "unchanged"


def test_prepare_failure_replaces_old_complete_status(tmp_path):
    from research.prepare import main

    (tmp_path / "status.json").write_text('{"state":"COMPLETE"}')
    with pytest.raises(FileNotFoundError):
        main(["--synthetic", "--candidate", str(tmp_path / "missing.py"),
              "--out", str(tmp_path / "result.json")])
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "FAILED"


@pytest.mark.synthetic
def test_skill_flow_baseline_candidate_and_monitor(tmp_path):
    from research.dashboard import read_run
    from research.prepare import main, record_main

    repo = tmp_path / "worktree"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"],
                   cwd=repo, check=True)
    training = repo / "src/research/train.py"
    training.parent.mkdir(parents=True)
    source = (Path(__file__).resolve().parents[2] /
              "src/research/train.py").read_text()
    training.write_text(source)
    run = tmp_path / "run"

    def experiment(index: int, description: str, baseline_score=None):
        subprocess.run(["git", "add", "src/research/train.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", description], cwd=repo, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True).strip()
        result_path = run / f"round-{index}.json"
        assert main(["--synthetic", "--candidate", str(training), "--out", str(result_path),
                     "--folds", "2", "--repeats", "1", "--budget", "20"]) == 0
        result = json.loads(result_path.read_text())
        improved = baseline_score is None or result["macro_f1"] > baseline_score
        decision = "KEEP" if improved else "REVERT"
        assert record_main(["--repo", str(repo), "--results", str(run / "results.tsv"),
                            "--commit", commit, "--result", str(result_path),
                            "--decision", decision, "--description", description]) == 0
        return commit, result, decision

    baseline_commit, baseline, _ = experiment(0, "synthetic baseline")
    training.write_text(source.replace("C=4.0", "C=3.0"))
    trial_commit, trial, decision = experiment(1, "synthetic trial", baseline["macro_f1"])
    if decision == "REVERT":
        subprocess.run(["git", "reset", "--hard", baseline_commit], cwd=repo,
                       check=True, capture_output=True)
    expected_head = baseline_commit if decision == "REVERT" else trial_commit
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip() == expected_head
    view = read_run(run)
    assert view["status"] == "COMPLETE"
    assert [row["decision"] for row in view["results"]] == ["KEEP", decision]
    assert float(view["results"][0]["value"]) == baseline["macro_f1"]
    assert float(view["results"][1]["value"]) == trial["macro_f1"]


# -- results.tsv policy and atomic boundaries ----------------------------------


def test_history_policy_and_crash_metadata(tmp_path):
    path = tmp_path / "results.tsv"
    _append(path, 1, value=None, decision="CRASH", protocol="")
    with pytest.raises(ValueError, match="metadata"):
        _append(path, 2, metric="accuracy")
    _append(path, 2, decision="KEEP")
    with pytest.raises(ValueError, match="improvement"):
        _append(path, 3, value=2.0, decision="KEEP")
    with pytest.raises(ValueError, match="metadata"):
        _append(path, 3, promotion="external")
    other = tmp_path / "external.tsv"
    _append(other, 1, decision="KEEP", promotion="external")
    _append(other, 2, value=1.0, decision="KEEP", promotion="external")


def test_atomic_history_and_result_boundaries(tmp_path):
    path = tmp_path / "results.tsv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: _append(path, n), range(1, 9)))
    assert len(path.read_text().splitlines()) == 9
    path.write_text(path.read_text() + "partial")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        _append(path, 9)
    assert path.read_bytes() == before
    result = tmp_path / "result.json"
    write_new_atomic(result, b"first")
    with pytest.raises(FileExistsError):
        write_new_atomic(result, b"second")
    assert result.read_bytes() == b"first"
    link = tmp_path / "link"
    link.symlink_to(result)
    with pytest.raises(FileExistsError):
        write_new_atomic(link, b"third")
    assert result.read_bytes() == b"first"
