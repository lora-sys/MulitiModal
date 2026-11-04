"""Regression checks for experiment history and container ownership."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from research.history import append_history, write_new_atomic
from research.sandbox import _remove_owned_container


def append(path, n, **changes):
    fields = dict(commit=f"{n:040x}", metric="loss", value=float(n), direction="minimize",
                  decision="REVERT", description="trial", protocol="a" * 64,
                  evidence="SMOKE_ONLY")
    fields.update(changes)
    append_history(path, **fields)


def test_history_policy_and_crash_metadata(tmp_path):
    path = tmp_path / "results.tsv"
    append(path, 1, value=None, decision="CRASH", protocol="")
    with pytest.raises(ValueError, match="metadata"):
        append(path, 2, metric="accuracy")
    append(path, 2, decision="KEEP")
    with pytest.raises(ValueError, match="improvement"):
        append(path, 3, value=2.0, decision="KEEP")
    with pytest.raises(ValueError, match="metadata"):
        append(path, 3, promotion="external")
    other = tmp_path / "external.tsv"
    append(other, 1, decision="KEEP", promotion="external")
    append(other, 2, value=1.0, decision="KEEP", promotion="external")


def test_atomic_history_and_result_boundaries(tmp_path):
    path = tmp_path / "results.tsv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: append(path, n), range(1, 9)))
    assert len(path.read_text().splitlines()) == 9
    path.write_text(path.read_text() + "partial")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        append(path, 9)
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


def test_cleanup_never_removes_a_foreign_container(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="different-owner\n", stderr="")

    monkeypatch.setattr("research.sandbox.subprocess.run", run)
    _remove_owned_container("candidate", "our-run")
    assert len(calls) == 1
    assert calls[0][1] == "inspect"
