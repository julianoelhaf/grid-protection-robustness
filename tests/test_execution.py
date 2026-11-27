import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_protection_robustness.jobs import claim, publish_json
from grid_protection_robustness import execution


def test_interrupted_worker_releases_its_case(tmp_path):
    script = """
import sys
from pathlib import Path
from grid_protection_robustness.jobs import claim
root = Path(sys.argv[1])
with claim(root, "case-230") as acquired:
    assert acquired
    (root / ".230-partial-test").mkdir()
    print("locked", flush=True)
    sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with claim(tmp_path, "case-230") as acquired:
            assert not acquired
        assert execution.read_receipt(tmp_path, SimpleNamespace(case=230)) is None
    finally:
        child.kill()
        child.wait(timeout=10)
    with claim(tmp_path, "case-230") as acquired:
        assert acquired
    assert execution.read_receipt(tmp_path, SimpleNamespace(case=230)) is None


def test_duplicate_does_not_fit(tmp_path, monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Duplicate case ran a second fit")

    monkeypatch.setattr(execution, "fit_case", unexpected_fit)
    with claim(tmp_path, "case-230") as acquired:
        assert acquired
        execution.run_case(tmp_path, 230, 1)
    assert execution.read_receipt(tmp_path, SimpleNamespace(case=230)) is None


def test_manifest_is_immutable(tmp_path):
    path = tmp_path / "run.json"
    publish_json(path, {"protocol": "v2"})
    publish_json(path, {"protocol": "v2"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Immutable"):
        publish_json(path, {"protocol": "modified"})
    assert path.read_bytes() == before


def test_unsealed_cache_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsealed"):
        execution.check_prepared(tmp_path, 50, "run", tmp_path / "absent.json")


def test_prepare_reclaims_interrupted_window(tmp_path, monkeypatch):
    base = tmp_path / "cache"
    abandoned = base / ".50-partial-abandoned"
    other = base / ".20-partial-other"
    for folder in (abandoned, other):
        folder.mkdir(parents=True)
        (folder / "x.npy").write_bytes(b"partial")
    monkeypatch.setattr(execution, "bind_run", lambda *args: "run")

    def interrupted(*args):
        (args[2] / "x.npy").write_bytes(b"incomplete")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(execution, "prepare", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        execution.prepare_case_inputs(tmp_path, tmp_path, 50)
    assert not list(base.glob(".50-partial-*"))
    assert other.is_dir()


def test_prepare_rejects_swapped_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "read_rows", lambda *args: pd.DataFrame({"sample_id": [0]}))
    monkeypatch.setattr(execution, "relay_buses", lambda *_: {1: [0]})
    (tmp_path / "windows.csv").write_text("row map")
    waveform = pd.read_csv(execution.REFERENCE / "episode_folds.csv").iloc[0].waveform_sha256
    meta = {"window_ms": 20, "rows": 1, "names": [], "relay_buses": {"1": [0]},
            "dataset_doi": execution.PROTOCOL["dataset_doi"], "episode_sha256": {"0": waveform}}
    publish_json(tmp_path / "meta.json", meta)
    with pytest.raises(ValueError, match="metadata"):
        execution.describe_cache(tmp_path, 50, "run", True)
    (tmp_path / "meta.json").unlink()
    publish_json(tmp_path / "meta.json", {**meta, "window_ms": 50})
    np.save(tmp_path / "x.npy", np.zeros((30, 128, 48), dtype=np.float32))
    with pytest.raises(ValueError, match="array shape"):
        execution.describe_cache(tmp_path, 50, "run", True)


def test_damaged_receipt_does_not_hide_other_cases(tmp_path, monkeypatch, capsys):
    matrix = execution.matrix().head(3)
    monkeypatch.setattr(execution, "matrix", lambda: matrix)
    case = tmp_path / "results" / "000"
    case.mkdir(parents=True)
    (case / "STATUS.json").write_text("{}")
    publish_json(tmp_path / "run.json", {})
    publish_json(execution.cache_manifest(tmp_path, int(matrix.iloc[0].window_ms)), {})
    execution.remaining(tmp_path, {2}, array=True)
    streams = capsys.readouterr()
    assert streams.out.strip() == "1"
    assert "Damaged case 0" in streams.err
    assert "Case 2 is active" in streams.err
    execution.repair_case(tmp_path, 0)
    assert not case.exists()
    assert next((tmp_path / "quarantine").glob("*/case/STATUS.json")).read_text() == "{}"
    execution.remaining(tmp_path, {2}, array=True)
    assert capsys.readouterr().out.splitlines()[-1] == "0,1"


def test_evidence_inventory_is_pinned(tmp_path):
    from grid_protection_robustness.audit import verify_archive

    (tmp_path / "contents.json").write_text('{"sha256": {}}')
    with pytest.raises(ValueError, match="pinned release"):
        verify_archive(tmp_path)
