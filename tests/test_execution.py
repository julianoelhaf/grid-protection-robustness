import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grid_protection_robustness.jobs import claim, publish_json
from grid_protection_robustness import execution
from grid_protection_robustness.waveforms import digest


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


def test_unreadable_case_can_be_quarantined(tmp_path, monkeypatch, capsys):
    matrix = execution.matrix().head(2)
    monkeypatch.setattr(execution, "matrix", lambda: matrix)
    result = tmp_path / "results" / "000"
    (result / "STATUS.json").mkdir(parents=True)
    publish_json(tmp_path / "run.json", {})
    publish_json(execution.cache_manifest(tmp_path, int(matrix.iloc[0].window_ms)), {})

    execution.remaining(tmp_path, set(), array=True)
    output = capsys.readouterr()
    assert output.out.strip() == "1"
    assert "Damaged case 0" in output.err
    execution.repair_case(tmp_path, 0)
    assert not result.exists()
    assert next((tmp_path / "quarantine").glob("*/case/STATUS.json")).is_dir()


def test_missing_window_manifest_only_blocks_that_window(tmp_path, monkeypatch, capsys):
    matrix = pd.DataFrame({"case": [0, 1, 2, 3], "window_ms": [50, 20, 50, 50]})
    monkeypatch.setattr(execution, "matrix", lambda: matrix)
    publish_json(tmp_path / "run.json", {})
    publish_json(execution.cache_manifest(tmp_path, 20), {})
    for case in (2, 3):
        (tmp_path / "results" / f"{case:03d}").mkdir(parents=True)
    execution.remaining(tmp_path, set(), array=True)
    output = capsys.readouterr()
    assert output.out.strip() == "1"
    assert output.err.count("Cache manifest unavailable for 50 ms") == 1
    with pytest.raises(execution.UnavailableWindow):
        execution.repair_case(tmp_path, 2)
    assert (tmp_path / "results" / "002").is_dir()


@pytest.mark.parametrize("matches", [True, False])
def test_prepare_restores_pruned_array_with_unchanged_receipt(tmp_path, monkeypatch, matches):
    window = 10
    root = tmp_path
    publish_json(root / "run.json", {})
    run_id = digest(root / "run.json")
    monkeypatch.setattr(execution, "bind_run", lambda *_: run_id)
    cache = root / "cache" / str(window)
    cache.mkdir(parents=True)
    array = cache / "x.npy"
    (cache / "windows.csv").write_bytes(b"row map")
    (cache / "meta.json").write_bytes(b"metadata")
    array.write_bytes(b"prepared array")

    def describe(path, size, run, smoke):
        files = {name: digest(path / name) for name in ("windows.csv", "meta.json", "x.npy")}
        return {"run": run, "window": size, "sha256": files}

    recorded = describe(cache, window, run_id, False)
    manifest = execution.cache_manifest(root, window)
    publish_json(manifest, recorded)
    publish_json(cache / "prepared.json", recorded)
    publish_json(cache / "array-stat.json", {
        "stat": execution.cache_stat(cache), "sha256": recorded["sha256"]["x.npy"],
    })
    case = next(execution.matrix().itertuples(index=False))
    result = root / "results" / f"{case.case:03d}"
    result.mkdir(parents=True)
    for name in ("fold-scores.csv", *(f"fold-{i}.npz" for i in range(1, 6))):
        (result / name).write_bytes(b"fold evidence")
    receipt = {"status": "complete", **case._asdict(), "run": run_id,
               "cache": digest(manifest), "sha256": {f.name: digest(f) for f in result.iterdir()}}
    publish_json(result / "STATUS.json", receipt)
    retained = {file: digest(file) for file in (manifest, cache / "prepared.json",
                cache / "windows.csv", cache / "meta.json", result / "STATUS.json")}
    array.unlink()

    def prepare(size, data, stage, smoke, sample_dir):
        for name in ("meta.json", "windows.csv"):
            (stage / name).write_bytes((cache / name).read_bytes())
        (stage / "x.npy").write_bytes(b"prepared array" if matches else b"wrong array")

    monkeypatch.setattr(execution, "prepare", prepare)
    monkeypatch.setattr(execution, "describe_cache", describe)
    if matches:
        assert execution.prepare_case_inputs(root, tmp_path, window) == cache
        assert digest(array) == recorded["sha256"]["x.npy"]
        assert execution.check_prepared(cache, window, run_id, manifest) == recorded
    else:
        with pytest.raises(ValueError, match="differs from its run manifest"):
            execution.prepare_case_inputs(root, tmp_path, window)
        assert not array.exists()
    assert {file: digest(file) for file in retained} == retained
    assert execution.read_receipt(root, case) == receipt
