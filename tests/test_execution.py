import subprocess
import sys
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
        assert execution.read_receipt(tmp_path, 230) is None
    finally:
        child.kill()
        child.wait(timeout=10)
    with claim(tmp_path, "case-230") as acquired:
        assert acquired
    assert execution.read_receipt(tmp_path, 230) is None


def test_duplicate_does_not_fit(tmp_path, monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Duplicate case ran a second fit")

    monkeypatch.setattr(execution, "fit_case", unexpected_fit)
    with claim(tmp_path, "case-230") as acquired:
        assert acquired
        execution.run_case(tmp_path, 230, 1)
    assert execution.read_receipt(tmp_path, 230) is None


def test_manifest_is_immutable(tmp_path):
    path = tmp_path / "run.json"
    publish_json(path, {"protocol": "v2"})
    publish_json(path, {"protocol": "v2"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Immutable"):
        publish_json(path, {"protocol": "modified"})
    assert path.read_bytes() == before
