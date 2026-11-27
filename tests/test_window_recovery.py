from types import SimpleNamespace

import numpy as np
import pytest

from sparse_fault_detect.data import create_windows


def test_failed_combine_keeps_batches(tmp_path, monkeypatch):
    config = SimpleNamespace(window_extraction=SimpleNamespace(
        window_length=0.02, step_length_seconds=0.005,
    ))
    x_batch = tmp_path / "X_batch0_0.02_0.005.npy"
    y_batch = tmp_path / "y_batch0_0.02_0.005.npy"
    np.save(x_batch, np.ones((2, 3)))
    np.save(y_batch, np.array([0, 1]))
    x_out, y_out = tmp_path / "X.npz", tmp_path / "y.npz"
    real_replace = type(x_out).replace

    def interrupted(path, destination):
        if destination == y_out:
            raise OSError("disk failure")
        return real_replace(path, destination)

    with monkeypatch.context() as mocked:
        mocked.setattr(type(x_out), "replace", interrupted)
        with pytest.raises(OSError, match="disk failure"):
            create_windows.combine_batches(config, x_out, y_out)
    assert x_batch.is_file() and y_batch.is_file()
    assert x_out.with_suffix(".pending").exists()
    create_windows.combine_batches(config, x_out, y_out)
    assert not x_batch.exists() and not y_batch.exists()
    assert not x_out.with_suffix(".pending").exists()
    with np.load(x_out) as x, np.load(y_out) as y:
        np.testing.assert_array_equal(x["X"], np.ones((2, 3)))
        np.testing.assert_array_equal(y["y"], [0, 1])
