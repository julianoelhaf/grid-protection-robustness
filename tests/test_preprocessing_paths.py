from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from sparse_fault_detect.data import create_windows, preprocessing


@pytest.mark.parametrize("absolute", [False, True])
def test_preprocessing_output_is_consumed_by_window_loader(tmp_path, monkeypatch, absolute):
    defaults = Path(preprocessing.__file__).parents[1] / "config/dataset/default.yaml"
    dataset = SimpleNamespace(**yaml.safe_load(defaults.read_text()))
    if absolute:
        dataset.preprocessed_data_directory = str(tmp_path / "absolute-output")
    config = SimpleNamespace(dataset=dataset, n_samples=1, full_dataset=False)
    monkeypatch.setattr(preprocessing.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    # Hydra may change the process directory; both stages must retain the original root.
    elsewhere = tmp_path / "hydra-output"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    raw = tmp_path / dataset.data_directory / dataset.raw_data_directory
    raw.mkdir(parents=True)
    (raw / "episode_7.txt").touch()
    expected = pd.DataFrame({"timestamp": [.001], "measurement": [42.]})
    monkeypatch.setattr(preprocessing, "process_file", lambda source, dest, cfg: expected.to_pickle(dest))
    preprocessing.process_files_in_parallel(config)
    frames, ids = next(create_windows.get_preprocessed_data(config))
    assert ids == [7]
    pd.testing.assert_frame_equal(frames[0], expected)
    output = tmp_path / ("absolute-output" if absolute else "data/preprocessed")
    assert list(output.glob("7_*.pkl"))
