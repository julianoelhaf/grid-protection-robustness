"""Check the channel contract at the point where EMT column names are discarded."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparse_fault_detect.data.create_windows import extract_windows_from_dataframe


@pytest.mark.parametrize("voltage_first", [False, True])
def test_window_extraction_requires_current_then_voltage_per_relay(voltage_first):
    current = [f"cur_L{phase}_A" for phase in (1, 2, 3)]
    voltage = [f"vol_L{phase}_V" for phase in (1, 2, 3)]
    columns = [
        f"relay_1_{name}" for name in (voltage + current if voltage_first else current + voltage)
    ]
    frame = pd.DataFrame(np.tile(np.arange(1, 7), (200, 1)), columns=columns)
    frame["timestamp"] = np.arange(200) / 20_000
    labels = pd.Series(
        {
            "t_evnt_start": 0.005,
            "t_evnt_end": 0.009,
            "fault_target": "line_1",
            "sc_location": 0.4,
            "phase_select": 3,
        }
    )
    config = SimpleNamespace(
        window_extraction=SimpleNamespace(
            window_length=0.01,
            step_length_seconds=0.005,
            time_diff=0.00005,
            period_of_interest=0.08,
        )
    )

    if voltage_first:
        with pytest.raises(ValueError, match="Episode 17: expected each relay"):
            extract_windows_from_dataframe(frame, labels, 17, config)
    else:
        windows, _ = extract_windows_from_dataframe(frame, labels, 17, config)
        assert windows.shape == (1, 200, 6)
        np.testing.assert_array_equal(windows[0, 0], np.arange(1, 7))
