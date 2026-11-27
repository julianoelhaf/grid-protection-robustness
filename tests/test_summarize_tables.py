"""Regression checks for the paper's window-level aggregation."""

import pytest

from sparse_fault_detect.visualization import summarize_tables


def test_summary_pairs_each_condition_with_its_own_window_reference():
    rows = [
        {
            "task": task,
            "condition": condition,
            "window_ms": window,
            "mean_f1_score": (window / 50 if condition == "baseline" else 0.2),
        }
        for task in summarize_tables.TASKS
        for condition in summarize_tables.CONDITIONS
        for window in summarize_tables.WINDOWS
    ]
    assert ("fault_target", "current_loss", "0.200", "0.000", "-54.33") in list(
        summarize_tables.summarize(rows)
    )
    assert ("fault", "baseline", "0.6000", "0.2828", "0.00") in list(
        summarize_tables.summarize(rows)
    )
    with pytest.raises(ValueError, match="duplicate"):
        list(summarize_tables.summarize([*rows, rows[0]]))
    with pytest.raises(ValueError, match="Missing task"):
        list(summarize_tables.summarize(rows[:-1]))
