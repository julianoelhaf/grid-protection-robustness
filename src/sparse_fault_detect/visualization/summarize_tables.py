"""Summarize selected window-level CV means for EUSIPCO v2 Tables II and III."""

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev

TASKS = ("fault", "fault_target")
CONDITIONS = ("baseline", "current_loss", "voltage_loss", "bus_1", "bus_2", "bus_3")
WINDOWS = (10, 20, 30, 40, 50)


def summarize(rows):
    expected = {
        (task, condition, window)
        for task in TASKS
        for condition in CONDITIONS
        for window in WINDOWS
    }
    scores = {}
    for row in rows:
        key = (row["task"], row["condition"], int(row["window_ms"]))
        score = float(row["mean_f1_score"])
        if key not in expected or key in scores or not 0 <= score <= 1:
            raise ValueError(f"Unexpected or duplicate row, or invalid F1: {row}")
        scores[key] = score
    if expected - scores.keys():
        raise ValueError(f"Missing task/condition/window rows: {sorted(expected - scores.keys())}")

    for task in TASKS:
        reference = [scores[task, "baseline", window] for window in WINDOWS]
        if not all(reference):
            raise ValueError(f"Complete-input F1 must be positive for every {task} window.")
        for condition in CONDITIONS:
            values = [scores[task, condition, window] for window in WINDOWS]
            change = 100 * mean((value - ref) / ref for value, ref in zip(values, reference))
            precision = 4 if task == "fault" and condition == "baseline" else 3
            yield (
                task,
                condition,
                f"{mean(values):.{precision}f}",
                f"{pstdev(values):.{precision}f}",
                f"{round(change, 2) or 0:.2f}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    try:
        with args.csv_path.open(encoding="utf-8-sig", newline="") as source:
            summary = list(summarize(csv.DictReader(source)))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("task", "condition", "mean_f1", "population_sd", "relative_change_pct"))
    writer.writerows(summary)


if __name__ == "__main__":
    main()
