import pandas as pd

from .run_matrix import REFERENCE


def summarize(output_dir, destination):
    matrix = pd.read_csv(
        REFERENCE / "matrix.csv", dtype={"value": str}
    )
    parts = []
    for case in matrix.case:
        root = output_dir / "results" / f"{case:03d}"
        assert (root / "STATUS.json").is_file(), case
        table = pd.read_csv(root / "fold-scores.csv")
        assert len(table) == 25 and not table.duplicated(["fold", "fault_type"]).any(), case
        assert table.macro_f1.between(0, 1).all(), case
        parts.append(table.assign(case=case))
    folds = pd.concat(parts).merge(matrix, on="case", validate="many_to_one")
    assert len(folds) == 6500
    folds.to_csv(destination / "all_fold_scores.csv", index=False)
    keys = ["case", "window_ms", "task", "condition", "value", "fault_type"]
    summary = folds.groupby(keys).macro_f1.agg(mean_f1="mean", sd_across_folds="std").reset_index()
    reference = summary.loc[
        summary.condition.eq("complete_input"), ["window_ms", "task", "fault_type", "mean_f1"]
    ].rename(columns={"mean_f1": "reference_f1"})
    summary = summary.merge(
        reference, on=["window_ms", "task", "fault_type"], validate="many_to_one"
    )
    summary["change_pct"] = 100 * (summary.mean_f1 / summary.reference_f1 - 1)
    summary.to_csv(destination / "matrix_summary.csv", index=False)
    print(f"Verified {len(matrix)} cases and {len(folds)} fold/subgroup score rows.")
