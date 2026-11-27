"""Verify the archived predictions independently of preprocessing and fitting."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from .jobs import read_json
from .waveforms import digest

PROTOCOL_DIR = Path(__file__).with_name("protocol")
GROUPS = {"all": None, "LLL": 0, "LL": 1, "SLG": 2, "LLG": 3}
TABLES = ("all_fold_scores.csv", "matrix_summary.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path, smoke=False):
    rows = pd.read_csv(path)
    episodes = pd.read_csv(PROTOCOL_DIR / "episode_folds.csv").set_index("episode_id")
    if smoke:
        selected = pd.read_csv(PROTOCOL_DIR / "smoke_episodes.csv").episode_id
        episodes = episodes.loc[selected]
    require(set(rows.sample_id) == set(episodes.index), f"Wrong episode set in {path}")
    require(not rows.duplicated(["sample_id", "start_time"]).any(), f"Duplicate windows in {path}")
    require(np.isfinite(rows.start_time).all(), f"Invalid start times in {path}")
    for field in ("fold", "sc_type", "fault_target"):
        expected = episodes[field].reindex(rows.sample_id).to_numpy()
        require(np.array_equal(rows[field], expected), f"Wrong {field} in {path}")
    require(rows.groupby("sample_id").fault.agg(lambda x: set(x) == {0, 1}).all(),
            f"Invalid detection labels in {path}")
    return rows


def macro_f1(truth, prediction, classes):
    scores = []
    for label in classes:
        true, predicted = truth == label, prediction == label
        total = int(true.sum()) + int(predicted.sum())
        scores.append(2 * int((true & predicted).sum()) / total if total else 0.0)
    return float(np.mean(scores))


def verify_case(case_dir, case, rows, smoke=False):
    eligible = rows if case.task == "fault" else rows.loc[rows.fault.eq(1)]
    expected_labels = eligible[case.task].astype(str)
    classes = expected_labels.unique()
    saved = pd.read_csv(case_dir / "fold-scores.csv")
    require(list(saved.columns) == ["fold", "fault_type", "windows", "macro_f1"]
            and len(saved) == 25 and not saved.duplicated(["fold", "fault_type"]).any()
            and set(zip(saved.fold, saved.fault_type)) ==
            {(fold, group) for fold in range(1, 6) for group in GROUPS},
            f"Invalid score table in {case_dir}")
    saved = saved.set_index(["fold", "fault_type"])
    for fold in range(1, 6):
        test = eligible.loc[eligible.fold.eq(fold)]
        train = eligible.loc[eligible.fold.ne(fold)]
        require(set(test.sample_id).isdisjoint(train.sample_id), f"Episode leak in {case_dir}")
        with np.load(case_dir / f"fold-{fold}.npz", allow_pickle=False) as data:
            row, truth, prediction = (data[key] for key in ("row", "truth", "prediction"))
        require(row.ndim == truth.ndim == prediction.ndim == 1
                and row.dtype.kind in "iu" and len(row) == len(truth) == len(prediction)
                and np.array_equal(row, test.index)
                and np.array_equal(truth, expected_labels.loc[test.index].to_numpy())
                and np.isin(prediction, classes).all(),
                f"Invalid held-out predictions in {case_dir}, fold {fold}")
        for name, code in GROUPS.items():
            member = np.ones(len(test), dtype=bool) if code is None else test.sc_type.eq(code).to_numpy()
            require(member.any() and (smoke or set(truth[member]) == set(classes)),
                    f"Missing cohort in {case_dir}, fold {fold}, {name}")
            score = macro_f1(truth[member], prediction[member], classes)
            entry = saved.loc[(fold, name)]
            require(member.sum() == entry.windows and abs(score - entry.macro_f1) <= 1e-12,
                    f"Score mismatch in {case_dir}, fold {fold}, {name}")


def verify_archive(root):
    from .summarize import summarize

    protocol = read_json(PROTOCOL_DIR / "manifest.json")
    contents = read_json(root / "contents.json")
    for name in ("matrix.csv", "episode_folds.csv"):
        require(digest(PROTOCOL_DIR / name) == protocol["reference_sha256"][name],
                f"Packaged protocol differs: {name}")
    matrix = pd.read_csv(PROTOCOL_DIR / "matrix.csv", dtype={"value": str})
    expected_files = {f"cache/{w}/{name}" for w in (10, 20, 30, 40, 50)
                      for name in ("windows.csv", "meta.json")}
    expected_files.update(f"results/{case:03d}/{name}" for case in matrix.case
                          for name in ("STATUS.json", "fold-scores.csv",
                                       *(f"fold-{i}.npz" for i in range(1, 6))))
    expected_files.update(("all_fold_scores.csv", "matrix_summary.csv", "execution/run_matrix.py",
                           "execution/matrix.csv", "execution/environment.txt",
                           "execution/episode_folds.csv", "execution/source.tar"))
    require(set(contents["sha256"]) == expected_files, "Incomplete archive inventory")
    for name, expected in contents["sha256"].items():
        require(digest(root / name) == expected, f"Archive checksum mismatch: {name}")
    for name, expected in protocol["executed_window_sha256"].items():
        require(contents["sha256"][name] == expected, f"Archived row map differs: {name}")
    for name, expected in {
        "run_matrix.py": protocol["executed_matrix_runner_sha256"],
        "source.tar": protocol["executed_source_archive_sha256"],
        "matrix.csv": protocol["executed_matrix_csv_sha256"],
        "environment.txt": protocol["reference_sha256"]["environment.txt"],
        "episode_folds.csv": protocol["reference_sha256"]["episode_folds.csv"],
    }.items():
        require(digest(root / "execution" / name) == expected,
                f"Executed provenance differs: {name}")
    episodes = pd.read_csv(PROTOCOL_DIR / "episode_folds.csv")
    waveform_hashes = dict(zip(episodes.episode_id.astype(str), episodes.waveform_sha256))
    for window, group in matrix.groupby("window_ms"):
        cache = root / "cache" / str(window)
        rows = read_rows(cache / "windows.csv")
        meta = read_json(cache / "meta.json")
        require(meta["episode_sha256"] == waveform_hashes and meta["rows"] == len(rows)
                and meta["window_ms"] == window, f"Invalid {window} ms cache metadata")
        for case in group.itertuples(index=False):
            case_dir = root / "results" / f"{case.case:03d}"
            status = read_json(case_dir / "STATUS.json")
            require(status.get("status") == "complete" and
                    all(status.get(key) == value for key, value in case._asdict().items()),
                    f"Case receipt differs: {case.case}")
            verify_case(case_dir, case, rows)
    with TemporaryDirectory() as scratch:
        destination = Path(scratch)
        summarize(root, destination)
        for name in TABLES:
            expected = protocol["reference_sha256"][name]
            require(digest(destination / name) == digest(root / name) == expected,
                    f"Reference table differs: {name}")
    print("Verified 1,300 held-out folds, 6,500 scores within 1e-12, and both reference tables.")
