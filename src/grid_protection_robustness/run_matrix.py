import hashlib
import json
import re
import zipfile
from tempfile import mkdtemp
from pathlib import Path
from types import SimpleNamespace as Config

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sparse_fault_detect.data.create_windows import extract_windows_from_dataframe
from sparse_fault_detect.models import data_sparsity
from sparse_fault_detect.models.run_model import create_model_from_name

from .waveforms import digest, read_episode

REFERENCE = Path(__file__).resolve().parent / "protocol"
PROTOCOL = json.loads((REFERENCE / "manifest.json").read_text())
SOURCE_BASELINE = "1f6eedb83ced6cf713a41d863e3f8080afe9ef1e"
WINDOWS = (10, 20, 30, 40, 50)
LABEL_FILE = "hv_double_line_90kv_labels.csv"
WAVEFORM_FILE = "hv_double_line_90kv_preprocessed_data.zip"
RATE = 6400
FAULT_TYPES = {0: "LLL", 1: "LL", 2: "SLG", 3: "LLG"}
LABEL_COUNTS = {0: 2273, 1: 2251, 2: 2279, 3: 2219}


def parameters(window):
    return Config(
        window_extraction=Config(
            window_length=window / 1000,
            step_length_seconds=0.005,
            period_of_interest=0.080,
            time_diff=1 / RATE,
        ),
        model=Config(model="random_forest_classifier"),
        dataset=Config(sampling_frequency=RATE),
    )


def episode_folds(labels):
    ids = labels.index.to_numpy()
    split = pd.Series(-1, index=labels.index, name="fold")
    for i, (train, test) in enumerate(GroupKFold(n_splits=5).split(ids, groups=ids), 1):
        assert set(ids[train]).isdisjoint(ids[test])
        assert set(labels.iloc[test].groupby(["sc_type", "fault_target"]).groups) == set(
            labels.groupby(["sc_type", "fault_target"]).groups
        )
        split.iloc[test] = i
    assert split.between(1, 5).all()
    return split


def relay_buses(names):
    relays = [
        re.fullmatch(r"Bus_([123])_Line_\d+_\d+[AB]_cur_L1_A", channel) for channel in names[::6]
    ]
    if len(names) != 48 or any(relay is None for relay in relays):
        raise ValueError("Unexpected PROTECT-90 relay channel order")
    mapping = {
        bus: [i for i, relay in enumerate(relays) if int(relay[1]) == bus] for bus in (1, 2, 3)
    }
    if list(map(len, mapping.values())) != [2, 4, 2]:
        raise ValueError(f"Unexpected relay count at each bus: {mapping}")
    return mapping


def prepare(window, data_dir, output_dir, smoke=False, sample_dir=None):
    labels_path = data_dir / LABEL_FILE
    if digest(labels_path, "md5") != PROTOCOL["labels_md5"]:
        raise ValueError("Labels must be from PROTECT-90 v1.0.0 (Zenodo 21109169)")
    labels = pd.read_csv(labels_path).set_index("sample_id").sort_index()
    if labels.sc_type.value_counts().to_dict() != LABEL_COUNTS or not labels.index.is_unique:
        raise ValueError("Wrong PROTECT-90 episode population")
    mode = "smoke-cache" if smoke else "cache"
    base = output_dir / mode
    base.mkdir(parents=True, exist_ok=True)
    target = base / str(window)
    if target.exists():
        if all((target / name).is_file() for name in ("meta.json", "windows.csv", "x.npy")):
            print(f"Reusing window cache {target}", flush=True)
            return
        raise ValueError(f"Incomplete window cache {target}. Use a fresh output directory.")
    staging = Path(mkdtemp(prefix=f".{window}-partial-", dir=base))
    expected = pd.read_csv(REFERENCE / "episode_folds.csv").set_index("episode_id")
    split = episode_folds(labels)
    pd.testing.assert_series_equal(split, expected.fold.reindex(split.index), check_names=False)
    if smoke:
        # One pinned episode per fault family and fold.
        selected = pd.read_csv(REFERENCE / "smoke_episodes.csv").episode_id
        labels = labels.loc[sorted(selected)]
    samples = round(window * RATE / 1000)
    max_rows = len(labels) * ((160 - window) // 5 + 2)
    output = np.lib.format.open_memmap(
        staging / "x.npy", mode="w+", dtype=np.float32, shape=(max_rows, samples, 48)
    )
    metadata, digests = [], {}
    names = None
    offset = 0
    archive = None if sample_dir else zipfile.ZipFile(data_dir / WAVEFORM_FILE)
    try:
        members = (
            {}
            if sample_dir
            else {Path(name).name: name for name in archive.namelist() if name.endswith(".pkl")}
        )
        cfg = parameters(window)
        for i, (episode_id, record) in enumerate(labels.iterrows(), 1):
            payload = (
                (sample_dir / f"{episode_id}.pkl").read_bytes()
                if sample_dir
                else archive.read(members[f"{episode_id}_sample_hv_double_line_90kv.pkl"])
            )
            digests[str(episode_id)] = hashlib.sha256(payload).hexdigest()
            episode, actual = read_episode(
                payload, expected.loc[episode_id, "waveform_sha256"], episode_id, names, RATE
            )
            if names is None:
                names = actual
                relay_buses(names)
            windows, targets = extract_windows_from_dataframe(
                episode, record, int(episode_id), cfg
            )
            if windows.shape[1:] != (samples, 48) or not np.isfinite(windows).all():
                raise ValueError(f"Invalid windows in episode {episode_id}: {windows.shape}")
            if set(targets.fault) != {0, 1}:
                raise ValueError(f"Episode {episode_id} has no positive or negative windows")
            stop = offset + len(windows)
            output[offset:stop] = windows
            offset = stop
            targets["sc_type"] = int(record.sc_type)
            targets["fold"] = int(split.loc[episode_id])
            metadata.append(targets)
            if i % 500 == 0:
                print(f"{window} ms: {i}/{len(labels)} episodes", flush=True)
    finally:
        if archive:
            archive.close()
    output.flush()
    del output
    pd.concat(metadata, ignore_index=True).to_csv(staging / "windows.csv", index=False)
    (staging / "meta.json").write_text(
        json.dumps(
            {
                "window_ms": window,
                "rows": offset,
                "names": names,
                "relay_buses": relay_buses(names),
                "episode_sha256": digests,
                "folds": "GroupKFold(5) on sorted episode IDs",
                "source_commit": SOURCE_BASELINE,
                "dataset_doi": "10.5281/zenodo.21109169",
            },
            indent=2,
        )
    )
    staging.rename(target)
    print(f"Prepared {offset} windows for {window} ms", flush=True)


def masked(windows, condition, value, config, mapping):
    if condition == "complete_input":
        return windows.reshape(len(windows), -1)
    config.data_sparsity = Config(
        downsampling_factor=1,
        current_loss=False,
        voltage_loss=False,
        relay_failure_ids=[0],
        bus_failure_id=0,
        zeroing_duration_s=0,
        phase_failure_id="None",
    )
    if condition == "decimation":
        config.data_sparsity.downsampling_factor = int(value)
        reduced = data_sparsity.downsample_data(windows, config)
        return np.ascontiguousarray(reduced).reshape(len(windows), -1)
    if condition == "missing_current":
        config.data_sparsity.current_loss = True
    if condition == "missing_voltage":
        config.data_sparsity.voltage_loss = True
    if condition == "relay_loss":
        config.data_sparsity.relay_failure_ids = [int(value)]
    if condition == "bus_loss":
        config.data_sparsity.relay_failure_ids = [relay + 1 for relay in mapping[str(value)]]
    if condition == "phase_loss":
        config.data_sparsity.phase_failure_id = value
    if condition == "gap_ms":
        config.data_sparsity.zeroing_duration_s = int(value) / 1000
    return data_sparsity.apply_sparsity_transform(windows, config)[0]


def fit_case(case, cache, out, jobs, smoke=False):
    matrix = pd.read_csv(REFERENCE / "matrix.csv", dtype={"value": str})
    row = matrix.loc[matrix.case.eq(case)].iloc[0]
    window, task, condition, value = int(row.window_ms), row.task, row.condition, row.value
    meta = json.loads((cache / "meta.json").read_text())
    y = pd.read_csv(cache / "windows.csv")
    windows = np.load(cache / "x.npy", mmap_mode="r")[: meta["rows"]]
    if len(windows) != len(y) or (
        not smoke and y.sample_id.nunique() != sum(LABEL_COUNTS.values())
    ):
        raise ValueError("The cache row count or episode population is wrong")
    ids = np.arange(len(y)) if task == "fault" else np.flatnonzero(y.fault.to_numpy() == 1)
    x = masked(
        windows if task == "fault" else windows[ids],
        condition,
        value,
        parameters(window),
        meta["relay_buses"],
    )
    target = y[task].to_numpy()[ids]
    folds, types = y.fold.to_numpy()[ids], y.sc_type.to_numpy()[ids]
    labels = np.unique(target)
    scores = []
    for i in range(1, 6):
        train, test = folds != i, folds == i
        model = create_model_from_name(parameters(window)).set_params(n_jobs=jobs)
        scaler = StandardScaler(copy=False)
        model.fit(scaler.fit_transform(x[train]), target[train])
        prediction = model.predict(scaler.transform(x[test]))
        np.savez_compressed(
            out / f"fold-{i}.npz",
            row=ids[test],
            truth=target[test].astype(str),
            prediction=prediction.astype(str),
        )
        for group, name in [
            (np.ones(test.sum(), dtype=bool), "all"),
            *[(types[test] == code, name) for code, name in FAULT_TYPES.items()],
        ]:
            if not group.any() or (not smoke and set(target[test][group]) != set(labels)):
                raise ValueError(f"Incomplete scoring cohort: {task}, fold {i}, {name}")
            scores.append(
                {
                    "fold": i,
                    "fault_type": name,
                    "windows": int(group.sum()),
                    "macro_f1": f1_score(
                        target[test][group],
                        prediction[group],
                        average="macro",
                        labels=labels,
                        zero_division=0,
                    ),
                }
            )
        pd.DataFrame(scores).to_csv(out / "fold-scores.csv", index=False)
        print(f"{task} {condition} {value}: fold {i}/5 complete", flush=True)


def smoke_checks(data_dir, output_dir, sample_dir):
    prepare(20, data_dir, output_dir, smoke=True, sample_dir=sample_dir)
    cache = output_dir / "smoke-cache" / "20"
    meta = json.loads((cache / "meta.json").read_text())
    x = np.load(cache / "x.npy", mmap_mode="r")[:2]
    for condition, value, expected_zero in [
        ("missing_current", 0, 24),
        ("missing_voltage", 0, 24),
        ("relay_loss", 8, 6),
        ("bus_loss", 2, 24),
        ("phase_loss", "B", 16),
    ]:
        a = masked(
            np.ones_like(x), condition, str(value), parameters(20), meta["relay_buses"]
        ).reshape(x.shape)
        assert (a[0, 0] == 0).sum() == expected_zero
    gap = masked(np.ones_like(x), "gap_ms", "15", parameters(20), meta["relay_buses"]).reshape(
        x.shape
    )
    assert np.flatnonzero((gap[0] == 0).all(axis=1)).tolist() == list(range(12, 108))
    assert masked(x, "decimation", "16", parameters(20), meta["relay_buses"]).shape == (2, 8 * 48)
    print("Verified 20 ms window extraction and all sparsity families", flush=True)
