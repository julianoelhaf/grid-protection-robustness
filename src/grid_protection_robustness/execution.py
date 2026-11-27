"""Bindings and atomic case publication for the installed EUSIPCO runner."""

import importlib.metadata
import os
import platform
import sys
from shutil import rmtree
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import pandas as pd

from .audit import read_rows, verify_case
from .jobs import claim, publish_json, read_json
from .run_matrix import (
    LABEL_FILE, PROTOCOL, RATE, REFERENCE, WAVEFORM_FILE, fit_case, prepare, relay_buses,
)
from .waveforms import digest


def matrix():
    return pd.read_csv(REFERENCE / "matrix.csv", dtype={"value": str})


def binding():
    source = Path(__file__).resolve().parent
    package_files = {
        f"{package}/{path.relative_to(source.parent / package).as_posix()}": digest(path)
        for package in (source.name, "sparse_fault_detect")
        for path in sorted((source.parent / package).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    return {
        "study": "eusipco",
        "benchmark_version": "2.0.0",
        "code": package_files,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.machine(),
            "system": platform.system(),
            "libc": platform.libc_ver(),
            "distributions": sorted({
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
            }),
        },
        "protocol": {path.name: digest(path) for path in sorted(REFERENCE.iterdir())},
        "input": {
            "dataset_version": PROTOCOL["dataset_version"],
            "dataset_doi": PROTOCOL["dataset_doi"],
            LABEL_FILE: PROTOCOL["labels_md5"],
            WAVEFORM_FILE: PROTOCOL["waveform_zip_md5"],
        },
    }


def bind_run(root, mode):
    path = root / "run.json"
    publish_json(path, {**binding(), "mode": mode})
    return digest(path)


def cache_manifest(root, window):
    return root / "cache-manifests" / f"{window}.json"


def cache_stat(cache):
    status = (cache / "x.npy").stat()
    return {
        "bytes": status.st_size, "inode": status.st_ino,
        "mtime_ns": status.st_mtime_ns, "ctime_ns": status.st_ctime_ns,
    }


def describe_cache(cache, window, run_id, smoke):
    rows = read_rows(cache / "windows.csv", smoke)
    meta = read_json(cache / "meta.json")
    episodes = pd.read_csv(REFERENCE / "episode_folds.csv").set_index("episode_id")
    waveforms = episodes.loc[rows.sample_id.unique(), "waveform_sha256"]
    expected_hashes = dict(zip(waveforms.index.astype(str), waveforms))
    buses = {str(bus): relays for bus, relays in relay_buses(meta["names"]).items()}
    if (meta["window_ms"] != window or meta["rows"] != len(rows)
            or meta["dataset_doi"] != PROTOCOL["dataset_doi"]
            or meta["relay_buses"] != buses or meta["episode_sha256"] != expected_hashes):
        raise ValueError(f"Invalid cache metadata: {cache}")

    array = np.load(cache / "x.npy", mmap_mode="r", allow_pickle=False)
    capacity = len(waveforms) * ((160 - window) // 5 + 2)
    if (array.shape != (capacity, round(window * RATE / 1000), 48)
            or array.dtype != np.float32 or len(rows) > capacity):
        raise ValueError(f"Invalid cache array shape or type: {cache}")

    files = {name: digest(cache / name) for name in ("windows.csv", "meta.json", "x.npy")}
    if not smoke and files["windows.csv"] != PROTOCOL["executed_window_sha256"][
        f"cache/{window}/windows.csv"
    ]:
        raise ValueError(f"Cache row map differs from the pinned {window} ms map")
    return {"run": run_id, "window": window, "sha256": files}


def check_prepared(cache, window, run_id, manifest):
    if not manifest.is_file():
        raise ValueError(f"Cache is unsealed: {cache}. Run prepare first.")
    recorded = read_json(manifest)
    if recorded["run"] != run_id or recorded["window"] != window:
        raise ValueError(f"Cache belongs to another run or window: {cache}")
    hashes = recorded["sha256"]
    for name in ("windows.csv", "meta.json"):
        if digest(cache / name) != hashes[name]:
            raise ValueError(f"Cache file changed: {cache / name}")
    stamp = read_json(cache / "array-stat.json")
    if stamp["sha256"] != hashes["x.npy"] or (
        stamp["stat"] != cache_stat(cache) and digest(cache / "x.npy") != hashes["x.npy"]
    ):
        raise ValueError(f"Cache array changed: {cache}")
    return recorded


def prepare_case_inputs(root, data, window, smoke=False, sample_dir=None):
    with claim(root, f"prepare-{window}", wait=True):
        run_id = bind_run(root, "smoke" if smoke else "matrix")
        base = root / ("smoke-cache" if smoke else "cache")
        base.mkdir(exist_ok=True)
        cache = base / str(window)
        manifest = cache_manifest(root, window)
        # Under the window lock, any abandoned stage for this window has no writer.
        for abandoned in base.glob(f".{window}-partial-*"):
            if abandoned.is_dir() and not abandoned.is_symlink():
                rmtree(abandoned)
        array = cache / "x.npy"
        if not array.exists():
            with TemporaryDirectory(prefix=f".{window}-partial-", dir=base) as temp:
                staging = Path(temp) / "cache"
                staging.mkdir()
                prepare(window, data, staging, smoke, sample_dir)
                recorded = describe_cache(staging, window, run_id, smoke)
                if manifest.exists() and read_json(manifest) != recorded:
                    raise ValueError(f"Regenerated cache differs from its run manifest: {cache}")
                if cache.exists():
                    if cache.is_symlink() or read_json(cache / "prepared.json") != recorded:
                        raise ValueError(f"Regenerated cache differs from its prepared manifest: {cache}")
                    for name in ("meta.json", "windows.csv"):
                        if digest(cache / name) != recorded["sha256"][name]:
                            raise ValueError(f"Retained cache file changed: {cache / name}")
                    # Publish the validated array independently of the retained manifests.
                    (staging / "x.npy").replace(array)
                    publish_json(staging / "array-stat.json", {
                        "stat": cache_stat(cache), "sha256": recorded["sha256"]["x.npy"],
                    })
                    (staging / "array-stat.json").replace(cache / "array-stat.json")
                else:
                    publish_json(staging / "prepared.json", recorded)
                    publish_json(staging / "array-stat.json", {
                        "stat": cache_stat(staging), "sha256": recorded["sha256"]["x.npy"],
                    })
                    staging.rename(cache)
        recorded = check_prepared(cache, window, run_id, cache / "prepared.json")
        publish_json(manifest, recorded)
        return cache


class DamagedCase(ValueError):
    pass


class UnavailableWindow(ValueError):
    def __init__(self, window):
        super().__init__(f"Cache manifest unavailable for {window} ms. Run prepare for that window.")


def read_receipt(root, case_row):
    folder = root / "results" / f"{case_row.case:03d}"
    if not folder.exists():
        return None
    run_id = digest(root / "run.json")
    try:
        manifest_id = digest(cache_manifest(root, case_row.window_ms))
    except OSError as exc:
        raise UnavailableWindow(case_row.window_ms) from exc
    expected = {"fold-scores.csv", *(f"fold-{i}.npz" for i in range(1, 6))}
    try:
        receipt = read_json(folder / "STATUS.json")
        if (receipt["status"] != "complete" or receipt["run"] != run_id
                or receipt["cache"] != manifest_id
                or any(receipt[key] != value for key, value in case_row._asdict().items())
                or set(receipt["sha256"]) != expected):
            raise ValueError("Receipt fields or file inventory differ")
        for name in expected:
            if digest(folder / name) != receipt["sha256"][name]:
                raise ValueError(f"Checksum mismatch: {name}")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise DamagedCase(f"Damaged case {case_row.case}: {exc}") from exc
    return receipt


def repair_case(root, case):
    row = next(row for row in matrix().itertuples(index=False) if row.case == case)
    with claim(root, f"case-{case}") as acquired:
        if not acquired:
            raise ValueError(f"Case {case} is active.")
        try:
            read_receipt(root, row)
        except DamagedCase:
            quarantine = root / "quarantine"
            quarantine.mkdir(exist_ok=True)
            destination = Path(mkdtemp(prefix=f"{case:03d}-", dir=quarantine)) / "case"
            (root / "results" / f"{case:03d}").rename(destination)
            print(f"Quarantined case {case} at {destination}")
            return
        raise ValueError(f"Case {case} has no damaged result to quarantine.")


def run_case(root, case, jobs, smoke=False):
    case_row = next(row for row in matrix().itertuples(index=False) if row.case == case)
    with claim(root, f"case-{case}") as acquired:
        if not acquired:
            print(f"Case {case} is active.", file=sys.stderr)
            return
        run_id = bind_run(root, "smoke" if smoke else "matrix")
        if read_receipt(root, case_row):
            print(f"Case {case} is complete.")
            return
        cache = root / ("smoke-cache" if smoke else "cache") / str(case_row.window_ms)
        manifest = cache_manifest(root, case_row.window_ms)
        check_prepared(cache, case_row.window_ms, run_id, manifest)
        cache_id = digest(manifest)
        results = root / "results"
        results.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=f".{case:03d}-partial-", dir=results) as staging:
            # Rename a child so TemporaryDirectory can still clean up normally.
            out = Path(staging) / "case"
            out.mkdir()
            fit_case(case, cache, out, jobs, smoke)
            rows = read_rows(cache / "windows.csv", smoke)
            verify_case(out, case_row, rows, smoke)
            files = {path.name: digest(path) for path in out.iterdir()}
            publish_json(out / "STATUS.json", {
                "status": "complete", **case_row._asdict(), "run": run_id,
                "cache": cache_id, "sha256": files,
                "workers": jobs, "slurm_job": os.getenv("SLURM_JOB_ID"),
                "slurm_array": os.getenv("SLURM_ARRAY_JOB_ID"),
            })
            out.rename(results / f"{case:03d}")


def remaining(root, active, array):
    pending = []
    unavailable = {}
    initialized = (root / "run.json").is_file()
    for row in matrix().itertuples(index=False):
        case, window = row.case, row.window_ms
        if window in unavailable:
            continue
        if initialized and not cache_manifest(root, window).is_file():
            unavailable[window] = UnavailableWindow(window)
            continue
        with claim(root, f"case-{case}") as available:
            if not available or case in active:
                print(f"Case {case} is active.", file=sys.stderr)
                continue
            try:
                if read_receipt(root, row) is None:
                    pending.append((case, window))
            except UnavailableWindow as exc:
                unavailable[window] = exc
            except DamagedCase as exc:
                print(f"{exc}. Use repair-case {case} before retrying.", file=sys.stderr)
    for error in unavailable.values():
        print(error, file=sys.stderr)
    eligible = [str(case) for case, window in pending if window not in unavailable]
    if eligible:
        print(("," if array else "\n").join(eligible))
