"""Bindings and atomic case publication for the installed EUSIPCO runner."""

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .audit import read_rows, verify_case
from .jobs import claim, publish_json, read_json
from .run_matrix import LABEL_FILE, PROTOCOL, REFERENCE, WAVEFORM_FILE, fit_case, prepare
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


def seal_cache(root, cache, window, run_id, smoke=False):
    read_rows(cache / "windows.csv", smoke)
    path = root / "cache-manifests" / f"{window}.json"
    files = {name: digest(cache / name) for name in ("windows.csv", "meta.json")}
    if path.is_file():
        expected = read_json(path)
        if {**expected, "run": run_id, "window": window,
            "sha256": {**expected["sha256"], **files}} != expected:
            raise ValueError(f"Cache manifest differs: {cache}")
        if (cache_stat(cache) != expected["array_stat"]
                and digest(cache / "x.npy") != expected["sha256"]["x.npy"]):
            raise ValueError(f"Cache array changed: {cache}")
    else:
        files["x.npy"] = digest(cache / "x.npy")
        publish_json(path, {
            "run": run_id, "window": window, "sha256": files, "array_stat": cache_stat(cache)
        })
    return digest(path)


def cache_stat(cache):
    status = (cache / "x.npy").stat()
    return {"bytes": status.st_size, "mtime_ns": status.st_mtime_ns}


def prepare_case_inputs(root, data, window):
    with claim(root, f"prepare-{window}", wait=True):
        run_id = bind_run(root, "matrix")
        # Each episode is checked against its pinned raw-member SHA-256 before decoding.
        prepare(window, data, root)
        seal_cache(root, root / "cache" / str(window), window, run_id)


def read_receipt(root, case):
    folder = root / "results" / f"{case:03d}"
    if not folder.exists():
        return None
    receipt = read_json(folder / "STATUS.json")
    if (receipt["case"] != case or receipt["status"] != "complete"
            or receipt["run"] != digest(root / "run.json")
            or receipt["cache"] != digest(root / "cache-manifests" / f"{receipt['window_ms']}.json")):
        raise ValueError(f"Invalid receipt: {folder}")
    for name, hashed in receipt["sha256"].items():
        if digest(folder / name) != hashed:
            raise ValueError(f"Invalid case output: {folder / name}")
    if set(receipt["sha256"]) != {"fold-scores.csv", *(f"fold-{i}.npz" for i in range(1, 6))}:
        raise ValueError(f"Incomplete case receipt: {folder}")
    return receipt


def run_case(root, case, jobs, smoke=False):
    case_row = next(row for row in matrix().itertuples(index=False) if row.case == case)
    with claim(root, f"case-{case}") as acquired:
        if not acquired:
            print(f"Case {case} is active.", file=sys.stderr)
            return
        run_id = bind_run(root, "smoke" if smoke else "matrix")
        if read_receipt(root, case):
            print(f"Case {case} is complete.")
            return
        cache = root / ("smoke-cache" if smoke else "cache") / str(case_row.window_ms)
        cache_id = seal_cache(root, cache, case_row.window_ms, run_id, smoke)
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
    for case in matrix().case:
        if read_receipt(root, case):
            continue
        with claim(root, f"case-{case}") as available:
            if available and case not in active:
                pending.append(str(case))
            else:
                print(f"Case {case} is active.", file=sys.stderr)
    if pending:
        print(("," if array else "\n").join(pending))
