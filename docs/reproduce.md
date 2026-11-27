# Reproduce EUSIPCO v2.0.0

Use CPython 3.12 on Linux x86_64. Download wheels while online, then use the same
wheelhouse and environment on every compute node.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
mkdir -p wheelhouse
bash tools/build-antlr-wheel.sh wheelhouse
python -m pip download --find-links wheelhouse --require-hashes --only-binary=:all: -r requirements.lock -d wheelhouse
python -m pip install --no-index --find-links wheelhouse --require-hashes -r requirements.lock
python -m pip wheel --no-build-isolation --no-deps --no-index -w dist .
python -m pip install --no-index --no-deps dist/grid_protection_robustness-2.0.0-py3-none-any.whl
```

The locked ANTLR wheel is built from the upstream 4.9.3 source distribution.
The bootstrap checks the source hash and builds with the locked tooling. The
resulting wheel must match the hash in `requirements.lock`. On Woody, stage the official PROTECT-90 v1.0.0 labels
and ZIP before submitting work. Keep the source ZIP and extracted caches outside
Git. The preparation step validates every episode's SHA-256 before loading its
pickle. Check the downloaded archives against the official file hashes.

```bash
cd /path/to/staged-data
md5sum -c <<'EOF'
5f015330f77ed53b76bd5db26e83c48d  hv_double_line_90kv_labels.csv
7cf176f169299b825ba6a6be102edca8  hv_double_line_90kv_preprocessed_data.zip
EOF
```

From an allocated compute node, run the 20-episode sample in its own directory.
Both tasks run with complete input and phase-A loss.

```bash
grid-protection-robustness eusipco smoke --run /scratch/eusipco-smoke --data /path/to/staged-data --jobs 4
```

Run the five preparation tasks on Slurm. Set the resource flags and working
directory for your site. The preparation launcher requests 16 GiB. Wait for preparation and inspect the manifests before
submitting the case array.

```bash
sbatch slurm/prepare.sbatch "$(command -v grid-protection-robustness)" /scratch/eusipco /path/to/staged-data
bash slurm/submit-remaining.sh /scratch/eusipco
grid-protection-robustness eusipco summarize --run /scratch/eusipco
```

The default retry throttle is four. A new submission includes only incomplete
cases, excludes pending cases in this launcher's Slurm arrays, and skips active
case locks. Damaged results are reported separately. Once a case is inactive,
quarantine it with `grid-protection-robustness eusipco repair-case CASE --run /scratch/eusipco`.
The next submission can retry that case. If a window's manifest is unavailable,
restore it with `prepare`; submissions for other windows can continue.
Duplicate tasks cannot publish a second result. Summarization reads
the case receipts across arrays. The output directory is bound to the installed
code, protocol, episode split, data identity and environment. Leave the run
manifest, completed cases, and cache manifests in place. Window CSV files retain
the row order needed for prediction verification. Before reclaiming space,
wait for fits using the window to finish. You can then remove just that window's
`x.npy`. Run `prepare` for the same window and run directory to restore the array
after matching its checksum against the retained manifest.

Before extracting the supplement, compare its SHA-256 against the release's
`SHA256SUMS.txt`. Then invoke the installed verifier, which also checks the
prediction inventory against its pinned hash.

```bash
grid-protection-robustness eusipco verify --supplement /path/to/extracted-supplement
```

The verifier checks the pinned fold assignment, row uniqueness, eligible
held-out rows, ground-truth labels, fault families and all reported macro F1
scores. It regenerates both published tables and compares their SHA-256
digests. The supplement contains prediction arrays, row maps and original
execution provenance. It excludes waveform files, derived arrays, and logs.
