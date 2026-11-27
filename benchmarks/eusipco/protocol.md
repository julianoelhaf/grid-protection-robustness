# PROTECT-90 sparsity evaluation

The v2.0.0 benchmark uses all 9,022 episodes of PROTECT-90 v1.0.0. The four fault
families comprise 2,279 single-line-to-ground (SLG), 2,251 line-to-line (LL),
2,219 double-line-to-ground (LLG), and 2,273 three-phase (LLL) episodes. The
fault-line identification target has four classes corresponding to the four line sections.
The corrected EUSIPCO paper documents the earlier three-phase study. Its historical code
snapshot is identified by `eusipco-arxiv-v2` in the original repository.

[PROTECT-90 v1.0.0, Zenodo](https://doi.org/10.5281/zenodo.21109169) ·
[Corrected EUSIPCO paper](https://arxiv.org/abs/2505.15560v2)

## Protocol

- The benchmark uses 10, 20, 30, 40, and 50 ms windows with a 5 ms stride in a
  ±80 ms interval around fault onset. Detection positives contain the onset with more
  than 1 ms on both sides. Line identification uses these positive windows.
- All cases use the same five episode folds. The scaler fits on training windows
  within each fold. The random forest is refit for each condition.
- At the native 6.4 kHz rate, the restrictions cover missing current or voltage,
  individual relays, physical bus masks based on the channel names, loss of one phase,
  and contiguous gaps that start 10% into the window. Gaps range from 5 to 45 ms in
  5 ms steps and are strictly shorter than their window. Decimation retains every
  second, fourth, eighth, or sixteenth sample without an anti-alias filter. The resulting
  rates are 3.2 kHz, 1.6 kHz, 800 Hz, and 400 Hz.
- Macro F1 is computed on held-out windows overall and per fault family. Fold tables
  show five-fold means and sample SD across folds. Changes are relative to the matching
  complete-input case at the same window length, task, and fault family.

## Reference results

Values below are equal-weight means of the five window-specific cross-validation means.
Relative changes discussed below are averaged over the five matched windows at full precision.

| Condition | Detection macro F1 | Line identification macro F1 |
|---|---:|---:|
| Complete input | 0.9929 | 0.9960 |
| Missing current | 0.9947 | 0.3591 |
| Missing voltage | 0.9923 | 0.9963 |
| 400 Hz decimation | 0.9863 | 0.9828 |

Missing current reduced line identification macro F1 by 63–65% across all four
fault families. At 400 Hz, line identification macro F1 fell by 3.25% for SLG faults,
compared with 1.32% overall. These are results for simulated episodes and the evaluated
refit protocol.

- [Case matrix](../../src/grid_protection_robustness/protocol/matrix.csv) lists all 260 experiment cases.
- [Fold scores](all_fold_scores.csv) contains 6,500 verified scores, covering
  1,300 folds and five reporting groups per fold.
- [Summary table](matrix_summary.csv) contains 1,300 case-by-group summaries.
- [Episode membership and waveform checksums](../../src/grid_protection_robustness/protocol/episode_folds.csv) fixes the
  split and exact inputs for every episode.
- [Provenance manifest](../../src/grid_protection_robustness/protocol/manifest.json) records the source baseline, audited
  outputs, and checksums. The [execution environment](environment.txt)
  lists the package versions used on Linux with Python 3.12.

See the [reproduction guide](../../docs/reproduce.md).
