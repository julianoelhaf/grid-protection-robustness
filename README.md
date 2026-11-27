# Grid Protection Robustness

How well can a model detect a fault and identify the faulted line when grid measurements go missing? This repository benchmarks random forests under reduced sampling rates and loss of current, voltage, phases, relays, buses, or time segments.

**EUSIPCO v2.0.0** extends the corrected EUSIPCO 2025 study from three-phase faults to all four fault families in PROTECT-90. It evaluates **9,022 simulated episodes**, **260 cases**, and **1,300 fits** across five fixed episode-level folds. [Paper (arXiv v2)](https://arxiv.org/abs/2505.15560v2) · [Dataset](https://zenodo.org/records/21109169) · [Protocol](benchmarks/eusipco/protocol.md)

DPSP Global 2026 is planned as a separately versioned benchmark. [DPSP release status](benchmarks/dpsp/README.md)

## Results

| Measurements | Fault detection | Faulted-line identification |
| --- | ---: | ---: |
| Complete input | 0.9929 | 0.9960 |
| Without current | 0.9947 | 0.3591 |
| Without voltage | 0.9923 | 0.9963 |
| Downsampled to 400 Hz | 0.9863 | 0.9828 |

Macro F1 averaged across the five window lengths, with each condition refit on the same episode splits. Losing current had little effect on detection in this simulation and sharply reduced line identification. [Full results](benchmarks/eusipco/matrix_summary.csv) · [Fold scores](benchmarks/eusipco/all_fold_scores.csv)

## Reproduce

Use the locked Python 3.12 environment on Linux. Download the PROTECT-90 labels and waveform ZIP into one directory, then run a 20-episode smoke test covering all four fault families. [Setup, full Slurm run, and evidence verification](docs/reproduce.md)

```bash
grid-protection-robustness eusipco smoke --data /path/to/protect90 --run ./runs/smoke
```

## Citation and licenses

Cite the EUSIPCO conference paper. Use the corrected arXiv v2 for the experimental details. [Paper (arXiv v2)](https://arxiv.org/abs/2505.15560v2) · [Citation metadata](CITATION.cff)

```bibtex
@inproceedings{oelhaf2025sparsity,
  author    = {Oelhaf, Julian and Kordowich, Georg and Kim, Changhun and
               P{\'e}rez-Toro, Paula Andrea and Maier, Andreas and
               J{\"a}ger, Johann and Bayer, Siming},
  title     = {Impact of Data Sparsity on Machine Learning for Fault Detection
               in Power System Protection},
  booktitle = {2025 33rd European Signal Processing Conference (EUSIPCO)},
  year      = {2025},
  pages     = {1997--2001},
  doi       = {10.23919/EUSIPCO63237.2025.11226584}
}
```

Also cite [PROTECT-90 v1.0.0](https://zenodo.org/records/21109169) when using the benchmark data. The software uses the [MIT License](LICENSE).

PROTECT-90 © 2026 Georg Kordowich, Julian Oelhaf, Christian Bergler, Andreas Maier, Siming Bayer, and Johann Jäger. The benchmark selects episode metadata, adds fold assignments and waveform checksums, and derives window-level labels and row maps. The underlying dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
