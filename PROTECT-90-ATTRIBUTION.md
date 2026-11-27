# PROTECT-90 attribution

*PROTECT-90: A Fault Dataset for Power System Protection*, v1.0.0.
Copyright © 2026 The Authors. Credit Georg Kordowich, Julian Oelhaf,
Christian Bergler, Andreas Maier, Siming Bayer, and Johann Jäger.
The dataset is licensed under Creative Commons Attribution 4.0 International.
[Dataset and DOI 10.5281/zenodo.21109169](https://zenodo.org/records/21109169) ·
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

## Material used and changes

- The packaged `episode_folds.csv` selects episode IDs, fault types and
  faulted-line labels from the dataset metadata. The benchmark adds fixed
  fold assignments and waveform checksums.
- The supplement's window row maps select and repeat episode metadata for
  the eligible windows and add window start times, detection labels and
  fold assignments. Held-out truth arrays use the corresponding line or
  detection labels. Prediction arrays and scores are benchmark outputs.
- The source and evidence downloads contain derived metadata and checksums.
  The waveforms are available from the dataset deposit.

The MIT notice in the release source and in the historical
`supplement/execution/source.tar` applies to the software. PROTECT-90
source material is credited and licensed as described above.
