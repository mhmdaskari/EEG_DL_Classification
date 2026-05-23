# Data

The public source of truth is BCI Competition IV, data set 2a.

- Dataset page: <https://www.bbci.de/competition/iv/>
- Raw GDF archive: <https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip>
- True-label archive: <https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip>

The original notebooks used six processed files:

- `X_train_valid.npy`
- `y_train_valid.npy`
- `person_train_valid.npy`
- `X_test.npy`
- `y_test.npy`
- `person_test.npy`

Post-processed `.npy` files under `data/processed/` are versioned with Git LFS so the
notebooks can run from a fresh clone. Raw archives, extracted GDF files, local caches,
and intermediate conversion folders stay out of Git.

Before adding or updating processed arrays, install Git LFS and initialize it once:

```bash
git lfs install
git lfs track "data/processed/*.npy"
```

The tracked files are:

- `data/processed/X_train_valid.npy`
- `data/processed/y_train_valid.npy`
- `data/processed/person_train_valid.npy`
- `data/processed/X_test.npy`
- `data/processed/y_test.npy`
- `data/processed/person_test.npy`

The processed arrays are intentionally excluded from PyPI distributions. PyPI users
should download or regenerate the data with the commands below.

## Download Raw Data

```bash
python -m eegclassify.cli download-bci2a --raw-dir data/raw
```

## Regenerate The Project Subset

```bash
python -m eegclassify.cli prepare-bci2a \
  --raw-dir data/raw \
  --output-dir data/processed \
  --manifest data/processed/manifest.json
```

The default converter uses the training GDF files `A01T.gdf` through `A09T.gdf`, extracts the first 22 EEG channels, uses 1000 samples starting one sample after cue onset, and applies the built-in course split map inferred from the original `.npy` cache. A trial-block split is still available with `--split-map none`.

## Compare With Local Cache

```bash
python -m eegclassify.cli compare-data \
  --generated-dir data/processed \
  --reference-dir data_temp \
  --output artifacts/data_comparison.json
```

The comparison report includes shapes, dtypes, SHA256 hashes, label counts, subject counts, exact-match flags, and numeric differences.

The notebook `notebooks/00_data_postprocessing.ipynb` runs the same workflow interactively.
