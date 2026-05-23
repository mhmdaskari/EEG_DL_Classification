from __future__ import annotations

from pathlib import Path

import numpy as np

from eegclassify.data import DatasetBundle, compare_processed_dirs, save_processed_arrays, summarize_processed_dir


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        X_train_valid=np.zeros((2, 22, 1000), dtype=np.float64),
        y_train_valid=np.array([769, 770], dtype=np.int32),
        person_train_valid=np.array([[0], [1]], dtype=np.float64),
        X_test=np.ones((1, 22, 1000), dtype=np.float64),
        y_test=np.array([771], dtype=np.int32),
        person_test=np.array([[0]], dtype=np.float64),
    )


def test_summary_and_compare(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    save_processed_arrays(_bundle(), left)
    save_processed_arrays(_bundle(), right)
    summary = summarize_processed_dir(left)
    assert summary["files"]["X_train_valid.npy"]["shape"] == [2, 22, 1000]
    report = compare_processed_dirs(left, right)
    assert report["all_files_match"] is True
    assert report["files"]["y_train_valid.npy"]["exact_match"] is True
