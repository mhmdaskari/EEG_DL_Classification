from __future__ import annotations

import numpy as np

from eegclassify.config import DataSplitConfig, PreprocessingConfig
from eegclassify.data import DatasetBundle
from eegclassify.preprocessing import augment_trials, labels_to_zero_based, prepare_splits


def _bundle() -> DatasetBundle:
    X_train = np.arange(12 * 22 * 20, dtype=np.float64).reshape(12, 22, 20)
    y_train = np.array([769, 770, 771, 772] * 3, dtype=np.int32)
    person_train = np.array([[0], [0], [1], [1]] * 3, dtype=np.float64)
    X_test = np.arange(4 * 22 * 20, dtype=np.float64).reshape(4, 22, 20)
    y_test = np.array([769, 770, 771, 772], dtype=np.int32)
    person_test = np.array([[0], [0], [1], [1]], dtype=np.float64)
    return DatasetBundle(X_train, y_train, person_train, X_test, y_test, person_test)


def test_labels_to_zero_based():
    np.testing.assert_array_equal(labels_to_zero_based(np.array([769, 770, 771, 772])), np.array([0, 1, 2, 3]))


def test_augment_trials_shape_without_noise():
    X = np.ones((3, 22, 20))
    y = np.array([0, 1, 2])
    config = PreprocessingConfig(max_time_step=20, sub_sample=2, average=2, noise=False)
    X_aug, y_aug = augment_trials(X, y, config, np.random.default_rng(1))
    assert X_aug.shape == (12, 22, 10)
    assert y_aug.shape == (12,)


def test_prepare_splits_all_subjects():
    bundle = _bundle()
    preprocess = PreprocessingConfig(max_time_step=20, noise=False)
    split = DataSplitConfig(validation_ratio=0.25, seed=1)
    prepared = prepare_splits(bundle, preprocess, split)
    assert prepared.x_train.shape[1:] == (10, 1, 22)
    assert prepared.y_train.shape[1] == 4
    assert prepared.x_valid.shape[1:] == (10, 1, 22)
    assert prepared.x_test.shape == (16, 10, 1, 22)


def test_prepare_splits_subject_zero_is_valid_filter():
    bundle = _bundle()
    preprocess = PreprocessingConfig(max_time_step=20, noise=False)
    split = DataSplitConfig(validation_ratio=0.25, subject_id=0, seed=1)
    prepared = prepare_splits(bundle, preprocess, split)
    assert prepared.x_test.shape[0] == 8
