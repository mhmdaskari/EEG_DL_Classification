"""Project constants and public dataset locations."""

from __future__ import annotations

from pathlib import Path

BCI2A_SOURCE_PAGE = "https://www.bbci.de/competition/iv/"
BCI2A_GDF_URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
BCI2A_TRUE_LABELS_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"

LOCAL_CACHE_CANDIDATES = (
    Path("data/processed"),
    Path("data_temp"),
    Path("/home/mammad/Documents/EDU/Projects/drive-download-20260523T061119Z-3-001"),
)

PROCESSED_FILENAMES = (
    "X_train_valid.npy",
    "y_train_valid.npy",
    "person_train_valid.npy",
    "X_test.npy",
    "y_test.npy",
    "person_test.npy",
)

EXPECTED_PROCESSED_SHAPES = {
    "X_train_valid.npy": (2115, 22, 1000),
    "y_train_valid.npy": (2115,),
    "person_train_valid.npy": (2115, 1),
    "X_test.npy": (443, 22, 1000),
    "y_test.npy": (443,),
    "person_test.npy": (443, 1),
}

CUE_LABELS = (769, 770, 771, 772)
N_CLASSES = 4
N_EEG_CHANNELS = 22
SAMPLING_RATE_HZ = 250
DEFAULT_WINDOW_SAMPLES = 1000
DEFAULT_TEST_TRIAL_START = 200
DEFAULT_TEST_TRIAL_STOP = 250
