"""BCI Competition IV data set 2a download and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import json
import shutil
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

from .constants import (
    BCI2A_GDF_URL,
    BCI2A_TRUE_LABELS_URL,
    CUE_LABELS,
    DEFAULT_TEST_TRIAL_START,
    DEFAULT_TEST_TRIAL_STOP,
    DEFAULT_WINDOW_SAMPLES,
    N_EEG_CHANNELS,
    SAMPLING_RATE_HZ,
)
from .data import DatasetBundle, save_processed_arrays, summarize_processed_dir, write_json_report


@dataclass(frozen=True)
class BCI2AConversionConfig:
    """Parameters for recreating the package processed `.npy` dataset.

    Attributes:
        window_samples: Number of time samples extracted per trial.
        window_offset_samples: Offset from cue onset before extraction starts.
        test_trial_start: First trial ordinal for block-split test mode.
        test_trial_stop: Stop trial ordinal for block-split test mode.
        eeg_channels: Number of EEG channels to extract.
        scale_to_microvolts: Whether to convert MNE's volt values to microvolts.
        quantize_microvolts: Whether to round onto the GDF microvolt grid.
        reject_artifacts: Whether to remove trials containing GDF artifact marker `1023`.
        split_map: Built-in package split, custom split-map JSON path, or `None` for block split.
    """

    window_samples: int = DEFAULT_WINDOW_SAMPLES
    window_offset_samples: int = 1
    test_trial_start: int = DEFAULT_TEST_TRIAL_START
    test_trial_stop: int = DEFAULT_TEST_TRIAL_STOP
    eeg_channels: int = N_EEG_CHANNELS
    scale_to_microvolts: bool = True
    quantize_microvolts: bool = True
    reject_artifacts: bool = False
    split_map: str | Path | None = "builtin"


def download_file(url: str, output_path: Path, force: bool = False) -> Path:
    """Download a URL unless it already exists.

    Args:
        url: Remote file URL.
        output_path: Local destination path.
        force: Redownload even if the destination already exists.

    Returns:
        Path to the downloaded file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    urlretrieve(url, tmp_path)
    tmp_path.replace(output_path)
    return output_path


def download_bci2a(raw_dir: Path | str = Path("data/raw"), force: bool = False) -> dict[str, Path]:
    """Download and extract official BCI Competition IV data set 2a archives.

    Args:
        raw_dir: Destination directory for archives and extracted files.
        force: Redownload archives even when they already exist.

    Returns:
        Paths to the downloaded archives and raw directory.
    """

    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    gdf_zip = download_file(BCI2A_GDF_URL, raw_root / "BCICIV_2a_gdf.zip", force=force)
    labels_zip = download_file(BCI2A_TRUE_LABELS_URL, raw_root / "true_labels.zip", force=force)
    for archive in (gdf_zip, labels_zip):
        with ZipFile(archive) as zipped:
            zipped.extractall(raw_root)
    return {"gdf_zip": gdf_zip, "true_labels_zip": labels_zip, "raw_dir": raw_root}


def copy_local_cache_to_processed(
    cache_dir: Path | str,
    processed_dir: Path | str = Path("data/processed"),
    manifest_path: Path | str | None = Path("data/processed/manifest.json"),
) -> None:
    """Copy an existing local `.npy` cache into the processed data directory.

    Args:
        cache_dir: Directory containing the six processed `.npy` files.
        processed_dir: Destination directory.
        manifest_path: Optional manifest path written after copying.
    """

    from .constants import PROCESSED_FILENAMES

    source = Path(cache_dir)
    destination = Path(processed_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in PROCESSED_FILENAMES:
        shutil.copy2(source / name, destination / name)
    if manifest_path is not None:
        write_json_report(summarize_processed_dir(destination), manifest_path)


def _import_mne():
    try:
        import mne  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BCI IV-2a conversion requires MNE. Install with `python -m pip install -e .[data]`."
        ) from exc
    return mne


def _event_code_maps(event_id: dict[str, int]) -> dict[int, int]:
    """Map MNE event integer IDs back to numeric GDF annotation codes."""

    mapped: dict[int, int] = {}
    for description, event_value in event_id.items():
        try:
            code = int(description)
        except ValueError:
            digits = "".join(ch for ch in description if ch.isdigit())
            if not digits:
                continue
            code = int(digits)
        mapped[int(event_value)] = code
    return mapped


def _extract_subject_trials(
    gdf_path: Path,
    subject_id: int,
    config: BCI2AConversionConfig,
) -> list[tuple[int, np.ndarray, int, int]]:
    mne = _import_mne()
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != SAMPLING_RATE_HZ:
        raise ValueError(f"Expected {SAMPLING_RATE_HZ} Hz data in {gdf_path}, got {raw.info['sfreq']}.")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    event_code_by_value = _event_code_maps(event_id)
    event_codes = np.array([event_code_by_value.get(int(value), int(value)) for value in events[:, 2]])

    trial_start_indices = np.where(event_codes == 768)[0]
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)[: config.eeg_channels]
    if len(eeg_picks) < config.eeg_channels:
        eeg_picks = np.arange(config.eeg_channels)

    trials: list[tuple[int, np.ndarray, int, int]] = []

    for trial_ordinal, event_position in enumerate(trial_start_indices):
        start_sample = int(events[event_position, 0])
        if trial_ordinal + 1 < len(trial_start_indices):
            stop_event_position = trial_start_indices[trial_ordinal + 1]
            stop_sample = int(events[stop_event_position, 0])
        else:
            stop_sample = raw.n_times

        interval_mask = (events[:, 0] >= start_sample) & (events[:, 0] < stop_sample)
        interval_codes = event_codes[interval_mask]
        interval_events = events[interval_mask]
        if config.reject_artifacts and 1023 in set(int(code) for code in interval_codes):
            continue

        cue_positions = np.where(np.isin(interval_codes, CUE_LABELS))[0]
        if cue_positions.size == 0:
            continue
        cue_index = int(cue_positions[0])
        label = int(interval_codes[cue_index])
        cue_sample = int(interval_events[cue_index, 0])
        window_start = cue_sample + config.window_offset_samples
        window_stop = window_start + config.window_samples
        if window_start < 0 or window_stop > raw.n_times:
            continue

        segment = raw.get_data(picks=eeg_picks, start=window_start, stop=window_stop)
        if config.scale_to_microvolts:
            segment = segment * 1e6
        if config.quantize_microvolts:
            segment = np.round(segment * 2048.0) / 2048.0
        segment = np.nan_to_num(segment, copy=False).astype(np.float64)

        trials.append((trial_ordinal, segment, label, subject_id))

    return trials


def _extract_subject_training_file(
    gdf_path: Path,
    subject_id: int,
    config: BCI2AConversionConfig,
) -> tuple[list[np.ndarray], list[int], list[int], list[np.ndarray], list[int], list[int]]:
    trials = _extract_subject_trials(gdf_path, subject_id, config)
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    train_person: list[int] = []
    test_X: list[np.ndarray] = []
    test_y: list[int] = []
    test_person: list[int] = []
    for trial_ordinal, segment, label, person in trials:
        if config.test_trial_start <= trial_ordinal < config.test_trial_stop:
            test_X.append(segment)
            test_y.append(label)
            test_person.append(person)
        else:
            train_X.append(segment)
            train_y.append(label)
            train_person.append(person)
    return train_X, train_y, train_person, test_X, test_y, test_person


def _load_split_map(split_map: str | Path | None) -> dict | None:
    if split_map is None:
        return None
    if split_map == "builtin":
        resource = resources.files("eegclassify.resources").joinpath("bci2a_course_split.json")
        with resource.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with Path(split_map).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_bci2a_training_subset(
    raw_dir: Path | str = Path("data/raw"),
    output_dir: Path | str = Path("data/processed"),
    config: BCI2AConversionConfig | None = None,
    manifest_path: Path | str | None = Path("data/processed/manifest.json"),
) -> DatasetBundle:
    """Convert official training GDF files into the six project `.npy` arrays.

    Args:
        raw_dir: Directory containing extracted BCI IV-2a `A01T.gdf` through `A09T.gdf`.
        output_dir: Destination directory for generated `.npy` files.
        config: Conversion settings. The default recreates the course project subset.
        manifest_path: Optional JSON manifest path.

    Returns:
        Generated data as a `DatasetBundle`.

    Raises:
        FileNotFoundError: If a required subject training GDF file is missing.
    """

    config = config or BCI2AConversionConfig()
    raw_root = Path(raw_dir)
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    train_person: list[int] = []
    test_X: list[np.ndarray] = []
    test_y: list[int] = []
    test_person: list[int] = []
    split_map = _load_split_map(config.split_map)

    if split_map is None:
        for subject_id in range(9):
            gdf_path = raw_root / f"A0{subject_id + 1}T.gdf"
            if not gdf_path.exists():
                candidates = list(raw_root.rglob(f"A0{subject_id + 1}T.gdf"))
                if not candidates:
                    raise FileNotFoundError(f"Could not find {gdf_path}. Run the download step first.")
                gdf_path = candidates[0]
            parts = _extract_subject_training_file(gdf_path, subject_id, config)
            sx, sy, sp, tx, ty, tp = parts
            train_X.extend(sx)
            train_y.extend(sy)
            train_person.extend(sp)
            test_X.extend(tx)
            test_y.extend(ty)
            test_person.extend(tp)
    else:
        trial_lookup: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
        for subject_id in range(9):
            gdf_path = raw_root / f"A0{subject_id + 1}T.gdf"
            if not gdf_path.exists():
                candidates = list(raw_root.rglob(f"A0{subject_id + 1}T.gdf"))
                if not candidates:
                    raise FileNotFoundError(f"Could not find {gdf_path}. Run the download step first.")
                gdf_path = candidates[0]
            trial_lookup.update(
                {
                    (subject_id, trial_ordinal): (segment, label, person)
                    for trial_ordinal, segment, label, person in _extract_subject_trials(gdf_path, subject_id, config)
                }
            )
        for item in split_map["train_valid"]:
            segment, label, person = trial_lookup[(int(item["subject"]), int(item["trial"]))]
            train_X.append(segment)
            train_y.append(label)
            train_person.append(person)
        for item in split_map["test"]:
            segment, label, person = trial_lookup[(int(item["subject"]), int(item["trial"]))]
            test_X.append(segment)
            test_y.append(label)
            test_person.append(person)

    bundle = DatasetBundle(
        X_train_valid=np.stack(train_X).astype(np.float64),
        y_train_valid=np.asarray(train_y, dtype=np.int32),
        person_train_valid=np.asarray(train_person, dtype=np.float64).reshape(-1, 1),
        X_test=np.stack(test_X).astype(np.float64),
        y_test=np.asarray(test_y, dtype=np.int32),
        person_test=np.asarray(test_person, dtype=np.float64).reshape(-1, 1),
    )
    save_processed_arrays(bundle, output_dir)
    if manifest_path is not None:
        write_json_report(summarize_processed_dir(output_dir), manifest_path)
    return bundle
