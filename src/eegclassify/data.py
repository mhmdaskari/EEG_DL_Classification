"""Load, save, summarize, and compare processed EEG arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

from .constants import (
    BCI2A_GDF_URL,
    BCI2A_SOURCE_PAGE,
    BCI2A_TRUE_LABELS_URL,
    EXPECTED_PROCESSED_SHAPES,
    LOCAL_CACHE_CANDIDATES,
    PROCESSED_FILENAMES,
)


@dataclass
class DatasetBundle:
    """Container for the six processed arrays used by the original notebooks.

    Attributes:
        X_train_valid: Training/validation EEG trials with shape `(n, 22, 1000)`.
        y_train_valid: BCI cue labels for `X_train_valid`, using labels `769-772`.
        person_train_valid: Subject IDs for `X_train_valid`, shaped `(n, 1)`.
        X_test: Held-out EEG trials with shape `(m, 22, 1000)`.
        y_test: BCI cue labels for `X_test`, using labels `769-772`.
        person_test: Subject IDs for `X_test`, shaped `(m, 1)`.
    """

    X_train_valid: np.ndarray
    y_train_valid: np.ndarray
    person_train_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    person_test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return arrays keyed by their canonical `.npy` filenames.

        Returns:
            Mapping from processed filename to array.
        """
        return {
            "X_train_valid.npy": self.X_train_valid,
            "y_train_valid.npy": self.y_train_valid,
            "person_train_valid.npy": self.person_train_valid,
            "X_test.npy": self.X_test,
            "y_test.npy": self.y_test,
            "person_test.npy": self.person_test,
        }


def find_processed_data_dir(candidates: tuple[Path, ...] = LOCAL_CACHE_CANDIDATES) -> Path:
    """Find a directory containing all six processed data files.

    Args:
        candidates: Candidate directories to inspect in order.

    Returns:
        First directory that contains every canonical processed `.npy` file.

    Raises:
        FileNotFoundError: If no candidate contains the complete processed dataset.
    """

    for candidate in candidates:
        if all((candidate / name).exists() for name in PROCESSED_FILENAMES):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find processed EEG arrays. Searched: {searched}")


def load_processed_arrays(data_dir: Path | str | None = None, mmap_mode: str | None = None) -> DatasetBundle:
    """Load processed project arrays.

    Args:
        data_dir: Directory containing the six `.npy` files. If omitted, known local cache
            locations are searched.
        mmap_mode: Optional NumPy memory-map mode such as `"r"`.

    Returns:
        A `DatasetBundle` with train/validation and test arrays.
    """

    root = Path(data_dir) if data_dir is not None else find_processed_data_dir()
    arrays = {name: np.load(root / name, mmap_mode=mmap_mode) for name in PROCESSED_FILENAMES}
    return DatasetBundle(
        X_train_valid=arrays["X_train_valid.npy"],
        y_train_valid=arrays["y_train_valid.npy"],
        person_train_valid=arrays["person_train_valid.npy"],
        X_test=arrays["X_test.npy"],
        y_test=arrays["y_test.npy"],
        person_test=arrays["person_test.npy"],
    )


def save_processed_arrays(bundle: DatasetBundle, output_dir: Path | str) -> None:
    """Save a processed data bundle in the original six-file layout.

    Args:
        bundle: Arrays to save.
        output_dir: Destination directory.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, array in bundle.as_dict().items():
        np.save(root / name, array)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 hash for a file.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read per chunk.

    Returns:
        Hex-encoded SHA256 digest.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def summarize_processed_dir(data_dir: Path | str, include_hashes: bool = True) -> dict[str, Any]:
    """Summarize processed arrays for manifests and provenance reports.

    Args:
        data_dir: Directory containing processed `.npy` arrays.
        include_hashes: Whether to compute SHA256 hashes for each file.

    Returns:
        JSON-serializable manifest with source URLs, shapes, dtypes, counts, and hashes.
    """

    root = Path(data_dir)
    summary: dict[str, Any] = {
        "data_dir": str(root),
        "source": {
            "dataset_page": BCI2A_SOURCE_PAGE,
            "gdf_archive": BCI2A_GDF_URL,
            "true_labels_archive": BCI2A_TRUE_LABELS_URL,
        },
        "files": {},
    }
    for name in PROCESSED_FILENAMES:
        path = root / name
        if not path.exists():
            summary["files"][name] = {"exists": False}
            continue
        array = np.load(path, mmap_mode="r")
        file_summary: dict[str, Any] = {
            "exists": True,
            "shape": list(array.shape),
            "expected_shape": list(EXPECTED_PROCESSED_SHAPES.get(name, ())),
            "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size,
        }
        if include_hashes:
            file_summary["sha256"] = sha256_file(path)
        if name.startswith("y_") or name.startswith("person_"):
            file_summary["counts"] = _counts(array)
        summary["files"][name] = file_summary
    return summary


def compare_processed_dirs(
    generated_dir: Path | str,
    reference_dir: Path | str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare generated processed arrays with a reference directory.

    Args:
        generated_dir: Directory containing regenerated `.npy` files.
        reference_dir: Directory containing reference/cache `.npy` files.
        atol: Absolute tolerance for floating-point comparisons.
        rtol: Relative tolerance for floating-point comparisons.

    Returns:
        JSON-serializable comparison report with hashes, shapes, counts, and numeric
        difference metrics.
    """

    generated = Path(generated_dir)
    reference = Path(reference_dir)
    report: dict[str, Any] = {
        "generated_dir": str(generated),
        "reference_dir": str(reference),
        "atol": atol,
        "rtol": rtol,
        "files": {},
    }
    all_match = True
    for name in PROCESSED_FILENAMES:
        generated_path = generated / name
        reference_path = reference / name
        item: dict[str, Any] = {
            "generated_exists": generated_path.exists(),
            "reference_exists": reference_path.exists(),
        }
        if not generated_path.exists() or not reference_path.exists():
            item["status"] = "missing"
            all_match = False
            report["files"][name] = item
            continue

        generated_array = np.load(generated_path, mmap_mode="r")
        reference_array = np.load(reference_path, mmap_mode="r")
        same_shape = generated_array.shape == reference_array.shape
        item.update(
            {
                "generated_shape": list(generated_array.shape),
                "reference_shape": list(reference_array.shape),
                "generated_dtype": str(generated_array.dtype),
                "reference_dtype": str(reference_array.dtype),
                "generated_sha256": sha256_file(generated_path),
                "reference_sha256": sha256_file(reference_path),
                "same_shape": same_shape,
                "same_dtype": generated_array.dtype == reference_array.dtype,
            }
        )
        if same_shape:
            exact = bool(np.array_equal(generated_array, reference_array))
            close = bool(np.allclose(generated_array, reference_array, atol=atol, rtol=rtol))
            diff = np.asarray(generated_array) - np.asarray(reference_array)
            item.update(
                {
                    "exact_match": exact,
                    "allclose": close,
                    "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                    "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
                }
            )
            if not (exact or close):
                all_match = False
        else:
            item["exact_match"] = False
            item["allclose"] = False
            all_match = False
        if name.startswith("y_") or name.startswith("person_"):
            item["generated_counts"] = _counts(generated_array)
            item["reference_counts"] = _counts(reference_array)
        report["files"][name] = item
    report["all_files_match"] = all_match
    return report


def write_json_report(report: dict[str, Any], output_path: Path | str) -> None:
    """Write a JSON report with stable formatting.

    Args:
        report: JSON-serializable report object.
        output_path: Destination JSON path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
