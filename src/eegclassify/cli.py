"""Command line interface for data and experiment utilities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import platform
from pathlib import Path
import site
import sys

from .bci2a import (
    BCI2AConversionConfig,
    convert_bci2a_training_subset,
    copy_local_cache_to_processed,
    download_bci2a,
)
from .config import DataSplitConfig, ModelConfig, PreprocessingConfig, TrainingConfig
from .data import compare_processed_dirs, summarize_processed_dir, write_json_report
from .data import load_processed_arrays
from .preprocessing import prepare_splits


def _site_package_roots() -> list[Path]:
    roots = [Path(site.getusersitepackages())]
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    return roots


def _tensorflow_nvidia_lib_dirs() -> list[Path]:
    lib_dirs: list[Path] = []
    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        lib_dirs.extend(sorted(path for path in nvidia_root.glob("*/lib") if path.exists()))
    return lib_dirs


def _tensorflow_cuda_data_dir() -> Path | None:
    for root in _site_package_roots():
        for candidate in (root / "nvidia" / "cuda_nvcc", root / "nvidia" / "cu13"):
            if (candidate / "nvvm" / "libdevice" / "libdevice.10.bc").exists():
                return candidate
    return None


def _ensure_tensorflow_gpu_environment(raw_argv: list[str]) -> None:
    """Restart the TensorFlow CLI once with pip-installed CUDA paths visible.

    TensorFlow's Linux GPU wheels load NVIDIA shared libraries at process startup.
    When the CUDA libraries come from pip extras, the CLI needs those directories on
    `LD_LIBRARY_PATH` before importing TensorFlow. The XLA CUDA data directory is also
    needed on RTX 50-series GPUs so PTX JIT compilation can find `libdevice`.
    """

    if os.environ.get("EEGCLASSIFY_DISABLE_TF_GPU_ENV") == "1":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return
    if os.environ.get("EEGCLASSIFY_TF_GPU_ENV_READY") == "1":
        return

    lib_dirs = _tensorflow_nvidia_lib_dirs()
    cuda_data_dir = _tensorflow_cuda_data_dir()
    if not lib_dirs and cuda_data_dir is None:
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    current_ld_parts = [part for part in current_ld.split(":") if part]
    updated_ld_parts = [str(path) for path in lib_dirs if str(path) not in current_ld_parts] + current_ld_parts
    if updated_ld_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(updated_ld_parts)

    if cuda_data_dir is not None and "xla_gpu_cuda_data_dir" not in os.environ.get("XLA_FLAGS", ""):
        current_xla = os.environ.get("XLA_FLAGS", "")
        xla_flag = f"--xla_gpu_cuda_data_dir={cuda_data_dir}"
        os.environ["XLA_FLAGS"] = f"{current_xla} {xla_flag}".strip()

    os.environ["EEGCLASSIFY_TF_GPU_ENV_READY"] = "1"
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        module_args = sys.argv[1:]
    else:
        module_args = ["train", *raw_argv]
    os.execvpe(sys.executable, [sys.executable, "-m", "eegclassify.cli", *module_args], os.environ)


def download_bci2a_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download official BCI Competition IV data set 2a archives.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    paths = download_bci2a(args.raw_dir, force=args.force)
    for key, value in paths.items():
        print(f"{key}: {value}")


def prepare_bci2a_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert BCI IV-2a GDF files into project `.npy` arrays.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument("--manifest", default="data/processed/manifest.json", type=Path)
    parser.add_argument("--window-samples", default=1000, type=int)
    parser.add_argument("--window-offset-samples", default=1, type=int)
    parser.add_argument("--test-trial-start", default=200, type=int)
    parser.add_argument("--test-trial-stop", default=250, type=int)
    parser.add_argument(
        "--split-map",
        default="builtin",
        help="Use `builtin` for the course split map, a JSON path, or `none` for trial-block splitting.",
    )
    parser.add_argument("--reject-artifacts", action="store_true")
    args = parser.parse_args(argv)
    config = BCI2AConversionConfig(
        window_samples=args.window_samples,
        window_offset_samples=args.window_offset_samples,
        test_trial_start=args.test_trial_start,
        test_trial_stop=args.test_trial_stop,
        reject_artifacts=args.reject_artifacts,
        split_map=None if args.split_map.lower() == "none" else args.split_map,
    )
    bundle = convert_bci2a_training_subset(args.raw_dir, args.output_dir, config, args.manifest)
    print(f"X_train_valid: {bundle.X_train_valid.shape}")
    print(f"X_test: {bundle.X_test.shape}")
    print(f"manifest: {args.manifest}")


def compare_data_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare generated processed arrays with a reference cache.")
    parser.add_argument("--generated-dir", default="data/processed", type=Path)
    parser.add_argument("--reference-dir", default="data_temp", type=Path)
    parser.add_argument("--output", default="artifacts/data_comparison.json", type=Path)
    args = parser.parse_args(argv)
    report = compare_processed_dirs(args.generated_dir, args.reference_dir)
    write_json_report(report, args.output)
    print(f"all_files_match: {report['all_files_match']}")
    print(f"report: {args.output}")


def summarize_data_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a manifest for processed EEG arrays.")
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--output", default="data/processed/manifest.json", type=Path)
    args = parser.parse_args(argv)
    write_json_report(summarize_processed_dir(args.data_dir), args.output)
    print(f"manifest: {args.output}")


def copy_cache_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy existing local `.npy` cache files into data/processed.")
    parser.add_argument("--cache-dir", default="data_temp", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--manifest", default="data/processed/manifest.json", type=Path)
    args = parser.parse_args(argv)
    copy_local_cache_to_processed(args.cache_dir, args.processed_dir, args.manifest)
    print(f"processed_dir: {args.processed_dir}")


def _json_history(history) -> list[dict[str, float]]:
    if hasattr(history, "history"):
        values = history.history
        length = len(next(iter(values.values()), []))
        return [{key: float(series[index]) for key, series in values.items()} for index in range(length)]
    return [{key: float(value) for key, value in row.items()} for row in history]


def train_main(argv: list[str] | None = None) -> None:
    """Run a local EEG classification experiment from the command line."""

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Train an eegclassify model and write a metrics report.")
    parser.add_argument("--framework", choices=("tensorflow", "pytorch", "jax"), required=True)
    parser.add_argument("--model", choices=("cnn", "lstm", "cnn_lstm", "gan_cnn"), default="cnn")
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--output-dir", default="artifacts/runs", type=Path)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gan-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gan-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gan-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gan-samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--use-gan-augmentation", action="store_true")
    args = parser.parse_args(argv)
    if args.framework == "tensorflow":
        _ensure_tensorflow_gpu_environment(raw_argv)

    model_name = "cnn" if args.model == "gan_cnn" else args.model
    use_gan_augmentation = args.use_gan_augmentation or args.model == "gan_cnn"
    preprocess = PreprocessingConfig(seed=args.seed)
    split = DataSplitConfig(validation_ratio=args.validation_ratio, subject_id=args.subject_id, seed=args.seed)
    training = TrainingConfig(
        epochs=args.epochs,
        gan_epochs=args.gan_epochs,
        batch_size=args.batch_size,
        gan_batch_size=args.gan_batch_size,
        learning_rate=args.learning_rate,
        gan_learning_rate=args.gan_learning_rate,
        use_gan_augmentation=use_gan_augmentation,
        gan_samples_per_class=args.gan_samples_per_class,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )
    model_config = ModelConfig(max_time_step=preprocess.max_time_step)
    bundle = load_processed_arrays(args.data_dir)
    prepared = prepare_splits(bundle, preprocess, split)
    x_train = prepared.x_train
    y_train = prepared.y_train
    gan_metadata = None
    gan_history: list[dict[str, float]] = []

    if args.framework == "tensorflow":
        from .models.tensorflow import TensorFlowClassifierFactory, TensorFlowGANAugmenter, TensorFlowTrainer

        if training.use_gan_augmentation:
            result = TensorFlowGANAugmenter(training, model_config).augment_training_data(x_train, y_train)
            x_train, y_train = result.x_train, result.y_train
            gan_metadata = result.metadata
            gan_history = result.history
        model = TensorFlowClassifierFactory(model_config).build(model_name, learning_rate=training.learning_rate)
        trainer = TensorFlowTrainer(training)
        classifier_history = _json_history(
            trainer.fit(model, x_train, y_train, prepared.x_valid, prepared.y_valid)
        )
        metrics = trainer.evaluate(model, prepared.x_test, prepared.y_test)
    elif args.framework == "pytorch":
        from .models.pytorch import PyTorchGANAugmenter, PyTorchTrainer, build_classifier

        if training.use_gan_augmentation:
            result = PyTorchGANAugmenter(training, model_config).augment_training_data(x_train, y_train)
            x_train, y_train = result.x_train, result.y_train
            gan_metadata = result.metadata
            gan_history = result.history
        model = build_classifier(model_name, model_config)
        trainer = PyTorchTrainer(training)
        classifier_history = _json_history(trainer.fit(model, x_train, y_train, prepared.x_valid, prepared.y_valid))
        metrics = trainer.evaluate(model, prepared.x_test, prepared.y_test)
    else:
        from .models.jax import JAXGANAugmenter, JAXTrainer, build_classifier

        if training.use_gan_augmentation:
            result = JAXGANAugmenter(training, model_config).augment_training_data(x_train, y_train)
            x_train, y_train = result.x_train, result.y_train
            gan_metadata = result.metadata
            gan_history = result.history
        model = build_classifier(model_name, model_config)
        trainer = JAXTrainer(training)
        classifier_history = _json_history(trainer.fit(model, x_train, y_train))
        metrics = trainer.evaluate(model, prepared.x_test, prepared.y_test)

    experiment_name = "gan_cnn" if training.use_gan_augmentation and model_name == "cnn" else model_name
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id / args.framework / experiment_name
    report = {
        "framework": args.framework,
        "model": model_name,
        "experiment": experiment_name,
        "metrics": metrics,
        "classifier_history": classifier_history,
        "gan_history": gan_history,
        "gan_metadata": gan_metadata,
        "config": {
            "training": training.__dict__,
            "preprocessing": preprocess.__dict__,
            "split": split.__dict__,
            "model": model_config.__dict__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    write_json_report(report, run_dir / "metrics.json")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"metrics: {run_dir / 'metrics.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eegclassify")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-bci2a")
    subparsers.add_parser("prepare-bci2a")
    subparsers.add_parser("compare-data")
    subparsers.add_parser("summarize-data")
    subparsers.add_parser("copy-cache")
    subparsers.add_parser("train")
    known, remaining = parser.parse_known_args(argv)
    if known.command == "download-bci2a":
        download_bci2a_main(remaining)
    elif known.command == "prepare-bci2a":
        prepare_bci2a_main(remaining)
    elif known.command == "compare-data":
        compare_data_main(remaining)
    elif known.command == "summarize-data":
        summarize_data_main(remaining)
    elif known.command == "copy-cache":
        copy_cache_main(remaining)
    elif known.command == "train":
        train_main(remaining)


if __name__ == "__main__":
    main()
