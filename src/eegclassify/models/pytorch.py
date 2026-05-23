"""PyTorch models for EEG classification."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from ..augmentation import (
    GANAugmentationResult,
    append_synthetic_training_data,
    classifier_images_to_gan_sequences,
    interpolation_labels,
)
from ..config import ModelConfig, TrainingConfig

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None
    TensorDataset = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


def _require_torch() -> None:
    """Raise a clear error if PyTorch is not installed."""

    if torch is None or DataLoader is None or TensorDataset is None:
        raise ImportError("Install PyTorch with `python -m pip install -e .[pytorch]`.")


def _to_nchw(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    return x.permute(0, 3, 1, 2).contiguous()


def _set_torch_seed(seed: int) -> None:
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _CNNBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        _require_torch()
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = config.channels
        for filters in config.cnn_filters:
            layers.extend(
                [
                    nn.Conv2d(in_channels, filters, kernel_size=config.cnn_kernel, padding="same"),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
                    nn.BatchNorm2d(filters),
                    nn.Dropout(config.dropout),
                ]
            )
            in_channels = filters
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(_to_nchw(x) if x.shape[-1] != x.shape[1] else x)


class PyTorchCNN(nn.Module):
    """CNN classifier implemented with PyTorch.

    Args:
        config: Optional architecture configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = _CNNBackbone(self.config)
        with torch.no_grad():
            dummy = torch.zeros(2, self.config.max_time_step // 2, 1, self.config.channels)
            features = self.backbone(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(features, self.config.n_classes)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        return self.classifier(features)


class PyTorchLSTM(nn.Module):
    """Stacked bidirectional LSTM classifier implemented with PyTorch.

    Args:
        config: Optional architecture configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        time_steps = self.config.max_time_step // 2
        self.flatten = nn.Flatten()
        self.project = nn.Linear(time_steps * self.config.channels, time_steps)
        self.lstm1 = nn.LSTM(1, self.config.lstm_units[0], batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.config.lstm_units[0] * 2, self.config.lstm_units[1], batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(self.config.lstm_units[1] * 2, self.config.lstm_units[2], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.config.lstm_units[2] * 2, self.config.n_classes)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        x = self.flatten(x)
        x = self.project(x).unsqueeze(-1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])
        return self.classifier(x)


class PyTorchCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM classifier implemented with PyTorch.

    Args:
        config: Optional architecture configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = _CNNBackbone(self.config)
        with torch.no_grad():
            dummy = torch.zeros(2, self.config.max_time_step // 2, 1, self.config.channels)
            features = self.backbone(dummy).flatten(1).shape[1]
        self.project = nn.Linear(features, 100)
        self.lstm = nn.LSTM(1, self.config.cnn_lstm_units, batch_first=True)
        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(self.config.cnn_lstm_units, self.config.n_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.project(x).unsqueeze(-1)
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x[:, -1, :]))


class PyTorchGenerator(nn.Module):
    """Conditional GAN generator implemented with PyTorch.

    Args:
        config: Optional architecture configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        input_dim = self.config.latent_dim + self.config.n_classes
        self.project = nn.Sequential(nn.Linear(input_dim, 10 * input_dim), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(
            nn.ConvTranspose1d(input_dim, 128, kernel_size=11, stride=5, padding=5, output_padding=4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 128, kernel_size=11, stride=5, padding=5, output_padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, self.config.channels, kernel_size=10, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, z_and_labels):
        x = self.project(z_and_labels).reshape(z_and_labels.shape[0], -1, 10)
        return self.net(x).permute(0, 2, 1)


class PyTorchDiscriminator(nn.Module):
    """Conditional GAN discriminator implemented with PyTorch.

    Args:
        config: Optional architecture configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        in_channels = self.config.channels + self.config.n_classes
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=5, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=9, stride=5, padding=4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x_and_labels):
        return self.net(x_and_labels.permute(0, 2, 1))


def _sequence_labels(labels, time_steps: int):
    labels = labels.float()
    return labels[:, None, :].repeat(1, time_steps, 1)


def build_classifier(name: str, config: ModelConfig | None = None) -> nn.Module:
    """Build a named PyTorch classifier.

    Args:
        name: Classifier name: `cnn`, `lstm`, or `cnn_lstm`.
        config: Optional architecture configuration.

    Returns:
        PyTorch module for the selected classifier.

    Raises:
        ValueError: If the classifier name is unknown.
        ImportError: If PyTorch is not installed.
    """

    _require_torch()
    builders = {"cnn": PyTorchCNN, "lstm": PyTorchLSTM, "cnn_lstm": PyTorchCNNLSTM}
    try:
        return builders[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown PyTorch classifier: {name}") from exc


@dataclass
class PyTorchGANAugmenter:
    """Train a PyTorch conditional GAN and append generated samples.

    The returned arrays use the same classifier layout as `prepare_splits`, so the
    augmentation can be applied before CNN, LSTM, or CNN-LSTM training.

    Attributes:
        training: Training and GAN augmentation configuration.
        model_config: Optional architecture configuration.
        device: Optional device string; defaults to CUDA when available, otherwise CPU.
    """

    training: TrainingConfig
    model_config: ModelConfig | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        _require_torch()
        self.model_config = self.model_config or ModelConfig()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, x: np.ndarray, y: np.ndarray) -> DataLoader:
        dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
        return DataLoader(dataset, batch_size=self.training.gan_batch_size, shuffle=True)

    def augment_training_data(self, x_train: np.ndarray, y_train: np.ndarray) -> GANAugmentationResult:
        """Create GAN-augmented training arrays.

        Args:
            x_train: Real training inputs shaped `(n, time, 1, channels)`.
            y_train: One-hot real training labels.

        Returns:
            Augmented training arrays plus synthetic sample metadata.
        """

        _set_torch_seed(self.training.seed)
        gan_sequences, scale = classifier_images_to_gan_sequences(x_train)
        labels = np.asarray(y_train, dtype=np.float32)
        generator = PyTorchGenerator(self.model_config).to(self.device)
        discriminator = PyTorchDiscriminator(self.model_config).to(self.device)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=self.training.gan_learning_rate)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.training.gan_learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        epochs = 1 if self.training.fast_dev_run else self.training.gan_epochs
        history: list[dict[str, float]] = []

        for _ in range(epochs):
            total_g_loss = 0.0
            total_d_loss = 0.0
            count = 0
            for real_sequences, one_hot_labels in self._loader(gan_sequences, labels):
                real_sequences = real_sequences.to(self.device)
                one_hot_labels = one_hot_labels.to(self.device)
                batch_size, time_steps, _ = real_sequences.shape
                image_labels = _sequence_labels(one_hot_labels, time_steps)

                z = torch.randn(batch_size, self.model_config.latent_dim, device=self.device)
                z_and_labels = torch.cat([z, one_hot_labels], dim=1)
                fake_sequences = generator(z_and_labels)
                fake_and_labels = torch.cat([fake_sequences.detach(), image_labels], dim=-1)
                real_and_labels = torch.cat([real_sequences, image_labels], dim=-1)
                combined = torch.cat([fake_and_labels, real_and_labels], dim=0)
                targets = torch.cat(
                    [torch.ones(batch_size, 1, device=self.device), torch.zeros(batch_size, 1, device=self.device)],
                    dim=0,
                )

                d_optimizer.zero_grad()
                d_loss = criterion(discriminator(combined), targets)
                d_loss.backward()
                d_optimizer.step()

                z = torch.randn(batch_size, self.model_config.latent_dim, device=self.device)
                z_and_labels = torch.cat([z, one_hot_labels], dim=1)
                misleading = torch.zeros(batch_size, 1, device=self.device)
                g_optimizer.zero_grad()
                fake_sequences = generator(z_and_labels)
                fake_and_labels = torch.cat([fake_sequences, image_labels], dim=-1)
                g_loss = criterion(discriminator(fake_and_labels), misleading)
                g_loss.backward()
                g_optimizer.step()

                total_g_loss += float(g_loss.item()) * batch_size
                total_d_loss += float(d_loss.item()) * batch_size
                count += batch_size
            history.append({"g_loss": total_g_loss / count, "d_loss": total_d_loss / count})

        synthetic_labels = interpolation_labels(
            n_classes=self.model_config.n_classes,
            samples_per_class=self.training.gan_samples_per_class,
        )
        generator.eval()
        with torch.no_grad():
            z = torch.randn(synthetic_labels.shape[0], self.model_config.latent_dim, device=self.device)
            label_tensor = torch.as_tensor(synthetic_labels, dtype=torch.float32, device=self.device)
            synthetic_sequences = generator(torch.cat([z, label_tensor], dim=1)).cpu().numpy()
        return append_synthetic_training_data(
            x_train,
            y_train,
            synthetic_sequences,
            synthetic_labels,
            scale,
            history=history,
            metadata={
                "framework": "pytorch",
                "real_samples": int(x_train.shape[0]),
                "synthetic_samples": int(synthetic_labels.shape[0]),
                "gan_epochs": int(epochs),
                "device": str(self.device),
            },
        )


@dataclass
class PyTorchTrainer:
    """Minimal PyTorch trainer for notebook reproduction.

    Attributes:
        training: Training configuration controlling epochs and batch size.
        device: Optional device string; defaults to CUDA when available, otherwise CPU.
    """

    training: TrainingConfig
    device: str | None = None

    def __post_init__(self):
        _require_torch()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, x, y, shuffle: bool) -> DataLoader:
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.long)
        dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), y_tensor)
        return DataLoader(dataset, batch_size=self.training.batch_size, shuffle=shuffle)

    def _loss(self, logits, targets):
        if targets.ndim == 2:
            return -(targets.to(self.device) * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
        return nn.CrossEntropyLoss()(logits, targets.to(self.device))

    def _target_classes(self, targets):
        return targets.argmax(dim=1) if targets.ndim == 2 else targets

    def fit(
        self,
        model: nn.Module,
        x_train: Any,
        y_train: Any,
        x_valid: Any | None = None,
        y_valid: Any | None = None,
    ) -> list[dict[str, float]]:
        """Train a PyTorch classifier.

        Args:
            model: PyTorch classifier.
            x_train: Training inputs.
            y_train: Training labels as class IDs or one-hot rows.
            x_valid: Reserved for API parity; currently unused.
            y_valid: Reserved for API parity; currently unused.

        Returns:
            Per-epoch metrics with loss and accuracy.
        """

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training.learning_rate)
        criterion = nn.CrossEntropyLoss()
        epochs = 1 if self.training.fast_dev_run else self.training.epochs
        history: list[dict[str, float]] = []
        for _ in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            count = 0
            for xb, yb in self._loader(x_train, y_train, shuffle=True):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = self._loss(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * xb.shape[0]
                correct += int((logits.argmax(dim=1) == self._target_classes(yb)).sum().item())
                count += xb.shape[0]
            history.append({"loss": total_loss / count, "accuracy": correct / count})
        return history

    def evaluate(self, model: nn.Module, x: Any, y: Any) -> dict[str, float]:
        """Evaluate a PyTorch classifier.

        Args:
            model: Trained PyTorch classifier.
            x: Evaluation inputs.
            y: Evaluation labels as class IDs or one-hot rows.

        Returns:
            Dictionary with `loss` and `accuracy`.
        """

        model.to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for xb, yb in self._loader(x, y, shuffle=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)
                loss = self._loss(logits, yb)
                total_loss += float(loss.item()) * xb.shape[0]
                correct += int((logits.argmax(dim=1) == self._target_classes(yb)).sum().item())
                count += xb.shape[0]
        return {"loss": total_loss / count, "accuracy": correct / count}
