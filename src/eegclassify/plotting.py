"""Plot helpers shared by notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: dict[str, list[float]], title: str) -> Any:
    """Plot training/validation accuracy and loss curves.

    Args:
        history: Keras-like history dictionary with accuracy/loss lists.
        title: Prefix used in subplot titles.

    Returns:
        Matplotlib figure containing accuracy and loss subplots.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("accuracy", []), label="train")
    axes[0].plot(history.get("val_accuracy", []), label="valid")
    axes[0].set_title(f"{title} accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].plot(history.get("loss", []), label="train")
    axes[1].plot(history.get("val_loss", []), label="valid")
    axes[1].set_title(f"{title} loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_channel_class_average(X: np.ndarray, y: np.ndarray, channel: int = 8) -> Any:
    """Plot the class-average signal for one EEG channel.

    Args:
        X: EEG trials shaped `(trials, channels, time)`.
        y: Raw BCI cue labels `769-772`.
        channel: Channel index to visualize.

    Returns:
        Matplotlib figure with one line per class.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = {769: "left", 770: "right", 771: "feet", 772: "tongue"}
    for label, name in labels.items():
        class_data = X[np.asarray(y).reshape(-1) == label, channel, :]
        if class_data.size:
            ax.plot(np.mean(class_data, axis=0), label=name)
    ax.set_xlabel("time step")
    ax.set_ylabel("amplitude")
    ax.legend()
    fig.tight_layout()
    return fig
