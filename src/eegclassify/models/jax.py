"""JAX/Flax models for EEG classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..augmentation import (
    GANAugmentationResult,
    append_synthetic_training_data,
    classifier_images_to_gan_sequences,
    interpolation_labels,
)
from ..config import ModelConfig, TrainingConfig

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import optax
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    optax = None

    class _MissingFlax:
        class Module:
            pass

        @staticmethod
        def compact(func):
            return func

    nn = _MissingFlax()


def _require_jax() -> None:
    """Raise a clear error if JAX/Flax/Optax are not installed."""

    if jax is None or jnp is None or optax is None:
        raise ImportError("Install JAX support with `python -m pip install -e .[jax]`.")


class FlaxCNN(nn.Module):
    """CNN classifier implemented with Flax.

    Attributes:
        config: Architecture configuration.
    """

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, x, train: bool = False):
        for filters in self.config.cnn_filters:
            x = nn.Conv(filters, self.config.cnn_kernel, padding="SAME")(x)
            x = nn.elu(x)
            x = nn.max_pool(x, window_shape=(3, 1), strides=(3, 1), padding="SAME")
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.Dropout(self.config.dropout, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(self.config.n_classes)(x)


class _SimpleLSTM(nn.Module):
    features: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x):
        if self.reverse:
            x = jnp.flip(x, axis=1)
        cell = nn.LSTMCell(features=self.features)
        carry = cell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0], x.shape[-1]))
        outputs = []
        for t in range(x.shape[1]):
            carry, y = cell(carry, x[:, t, :])
            outputs.append(y)
        y = jnp.stack(outputs, axis=1)
        return jnp.flip(y, axis=1) if self.reverse else y


class _BiLSTM(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        forward = _SimpleLSTM(self.features)(x)
        backward = _SimpleLSTM(self.features, reverse=True)(x)
        return jnp.concatenate([forward, backward], axis=-1)


class FlaxLSTM(nn.Module):
    """Stacked bidirectional LSTM classifier implemented with Flax.

    Attributes:
        config: Architecture configuration.
    """

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, x, train: bool = False):
        batch = x.shape[0]
        x = x.reshape((batch, -1))
        x = nn.Dense(self.config.max_time_step // 2)(x)
        x = x.reshape((batch, self.config.max_time_step // 2, 1))
        for units in self.config.lstm_units:
            x = _BiLSTM(units)(x)
        x = x[:, -1, :]
        x = nn.Dropout(0.4, deterministic=not train)(x)
        return nn.Dense(self.config.n_classes)(x)


class FlaxCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM classifier implemented with Flax.

    Attributes:
        config: Architecture configuration.
    """

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, x, train: bool = False):
        for filters in self.config.cnn_filters:
            x = nn.Conv(filters, self.config.cnn_kernel, padding="SAME")(x)
            x = nn.elu(x)
            x = nn.max_pool(x, window_shape=(3, 1), strides=(3, 1), padding="SAME")
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.Dropout(self.config.dropout, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(100)(x).reshape((x.shape[0], 100, 1))
        x = _SimpleLSTM(self.config.cnn_lstm_units)(x)[:, -1, :]
        x = nn.Dropout(0.6, deterministic=not train)(x)
        return nn.Dense(self.config.n_classes)(x)


class FlaxGenerator(nn.Module):
    """Conditional GAN generator implemented with Flax.

    Attributes:
        config: Architecture configuration.
    """

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, z_and_labels):
        input_dim = self.config.latent_dim + self.config.n_classes
        x = nn.Dense(10 * input_dim)(z_and_labels)
        x = nn.leaky_relu(x, 0.2)
        x = x.reshape((x.shape[0], 10, input_dim))
        x = nn.ConvTranspose(128, (11,), strides=(5,), padding="SAME")(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.ConvTranspose(128, (11,), strides=(5,), padding="SAME")(x)
        x = nn.leaky_relu(x, 0.2)
        return nn.sigmoid(nn.Conv(self.config.channels, (10,), padding="SAME")(x))


class FlaxDiscriminator(nn.Module):
    """Conditional GAN discriminator implemented with Flax.

    Attributes:
        config: Architecture configuration.
    """

    config: ModelConfig = ModelConfig()

    @nn.compact
    def __call__(self, x_and_labels):
        x = nn.Conv(64, (9,), strides=(5,), padding="SAME")(x_and_labels)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Conv(128, (9,), strides=(5,), padding="SAME")(x)
        x = nn.leaky_relu(x, 0.2)
        x = jnp.max(x, axis=1)
        return nn.Dense(1)(x)


def build_classifier(name: str, config: ModelConfig | None = None) -> nn.Module:
    """Build a named JAX/Flax classifier.

    Args:
        name: Classifier name: `cnn`, `lstm`, or `cnn_lstm`.
        config: Optional architecture configuration.

    Returns:
        Flax module for the selected classifier.

    Raises:
        ValueError: If the classifier name is unknown.
        ImportError: If JAX/Flax/Optax are not installed.
    """

    _require_jax()
    config = config or ModelConfig()
    builders = {"cnn": FlaxCNN, "lstm": FlaxLSTM, "cnn_lstm": FlaxCNNLSTM}
    try:
        return builders[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown JAX classifier: {name}") from exc


def _sequence_labels(labels: Any, time_steps: int) -> Any:
    return jnp.repeat(labels[:, None, :], repeats=time_steps, axis=1)


@dataclass
class JAXGANAugmenter:
    """Train a JAX/Flax conditional GAN and append generated samples.

    The augmenter is independent of the downstream classifier and can be used before
    CNN, LSTM, or CNN-LSTM training.

    Attributes:
        training: Training and GAN augmentation configuration.
        model_config: Optional architecture configuration.
    """

    training: TrainingConfig
    model_config: ModelConfig | None = None

    def __post_init__(self) -> None:
        _require_jax()
        self.model_config = self.model_config or ModelConfig()

    def augment_training_data(self, x_train: np.ndarray, y_train: np.ndarray) -> GANAugmentationResult:
        """Create GAN-augmented training arrays.

        Args:
            x_train: Real training inputs shaped `(n, time, 1, channels)`.
            y_train: One-hot real training labels.

        Returns:
            Augmented training arrays plus synthetic sample metadata.
        """

        _require_jax()
        gan_sequences, scale = classifier_images_to_gan_sequences(x_train)
        labels = np.asarray(y_train, dtype=np.float32)
        generator = FlaxGenerator(self.model_config)
        discriminator = FlaxDiscriminator(self.model_config)
        rng = jax.random.PRNGKey(self.training.seed)
        rng, gen_key, disc_key = jax.random.split(rng, 3)
        gen_vars = generator.init(gen_key, jnp.zeros((1, self.model_config.latent_dim + self.model_config.n_classes)))
        disc_vars = discriminator.init(
            disc_key,
            jnp.zeros((1, gan_sequences.shape[1], self.model_config.channels + self.model_config.n_classes)),
        )
        gen_params = gen_vars["params"]
        disc_params = disc_vars["params"]
        gen_optimizer = optax.adam(self.training.gan_learning_rate)
        disc_optimizer = optax.adam(self.training.gan_learning_rate)
        gen_opt_state = gen_optimizer.init(gen_params)
        disc_opt_state = disc_optimizer.init(disc_params)
        epochs = 1 if self.training.fast_dev_run else self.training.gan_epochs
        history: list[dict[str, float]] = []
        np_rng = np.random.default_rng(self.training.seed)

        for _ in range(epochs):
            order = np_rng.permutation(gan_sequences.shape[0])
            total_g_loss = 0.0
            total_d_loss = 0.0
            count = 0
            for start in range(0, order.shape[0], self.training.gan_batch_size):
                batch_idx = order[start : start + self.training.gan_batch_size]
                real_sequences = jnp.asarray(gan_sequences[batch_idx], dtype=jnp.float32)
                one_hot_labels = jnp.asarray(labels[batch_idx], dtype=jnp.float32)
                batch_size = real_sequences.shape[0]
                image_labels = _sequence_labels(one_hot_labels, real_sequences.shape[1])
                rng, z_key, g_key = jax.random.split(rng, 3)

                def discriminator_loss(params):
                    z = jax.random.normal(z_key, (batch_size, self.model_config.latent_dim))
                    fake_sequences = generator.apply(
                        {"params": gen_params},
                        jnp.concatenate([z, one_hot_labels], axis=1),
                    )
                    fake_and_labels = jnp.concatenate([fake_sequences, image_labels], axis=-1)
                    real_and_labels = jnp.concatenate([real_sequences, image_labels], axis=-1)
                    combined = jnp.concatenate([fake_and_labels, real_and_labels], axis=0)
                    targets = jnp.concatenate([jnp.ones((batch_size, 1)), jnp.zeros((batch_size, 1))], axis=0)
                    logits = discriminator.apply({"params": params}, combined)
                    return optax.sigmoid_binary_cross_entropy(logits, targets).mean()

                d_loss, d_grads = jax.value_and_grad(discriminator_loss)(disc_params)
                disc_updates, disc_opt_state = disc_optimizer.update(d_grads, disc_opt_state, disc_params)
                disc_params = optax.apply_updates(disc_params, disc_updates)

                def generator_loss(params):
                    z = jax.random.normal(g_key, (batch_size, self.model_config.latent_dim))
                    fake_sequences = generator.apply(
                        {"params": params},
                        jnp.concatenate([z, one_hot_labels], axis=1),
                    )
                    fake_and_labels = jnp.concatenate([fake_sequences, image_labels], axis=-1)
                    misleading = jnp.zeros((batch_size, 1))
                    logits = discriminator.apply({"params": disc_params}, fake_and_labels)
                    return optax.sigmoid_binary_cross_entropy(logits, misleading).mean()

                g_loss, g_grads = jax.value_and_grad(generator_loss)(gen_params)
                gen_updates, gen_opt_state = gen_optimizer.update(g_grads, gen_opt_state, gen_params)
                gen_params = optax.apply_updates(gen_params, gen_updates)

                total_g_loss += float(g_loss) * batch_size
                total_d_loss += float(d_loss) * batch_size
                count += batch_size
            history.append({"g_loss": total_g_loss / count, "d_loss": total_d_loss / count})

        synthetic_labels = interpolation_labels(
            n_classes=self.model_config.n_classes,
            samples_per_class=self.training.gan_samples_per_class,
        )
        rng, sample_key = jax.random.split(rng)
        z = jax.random.normal(sample_key, (synthetic_labels.shape[0], self.model_config.latent_dim))
        synthetic_sequences = generator.apply(
            {"params": gen_params},
            jnp.concatenate([z, jnp.asarray(synthetic_labels, dtype=jnp.float32)], axis=1),
        )
        return append_synthetic_training_data(
            x_train,
            y_train,
            np.asarray(synthetic_sequences),
            synthetic_labels,
            scale,
            history=history,
            metadata={
                "framework": "jax",
                "real_samples": int(x_train.shape[0]),
                "synthetic_samples": int(synthetic_labels.shape[0]),
                "gan_epochs": int(epochs),
            },
        )


@dataclass
class FlaxTrainState:
    """Small training-state container for Flax examples.

    Attributes:
        params: Model parameters.
        batch_stats: Optional batch-normalization statistics.
        optimizer_state: Optax optimizer state.
    """

    params: dict
    batch_stats: dict | None
    optimizer_state: optax.OptState


@dataclass
class JAXTrainer:
    """Minimal JAX/Flax trainer for notebook reproduction.

    Attributes:
        training: Training configuration controlling epochs and learning rate.
    """

    training: TrainingConfig
    state: FlaxTrainState | None = field(default=None, init=False)

    def init_state(self, model: nn.Module, rng: Any, sample_batch: Any) -> FlaxTrainState:
        """Initialize model parameters and optimizer state.

        Args:
            model: Flax model.
            rng: JAX random key.
            sample_batch: Example input batch used for shape inference.

        Returns:
            Initialized training state.
        """

        _require_jax()
        variables = model.init({"params": rng, "dropout": rng}, sample_batch, train=True)
        optimizer = optax.adam(self.training.learning_rate)
        params = variables["params"]
        batch_stats = variables.get("batch_stats")
        return FlaxTrainState(params=params, batch_stats=batch_stats, optimizer_state=optimizer.init(params))

    def train_step(
        self,
        model: nn.Module,
        state: FlaxTrainState,
        x: Any,
        y: Any,
        rng: Any,
    ) -> tuple[FlaxTrainState, dict[str, float]]:
        """Run one full-batch optimization step.

        Args:
            model: Flax model.
            state: Current training state.
            x: Input batch.
            y: One-hot labels.
            rng: JAX random key for dropout.

        Returns:
            Updated state and scalar metrics.
        """

        _require_jax()
        optimizer = optax.adam(self.training.learning_rate)

        def loss_fn(params):
            variables = {"params": params}
            mutable = []
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats
                mutable = ["batch_stats"]
            outputs = model.apply(variables, x, train=True, rngs={"dropout": rng}, mutable=mutable)
            if mutable:
                logits, updates = outputs
            else:
                logits, updates = outputs, {}
            labels = jnp.argmax(y, axis=1)
            loss = optax.softmax_cross_entropy(logits, y).mean()
            accuracy = (jnp.argmax(logits, axis=1) == labels).mean()
            return loss, (accuracy, updates)

        (loss, (accuracy, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates_opt, optimizer_state = optimizer.update(grads, state.optimizer_state, state.params)
        params = optax.apply_updates(state.params, updates_opt)
        batch_stats = updates.get("batch_stats", state.batch_stats)
        return FlaxTrainState(params, batch_stats, optimizer_state), {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }

    def fit(self, model: nn.Module, x_train: np.ndarray, y_train: np.ndarray) -> list[dict[str, float]]:
        """Train a Flax classifier.

        Args:
            model: Flax classifier.
            x_train: Training inputs.
            y_train: One-hot training labels.

        Returns:
            Per-epoch metrics with loss and accuracy.
        """

        _require_jax()
        rng = jax.random.PRNGKey(self.training.seed)
        state = self.init_state(model, rng, jnp.asarray(x_train[:1], dtype=jnp.float32))
        epochs = 1 if self.training.fast_dev_run else self.training.epochs
        np_rng = np.random.default_rng(self.training.seed)
        history: list[dict[str, float]] = []
        for _ in range(epochs):
            order = np_rng.permutation(x_train.shape[0])
            total_loss = 0.0
            total_accuracy = 0.0
            count = 0
            for start in range(0, order.shape[0], self.training.batch_size):
                batch_idx = order[start : start + self.training.batch_size]
                rng, step_rng = jax.random.split(rng)
                state, metrics = self.train_step(
                    model,
                    state,
                    jnp.asarray(x_train[batch_idx], dtype=jnp.float32),
                    jnp.asarray(y_train[batch_idx], dtype=jnp.float32),
                    step_rng,
                )
                batch_size = len(batch_idx)
                total_loss += metrics["loss"] * batch_size
                total_accuracy += metrics["accuracy"] * batch_size
                count += batch_size
            history.append({"loss": total_loss / count, "accuracy": total_accuracy / count})
        self.state = state
        return history

    def evaluate(self, model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate a trained Flax classifier.

        Args:
            model: Flax classifier.
            x: Evaluation inputs.
            y: One-hot or soft-label evaluation targets.

        Returns:
            Dictionary with `loss` and `accuracy`.

        Raises:
            RuntimeError: If `fit` has not been called and no state is available.
        """

        _require_jax()
        if self.state is None:
            raise RuntimeError("Call `fit` before `evaluate` so model parameters are available.")
        variables = {"params": self.state.params}
        if self.state.batch_stats is not None:
            variables["batch_stats"] = self.state.batch_stats
        total_loss = 0.0
        total_accuracy = 0.0
        count = 0
        for start in range(0, x.shape[0], self.training.batch_size):
            xb = jnp.asarray(x[start : start + self.training.batch_size], dtype=jnp.float32)
            yb = jnp.asarray(y[start : start + self.training.batch_size], dtype=jnp.float32)
            logits = model.apply(variables, xb, train=False)
            labels = jnp.argmax(yb, axis=1)
            loss = optax.softmax_cross_entropy(logits, yb).mean()
            accuracy = (jnp.argmax(logits, axis=1) == labels).mean()
            batch_size = xb.shape[0]
            total_loss += float(loss) * batch_size
            total_accuracy += float(accuracy) * batch_size
            count += batch_size
        return {"loss": total_loss / count, "accuracy": total_accuracy / count}
