"""TensorFlow/Keras models for EEG classification."""

from __future__ import annotations

from dataclasses import dataclass
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
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing.
    tf = None
    keras = None
    layers = None


def _require_tensorflow() -> None:
    """Raise a clear error if TensorFlow is not installed."""

    if keras is None or layers is None or tf is None:
        raise ImportError("Install TensorFlow with `python -m pip install -e .[tensorflow]`.")


class TensorFlowClassifierFactory:
    """Build Keras classifiers matching the original notebook architectures.

    Args:
        model_config: Optional architecture configuration.
    """

    def __init__(self, model_config: ModelConfig | None = None):
        _require_tensorflow()
        self.config = model_config or ModelConfig()

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.config.max_time_step // 2, 1, self.config.channels)

    def _compile(self, model: keras.Model, learning_rate: float) -> keras.Model:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        return model

    def _cnn_backbone(self, model: keras.Sequential) -> keras.Sequential:
        for index, filters in enumerate(self.config.cnn_filters):
            kwargs = {"input_shape": self.input_shape} if index == 0 else {}
            model.add(
                layers.Conv2D(
                    filters=filters,
                    kernel_size=self.config.cnn_kernel,
                    padding="same",
                    activation="elu",
                    **kwargs,
                )
            )
            model.add(layers.MaxPooling2D(pool_size=(3, 1), padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.config.dropout))
        return model

    def build_cnn(self, learning_rate: float = 1e-3) -> keras.Model:
        """Build and compile the CNN classifier.

        Args:
            learning_rate: Adam optimizer learning rate.

        Returns:
            Compiled Keras model.
        """

        model = keras.Sequential(name="eeg_cnn")
        self._cnn_backbone(model)
        model.add(layers.Flatten())
        model.add(layers.Dense(self.config.n_classes, activation="softmax"))
        return self._compile(model, learning_rate)

    def build_lstm(self, learning_rate: float = 1e-3) -> keras.Model:
        """Build and compile the naive LSTM classifier.

        Args:
            learning_rate: Adam optimizer learning rate.

        Returns:
            Compiled Keras model.
        """

        time_steps = self.config.max_time_step // 2
        model = keras.Sequential(name="eeg_lstm")
        model.add(layers.Flatten(input_shape=self.input_shape))
        model.add(layers.Dense(time_steps))
        model.add(layers.Reshape((time_steps, 1)))
        model.add(layers.Bidirectional(layers.LSTM(self.config.lstm_units[0], return_sequences=True)))
        model.add(layers.Bidirectional(layers.LSTM(self.config.lstm_units[1], return_sequences=True)))
        model.add(layers.Bidirectional(layers.LSTM(self.config.lstm_units[2])))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(self.config.n_classes, activation="softmax"))
        return self._compile(model, learning_rate)

    def build_cnn_lstm(self, learning_rate: float = 1e-3) -> keras.Model:
        """Build and compile the hybrid CNN-LSTM classifier.

        Args:
            learning_rate: Adam optimizer learning rate.

        Returns:
            Compiled Keras model.
        """

        model = keras.Sequential(name="eeg_cnn_lstm")
        self._cnn_backbone(model)
        model.add(layers.Flatten())
        model.add(layers.Dense(100))
        model.add(layers.Reshape((100, 1)))
        model.add(
            layers.LSTM(
                self.config.cnn_lstm_units,
                dropout=0.6,
                recurrent_dropout=0.1,
                return_sequences=False,
            )
        )
        model.add(layers.Dense(self.config.n_classes, activation="softmax"))
        return self._compile(model, learning_rate)

    def build(self, name: str, learning_rate: float = 1e-3) -> keras.Model:
        """Build a named TensorFlow classifier.

        Args:
            name: Classifier name: `cnn`, `lstm`, or `cnn_lstm`.
            learning_rate: Adam optimizer learning rate.

        Returns:
            Compiled Keras model.

        Raises:
            ValueError: If the classifier name is unknown.
        """

        builders = {
            "cnn": self.build_cnn,
            "lstm": self.build_lstm,
            "cnn_lstm": self.build_cnn_lstm,
        }
        try:
            return builders[name](learning_rate=learning_rate)
        except KeyError as exc:
            raise ValueError(f"Unknown TensorFlow classifier: {name}") from exc


def build_generator(config: ModelConfig | None = None) -> keras.Model:
    """Build the conditional GAN generator.

    Args:
        config: Optional architecture configuration.

    Returns:
        Keras generator model.
    """

    _require_tensorflow()
    config = config or ModelConfig()
    generator_in_channels = config.latent_dim + config.n_classes
    return keras.Sequential(
        [
            layers.InputLayer((generator_in_channels,)),
            layers.Dense(10 * generator_in_channels),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((10, generator_in_channels)),
            layers.Conv1DTranspose(128, 11, strides=5, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1DTranspose(128, 11, strides=5, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(config.channels, 10, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )


def build_discriminator(config: ModelConfig | None = None) -> keras.Model:
    """Build the conditional GAN discriminator.

    Args:
        config: Optional architecture configuration.

    Returns:
        Keras discriminator model.
    """

    _require_tensorflow()
    config = config or ModelConfig()
    discriminator_in_channels = config.channels + config.n_classes
    return keras.Sequential(
        [
            layers.InputLayer((config.max_time_step // 2, discriminator_in_channels)),
            layers.Conv1D(64, 9, strides=5, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv1D(128, 9, strides=5, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )


if keras is not None:
    _TensorFlowGANBase = keras.Model
else:  # pragma: no cover
    _TensorFlowGANBase = object


class TensorFlowConditionalGAN(_TensorFlowGANBase):
    """Conditional GAN augmentation model from the original notebook.

    Args:
        discriminator: Optional discriminator model.
        generator: Optional generator model.
        config: Optional architecture configuration.
    """

    def __init__(
        self,
        discriminator: keras.Model | None = None,
        generator: keras.Model | None = None,
        config: ModelConfig | None = None,
    ):
        _require_tensorflow()
        super().__init__()
        self.config = config or ModelConfig()
        self.discriminator = discriminator or build_discriminator(self.config)
        self.generator = generator or build_generator(self.config)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):  # type: ignore[override]
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):  # type: ignore[override]
        real_images, one_hot_labels = data
        image_one_hot_labels = one_hot_labels[:, :, None]
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=[self.config.max_time_step // 2])
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.config.max_time_step // 2, self.config.n_classes)
        )

        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.config.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)
        generated_images = self.generator(random_vector_labels)

        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.config.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {"g_loss": self.gen_loss_tracker.result(), "d_loss": self.disc_loss_tracker.result()}


@dataclass
class TensorFlowGANAugmenter:
    """Train a TensorFlow conditional GAN and append generated samples.

    The augmenter is classifier-independent: it receives prepared training arrays and
    returns augmented arrays that can be passed to CNN, LSTM, or CNN-LSTM classifiers.

    Attributes:
        training: Training and GAN augmentation configuration.
        model_config: Optional architecture configuration.
    """

    training: TrainingConfig
    model_config: ModelConfig | None = None

    def __post_init__(self) -> None:
        _require_tensorflow()
        self.model_config = self.model_config or ModelConfig()

    def augment_training_data(self, x_train: np.ndarray, y_train: np.ndarray) -> GANAugmentationResult:
        """Create GAN-augmented training arrays.

        Args:
            x_train: Real training inputs shaped `(n, time, 1, channels)`.
            y_train: One-hot real training labels.

        Returns:
            Augmented training arrays plus synthetic sample metadata.
        """

        tf.keras.utils.set_random_seed(self.training.seed)
        gan_sequences, scale = classifier_images_to_gan_sequences(x_train)
        labels = np.asarray(y_train, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((gan_sequences, labels))
        dataset = dataset.shuffle(buffer_size=min(256, gan_sequences.shape[0]), seed=self.training.seed).batch(
            self.training.gan_batch_size
        )

        gan = TensorFlowConditionalGAN(config=self.model_config)
        gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=self.training.gan_learning_rate),
            g_optimizer=keras.optimizers.Adam(learning_rate=self.training.gan_learning_rate),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        epochs = 1 if self.training.fast_dev_run else self.training.gan_epochs
        history_object = gan.fit(dataset, epochs=epochs, verbose=True)
        history = [
            {key: float(values[index]) for key, values in history_object.history.items()}
            for index in range(len(next(iter(history_object.history.values()), [])))
        ]

        synthetic_labels = interpolation_labels(
            n_classes=self.model_config.n_classes,
            samples_per_class=self.training.gan_samples_per_class,
        )
        noise = tf.random.normal(shape=(synthetic_labels.shape[0], self.model_config.latent_dim))
        noise_and_labels = tf.concat([noise, tf.convert_to_tensor(synthetic_labels, dtype=tf.float32)], axis=1)
        synthetic_sequences = gan.generator.predict(noise_and_labels, verbose=0)
        return append_synthetic_training_data(
            x_train,
            y_train,
            synthetic_sequences,
            synthetic_labels,
            scale,
            history=history,
            metadata={
                "framework": "tensorflow",
                "real_samples": int(x_train.shape[0]),
                "synthetic_samples": int(synthetic_labels.shape[0]),
                "gan_epochs": int(epochs),
            },
        )


@dataclass
class TensorFlowTrainer:
    """Thin TensorFlow trainer wrapper used by notebooks and examples.

    Attributes:
        training: Training configuration controlling batch size, epochs, and dev mode.
    """

    training: TrainingConfig

    def fit(self, model: keras.Model, x_train: Any, y_train: Any, x_valid: Any, y_valid: Any) -> Any:
        """Train a compiled Keras model.

        Args:
            model: Compiled classifier.
            x_train: Training inputs.
            y_train: Training one-hot labels.
            x_valid: Validation inputs.
            y_valid: Validation one-hot labels.

        Returns:
            Keras `History` object.
        """

        epochs = 1 if self.training.fast_dev_run else self.training.epochs
        return model.fit(
            x_train,
            y_train,
            batch_size=self.training.batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            verbose=True,
        )

    def evaluate(self, model: keras.Model, x_test: Any, y_test: Any) -> dict[str, float]:
        """Evaluate a trained Keras model.

        Args:
            model: Trained classifier.
            x_test: Test inputs.
            y_test: Test one-hot labels.

        Returns:
            Dictionary with `loss` and `accuracy`.
        """

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return {"loss": float(loss), "accuracy": float(accuracy)}
