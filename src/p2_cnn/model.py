import sys
from typing import List, Optional

import numpy as np

from p2_cnn.core.layer import Layer
from p2_cnn.layers.conv2d import Conv2d
from p2_cnn.layers.dense import Dense
from p2_cnn.layers.Flatten import Flatten
from p2_cnn.layers.pooling import Pooling


class Model:
    def __init__(
        self,
        layers: List[Layer],
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ):
        self.layers = layers
        self.rng = rng

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward_batch(x)
        return x

    def backward(self, d_loss: np.ndarray, learning_rate: float):
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, learning_rate=learning_rate)

    def backward_batch(self, d_loss: np.ndarray, learning_rate: float):
        for layer in reversed(self.layers):
            d_loss = layer.backward_batch(d_loss, learning_rate=learning_rate)

    def train(
        self,
        train_x: np.ndarray,
        train_y_one_hot: np.ndarray,
        epochs: int,
        learning_rate: float,
        epoch_size: int,
        val_data: tuple = None,
    ):
        # train_x shape : (n_samples, sample_h, sample_w)
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            x_shuffled = train_x[indices]
            x_epoch = x_shuffled[:epoch_size]
            y_shuffled = train_y_one_hot[:, indices]
            y_epoch = y_shuffled[:epoch_size]

            learning_rate *= 0.8  # simple learning rate decay

            for i in range(0, epoch_size):
                x = x_epoch[i, :, :]
                y = y_epoch[:, i].reshape(-1, 1)

                y_hat = self.forward(x)

                d_y = y_hat - y  # softmax + cross correlation cost function

                self.backward(d_y, learning_rate=learning_rate)

                # Update progress meter
                if i % 100 == 0 or i == epoch_size - 1:
                    percent = (i) / epoch_size * 100
                    sys.stdout.write(
                        f"\rEpoch {epoch + 1}/{epochs} | Progress: {percent:.1f}% [{i}/{epoch_size}]"
                    )
                    sys.stdout.flush()

            if (epoch + 1) % 1 == 0:
                sr = self.success_rate(*val_data)
                print(f"[PROGRESS] Epoch {epoch + 1}/{epochs} - Success Rate: {sr:.4f}")
            pass

    def train_batch(
        self,
        train_x: np.ndarray,
        train_y_one_hot: np.ndarray,
        epochs: int,
        learning_rate: float,
        learning_decay: float,
        batch_size: int,
        val_data: tuple = None,
    ):
        # train_x shape : (n_samples, sample_h, sample_w)
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            x_shuffled = train_x[indices]
            y_shuffled = train_y_one_hot[:, indices]

            learning_rate *= 0.8  # simple learning rate decay

            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size, :, :]
                y_batch = y_shuffled[:, i : i + batch_size].T.reshape(-1, 10, 1)

                y_hat_batch = self.forward_batch(x_batch)

                d_y_batch = (
                    y_hat_batch - y_batch
                )  # softmax + cross correlation cost function

                self.backward_batch(d_y_batch, learning_rate=learning_rate)

                # Update progress meter
                percent = (i) / n_samples * 100
                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{epochs} | Progress: {percent:.1f}% [{i}/{n_samples}]"
                )
                sys.stdout.flush()

            if (epoch + 1) % 1 == 0:
                sr = self.success_rate(*val_data)
                print(f"[PROGRESS] Epoch {epoch + 1}/{epochs} - Success Rate: {sr:.4f}")
            pass

            learning_rate *= learning_decay

    def success_rate(self, test_labels: np.ndarray, test_examples: np.ndarray) -> float:
        # test_exampls shape: (n_examples, example_h, example_w)
        n_examples = test_examples.shape[0]
        correct_guesses = 0
        for i in range(n_examples):
            y_hat = self.forward(test_examples[i])
            if np.argmax(y_hat) == test_labels[i]:
                correct_guesses += 1
        return correct_guesses / n_examples

    def success_rate_batch(
        self, test_labels: np.ndarray, test_examples: np.ndarray
    ) -> float:
        # test_exampls shape: (n_examples, example_h, example_w)
        y_hat_batch = self.forward_batch(test_examples)
        predictions = np.argmax(y_hat_batch.squeeze(axis=2), axis=1)
        correct_guesses = np.sum(predictions == test_labels)
        return correct_guesses / test_examples.shape[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def save(self, path: str):
        dic = {}
        dic["num_layers"] = np.array(len(self.layers))

        for i, layer in enumerate(self.layers):
            dic[f"type_{i}"] = np.array(layer.__class__.__name__)

            # Save activation if the layer has one
            if hasattr(layer, "activation_type"):
                dic[f"act_{i}"] = np.array(layer.activation_type)

            if hasattr(layer, "kernels"):
                dic[f"k_{i}"] = layer.kernels
                dic[f"b_{i}"] = layer.biases
                dic[f"meta_{i}"] = np.array(
                    [
                        layer.in_channels,
                        layer.out_channels,
                        layer.kernel_h,
                        layer.kernel_w,
                    ]
                )
            elif hasattr(layer, "weights"):
                dic[f"w_{i}"] = layer.weights
                dic[f"b_{i}"] = layer.biases
                dic[f"meta_{i}"] = np.array(
                    [layer.weights.shape[1], layer.weights.shape[0]]
                )
            elif hasattr(layer, "channels") or hasattr(layer, "in_channels"):
                val = (
                    layer.in_channels
                    if hasattr(layer, "in_channels")
                    else layer.channels
                )
                dic[f"meta_{i}"] = np.array([val])

        np.savez(path, **dic)
        print(f"[SUCCESS] Model saved to {path}")

    @classmethod
    def load(cls, path: str, rng=None):
        # If no rng is provided, create a default one to prevent the NoneType error
        if rng is None:
            rng = np.random.default_rng()

        data = np.load(path)
        num_layers = int(data["num_layers"])
        layers = []

        for i in range(num_layers):
            l_type = str(data[f"type_{i}"])
            meta = data[f"meta_{i}"]
            # Retrieve activation if it was saved, otherwise default to a safe value [cite: 56]
            act = str(data[f"act_{i}"]) if f"act_{i}" in data else None

            if l_type == "Conv2d":
                l = Conv2d(
                    int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3]), act, rng
                )
                l.kernels = data[f"k_{i}"]
                l.biases = data[f"b_{i}"]
            elif l_type == "Pooling":
                l = Pooling(int(meta[0]))
            elif l_type == "Flatten":
                l = Flatten(int(meta[0]))
            elif l_type == "Dense":
                l = Dense(int(meta[0]), int(meta[1]), act, rng)
                l.weights = data[f"w_{i}"]
                l.biases = data[f"b_{i}"]

            layers.append(l)

        return cls(layers, rng)
