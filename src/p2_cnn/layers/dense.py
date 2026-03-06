from typing import Optional

import numpy as np

from p2_cnn.core.activation import ACTIVATIONS
from p2_cnn.core.layer import Layer
from p2_cnn.core.utils import init_he, init_xavier


class Dense(Layer):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        activation_type: str,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ):
        super().__init__()

        self.z = None

        self.activation_type = activation_type
        self.act_fn, self.der_act_fn, init_mode = ACTIVATIONS[activation_type]
        if init_mode == "xavier":
            self.weights = init_xavier((fan_out, fan_in), rng)
        elif init_mode == "he":
            self.weights = init_he((fan_out, fan_in), rng)

        self.biases = np.zeros(shape=(fan_out, 1))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.z = np.dot(self.weights, self.input) + self.biases
        return self.act_fn(self.z)

    def forward_batch(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.z = self.weights @ self.input + self.biases.reshape(1, -1, 1)
        return self.act_fn(self.z)

    # in dense layer d_output means duC/dua (a is the output activations)
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        delta = self.der_act_fn(self.z) * d_output

        # calculating dw db d_input
        dw = np.outer(delta, self.input)
        db = delta
        d_input = np.dot(self.weights.T, delta)

        # learning step (Stochastic Grediant Descent)
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return d_input

    # in dense layer d_output means duC/dua (a is the output activations)
    def backward_batch(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = d_output.shape[0]

        delta = self.der_act_fn(self.z) * d_output

        # calculating dw db d_input
        # delta shape: (batch_size, fan_out, 1)
        # input shape: (batch_size, fan_in, 1)
        dw_samples = delta * self.input.transpose(0, 2, 1)
        dw = np.sum(dw_samples, axis=0)
        db = np.sum(delta, axis=0)  # sum over batch examples
        d_input = self.weights.T @ delta

        # learning step (Stochastic Grediant Descent)
        self.weights -= learning_rate * (dw / batch_size)
        self.biases -= learning_rate * (db / batch_size)

        return d_input
