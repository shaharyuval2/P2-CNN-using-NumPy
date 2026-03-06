from typing import Optional

import numpy as np
from scipy import signal

from p2_cnn.core.activation import ACTIVATIONS
from p2_cnn.core.layer import Layer
from p2_cnn.core.utils import ML_conv2d, init_he, init_xavier, rotate180


class Conv2d(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        activation_type: str,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ):
        super().__init__()

        self.z = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w

        self.activation_type = activation_type
        self.act_fn, self.der_act_fn, init_mode = ACTIVATIONS[activation_type]
        if init_mode == "xavier":
            self.kernels = init_xavier(
                (in_channels, out_channels, kernel_h, kernel_w), rng
            )
        elif init_mode == "he":
            self.kernels = init_he((in_channels, out_channels, kernel_h, kernel_w), rng)

        self.biases = np.zeros(shape=(out_channels, 1))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (in_channels, H, W)
        self.input = input_data

        _, in_h, in_w = input_data.shape
        out_h = in_h - self.kernel_h + 1
        out_w = in_w - self.kernel_w + 1

        self.z = np.zeros(shape=(self.out_channels, out_h, out_w))

        for q in range(self.out_channels):
            for p in range(self.in_channels):
                self.z[q] += ML_conv2d(input_data[p], self.kernels[p][q])
            self.z[q] += self.biases[q]

        return self.act_fn(self.z)

    def forward_batch(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (batch_size, in_channels, H, W)
        self.input = input_data

        batch_size, _, in_h, in_w = input_data.shape
        out_h = in_h - self.kernel_h + 1
        out_w = in_w - self.kernel_w + 1

        self.z = np.zeros(shape=(batch_size, self.out_channels, out_h, out_w))

        for q in range(self.out_channels):
            for p in range(self.in_channels):
                # passes into 3d correlate I_p = (batch_size, Ih, Iw) , K_p,q = (1, Kh, Kw)
                self.z[:, q] += signal.correlate(
                    self.input[:, p, :, :],
                    self.kernels[p, q, np.newaxis, :, :],
                    mode="valid",
                )
            self.z[:, q] += self.biases[q]

        return self.act_fn(self.z)

    # d_output shape: (out_channel, out_h, out_w)
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        delta = self.der_act_fn(self.z) * d_output

        # calculating dk db d_input
        dk = np.zeros(shape=self.kernels.shape)
        d_input = np.zeros(shape=self.input.shape)

        for p in range(self.in_channels):
            for q in range(self.out_channels):
                dk[p][q] = ML_conv2d(rotate180(self.input[p]), delta[q])
                d_input[p] += ML_conv2d(
                    delta[q], rotate180(self.kernels[p][q]), mode="full"
                )

        db = np.sum(delta, axis=(1, 2)).reshape(-1, 1)

        # rescaling db and dk according to layer size to not overshoot
        area = delta.shape[1] * delta.shape[2]
        dk /= area
        db /= area

        # learning step (Stochastic Grediant Descent)
        self.kernels -= learning_rate * dk
        self.biases -= learning_rate * db

        return d_input

    # d_output shape: (batch_size, out_channel, out_h, out_w)
    def backward_batch(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = d_output.shape[0]

        delta = self.der_act_fn(self.z) * d_output

        # calculating dk db d_input
        dk = np.zeros_like(self.kernels)
        d_input = np.zeros_like(self.input)

        for p in range(self.in_channels):
            input_p_rotated = np.rot90(self.input[:, p], k=2, axes=(1, 2))
            for q in range(self.out_channels):
                # passes into 3d correlate rot_I_p = (batch_size, Ih, Iw) , delta_q = (batch_size, Oh, Ow)
                # (note that it sums over batch examples due to convolution nature)
                dk[p, q] = signal.correlate(input_p_rotated, delta[:, q], mode="valid")[
                    0
                ]

                rot_kernel = np.rot90(self.kernels[p, q], k=2)
                # passes into 3d correlate
                # delta_q = (batch_size, Oh, Ow)
                # rot_kenel = (1, Kh, Kw)
                d_input[:, p] = signal.convolve(
                    delta[:, q], rot_kernel[np.newaxis, :, :], mode="full"
                )

        db = np.sum(delta, axis=(0, 2, 3)).reshape(-1, 1)

        # rescaling db and dk according to layer size and batch size to not overshoot
        norm_factor = batch_size * delta.shape[1] * delta.shape[2]
        dk /= norm_factor
        db /= norm_factor

        # learning step (Stochastic Grediant Descent)
        self.kernels -= learning_rate * dk
        self.biases -= learning_rate * db

        return d_input
