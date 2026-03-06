import numpy as np

from p2_cnn.core.layer import Layer


class Pooling(Layer):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.channels = channels

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (channels, H, W)
        self.input = input_data

        _, H, W = self.input.shape

        pooling_size = 2
        reshaped = self.input.reshape(
            self.channels,
            H // pooling_size,
            pooling_size,
            W // pooling_size,
            pooling_size,
        )

        return reshaped.mean(axis=(2, 4))

    def forward_batch(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (batch_size, channels, H, W)
        self.input = input_data

        batch_size, _, H, W = self.input.shape

        pooling_size = 2
        reshaped = self.input.reshape(
            batch_size,
            self.channels,
            H // pooling_size,
            pooling_size,
            W // pooling_size,
            pooling_size,
        )

        return reshaped.mean(axis=(3, 5))

    # d_output shape: (channels, out_h//2, out_w//2)
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return d_output.repeat(2, axis=1).repeat(2, axis=2) / 4

    # d_output shape: (batch_size, channels, out_h//2, out_w//2)
    def backward_batch(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return d_output.repeat(2, axis=2).repeat(2, axis=3) / 4
