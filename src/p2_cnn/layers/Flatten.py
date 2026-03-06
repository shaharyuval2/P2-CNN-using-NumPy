import numpy as np

from p2_cnn.core.layer import Layer


class Flatten(Layer):
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.shape = None
        self.in_channels = in_channels

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (in_channels, H, W)
        self.shape = input_data.shape
        return input_data.flatten().reshape(-1, 1)

    def forward_batch(self, input_data: np.ndarray) -> np.ndarray:
        # input_data shape: (batch_size, in_channels, H, W)
        self.shape = input_data.shape
        return input_data.flatten().reshape(self.shape[0], -1, 1)

    # d_output shape: (in_channel * out_h * out_w, 1)
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return d_output.reshape(self.shape)

    # d_output shape: (batch_size, in_channel * out_h * out_w, 1)
    def backward_batch(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return d_output.reshape(self.shape)
