import numpy as np
from scipy import signal


def splitData(file_path):
    raw_data = np.loadtxt(file_path, delimiter=",")

    labels = raw_data[:, 0]
    examples = raw_data[:, 1:] / 255

    return labels, examples


"""
def showImage_reshape_from_batch(examples, index):
    plt.imshow(examples[index].reshape(28, 28), cmap="grey")
    plt.show()


def showImage(x):
    plt.imshow(x, cmap="grey")
    plt.show()
"""


def one_hot(labels, classes=10):
    return np.eye(classes)[labels.astype(int)].T


def get_fans(shape):
    fan_in, fan_out = None, None
    if len(shape) == 2:  # dense - (fan_out, fan_in)
        fan_in = shape[1]
        fan_out = shape[0]
    elif len(shape) == 4:  # conve2d - (in_channels, out_channels, kernel_h, kernel_w)
        kernel_size = shape[2] * shape[3]
        fan_in = shape[0] * kernel_size
        fan_out = shape[1] * kernel_size
    else:
        raise ValueError(f"Unknown shape length: {len(shape)}")
    return fan_in, fan_out


# for Relu activation
def init_he(shape, rng):
    fan_in, _ = get_fans(shape)
    std = np.sqrt(2 / fan_in)
    return rng.normal(loc=0, scale=std, size=shape)


# for sigmoid / tanh activation
def init_xavier(shape, rng):
    fan_in, fan_out = get_fans(shape)
    std = np.sqrt(2 / (fan_in + fan_out))
    return rng.normal(loc=0, scale=std, size=shape)


# CNN actually does cross correlation (no need for flipping since Kernel weights are learned)
def ML_conv2d(I: np.ndarray, K: np.ndarray, mode="valid") -> np.ndarray:
    return signal.correlate2d(I, K, mode=mode)


def rotate180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=2)
