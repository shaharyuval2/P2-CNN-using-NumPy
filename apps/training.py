import numpy as np

from p2_cnn.core.utils import one_hot, splitData
from p2_cnn.layers.conv2d import Conv2d
from p2_cnn.layers.dense import Dense
from p2_cnn.layers.Flatten import Flatten
from p2_cnn.layers.pooling import Pooling
from p2_cnn.model import Model


def main():
    # Load Data
    print("[INFO] Loading MNIST dataset...")
    train_labels, train_images = splitData("data/mnist_train.csv")
    test_labels, test_images = splitData("data/mnist_test.csv")

    # Preprocess
    train_y_one_hot = one_hot(train_labels, 10)
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)

    # Initialize Model
    seed = 42
    rng = np.random.default_rng(seed)
    conv2d_1 = Conv2d(1, 6, 5, 5, "relu", rng)
    pooling_1 = Pooling(6)
    conv2d_2 = Conv2d(6, 12, 5, 5, "relu", rng)
    pooling_2 = Pooling(12)
    flatten = Flatten(12)
    dense = Dense(192, 10, "softmax", rng)

    layers = [conv2d_1, pooling_1, conv2d_2, pooling_2, flatten, dense]
    model = Model(layers, rng)

    # Check baseline model performance
    initial_sr = model.success_rate_batch(test_labels, test_images)
    print(f"[INFO] Initial Success batch Rate: {initial_sr:.4f}")

    # Train the model
    model.train_batch(
        train_images, train_y_one_hot, 5, 0.3, 1.2, 128, (test_labels, test_images)
    )

    # Final Evaluation & Saving
    final_sr = model.success_rate(test_labels, test_images)
    print(f"[SUCCESS] Final Success Rate: {final_sr:.4f}")

    model_name = "models/mnist_cnn_v3.npz"
    model.save(model_name)
    print(f"[INFO] Model weights saved to {model_name}")


if __name__ == "__main__":
    main()
