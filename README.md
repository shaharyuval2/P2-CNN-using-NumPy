# CNN from Scratch: Modular NumPy Implementation
## Developed by Shahar — **The Stupid Guy**

This project is a modular, high-performance **Convolutional Neural Network (CNN)** engine built entirely from the ground up using only **NumPy**. It marks a significant evolution from my previous DNN implementation, moving from flat vectors to 3D spatial feature maps to master the mechanics of modern Computer Vision.

## Project Overview
To truly understand computer vision, I decided to "break the black box" of high-level frameworks like PyTorch or TensorFlow. This implementation is heavily based on the mathematical frameworks found in **Zhifei Zang’s "Derivation of Backpropagation in Convolutional Neural Network,"** ensuring my understanding in every gradient flow and weight update.

* **Custom CNN Engine:** A fully modular architecture supporting Conv2D, Max-Pooling, Flatten, and Dense layers.
* **4D Tensor Calculus:** Manual implementation of backpropagation through 4D tensors $$(Batch, Channel, Height, Width)$$, handling the non-trivial spatial dependencies of filters.
* **Professional Architecture:** sticking to a "Source Layout" design, treating the engine as a reusable, production-ready Python package.
* **Live Activation Visualization:** A Pygame-based GUI that renders internal feature maps in real-time, visualizing how the data flows through the model.


## My Learnings
This project served as a rigorous deep dive into both the mathematical and engineering challenges of Deep Learning. Through this implementation, I have honed several key skills:

* **Generality of Backpropagation:** I developed a much deeper intuition for the Chain Rule. Moving beyond simple matrices to convolutional kernels taught me how backpropagation is a universal flow of information, regardless of the layer's geometric structure.
* **Advanced Modular Design:** I shifted from "script-based" coding to a decoupled, object-oriented approach. Each layer is an independent module with its own state, making the engine scalable and easy to debug.
* **Mathematical Resilience:** Working through the Zang derivation strengthened my ability to translate research papers into working code.

## Project Architecture

The project is organized into a clean, decoupled structure:
* **`src/p2_cnn/`**: The core engine, including layers modules and model logic.
* **`apps/`**: High-level applications including the training pipeline and the interactive Showcase.
* **`data/`**: Scripts for MNIST data generation.
* **`models/`**: Storage for trained weights in `.npz` format.

### Installation & Setup

1.  **Clone the Repository**
    
```bash
    git clone https://github.com/shaharyuval2/P2-CNN-using-NumPy.git
    cd P2_CNN
```

2. **Create and Activate Virtual Environment**
```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3.  **Install in Editable Mode**
    
```bash
    pip install -e .
```

This uses the `pyproject.toml` configuration to set up the `p2_cnn` package and its dependencies (NumPy, Pygame, Scipy).


### How to Use

1.  **Generate the Dataset**
    Before training, you must download and convert the MNIST data:
    
```bash
    python3 data/generate_mnist_csv.py
```

2.  **Train the Model**
    By running the training script:
    
```bash
    python3 apps/training.py
```

3.  **Live Showcase**
    Launch the interactive drawing board to test your trained model:
    
```bash
    python3 apps/ShowCase/ShowCase.py
```

