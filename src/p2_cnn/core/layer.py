from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        self.training = True

    @abstractmethod
    def forward(self, input_data):
        # Store input for backprop
        # Return transformed output
        pass

    @abstractmethod
    def forward_batch(self, input_data):
        # Store input for backprop
        # Return transformed output
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        # 1. Update weights/biases using output_gradient and self.input
        # 2. Return the gradient for the previous layer
        pass

    @abstractmethod
    def backward_batch(self, output_gradient, learning_rate):
        # 1. Update weights/biases using output_gradient and self.input
        # 2. Return the gradient for the previous layer
        pass
