import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # weights are set with the number of rows as n_inputs such that we do not have to transpose the input matrix during the dot product
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)  # 0 - inputs, 1 - neurons

activation1 = Activation_ReLU()


layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
print(activation1.output)

# the output of the first layer should match the input of the second layer
# layer2 = Layer_Dense(5, 2)
# layer2.forward(layer1.output)
# print(layer2.output)
