import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


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


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # subtracting the max value from the inputs to avoid overflow # since e^x can be exponential, but when restricted 0, it becomes a range between 0 < x <= 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)  # 100 samples for 3 classes each

dense1 = Layer_Dense(
    2, 3
)  # x,y cordinates as inputs, & 3 classes as outputs for this layer
activation1 = Activation_ReLU()

dense2 = Layer_Dense(
    3, 3
)  # 3 inputs from the previous layer, 3 outputs as final outputs
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


"""
layer1 = Layer_Dense(2, 5)  # 0 - inputs, 1 - neurons
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
print(activation1.output)

# the output of the first layer should match the input of the second layer
layer2 = Layer_Dense(5, 2)
layer2.forward(layer1.output)
print(layer2.output)
"""
