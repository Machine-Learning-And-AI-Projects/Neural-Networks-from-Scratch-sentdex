import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

print(exp_values)

norm_values = exp_values / np.sum(layer_outputs, axis=1, keepdims=True)

print(norm_values)

"""


print(norm_values)
print(sum(norm_values))

# input -> exponential -> normalize -> output
# softmax function = exponential function + normalization function

softmax_function = np.exp(layer_outputs) / np.sum(np.exp(layer_outputs))
"""
