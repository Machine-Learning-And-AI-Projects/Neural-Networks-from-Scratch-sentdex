import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

# here we extract the label values from the softmax outputs
# EG: if the label was cat, then we would extract the value of the element that shows cat in the softmax output
# print(softmax_outputs[[0, 1, 2], class_targets])
print(
    softmax_outputs[[i for i in range(len(softmax_outputs))], class_targets]
)  # same as above

# calculate the loss for each predicted value
neg_log = -np.log(
    softmax_outputs[[i for i in range(len(softmax_outputs))], class_targets]
)
average_loss = np.mean(neg_log)
print(average_loss)
