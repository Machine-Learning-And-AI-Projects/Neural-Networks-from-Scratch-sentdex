import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]  # one-hot encoded

# here we loop through each output and calculate the loss
# but we are looping through the output and just multiplying the softmax output with the target output
# the one that matches will be 1 and the rest will be 0
loss = -sum(
    [target_output[i] * math.log(softmax_output[i]) for i in range(len(target_output))]
)

print(loss)

# to reduce this unnecessary looping, we can just extract the only value that is 1 from the target output
loss = -math.log(
    softmax_output[0]
)  # since the target output is [1, 0, 0] & 0 is the index of the 1


print(loss)

# in Loss function,
# x -> output of the softmax function i.e. confidence of the model [0, 1]
# y -> loss (-inf, 1] # the lower the loss, the better the model

# y = ln(x) # y -> loss, x -> confidence

# higher confidence -> lower loss -> better model & vice versa
