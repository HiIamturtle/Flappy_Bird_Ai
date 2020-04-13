# Libraries
import numpy as np


# Sigmoid (maps x btwn 0 and 1)
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Tanh (maps x btwn -1 and 1)
def tanh(x):
    return np.tanh(x)


# Relu (either 0 or normal num)
def relu(x):
    return np.maximum(0, x)


