# Libraries
import numpy as np
import random


# A fully connected layer
class Dense:
    def __init__(self, n_nodes, activation_func):
        self.nodes = n_nodes  # Number of nodes in the layer
        self.w = None  # The weights of the layer
        self.b = None  # The biases of the layer

        self.dot_prod = None  # The answer to the dot product before activation
        self.activated = None  # The value after the dot_prod has been activated

        self.activation = activation_func  # Activation function used on the layer
        self.error = None  # how much a layer attributed to the total error

        self.d_w = None  # The delta weights of the layer
        self.d_b = None  # The delta biases of the layer

    # Generating the weights
    def init_params(self, next_layer):
        # New seed
        seed_choice = random.randint(0, 1000000000)
        #print(seed_choice)
        np.random.seed(seed_choice)

        # Creating a matrix that's shape is (next_layer, current_layer)
        self.w = np.random.randn(next_layer.nodes, self.nodes) * 0.01
        self.b = np.random.randn(next_layer.nodes, 1) * 0.01

    # Setting deltas to zero
    def zero_deltas(self):
        # Generating a matrices of zeros the shape of this layers parameters
        self.d_w = np.zeros(self.w.shape)
        self.d_b = np.zeros(self.b.shape)

    # Calculating this layers error
    def calc_error(self, input=None, index=0, last_error=None):

        # If the output layer
        if index == 0:
            self.error = input - last_error
        # For the hidden layers
        else:
            self.error = np.dot(self.w.T, last_error)

        return self.error
