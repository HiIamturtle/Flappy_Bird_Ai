# Libraries
import numpy as np
import pickle
import random

from toy_framework.optimizers import *
from toy_framework.activations import *


# Creating a class to handle the different layers
class Network:
    def __init__(self, structure=None, optimizer=gradient_decent, lr=0.01):
        if structure is None:
            structure = []
        self.struct = structure  # Variable to store the different layers
        #print(f'struct: {self.struct}')
        self.lr = lr  # Learning rate
        self.optimizer = optimizer

    # Initializing the weights of the entire network
    def init_network(self):
        # Looping over each layer in structure except the output, which has no weights
        for index, i in enumerate(self.struct[:-1]):
            #print(f'i: {i}')
            # Getting the next layers nodes
            next_layer = self.struct[index + 1]

            # Generating the weights
            i.init_params(next_layer)

    # Generating delta weight matrices
    def delta_weights(self):
        for i in self.struct[:-1]:
            i.zero_deltas()

    # Feeding an input through the neural network
    def feed_forward(self, inputs, training=False):
        # Formatting the data when not training
        if not training:
            # Working with the input data
            if isinstance(inputs, list):
                inputs = np.array(inputs, ndmin=2).T
            elif isinstance(inputs, np.ndarray):
                inputs = inputs.flatten()
                inputs = np.array(inputs, ndmin=2).T
            else:
                raise TypeError('Input needs to be either an np.array or a list')

        # Send input through the network
        layer_input = inputs
        for index, i in enumerate(self.struct[:-1]):

            # Calculating the layer output
            activation = self.struct[index + 1].activation
            layer_dot = np.dot(i.w, layer_input) + i.b
            layer_out = activation(layer_dot)

            # Saving the dot and out
            i.dot_prod = layer_dot
            i.activated = layer_out

            # Setting next input
            layer_input = layer_out

        # Returning the net_output
        return layer_input

    # Train on data
    def train(self, batch=None, network=None):
        if network is None:
            network = self
        self.optimizer(batch=batch, network=network)

    # Saving the model to a .pkl file
    def save(self, file_path):
        file = open(file_path, 'wb')
        pickle.dump(self.struct, file)
        file.close()

    # Load saved model from a .pkl file
    def load(self, file_path):
        file = open(file_path, 'wb')
        self.struct = pickle.load(file)
        file.close()


if __name__ == '__main__':
    from toy_framework.layers import *

    a = Network(structure=[
        Dense(4, None),
        Dense(8, tanh),
        Dense(2, sigmoid)
    ])

    b = Network(structure=[
        Dense(4, None),
        Dense(8, tanh),
        Dense(2, sigmoid)
    ])
    a.init_network()
    b.init_network()
    print(f'a: {a.struct[0].w}')
    print(f'b: {b.struct[0].w}')