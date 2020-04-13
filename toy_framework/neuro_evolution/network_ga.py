# Libraries
import numpy as np
import config

from toy_framework.optimizers import *
from toy_framework.layers import *
from toy_framework.activations import *
from toy_framework.networks import Network


# Neural network for genetic algorithms
class NetworkGA:
    def __init__(self, structure=None, optimizer=gen_algorithm, mutation_rate=0.01):
        print(structure)
        # Initializing the base network
        self.network = Network(structure=structure, optimizer=optimizer)
        self.network.__init__(structure=structure, optimizer=optimizer)

        # Setting the variables for the genetic algorithm
        self.mr = mutation_rate
        self.fitness = None  # How likely is this agent to be chosen for crossover

    def cross_over(self, parent):
        pass

    def mutate(self):
        pass


if __name__ == '__main__':
    architecture = [
        Dense(5, None),
        Dense(16, relu),
        Dense(2, tanh)
    ]

    gn = NetworkGA(structure=architecture)
    ga = NetworkGA(structure=architecture)
    gn.network.init_network()
    ga.network.init_network()
    print(f'gn: {gn.network.struct[0].w}, ga {ga.network.struct[0].w}')
