# Libraries
from toy_framework.layers import *
from toy_framework.activations import *

# Window dimensions
width = 920
height = 720

# Genetic Algorithm
architect = [
    Dense(4, None),
    Dense(2, tanh),
    Dense(2, sigmoid)
]
popu_size = 2
mutation_rate = 0.01
