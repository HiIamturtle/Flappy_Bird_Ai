from toy_framework.neuro_evolution.network_ga import NetworkGA
from toy_framework.networks import *
from toy_framework.layers import *
from toy_framework.activations import *

a = Network(structure=[
    Dense(4, None),
    Dense(16, tanh),
    Dense(2, sigmoid)
])

b = Network(structure=[
    Dense1(4, None),
    Dense1(16, tanh),
    Dense1(2, sigmoid)
])

a.init_network()
b.init_network()

if __name__ == '__main__':
    print(f'a: {a.struct[0].w}, b: {b.struct[0].w}')
