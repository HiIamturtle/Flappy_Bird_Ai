# Libraries
import numpy as np
import config
from toy_framework.neuro_evolution.network_ga import NetworkGA
from toy_framework.networks import *
import random


# Creating a class as a base to manage a population of NNs
class PopulationBase:
    def __init__(self, agent=None, mutation_rate=0.01, popu_size=50, nn_architecture=None):
        self.popu = []  # List to hold all the neural networks
        self.mating_pool = []  # List holding NNs in relation to fitness

        self.mr = mutation_rate  # Percentage chance for a weight to randomly mutate
        self.popu_size = popu_size  # Number of agents in a population

        for _ in range(popu_size):
            temp_net = Network(structure=nn_architecture)
            temp_net.init_network()
            print(f'before going to agent: {temp_net.struct[0].w}')
            temp_agent = agent(network=temp_net)
            print(f'after going to agent: {temp_agent.net.struct[0].w}')
            self.popu.append(temp_agent)

        for index, i in enumerate(self.popu):
            print(f'Printing all agent network structures: {i.net.struct[0].w}')

    # Combing 2 parents
    def cross_over(self):
        temp_popu = []
        for _ in range(len(self.popu)):
            parent_a = random.choice(self.mating_pool)
            # Making sure the same parent is not chosen twice
            while True:
                parent_b = random.choice(self.mating_pool)
                if parent_a != parent_b:
                    break
            child = parent_a.cross(parent_b)
            child.mutate()

            temp_popu.append(child)
