# Libraries
from p5 import *
from bird import Bird
import config
from pipe import Pipe

from toy_framework.layers import *
from toy_framework.activations import *
from toy_framework.neuro_evolution.population import PopulationBase


# Getting the inputs for the neural network (t_dist, b_dist), we get the y_pos from the bird
def get_inputs(bird, pipe):
    t_dist = pipe.top_pos.y - bird.pos.y  # Distance to the top pipe
    b_dist = pipe.bot_pos.y - bird.pos.y  # Distance to the bottom pipe
    x_dist = pipe.bot_pos.x - bird.pos.x  # X_distance

    return t_dist, b_dist, x_dist


# Custom population
class Population(PopulationBase):
    def __init__(self):
        PopulationBase.__init__(self, agent=Bird, popu_size=config.popu_size,
                                nn_architecture=config.architect)

    # Updating each agent
    def update(self, pipe, pipes):
        for i in self.popu:
            t_dist, b_dist, x_dist = get_inputs(i, pipe)
            i.update(pipes, t_dist, b_dist, x_dist)


popu = Population()
pipes = []
next_pipe = None
pipe_count = 0


# Setup
def setup():
    global next_pipe
    size(config.width, config.height)
    pipes.append(Pipe())
    next_pipe = pipes[0]


# Draw Loop
def draw():
    global pipe_count, next_pipe
    background(51)

    # Generating new pipes
    pipe_count += pipes[0].vel.x
    if pipe_count % 125 == 0:
        pipes.append(Pipe())

    # Updating each pipe
    for pipe in pipes:
        pipe_done = pipe.update()
        if pipe_done:
            del pipe

    # Finding teh next pipe
    if pipes[0].top_pos.x + pipes[0].w < popu.popu[0].pos.x - (popu.popu[0].w / 2):
        next_pipe = pipes[1]
    else:
        next_pipe = pipes[0]

    # Updating each bird
    popu.update(next_pipe, pipes)


if __name__ == '__main__':
    run()
