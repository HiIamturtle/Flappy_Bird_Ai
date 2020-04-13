# Libraries
from p5 import *
import config
import random
from toy_framework.neuro_evolution.network_ga import NetworkGA
from toy_framework.networks import *


# Bird object
class Bird:
    def __init__(self, network=None):
        self.w = 25
        self. dead = False

        self.pos = Vector(75, config.height/2)
        self.vel = Vector(0, 0)
        self.grav = 0.5
        self.jump_force = Vector(0, -11)

        self.net = network

    # Drawing the image to the canvas
    def draw(self):
        if not self.dead:
            fill(255, 255, 255, 100)
        else:
            fill(255, 0, 0, 175)
        no_stroke()
        ellipse_mode(CENTER)
        ellipse(self.pos, self.w, self.w)

    # Checking the collision of the bird
    def collide(self, pipes):
        for pipe in pipes:
            if pipe.top_pos.x < self.pos.x < pipe.top_pos.x + pipe.w:
                if self.pos.y + (self.w/2) > pipe.bot_pos.y \
                        or self.pos.y - (self.w/2) < pipe.gap_start:
                    self.dead = True
                    return True

            if self.pos.y + (self.w/2) > config.height:
                self.pos.y = config.height - (self.w/2)
                self.dead = True

    # Using neural network to take an action
    def action(self, t_dist, b_dist, x_dist): # 4th is the self.pos.y
        # Taking in the inputs and formatting the,
        inputs = np.array([t_dist, b_dist, x_dist, self.pos.y], ndmin=2).T

        # Taking the output of the neural network
        #print(self.net.struct[0].w)
        out = np.argmax(self.net.feed_forward(inputs, training=True))
        #print(out)

        # Interpreting the output
        if out == 1:
            #print('yu')
            self.vel = self.jump_force
        else:
            #print('yo')
            self.vel += Vector(0, 0)

    def update(self, pipes, t_dist, b_dist, x_dist):
        # Applying gravity
        if self.vel.y < 8:
            self.vel.y += self.grav

        # Getting the output from the Neural network
        self.action(t_dist, b_dist, x_dist)

        # Updating the position
        self.pos += self.vel
        self.collide(pipes)

        # Drawing the new position
        self.draw()
