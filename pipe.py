# Libraries
from p5 import *
import config
import random


# Pipe_object
class Pipe:
    def __init__(self):
        self.w = 65
        self.gap_dist = 145
        
        # Generating the gap btwn the 2 pipes
        self.gap_start = random.choice(range(25, config.height - 165))
        self.top_pos = Vector(config.width + self.w, 0)
        self.bot_pos = Vector(config.width + self.w, self.gap_start + self.gap_dist)

        self.vel = Vector(-3, 0)

    def draw(self):
        fill(255)
        no_stroke()
        # Drawing the top pipe
        rect(self.top_pos, self.w, self.gap_start)

        # Drawing the bottom pipe
        rect(self.bot_pos, self.w, config.height)

    def update(self):
        self.top_pos += self.vel
        self.bot_pos += self.vel
        self.draw()

        if self.top_pos.x + self.w < 0:
            return True
