import random
import numpy as np


class Region():
    def __init__(self, 
                    name, 
                    code=None,
                    min_instances=None,
                    max_instances=None,
                    min_instace_size=None,
                    max_instance_size=None,
                    ):
        self.name = name
        self.code = code
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.min_instace_size = min_instace_size
        self.max_instance_size = max_instance_size
        self.instances = 0

class River(Region):
    def __init__(self, path = 'top-bottom'):
        super().__init__('river', 'r', 0, 2, 0.01, 0.075)
        self.path = path
    def prompt(self):
        path_text = ""
        if self.path == 'top-bottom':
            path_text = "The river runs from the top of the map to the bottom."
        elif self.path == 'left-right':
            path_text = "The river runs from the left of the map to the right."
        elif self.path == 'curved': # Can mean any curve
            path_text = "The river runs from one edge to another in a curve."
        text = "Draw a river!" + path_text
        return text
    
class Forest(Region):
    def __init__(self):
        super().__init__('forest', 'f', 1, 3, 0.05, 0.2)

    def prompt(self):
        text = "Draw a forest! It should look like a closed blob"
        return text
    
class Road(Region):
    def __init__(self, path = 'top-bottom'):
        super().__init__('river', 'r', 0, 2, 0.01, 0.075)
        self.path = path
    def prompt(self):
        path_text = ""
        if self.path == 'top-bottom':
            path_text = "The road runs from the top of the map to the bottom."
        elif self.path == 'left-right':
            path_text = "The road runs from the left of the map to the right."
        elif self.path == 'curved':
            path_text = "The road runs from top/bottom to left/right in a curve."
        text = "Draw a road!" + path_text
        return text

# #TODO Bush is not region, but object above map
class Bush(Region):
    def __init__(self):
        super().__init__('bush', 'b', 1, 4, 0.005, 0.025)

    def prompt(self):
        text = "Draw a bush! It should look like small a closed blob"
        return text

# All unclassified regions are grass    
class Grass(Region):
    def __init__(self):
        super().__init__('grass', 'g', 1, None, 0.1, 0.5)


def generate_region(regions):
    counter = 0 # Prevent infinite loop
    while counter < 100:
        region = random.choice(regions)
        if region.instances > region.max_instances:
            continue
        else:
            return region

