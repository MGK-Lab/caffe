from pyswmm import Simulation, Nodes
import numpy as np


class swmm:
    def __init__(self, inpfile):
        self.sim = Simulation(inpfile)

    def load_nodes(self):
        self.nodes = Nodes(self.sim)

    def nodes_xy_check(self):
        for n in self.nodes:
            print(n.invert_elevation)
            # print(n.xcoord)
