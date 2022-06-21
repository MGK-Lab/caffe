from pyswmm import Simulation, Nodes


class swmm:
    def __init__(self, inpfile):
        self.sim = Simulation(inpfile)
