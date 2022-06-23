from pyswmm import Simulation, Nodes
import numpy as np
import hymo
import pandas as pd


class swmm:
    def __init__(self, inpfile):
        # load PySWMM
        self.sim = Simulation(inpfile)
        print("\n")
        # load SWMM input file using hyno package
        self._hymo_inp = hymo.SWMMInpFile(inpfile)

    def LoadNodes(self):
        self.nodes = Nodes(self.sim)

        c = ["Inv_Elv", "F_Depth"]
        self.nodes_info = pd.DataFrame(columns=c)

        for n in self.nodes:
            self.nodes_info.loc[n.nodeid] = pd.DataFrame(
                [[n.invert_elevation, n.full_depth]], columns=c).loc[0]

        self.nodes_info = pd.concat(
            [self._hymo_inp.coordinates, self.nodes_info], axis=1)

    def MapInfo(self, dim, unit):
        # should be first implemented in hymo
        pass

    def InteractionInterval(self, sec):
        self.sim.step_advance(sec)

    def NodeFunction(self, function):
        function(self.nodes)

    def StartSimulation(self):
        for step in self.sim:
            for n in self.nodes:
                self.NodeFunction(n)

    def FinishSimulation(self):
        self.sim.report()
        self.sim.close()
        print("\n")
