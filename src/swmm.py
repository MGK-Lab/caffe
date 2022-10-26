from pyswmm import Simulation, Nodes
import hymo
import numpy as np
import pandas as pd


class swmm:
    def __init__(self, inp_file):
        # load PySWMM
        print("\n .....loading SWMM inputfile using PySWMM.....")
        self.sim = Simulation(inp_file)
        print("\n")
        # load SWMM input file using hyno package
        self._hymo_inp = hymo.SWMMInpFile(inp_file)

    def LoadNodes(self):
        self.nodes = Nodes(self.sim)

        c = ["Inv_Elv", "F_Depth", "Outfall"]
        self.nodes_info = pd.DataFrame(columns=c)

        for n in self.nodes:
            self.nodes_info.loc[n.nodeid] = pd.DataFrame(
                [[n.invert_elevation, n.full_depth, n.is_outfall()]], columns=c).loc[0]

        self.nodes_info = pd.concat(
            [self._hymo_inp.coordinates, self.nodes_info], axis=1)

        self.node_list = list(self.nodes_info.index.values)

    def InteractionInterval(self, sec):
        self.sim.step_advance(sec)

    def setNodesInflow(self, N_In):
        i = 0
        for n in self.node_list:
            self.nodes[n].generated_inflow(N_In[i])
            i += 1

    def getNodesHead(self):
        H = np.zeros(len(self.node_list))
        i = 0
        for n in self.node_list:
            H[i] = self.nodes[n].head
            i += 1

        return H

    def getNodesTotalInflow(self):
        TI = np.zeros(len(self.node_list))
        i = 0
        for n in self.node_list:
            TI[i] = self.nodes[n].total_inflow
            i += 1

        return TI

    def getNodesFlooding(self):
        TI = np.zeros(len(self.node_list))
        i = 0
        for n in self.node_list:
            TI[i] = self.nodes[n].flooding
            i += 1

        return TI

    def getNodesTotalOutflow(self):
        TO = np.zeros(len(self.node_list))
        i = 0
        for n in self.node_list:
            TO[i] = self.nodes[n].total_outflow
            i += 1

        return TO

    def CloseSimulation(self):
        self.sim.report()
        self.sim.close()
        print("\n")
