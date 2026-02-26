from pyswmm import Simulation, Nodes, Output
from swmm.toolkit.shared_enum import NodeAttribute
from datetime import datetime, timedelta
from swmm_api import read_inp_file
# import hymo
import numpy as np
import pandas as pd



class swmm:
    def __init__(self, inp_file):
        # load PySWMM
        print("\n .....loading SWMM inputfile using PySWMM.....")
        self.sim = Simulation(inp_file)
        print("\n")
        # load SWMM input file using hyno package
        # self._hymo_inp = hymo.SWMMInpFile(inp_file)
        self._inp = read_inp_file(inp_file)
        self.bounds = np.zeros(4)
        self.input_file = inp_file

    def LoadNodes(self):
        self.nodes = Nodes(self.sim)

        c = ["Inv_Elv", "F_Depth", "Outfall"]
        self.nodes_info = pd.DataFrame(columns=c)

        for n in self.nodes:
            self.nodes_info.loc[n.nodeid] = pd.DataFrame(
                [[n.invert_elevation, n.full_depth, n.is_outfall()]], columns=c).loc[0]
        
        coords_dict = self._inp.COORDINATES
        df_coords = pd.DataFrame({
            "X_Coord": {k: v.x for k, v in coords_dict.items()},
            "Y_Coord": {k: v.y for k, v in coords_dict.items()}
            })

        self.nodes_info = pd.concat(
            [df_coords, self.nodes_info], axis=1)
                
        self.node_list = list(self.nodes_info.index.values)
        self.No_Nodes = len(self.node_list)

        self.bounds[0] = self.nodes_info['X_Coord'].min()   # xmin
        self.bounds[1] = self.nodes_info['Y_Coord'].max()   # ymax
        self.bounds[2] = self.nodes_info['X_Coord'].max()   # xmax
        self.bounds[3] = self.nodes_info['Y_Coord'].min()   # ymin

    def Output_getNodesFlooding(self):
        # load PySWMM output file with the same file name
        out_file = self.input_file[:-3] + "out"
        self.out = Output(out_file)

        flood_volume = np.zeros(self.No_Nodes)
        report_timestep=self.out.times[-1]-self.out.times[0]
        report_timestep=report_timestep.total_seconds()/(len(self.out.times)-1)

        for n in range(self.No_Nodes):
            temp=self.out.node_series(n, NodeAttribute.FLOODING_LOSSES)
            flood_volume[n] = np.sum(np.array(list(temp.values())))
        
        flood_volume *= report_timestep

        print('\nTotal flood Volume = ',np.sum(flood_volume))
        self.out.close()

        return flood_volume

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
