from .caffe import caffe
from .swmm import swmm
import numpy as np
import sys


class csc:
    def __init__(self):
        print("\n .....Initiating 2D-1D modelling using coupled SWMM & CA-ffé.....")

    def LoadCaffe(self, DEM_file, hf, increment_constant, EV_threshold):
        self.caffe = caffe(DEM_file)
        self.caffe.setConstants(hf, increment_constant, EV_threshold)
        self.caffe_output_name = self.caffe.outputs_name

    def LoadSwmm(self, SWMM_inp):
        self.swmm = swmm(SWMM_inp)
        self.swmm.LoadNodes()

    def NodeElvCoordChecks(self):
        self.swmm_node_info = self.swmm.nodes_info.to_numpy()
        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info[:, 0], self.swmm_node_info[:, 1],
             self.swmm_node_info[:, 2] + self.swmm_node_info[:, 3], self.swmm_node_info[:, 4]))

        self.swmm_node_info[:, 0] = np.int_(
            self.swmm_node_info[:, 0] / self.caffe.length)
        self.swmm_node_info[:, 1] = np.int_(
            self.swmm_node_info[:, 1] / self.caffe.length)

        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((r[0] < 0 or r[0] > self.caffe.DEMshape[0]
                    or r[1] < 0 or r[1] > self.caffe.DEMshape[1]) and r[3] == False):
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions coordinates are out of the range in the provided DEM")

        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((abs(r[2] - self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                    > 0.01 * self.caffe.DEM[np.int_(r[0]), np.int_(r[1])]) and r[3] == False):
                print(self.swmm.node_list[i], " surface Elv != ",
                      self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions surface elevation have > 1% difference with the provided DEM")

        print("-----Nodes elevation and coordinates are compatible in both models (outfalls excluded)")

    def InteractionInterval(self, sec):
        self.swmm.InteractionInterval(sec)
        self.IntTimeStep = sec

    def RunOne_SWMMtoCaffe(self):
        floodvolume = 0
        for step in self.swmm.sim:
            floodvolume += self.swmm.getNodesFlooding()
        floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
                                       np.transpose(floodvolume)*self.IntTimeStep))
        self.swmm.CloseSimulation()

        self.caffe.ExcessVolumeMapArray(floodvolume)
        self.caffe.RunSimulation()
        self.caffe.CloseSimulation()

    def RunMulti_SWMMtoCaffe(self):
        origin_name = self.caffe.outputs_name

        for step in self.swmm.sim:
            tmp = self.swmm.getNodesFlooding()
            if (np.sum(tmp) > 0):
                floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
                                               np.transpose(tmp)*self.IntTimeStep))
                self.caffe.ExcessVolumeMapArray(floodvolume)
                self.caffe.outputs_name = origin_name + \
                    "_" + str(self.swmm.sim.current_time)
                self.caffe.RunSimulation()
                self.caffe.CloseSimulation()

        self.swmm.CloseSimulation()

        # def RunMulti_SWMMtoCaffe(self):
        #     origin_name = self.caffe.outputs_name
        #     water_levels = self.caffe.water_levels
        #
        #     for step in self.swmm.sim:
        #         tmp = self.swmm.getNodesFlooding()
        #         if (np.sum(tmp) > 0):
        #             floodvolume = np.column_stack((self.swmm_node_info[:, 0:2],
        #                                            np.transpose(tmp)*self.IntTimeStep))
        #
        #             caffe_tmp = caffe(self.DEM_file)
        #             caffe_tmp = deepcopy(self.caffe)
        #             caffe_tmp.water_levels = water_levels
        #             caffe_tmp.ExcessVolumeMapArray(floodvolume)
        #             caffe_tmp.outputs_name = origin_name + \
        #                 "_" + str(self.swmm.sim.current_time)
        #             caffe_tmp.RunSimulation()
        #             water_levels = self.caffe.water_levels
        #             caffe_tmp.CloseSimulation()
        #             del caffe_tmp
        #
        #     self.swmm.CloseSimulation()
