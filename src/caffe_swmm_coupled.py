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
        swmm_node_info = self.swmm.nodes_info.to_numpy()
        swmm_node_info = np.column_stack(
            (swmm_node_info[:, 0], swmm_node_info[:, 1],
             swmm_node_info[:, 2] + swmm_node_info[:, 3]))

        swmm_node_info[:, 0] = np.int_(
            swmm_node_info[:, 0] / self.caffe.length) - 1
        swmm_node_info[:, 1] = np.int_(
            swmm_node_info[:, 1] / self.caffe.length) - 1

        err = False
        i = 0
        for r in swmm_node_info:
            if (r[0] < 0 or r[0] > self.caffe.DEMshape[0]
                    or r[1] < 0 or r[1] > self.caffe.DEMshape[1]):
                print(self.swmm.node_list[i])
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions coordinates are out of the range in the provided DEM")

        err = False
        i = 0
        for r in swmm_node_info:
            if (abs(r[2] - self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                    > 0.01 * self.caffe.DEM[np.int_(r[0]), np.int_(r[1])]):
                print(self.swmm.node_list[i], " surface Elv != ",
                      self.caffe.DEM[np.int_(r[0]), np.int_(r[1])])
                err = True
            i += 1
        if err:
            sys.exit(
                "The above SWMM junctions surface elevation have > 1% difference with the provided DEM")

        print(swmm_node_info)
