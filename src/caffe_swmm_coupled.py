from .caffe import caffe
from .swmm import swmm
import numpy as np


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

    def NodeElevationCheck(self):
        swmm_node_info = self.swmm.nodes_info.to_numpy()
        swmm_node_info = np.column_stack(
            (swmm_node_info[:, 0], swmm_node_info[:, 1],
             swmm_node_info[:, 2] + swmm_node_info[:, 3]))

        swmm_node_info[:, 0] = np.int_(
            swmm_node_info[:, 0] / self.caffe.length) - 1
        swmm_node_info[:, 1] = np.int_(
            swmm_node_info[:, 1] / self.caffe.length) - 1
        print(swmm_node_info)
        # EVM_np[:, 1] = EVM_np[:, 1] / self.length - 1
        # for r in EVM_np:
        #     self.excess_volume_map[r[0], r[1]] = r[2]
