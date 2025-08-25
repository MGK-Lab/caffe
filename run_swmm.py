import numpy as np
import sys
sys.path.append("./src")
from src.caffe_swmm_coupled import csc  # NOQA
from cswmm import swmm  # NOQA

if __name__ == "__main__":

    swmm_obj = swmm('./tests/swmm_test.inp')
    swmm_obj.LoadNodes()
    for step in swmm_obj.sim:
        print("Node heads: " + str(swmm_obj.getNodesHead()))
        swmm_obj.setNodesInflow(np.zeros(len(swmm_obj.node_list)))
    swmm_obj.CloseSimulation()
