from src.swmm import swmm
import numpy as np

if __name__ == "__main__":

    swmm_obj = swmm('./tests/swmm_test.inp')
    swmm_obj.LoadNodes()
    for step in swmm_obj.sim:
        print(swmm_obj.getNodesHead())
        swmm_obj.setNodesInflow(np.zeros(len(swmm_obj.node_list)))
    swmm_obj.FinishSimulation()
