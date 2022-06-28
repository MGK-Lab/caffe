from src.swmm import swmm
import numpy as np

swmm_obj = swmm('./tests/tutorial.inp')
swmm_obj.LoadNodes()
for step in swmm_obj.sim:
    print(swmm_obj.getJunctionHead())
    swmm_obj.setJunctionInflow(np.zeros(len(swmm_obj.node_list)))
swmm_obj.FinishSimulation()
