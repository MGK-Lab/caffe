from src.swmm import swmm
import pyswmm

swmm_obj = swmm('./tests/tutorial.inp')
swmm_obj.LoadNodes()
swmm_obj.StartSimulation()
swmm_obj.FinishSimulation()
# swmm_obj.NodeFunction(maz)
