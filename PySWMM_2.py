from src import swmm
import hymo

swmm_obj = swmm.swmm('./tests/tutorial.inp')

swmm_obj.load_nodes()
swmm_obj.nodes_xy_check()

inp = hymo.SWMMInpFile('./tests/tutorial.inp')

print(inp.coordinates)
