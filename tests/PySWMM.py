from pyswmm import Simulation, Nodes

with Simulation('tutorial.inp') as sim:

    # sim.step_advance(100)

    node_object = Nodes(sim)
    J2 = node_object["J2"]
    print('\n')
    print(J2.invert_elevation)
    print(J2.is_junction())
    # for step in sim:
    #     print(sim.current_time, "{:.3f}".format(
    #         J2.total_inflow-J2.total_outflow))
    a = 0
    for step in sim:
        J2.generated_inflow(a)
        a = a - 0.00
        # print(J2.statistics["flooding_volume"])
        print(J2.flooding)
