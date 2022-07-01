from src.caffe import caffe
import numpy as np
if __name__ == "__main__":

    # result_path = './tests/'
    # result_name = "hf" + str(hf)+"_IC_" + str(increment_constant)

    # run model
    # run_caffe(input_DEM_file, increment_constant, hf,
    # result_path, result_name, EV_threshold)

    input_DEM_file = './tests/dem_s1.tif'
    hf = 0.09                   # First CAffe model parameter selected by user
    increment_constant = 0.0005  # Second CAffe model parameter selected by user
    EV_threshold = 0.00002

    # to make a caffe model
    sim = caffe(input_DEM_file)
    # to set the caffe constants
    sim.setConstants(hf, increment_constant, EV_threshold)
    # to set DEM cell size (the default is 1)
    sim.setDEMCellSize(1)
    sim.ExtraVolumeMapArray(np.array([[119, 704, 8000]]))
    sim.RunSimulation()
    sim.setOutputPath("./tests/")
    sim.CloseSimulation()
