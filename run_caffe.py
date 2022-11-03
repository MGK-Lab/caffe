from src.caffe import caffe
import numpy as np


if __name__ == "__main__":

    input_DEM_file = './tests/caffe_test.tif'
    hf = 0.09
    increment_constant = 5e-4
    EV_threshold = 2e-5

    # to make a caffe model
    sim = caffe(input_DEM_file)
    # to set the caffe constants
    sim.setConstants(hf, increment_constant, EV_threshold)
    # to set DEM cell size (the default is 1)
    sim.setDEMCellSize(1)
    sim.setSpreadVolumeCutoff(0.0001)
    # to set excess volume map using coordinate
    sim.ExcessVolumeMapArray(np.array([[499, 499, 15000]]))
    sim.OpenBCMapArray(np.array([[300, 300]]))
    sim.RunSimulation()
    sim.setOutputPath("./tests/")
    sim.CloseSimulation()
