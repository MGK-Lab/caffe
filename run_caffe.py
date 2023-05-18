import numpy as np
import sys
sys.path.append("./src")
from caffe import caffe  # NOQA


if __name__ == "__main__":

    input_DEM_file = './tests/caffe_test.tif'
    hf = 0.09
    increment_constant = 1e-4
    EV_threshold = 2e-5

    # to make an instance of the caffe class model
    sim = caffe(input_DEM_file)
    sim.setConstants(hf, increment_constant, EV_threshold)
    sim.ExcessVolumeArray(np.array([[499, 499, 10000]]))
    sim.OpenBCArray(np.array([[300, 300]]))
    sim.RunSimulation()
    sim.setOutputPath("./tests/")
    sim.CloseSimulation()
