from src.caffe_swmm_coupled import csc
import numpy as np

if __name__ == "__main__":

    DEM_file = './tests/caffe_test.tif'
    SWMM_inp = './tests/swmm_test.inp'
    hf = 0.09
    increment_constant = 0.0005
    EV_threshold = 0.00002

    csc_obj = csc()
    csc_obj.LoadCaffe(DEM_file, hf, increment_constant, EV_threshold)
    csc_obj.caffe.setDEMCellSize(15)
    csc_obj.LoadSwmm(SWMM_inp)
    csc_obj.NodeElvCoordChecks()
