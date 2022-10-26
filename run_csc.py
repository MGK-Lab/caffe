from src.caffe_swmm_coupled import csc
import numpy as np

if __name__ == "__main__":

    DEM_file = './tests/csc_dem.tif'
    hf = 0.09
    increment_constant = 0.0001
    EV_threshold = 0.000002

    SWMM_inp = './tests/csc_test.inp'

    csc_obj = csc()
    csc_obj.LoadCaffe(DEM_file, hf, increment_constant, EV_threshold)
    # an example of calling caffe class members
    csc_obj.caffe.setDEMCellSize(1)
    csc_obj.caffe.setOutputPath("./maz/")

    csc_obj.LoadSwmm(SWMM_inp)

    csc_obj.NodeElvCoordChecks()
    csc_obj.InteractionInterval(600)
    csc_obj.caffe.setSpreadVolumeCutoff(0.01)
    # csc_obj.RunOne_SWMMtoCaffe()
    csc_obj.RunMulti_SWMMtoCaffe()
