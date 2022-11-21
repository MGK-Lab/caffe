from src.caffe import caffe

if __name__ == "__main__":

    input_DEM_file = './tests/caffe_test.tif'
    input_WD_file = './tests/caffe_test_out_wd.tif'

    hf = 0.09
    increment_constant = 5e-4
    EV_threshold = 2e-5

    # to make a caffe model
    sim = caffe(input_DEM_file)
    # to set the caffe constants
    sim.setConstants(hf, increment_constant, EV_threshold)
    # to set DEM cell size (the default is 1)
    sim.setDEMCellSize(1)
    # to set excess volume map using coordinate
    sim.LoadInitialExcessWaterDepthFile(input_WD_file)
    sim.setSpreadVolumeCutoff(0.0001)
    sim.RunSimulation()
    sim.setOutputName("test_loadWD")
    sim.setOutputPath("./tests/")
    sim.CloseSimulation()
