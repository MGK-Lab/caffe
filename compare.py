import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./src")
from caffe import caffe  # NOQA
import util as ut  # NOQA
import visual as vs  # NOQA


if __name__ == "__main__":

    WD1_file = './tests/caffe_test_out_wd.tif'
    WD2_file = './tests/test_loadWD_wd.tif'
    diff_file = './tests/diff.tif'

    WD1, tmp = ut.DEMRead(WD1_file)
    WD2, tmp = ut.DEMRead(WD2_file)

    diff = (WD1-WD2)
    ut.DEMGenerate(diff, diff_file)

    diff = diff[np.logical_not(np.isnan(diff))]
    diff = diff[np.logical_not(np.isinf(diff))]
    print("Max difference: ", np.amax(diff))
    print("Min difference: ", np.amin(diff))
    print("Avg difference: ", np.mean(diff))
    print("Sum difference: ", np.sum(diff))

    vs.PlotDEM3d(diff_file, 1)
    plt.show()
