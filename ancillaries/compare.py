import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./src")
from caffe import caffe  # NOQA
import util as ut  # NOQA
import visual as vs  # NOQA


if __name__ == "__main__":

    WD1_file = './tests/c_ver_wd.tif'
    WD2_file = './tests/serial_wd.tif'
    diff_file = './tests/diff.tif'

    WD1, a, b, c = ut.RasterToArray(WD1_file)
    WD2, a, b, c = ut.RasterToArray(WD2_file)

    diff = (WD1-WD2)
    # ut.ArrayToRaster(diff, diff_file, WD2_file)

    # plt.imshow(np.flipud(diff), origin='lower',
    #            interpolation='nearest', cmap="gist_earth")
    # plt.colorbar()
    # plt.show()

    diff = diff[np.logical_not(np.isnan(diff))]
    diff = diff[np.logical_not(np.isinf(diff))]
    print("Max difference: ", np.amax(diff))
    print("Min difference: ", np.amin(diff))
    print("Avg difference: ", np.mean(diff))
    print("Sum difference: ", np.sum(diff))
    print("Max WD1: ", np.amax(WD1))
    print("Min WD1: ", np.amin(WD1))
    print("Avg WD1: ", np.mean(WD1))
    print("Sum WD1: ", np.sum(WD1))
    print("Max WD2: ", np.amax(WD2))
    print("Min WD2: ", np.amin(WD2))
    print("Avg WD2: ", np.mean(WD2))
    print("Sum WD2: ", np.sum(WD2))
