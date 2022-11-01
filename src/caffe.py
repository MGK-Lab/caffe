import caffe_core
import numpy as np
import time
import sys
from . import util


class caffe():
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.DEM, self.mask = util.DEMRead(dem_file)
        self.DEMshape = self.DEM.shape

        # to initialise a CAffe model
        self.length = 1
        self.cell_area = 1
        self.vol_cutoff = 0.1
        self.setConstants_has_been_called = False
        self.water_levels = np.copy(self.DEM).astype(np.double)
        self.excess_volume_map = np.zeros_like(self.DEM, dtype=np.double)
        self.user_waterdepth_file = False
        self.outputs_path = "./"
        name = dem_file.split('.')
        name = name[1].split('/')
        self.outputs_name = name[-1] + "_out"

    def CloseSimulation(self):
        print("\n .....closing and reporting the CA-ffé simulation.....")
        self.Report()
        print("\n", time.ctime(), "\n")

    def setOutputPath(self, fp):
        self.outputs_path = fp

    def setSpreadVolumeCutoff(self, vc):
        self.vol_cutoff = vc

    def setOutputName(self, fn):
        self.outputs_name = fn

    def ExcessVolumeMapArray(self, EVM_np):
        EVM_np[:, 0] = EVM_np[:, 0] / self.length
        EVM_np[:, 1] = EVM_np[:, 1] / self.length
        for r in EVM_np:
            self.excess_volume_map[int(
                np.ceil(r[0])), int(np.ceil(r[1]))] = r[2]

    def LoadInitialExcessWaterDepthfile(self, wd_file):
        self.user_waterdepth_file = True
        self.excess_volume_map, tmp = util.DEMRead(wd_file)

    def setConstants(self, hf, ic, EVt):
        self.setConstants_has_been_called = True
        # First CAffe model parameter selected by user
        self.hf = hf
        # Second CAffe model parameter selected by user
        self.ic = ic
        self.EVt = EVt

    def setDEMCellSize(self, length):
        self.length = length
        self.cell_area = length**2

    def RunSimulation(self):
        self.begining = time.time()
        if not self.setConstants_has_been_called:
            sys.exit(
                "CA-ffé constants were not set by the user, use setConstants",
                " method and try again")

        if self.user_waterdepth_file:
            self.excess_total_volume = np.sum(
                self.excess_volume_map) * self.cell_area
            print("total volume to be spread (m3) =", self.excess_total_volume)
            self.excess_water_column_map = self.excess_volume_map
        else:
            self.excess_total_volume = np.sum(self.excess_volume_map)
            print("total volume to be spread (m3) =", self.excess_total_volume)
            self.excess_water_column_map = self.excess_volume_map / self.cell_area

        # For masked areas (i.e. borders of the calculation domain),
        # a very large negative number is picked in order to make sure that
        # these cells will never be activated during the simulation
        # self.excess_water_column_map[self.mask] = -1*(10**6)
        # self.DEM[self.mask] += 10000

        # CAffe_engine works with 1D arrays to provide faster simulations
        self.DEM1d = self.DEM.ravel()
        self.excess_water_column_map = self.excess_water_column_map.ravel()
        self.max_f = np.zeros_like(
            self.excess_water_column_map, dtype=np.double)
        self.water_levels = self.water_levels.ravel()

        caffe_core.CAffe_engine(self.water_levels, self.excess_water_column_map,
                                self.max_f, np.asarray(
                                    self.DEMshape), self.cell_area,
                                self.excess_total_volume, self.ic, self.hf,
                                self.EVt, self.vol_cutoff)

        self.water_levels = self.water_levels.reshape(self.DEMshape)
        # self.water_levels[self.mask] = 0
        # self.DEM[self.mask] -= 10000

        self.water_depths = self.water_levels - self.DEM
        # self.water_depths[self.mask] = 0

        self.max_water_levels = self.max_f.reshape(self.DEMshape)
        self.max_water_levels = np.maximum(
            self.water_levels, self.max_water_levels)
        # self.max_water_levels[self.mask] = 0

        self.max_water_depths = self.max_water_levels - self.DEM
        # self.max_water_depths[self.mask] = 0

        # self.DEM[self.mask] = 0

        print("\nSimulation finished in", (time.time() - self.begining),
              "seconds")

    def Report(self):
        print("\n")
        print("water depth (min, max):     ", np.min(
            self.water_depths), ", ", np.max(self.water_depths))

        print("max water depth (min, max): ", np.min(self.max_water_depths),
              ", ", np.max(self.max_water_depths))

        print("water level (min, max):     ", np.min(
            self.water_levels), ", ", np.max(self.water_levels))

        print("max water level (min, max): ", np.min(self.max_water_levels),
              ", ", np.max(self.max_water_levels))

        print("Sum spreaded water volume:  ", np.sum(
            self.water_depths) * self.cell_area)
        self.excess_water_column_map = self.excess_water_column_map.reshape(
            self.DEMshape)
        # self.excess_water_column_map[self.mask] += 1 * (10 ** 6)
        print("Total volume to out:        ", np.sum(
            self.excess_water_column_map[self.mask]) * self.cell_area)
        print("Left excess volume:         ", np.sum(
            self.excess_water_column_map[self.mask == False]) * self.cell_area)

        fn = self.outputs_path + self.outputs_name + '_wl.tif'
        util.arraytoRasterIO(self.water_levels, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_wd.tif'
        util.arraytoRasterIO(self.water_depths, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_mwd.tif'
        util.arraytoRasterIO(self.max_water_depths, self.dem_file, fn)
        util.arraytoRasterIO(self.max_water_depths, self.dem_file, fn)
