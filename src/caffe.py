import caffe_core as cc
import numpy as np
import warnings as ws
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
        self.setConstants_has_been_called = False
        self.water_levels = np.copy(self.DEM).astype(np.float32)
        self.extra_volume_map = np.zeros_like(self.DEM, dtype=np.float32)
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

    def setOutputName(self, fn):
        self.outputs_name = fn

    def ExtraVolumeMapArray(self, EVM_np):
        EVM_np[:, 0] = EVM_np[:, 0] / self.length - 1
        EVM_np[:, 1] = EVM_np[:, 1] / self.length - 1
        for r in EVM_np:
            self.extra_volume_map[r[0], r[1]] = r[2]

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
                "CA-ffé constants were not set by the user, use setConstants method and try again")

        self.extra_total_volume = np.sum(self.extra_volume_map)
        print("total volume to be spread (m3) =", self.extra_total_volume)
        self.extra_water_column_map = self.extra_volume_map / self.cell_area
        # For masked areas (i.e. borders of the calculation domain), a very large negative number is picked in order to make sure that these cells will never be activated during the simulation
        self.extra_water_column_map[self.mask] = -1*(10**6)

        # CAffe_engine is designed to work with 1D arrays to provide faster simulations
        self.DEM1d = self.DEM.ravel()
        self.extra_water_column_map = self.extra_water_column_map.ravel()
        self.max_f = np.zeros_like(
            self.extra_water_column_map, dtype=np.float32)
        self.water_levels = self.water_levels.ravel()
        l = int(len(self.extra_water_column_map) - self.DEMshape[1])

        cc.CAffe_engine(self.water_levels, self.extra_water_column_map,
                        self.max_f, self.DEMshape[0], self.DEMshape[1],
                        self.cell_area, self.extra_total_volume, self.ic,
                        self.hf, self.EVt, l)

        self.water_levels = self.water_levels.reshape(self.DEMshape)
        self.water_levels[self.mask] = 0

        self.water_depths = self.water_levels - self.DEM
        self.water_depths[self.mask] = 0

        self.max_water_levels = self.max_f.reshape(self.DEMshape)
        self.max_water_levels = np.maximum(
            self.water_levels, self.max_water_levels)
        self.max_water_levels[self.mask] = 0

        self.max_water_depths = self.max_water_levels - self.DEM
        self.max_water_depths[self.mask] = 0

        self.DEM[self.mask] = 0

        print("Simulation finished in", (time.time() - self.begining), "seconds")

    def Report(self):
        print("\n")
        print("water depth (min, max): ", np.min(
            self.water_depths), ", ", np.max(self.water_depths))

        print("max water depth (min, max): ", np.min(self.max_water_depths),
              ", ", np.max(self.max_water_depths))

        print("water level (min, max): ", np.min(
            self.water_levels), ", ", np.max(self.water_levels))

        print("max water level (min, max): ", np.min(self.max_water_levels),
              ", ", np.max(self.max_water_levels))

        print("Sum spreaded water volume: ", np.sum(
            self.water_depths) * self.cell_area)
        self.extra_water_column_map = self.extra_water_column_map.reshape(
            self.DEMshape)
        self.extra_water_column_map[self.mask] += 1 * (10 ** 6)
        print("Total volume to out:", np.sum(
            self.extra_water_column_map[self.mask]) * self.cell_area)
        print("Left excess volume:", np.sum(
            self.extra_water_column_map[self.mask == False]) * self.cell_area)

        fn = self.outputs_path + self.outputs_name + '_wl.tif'
        util.arraytoRasterIO(self.water_levels, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_wd.tif'
        util.arraytoRasterIO(self.water_depths, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_mwd.tif'
        util.arraytoRasterIO(self.max_water_depths, self.dem_file, fn)
