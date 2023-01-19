import caffe_core
import numpy as np
import time
import sys
from . import util
import os


class caffe():
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.DEM, self.ClosedBC, self.bounds = util.DEMRead(dem_file)
        self.DEMshape = self.DEM.shape

        # to initialise a CAffe model
        self.BCtol = 1e9
        self.length = 1
        self.cell_area = 1
        self.vol_cutoff = 0.1
        self.setConstants_has_been_called = False
        self.water_levels = np.copy(self.DEM).astype(np.double)
        self.excess_volume_map = np.zeros_like(self.DEM, dtype=np.double)
        self.OpenBC = np.zeros_like(self.DEM, dtype=np.bool)
        self.user_waterdepth_file = False
        self.outputs_path = "./"
        name = dem_file.split('.')
        name = name[1].split('/')
        self.outputs_name = name[-1] + "_out"
        self.CBC_cells = np.array([])
        self.OBC_cells = np.array([])

    def CloseSimulation(self):
        print("\n .....closing and reporting the CA-ffé simulation.....")
        self.ReportScreen()
        self.ReportFile()
        print("\n", time.ctime(), "\n")

    def setOutputPath(self, fp):
        self.outputs_path = fp

    def setSpreadVolumeCutoff(self, vc):
        self.vol_cutoff = vc

    def setOutputName(self, fn):
        self.outputs_name = fn

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

    def ExcessVolumeMapArray(self, EVM_np, add=False):
        EVM_np[:, 0] = EVM_np[:, 0] / self.length
        EVM_np[:, 1] = EVM_np[:, 1] / self.length
        for r in EVM_np:
            if add:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] += r[2]
            else:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = r[2]

    def LoadInitialExcessWaterDepthFile(self, wd_file):
        self.user_waterdepth_file = True
        self.excess_volume_map, tmp, bounds = util.DEMRead(wd_file)

    def LoadInitialExcessWaterDepthArray(self, wd_np):
        self.user_waterdepth_file = True
        self.excess_volume_map = wd_np

    def OpenBCMapArray(self, OBCM_np):
        self.OBC_cells = np.zeros_like(OBCM_np, dtype=np.int)
        OBCM_np[:, 0] = OBCM_np[:, 0] / self.length
        OBCM_np[:, 1] = OBCM_np[:, 1] / self.length

        i = 0
        for r in OBCM_np:
            self.OpenBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.OBC_cells[i, 0] = int(np.ceil(r[0]))
            self.OBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1

    def ClosedBCMapArray(self, CBCM_np):
        self.CBC_cells = np.zeros_like(CBCM_np, dtype=np.int)
        CBCM_np[:, 0] = CBCM_np[:, 0] / self.length
        CBCM_np[:, 1] = CBCM_np[:, 1] / self.length

        i = 0
        for r in CBCM_np:
            self.ClosedBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.CBC_cells[i, 0] = int(np.ceil(r[0]))
            self.CBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1

    def SetBCs(self):
        self.water_levels[self.ClosedBC] += self.BCtol
        self.water_levels[self.OpenBC] -= self.BCtol

    def ResetBCs(self):
        self.water_levels[self.ClosedBC] -= self.BCtol
        self.water_levels[self.OpenBC] += self.BCtol

    def Reset_WL_EVM(self):
        self.water_levels = np.copy(self.DEM).astype(np.double)
        self.excess_volume_map = np.zeros_like(self.DEM, dtype=np.double)

        for r in self.CBC_cells:
            self.ClosedBC[r[0], r[1]] = False

        for r in self.OBC_cells:
            self.OpenBC[r[0], r[1]] = False

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

        # CAffe_engine works with 1D arrays to provide faster simulations
        self.SetBCs()
        self.ClosedBC = self.ClosedBC.ravel()
        self.excess_water_column_map = self.excess_water_column_map.ravel()
        self.water_levels = self.water_levels.ravel()

        self.max_f = np.zeros_like(
            self.excess_water_column_map, dtype=np.double)

        caffe_core.CAffe_engine(self.water_levels, self.ClosedBC,
                                self.excess_water_column_map,
                                self.max_f, np.asarray(self.DEMshape),
                                self.cell_area, self.ic, self.hf, self.EVt,
                                self.vol_cutoff)

        self.ClosedBC = self.ClosedBC.reshape(self.DEMshape)
        self.water_levels = self.water_levels.reshape(self.DEMshape)
        self.ResetBCs()
        self.water_depths = self.water_levels - self.DEM
        self.max_water_levels = self.max_f.reshape(self.DEMshape)
        self.max_water_levels = np.maximum(
            self.water_levels, self.max_water_levels)
        self.max_water_depths = self.max_water_levels - self.DEM
        self.excess_water_column_map = self.excess_water_column_map.reshape(
            self.DEMshape)

        print("\nSimulation finished in", (time.time() - self.begining),
              "seconds")

    def ReportScreen(self):
        print("\n")
        print("water depth (min, max):       ", np.min(
            self.water_depths), ", ", np.max(self.water_depths))

        print("max water depth (min, max):   ", np.min(self.max_water_depths),
              ", ", np.max(self.max_water_depths))

        print("water level (min, max):       ", np.min(
            self.water_levels), ", ", np.max(self.water_levels))

        print("max water level (min, max):   ", np.min(self.max_water_levels),
              ", ", np.max(self.max_water_levels))

        print("Sum of total spreaded volume: ", np.sum(
            self.water_depths) * self.cell_area)

        print("Sum of non-spreaded volume:   ", np.sum(
            self.excess_water_column_map[self.ClosedBC == False])
            * self.cell_area)

        print("Left volume at Closed BC:     ", np.sum(
            self.excess_water_column_map[self.ClosedBC]) * self.cell_area)

        print("Left volume at Open BC:       ", np.sum(
            self.water_depths[self.OpenBC]) * self.cell_area)

    def ReportFile(self):
        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)

        fn = self.outputs_path + self.outputs_name + '_wl.tif'
        util.arraytoRasterIO(self.water_levels, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_wd.tif'
        util.arraytoRasterIO(self.water_depths, self.dem_file, fn)

        fn = self.outputs_path + self.outputs_name + '_mwd.tif'
        util.arraytoRasterIO(self.max_water_depths, self.dem_file, fn)
        util.arraytoRasterIO(self.max_water_depths, self.dem_file, fn)
