import caffe_core
import numpy as np
import math
import time
import sys
import util
import os
from dateutil.parser import parse
import pandas as pd


class caffe():
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.DEM, self.mask_dem, self.bounds, self.length = util.RasterToArray(
            dem_file)
        self.ClosedBC = self.mask_dem
        self.DEMshape = self.DEM.shape

        self.DEM[self.ClosedBC == True] = np.amax(self.DEM)

        # to initialise a CAffe model
        self.BCtol = 1.0e6
        self.cell_area = self.length**2
        self.vol_cutoff = 0.1
        self.setConstants_has_been_called = False
        self.water_levels = np.copy(self.DEM).astype(np.double)
        self.water_depths = np.zeros_like(self.DEM, dtype=np.double)
        self.excess_volume_map = np.zeros_like(self.DEM, dtype=np.double)
        self.OpenBC = np.zeros_like(self.DEM, dtype=np.bool)
        self.waterdepth_excess = False
        self.outputs_path = "./"
        name = dem_file.split('/')
        name = name[-1].split('.')
        self.outputs_name = name[0] + "_out"
        self.CBC_cells = np.array([])
        self.OBC_cells = np.array([])
        self.start_date_time = None
        self.end_date_time = None
        self.initialised = False
        self.RainOnGrid = False
        self.rain = None

    def CloseSimulation(self):
        print("\n .....closing and reporting the CA-ffé simulation.....")
        self.ReportScreen()
        self.ReportFile()
        print("\n", time.ctime(), "\n")

    def setOutputPath(self, fp):
        self.outputs_path = fp

    def setOutputName(self, fn):
        self.outputs_name = fn

    def set_simulation_dates(self, start_date_time, end_date_time):
        # the format should be "year-month-day hour:minute"
        self.start_date_time = parse(start_date_time)
        self.end_date_time = parse(end_date_time)

    def setConstants(self, hf, ic, EVt):
        self.setConstants_has_been_called = True
        # First CAffe model parameter selected by user
        self.hf = hf
        # Second CAffe model parameter selected by user
        self.ic = ic
        self.EVt = EVt

    def readRainfallSeries(self, filename, type='depth'):
        # the first data should be zero in rain column
        # the depth should be in mm while intensity in mm/hr

        self.RainOnGrid = True
        intensity = False
        if type.lower() == 'intensity':
            intensity = True

        data = pd.read_csv(filename, parse_dates=[
                           0], dtype={1: float}, header=None)
        data.columns = ['Time', 'Rain']  # Renaming columns
        time_diff = data['Time'].diff().dt.total_seconds()
        data['TimeDiff'] = time_diff

        if intensity:
            data['Rain'] = data.apply(
                lambda row: row['Rain']*row['TimeDiff']/3600, axis=1)

        data['Rain'] = data.apply(lambda row: row['Rain']/1000, axis=1)
        self.rain = data

    def setDEMCellSize(self, length):
        # the cell size is extrated from the opened DEM file
        # if the user likes to set it manually, this function should be used
        self.length = length
        self.cell_area = length**2

    def ExcessVolumeArray(self, EVM_np, add=False):
        # this function gets an array with 3 columns: x coord, y coord, volume
        # note: it uses local coordinates of the array not Georeference
        EVM_np = EVM_np / self.length
        for r in EVM_np:
            if add:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] += r[2]
            else:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = r[2]

    def ExcessWaterDepthFile(self, wd_file):
        # this takes water depths from a file to spread
        self.waterdepth_excess = True
        self.excess_volume_map, mask, tmp1, tmp2 = util.RasterToArray(wd_file)

    def ExcessWaterDepthArray(self, wd_np):
        # this takes water depths from an array to spread
        self.waterdepth_excess = True
        self.excess_volume_map = wd_np

    def OpenBCArray(self, OBCM_np):
        # it takes an array of x and y coords (local) to create a raster-like
        # array of open boundary cells (as big wells)
        self.OBC_cells = np.zeros_like(OBCM_np, dtype=np.int)
        OBCM_np = OBCM_np / self.length

        i = 0
        for r in OBCM_np:
            self.OpenBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.OBC_cells[i, 0] = int(np.ceil(r[0]))
            self.OBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1

    def ClosedBCArray(self, CBCM_np):
        # it takes an array of x and y coords (local) to create a raster-like
        # array of closed boundary cells (as big spikes)
        self.CBC_cells = np.zeros_like(CBCM_np, dtype=np.int)
        CBCM_np = CBCM_np / self.length

        i = 0
        for r in CBCM_np:
            self.ClosedBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.CBC_cells[i, 0] = int(np.ceil(r[0]))
            self.CBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1

    def SetBCs(self):
        # it modified the defined boundary cells' elevation
        self.water_levels[self.ClosedBC] += self.BCtol
        self.water_levels[self.OpenBC] -= self.BCtol

    def ResetBCs(self):
        # it restore boundary cells' elevation to normal
        self.water_levels[self.ClosedBC] -= self.BCtol
        self.water_levels[self.OpenBC] += self.BCtol

    def Reset_WL_EVM(self):
        # it resets water levels and excess volume map and it also removes
        # defined boundary cells (closed and open) from the lists
        self.water_levels = np.copy(self.DEM).astype(np.double)
        self.excess_volume_map = np.zeros_like(self.DEM, dtype=np.double)

        for r in self.CBC_cells:
            self.ClosedBC[r[0], r[1]] = False

        for r in self.OBC_cells:
            self.OpenBC[r[0], r[1]] = False

    def InitialChecks(self):
        # any initial checks should be performed here
        if not self.setConstants_has_been_called:
            sys.exit(
                "CA-ffé constants were not set by the user, use setConstants",
                " method and try again")

        if ((self.start_date_time == None or self.start_date_time == None) and self.RainOnGrid):
            self.start_date_time = self.rain['Time'].iloc[0]
            self.end_date_time = self.rain['Time'].iloc[-1]
            print(
                "\nWarning: simulation times were not set, so the rain times were used instead\n")

        if self.RainOnGrid:
            if self.start_date_time <= self.rain['Time'].iloc[0]:
                self.start_date_time = self.rain['Time'].iloc[0]
            else:
                sys.exit(
                    "The simulation start time is after the begining of the rain")

            if not self.end_date_time >= self.rain['Time'].iloc[-1]:
                sys.exit(
                    "The simulation end time is before the end of the rain")

        self.initialised = True

    def RunSimulation(self):
        self.begining = time.time()
        if not self.initialised:
            self.InitialChecks()

        # it prepares excess_water_column_map array based on input type
        if self.waterdepth_excess:
            self.excess_water_column_map = self.excess_volume_map
        else:
            self.excess_water_column_map = self.excess_volume_map / self.cell_area

        self.excess_total_volume = np.sum(
            self.excess_water_column_map) * self.cell_area
        print("total volume to be spread (m3) =", self.excess_total_volume)

        # it avoids decimal rounding otherwise a very big value would be used
        self.BCtol = math.ceil(self.excess_total_volume / self.cell_area)
        self.SetBCs()

        # CAffe_engine works with 1D arrays to provide faster simulations
        # all 2D arrays are converted to 1D
        ClosedBC = self.ClosedBC.ravel()
        self.excess_water_column_map = self.excess_water_column_map.ravel()
        self.water_levels = self.water_levels.ravel()

        self.max_f = np.zeros_like(
            self.excess_water_column_map, dtype=np.double)

        caffe_core.CAffe_engine(self.water_levels, ClosedBC,
                                self.excess_water_column_map,
                                self.max_f, np.asarray(self.DEMshape),
                                self.cell_area, self.ic, self.hf, self.EVt,
                                self.vol_cutoff)

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

    def RunSimulationROG(self):
        self.waterdepth_excess = True
        rog_arr = np.ones_like(self.DEM) * ~self.mask_dem
        name = self.outputs_name
        print(np.sum(rog_arr))

        for i in range(1, len(self.rain)):
            print("\nRain time: ", self.rain['Time'].iloc[i])
            self.excess_volume_map = self.rain['Rain'].iloc[i] * \
                rog_arr + self.water_depths
            self.RunSimulation()
            self.ReportScreen()
            self.outputs_name = name + "_" + \
                self.rain['Time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
            self.ReportFile()
            self.Reset_WL_EVM()

    def ReportScreen(self):
        indices = np.logical_and(self.ClosedBC == False, self.OpenBC == False)
        print("\n")
        print("water depth (min, max):       ", np.min(
            self.water_depths[indices]), ", ", np.max(self.water_depths[indices]))

        print("max water depth (min, max):   ", np.min(self.max_water_depths[indices]),
              ", ", np.max(self.max_water_depths[indices]))

        print("water level (min, max):       ", np.min(
            self.water_levels[indices]), ", ", np.max(self.water_levels[indices]))

        print("max water level (min, max):   ", np.min(self.max_water_levels[indices]),
              ", ", np.max(self.max_water_levels[indices]))

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

        indices = np.logical_and(self.ClosedBC == False, self.OpenBC == False)
        indices = ~indices

        fn = self.outputs_path + self.outputs_name + '_wl.tif'
        wl = self.water_levels
        wl[indices] = self.DEM[indices]
        util.ArrayToRaster(wl, fn, self.dem_file)

        wd = self.water_depths
        wd[indices] = 0
        mask = np.where(wd == 0, True, False)
        print(np.sum(wd), " ", np.sum(mask))
        fn = self.outputs_path + self.outputs_name + '_wd.tif'
        util.ArrayToRaster(wd, fn, self.dem_file, ~mask)

        mwd = self.max_water_depths
        mwd[indices] = 0
        mask = np.where(mwd == 0, True, False)
        print(np.sum(mwd), " ", np.sum(mask))
        fn = self.outputs_path + self.outputs_name + '_mwd.tif'
        util.ArrayToRaster(mwd, fn, self.dem_file, ~mask)
