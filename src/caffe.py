# ------------------------------------------------------------------------------
# Dynamic CA-ffe
# Copyright (C) 2022–2026 Maziar Gholami Korzani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

import caffe_core as core_serial
import numpy as np
import math
import time
import sys
import util
import os
from dateutil.parser import parse
import pandas as pd
import ctypes
from copy import deepcopy
import platform


class caffe():
    def __init__(self, dem_file):
        print("\n .....loading DEM file using CA-ffé.....")
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.DEM, self.mask_dem, self.bounds, self.length = util.RasterToArray(
            dem_file)
        self.ClosedBC = deepcopy(self.mask_dem)
        self.DEMshape = self.DEM.shape

        self.DEM[self.ClosedBC == True] = np.amax(self.DEM)

        # to initialise a CAffe model
        self.BCtol = 1.0e6
        self.cell_area = self.length**2
        self.vol_cutoff = 0.1
        self.setConstants_has_been_called = False
        self.water_levels = deepcopy(self.DEM)
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
        self.threads = 0

    def EnableParallelRun(self, threads, libpath=''):
        # Determine the correct library file extension based on OS
        system = platform.system()
        if system == "Linux":
            lib_ext = ".so"
        elif system == "Darwin":
            lib_ext = ".dylib"
        elif system == "Windows":
            lib_ext = ".dll"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        # If no custom path provided, use default filename in current folder
        if libpath == '':
            self.parallel_lib_path = os.path.join('.', f'caffe_core_parallel{lib_ext}')
        else:
            self.parallel_lib_path = libpath

        # Load the library
        self.lib = ctypes.CDLL(self.parallel_lib_path)
        self.threads = threads

        # Define the argument types for the C++ function for parallel computation
        self.lib.CAffe_engine.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # water_levels
            np.ctypeslib.ndpointer(
                dtype=np.bool, ndim=1, flags='C_CONTIGUOUS'),  # mask
            np.ctypeslib.ndpointer(
                dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # extra_volume_map
            np.ctypeslib.ndpointer(
                dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # max_f
            np.ctypeslib.ndpointer(
                dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),  # DEMshape
            ctypes.c_double,  # total_volume
            ctypes.c_double,  # cell_area
            ctypes.c_double,  # increment_constant
            ctypes.c_double,  # hf
            ctypes.c_double,  # EV_threshold
            ctypes.c_int     # threads
        ]

    def CloseSimulation(self, name=None):
        print("\n .....closing and reporting the CA-ffé simulation.....")
        self.ReportScreen()
        self.ReportFile(name)
        print("\n", time.ctime(), "\n")

    def setOutputPath(self, fp):
        self.outputs_path = fp
        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)

    def setOutputName(self, fn):
        self.outputs_name = fn

    def setSimulationDates(self, start_date_time, end_date_time):
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
        EVM_np[:, 0] = EVM_np[:, 0] / self.length
        EVM_np[:, 1] = EVM_np[:, 1] / self.length
        for r in EVM_np:
            if add:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] += r[2]
            else:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = r[2]

    def ExcessWaterDepthRasterFile(self, wd_file):
        # this takes water depths from a file to spread
        self.waterdepth_excess = True
        self.excess_volume_map, mask, tmp1, tmp2 = util.RasterToArray(wd_file)

    def ExcessWaterDepthRaster(self, wd_np):
        # this takes water depths from an array to spread
        self.waterdepth_excess = True
        self.excess_volume_map = deepcopy(wd_np)

    def OpenBCArray(self, OBCM_np):
        # it takes an array of x and y coords (local) to create a raster-like
        # array of open boundary cells (as big wells)
        self.OBC_cells = np.zeros_like(OBCM_np, dtype=np.int64)
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
        self.CBC_cells = np.zeros_like(CBCM_np, dtype=np.int64)
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
        self.water_levels = deepcopy(self.DEM)
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
            self.excess_water_column_map = deepcopy(self.excess_volume_map)
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

        if self.threads < 2:
            print("Serial version used")
            core_serial.CAffe_engine(
                self.water_levels, ClosedBC, self.excess_water_column_map,
                self.max_f, np.asarray(self.DEMshape), self.excess_total_volume,
                self.cell_area, self.ic, self.hf, self.EVt)
        else:
            print("Parallel version used")
            self.lib.CAffe_engine(
                self.water_levels, ClosedBC, self.excess_water_column_map,
                self.max_f, np.asarray(self.DEMshape),
                self.excess_total_volume, self.cell_area, self.ic, self.hf,
                self.EVt, self.threads)

        self.water_levels = self.water_levels + self.excess_water_column_map
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

        for i in range(1, len(self.rain)):
            name = self.StepSimulationROG(i)
            self.ReportFile(name)
            self.Reset_WL_EVM()

    def StepSimulationROG(self, i):
        print("\nRain time: ", self.rain['Time'].iloc[i])
        rog_arr = np.ones_like(self.DEM) * ~self.mask_dem
        self.excess_volume_map = self.rain['Rain'].iloc[i] * \
            rog_arr + self.water_depths
        self.RunSimulation()
        self.ReportScreen()
        name = self.outputs_name + "_rain_" + \
            self.rain['Time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')

        return name

    def ReportScreen(self):
        indices = np.logical_and(self.ClosedBC == False, self.OpenBC == False)
        print("\n")
        print("water depth (min, max):       ", np.min(
            self.water_depths[indices]),
            ", ", np.max(self.water_depths[indices]))

        print("max water depth (min, max):   ", np.min(
            self.max_water_depths[indices]),
            ", ", np.max(self.max_water_depths[indices]))

        print("water level (min, max):       ", np.min(
            self.water_levels[indices]),
            ", ", np.max(self.water_levels[indices]))

        print("max water level (min, max):   ", np.min(
            self.max_water_levels[indices]),
            ", ", np.max(self.max_water_levels[indices]))

        print("Sum of total spreaded volume: ", (np.sum(
            self.water_depths) - np.sum(self.excess_water_column_map)) *
            self.cell_area)

        print("Sum of non-spreaded volume:   ", np.sum(
            self.excess_water_column_map[self.ClosedBC == False])
            * self.cell_area)

        print("Left volume at Closed BC:     ", np.sum(
            self.excess_water_column_map[self.ClosedBC]) * self.cell_area)

        print("Left volume at Open BC:       ", np.sum(
            self.water_depths[self.OpenBC]) * self.cell_area)

    def ReportFile(self, name=None):
        if name == None:
            name = self.outputs_name

        indices = np.logical_and(self.ClosedBC == False, self.OpenBC == False)
        indices = ~indices

        fn = self.outputs_path + name + '_wl.tif'
        wl = deepcopy(self.water_levels)
        wl[indices] = self.DEM[indices]
        util.ArrayToRaster(wl, fn, self.dem_file)

        wd = deepcopy(self.water_depths)
        wd[indices] = 0
        mask = np.where(wd == 0, True, False)
        fn = self.outputs_path + name + '_wd.tif'
        util.ArrayToRaster(wd, fn, self.dem_file, ~mask)

        mwd = deepcopy(self.max_water_depths)
        mwd[indices] = 0
        mask = np.where(mwd == 0, True, False)
        fn = self.outputs_path + name + '_mwd.tif'
        util.ArrayToRaster(mwd, fn, self.dem_file, ~mask)
