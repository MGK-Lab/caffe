import caffe_core as cc
import numpy as np
import time
from . import util


class caffe():
    def __init__(self, dem_file):
        self.begining = time.time()
        print("\n .....loading DEM file using CA-ffé.....")
        self.DEM, self.mask = util.DEMRead(dem_file)
        print("\n ")

    def setWaterDepthZero(self):
        self.water_levels = np.copy(self.DEM).astype(np.float32)

    def setDEMCellSize(self, length):
        self.cell_area = length**2

    def RunSimulation(slf):
        pass


def run_caffe(dem_file, increment_constant, hf, result_path, result_name, EV_threshold):
    """ This function has three main actions:
        2. running CAffe_engine
        3. Post-processing which is saving 2D arrays of estimated results into gis raster maps"""

    now = time.time()

    # Pre-processing
    DEM, mask = util.DEMRead(dem_file)

    DEMshape0, DEMshape1 = DEM.shape

    cell_area = 1  # we know the test data has a cell area of 1 meter

    # CAffe_engine is designed to work with 1D arrays to provide faster simulations
    dem = DEM.ravel()
    # at the start of simulation, water level is equal to ground level (i.e. water depeth = 0)
    water_levels = np.copy(dem).astype(np.float32)
    # extra_volume_map is an array that include flood water volumes to be spread in the simulation
    extra_volume_map = np.zeros_like(DEM, dtype=np.float32)
    # for this case we assume that at point (118, 703) we have a flooding event that generated 8,000 m3 flood water.
    extra_volume_map[118, 703] = 8000

    total_vol = np.sum(extra_volume_map)
    print("\nstarted run at:", time.ctime())
    print("total volume to be spread (m3) =", total_vol)

    # turn volume into depth of water column
    extra_volume_map = extra_volume_map / cell_area
    # for masked areas (i.e. borders of the calculation domain) we pick a very large negative number in order to make sure that these cells will never be activated during the simulation
    extra_volume_map[mask] = -1*(10**6)

    extra_volume_map = extra_volume_map.ravel()
    max_f = np.zeros_like(extra_volume_map, dtype=np.float32)

    length = int(len(extra_volume_map) - DEMshape1)

    # 2. running CAffe_engine
    cc.CAffe_engine(water_levels, extra_volume_map, max_f, DEMshape0,
                    DEMshape1, cell_area, total_vol, increment_constant, hf, EV_threshold, length)

    # 3. Post-processing
    water_levels = water_levels.reshape(DEMshape0, DEMshape1)
    water_depth = water_levels - DEM
    print(np.min(water_depth), np.max(water_depth))
    water_depth[mask] = 0
    water_levels[mask] = 0
    DEM[mask] = 0
    max_water_level = max_f.reshape(DEMshape0, DEMshape1)
    print(np.min(max_water_level), np.max(max_water_level))
    max_water_level = np.maximum(water_levels, max_water_level)
    max_water_depth = max_water_level - DEM
    max_water_depth[mask] = 0
    print(np.min(max_water_level), np.max(max_water_level))
    print(np.min(max_water_depth), np.max(max_water_depth))

    print("sum water depths:", np.sum(water_depth) * cell_area)
    extra_volume_map = extra_volume_map.reshape(DEMshape0, DEMshape1)
    extra_volume_map[mask] += 1 * (10 ** 6)
    print("total volume to out:", np.sum(extra_volume_map[mask]))
    print("left excess volume:", np.sum(extra_volume_map[mask == False]))

    wl_filename = result_path + result_name + 'wl.tif'
    util.arraytoRasterIO(water_levels, dem_file, wl_filename)

    depth_filename = result_path + result_name + 'depth.tif'
    util.arraytoRasterIO(np.array(water_depth, dtype=np.float32),
                         dem_file, depth_filename)

    max_depth_filename = result_path + result_name + 'max_depth.tif'
    util.arraytoRasterIO(np.array(max_water_depth, dtype=np.float32),
                         dem_file, max_depth_filename)

    print(max_water_depth.shape)
    print("Simulation finished in", (time.time() - now), "seconds")
