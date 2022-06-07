import Caffe
import numpy as np
import time
import rasterio as rio

now = time.time()

def arraytoRasterIO(array, existingraster, dst_filename):
    """this function is used to save a 2D numpy array as a GIS raster map"""
    with rio.open(existingraster) as src:
        naip_meta = src.profile

    naip_meta['count'] = 1
    naip_meta['nodata'] = -999

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        dst.write(array, 1)

def run_CAffe(dem_file, increment_constant, hf, result_path,result_name, EV_threshold):

    """ This function has three main actions:
        1. Preprocessing: which involves reading digital elevation model (DEM) raster files of topography into numpy arrays and preparing pther necessary inputs for the CAffe_engine
        2. running CAffe_engine
        3. Post-processing which is saving 2D arrays of estimated results into gis raster maps"""


    # Pre-processing
    src = rio.open(dem_file) # reads digital elevation model (DEM file)
    dtypee = src.profile['dtype']
    print(dtypee)

    cell_area = 1 #we know the test data has a cell area of 1 meter
    dem_array = src.read(1)
    mask = np.zeros_like(dem_array, dtype=bool)
    msk = src.read_masks(1)
    dem_array[msk == 0] = 100000
    mask[msk == 0] = True
    DEM = np.array(dem_array, dtype=np.float32)

    DEMshape0, DEMshape1 = DEM.shape[0], DEM.shape[1]

    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True


    dem = DEM.ravel() # CAffe_engine is designed to work with 1D arrays to provide faster simulations
    water_levels = np.copy(dem).astype(np.float32) # at the start of simulation, water level is equal to ground level (i.e. water depeth = 0)
    extra_volume_map = np.zeros_like(DEM, dtype=np.float32) #extra_volume_map is an array that include flood water volumes to be spread in the simulation
    extra_volume_map[118,703] = 8000 # for this case we assume that at point (118, 703) we have a flooding event that generated 8,000 m3 flood water.

    total_vol = np.sum(extra_volume_map)
    print("\nstarted run at:", time.ctime())
    print("total volume to be spread (m3) =", total_vol)

    extra_volume_map = extra_volume_map / cell_area # turn volume into depth of water column
    extra_volume_map[mask] = -1*(10**6) # for masked areas (i.e. borders of the calculation domain) we pick a very large negative number in order to make sure that these cells will never be activated during the simulation

    extra_volume_map = extra_volume_map.ravel()
    max_f = np.zeros_like(extra_volume_map, dtype=np.float32)


    length = int(len(extra_volume_map) - DEMshape1)



    # 2. running CAffe_engine
    Caffe.CAffe_engine(water_levels, extra_volume_map, max_f, DEMshape0,
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
    print("total_volume_to_out:", np.sum(extra_volume_map[mask]))
    print("left excess volume:", np.sum(extra_volume_map[mask==False]))

    wl_filename = result_path + result_name + 'wl.tif'


    wl_filename = result_path + result_name + 'wl.tif'
    arraytoRasterIO(water_levels, dem_file, wl_filename)
    depth_filename = result_path + result_name + 'depth.tif'
    arraytoRasterIO(np.array(water_depth, dtype=np.float32), dem_file, depth_filename)
    max_depth_filename = result_path + result_name + 'max_depth.tif'
    arraytoRasterIO(np.array(max_water_depth, dtype=np.float32), dem_file, max_depth_filename)

    print(max_water_depth.shape)
    print("Simulation finished in", (time.time() - now), "seconds")


if __name__ == "__main__":

    # user input data:
    input_DEM_file = './data/dem_s1.tif'
    hf = 0.09                   # First CAffe model parameter selected by user
    increment_constant = 0.0005 # Second CAffe model parameter selected by user
    EV_threshold = 0.00002
    result_path = './results/'
    result_name = "hf" + str(hf)+"_IC_" + str(increment_constant)

    # run model
    run_CAffe(input_DEM_file, increment_constant, hf, result_path,result_name, EV_threshold)
