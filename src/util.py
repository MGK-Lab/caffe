import rasterio as rio
import numpy as np


def arraytoRasterIO(array, existingraster, dst_filename):
    """this function is used to save a 2D numpy array as a GIS raster map"""
    with rio.open(existingraster) as src:
        naip_meta = src.profile

    naip_meta['count'] = 1
    naip_meta['nodata'] = -999

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        dst.write(array, 1)


def DEMRead(dem_file):
    """this function is used to read digital elevation model (DEM file) of
    the 1st layer (band = 1)"""
    src = rio.open(dem_file)
    band = 1
    dem_array = src.read(band)
    msk = src.read_masks(band)

    # unrealistic height to invalid data
    dem_array[msk == 0] = 100000
    DEM = np.array(dem_array, dtype=np.float32)

    mask = np.zeros_like(dem_array, dtype=bool)
    mask[msk == 0] = True
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return DEM, mask
