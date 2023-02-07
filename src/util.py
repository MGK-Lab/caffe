from __future__ import division
import rasterio as rio
from rasterio.profiles import DefaultGTiffProfile
import numpy as np
import warnings


def arraytoRasterIO(array, existingraster, dst_filename):
    """this function is used to save a 2D numpy array as a GIS raster map"""
    with rio.open(existingraster) as src:
        naip_meta = src.profile
        mask = src.dataset_mask()

    naip_meta['count'] = 1
    naip_meta['nodata'] = -999
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        dst.write(np.ma.masked_array(array, mask), 1)


def DEMRead(dem_file):
    """this function is used to read digital elevation model (DEM file) of
    the 1st layer (band = 1)"""
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    src = rio.open(dem_file)
    band = 1
    DEM = src.read(band)
    msk = src.read_masks(band)
    DEM = DEM.astype(np.double)
    bounds = np.zeros(4)
    bounds = [src.bounds.left, src.bounds.top, src.bounds.right, src.bounds.bottom]
 
    mask = np.zeros_like(DEM, dtype=bool)
    mask[msk == 0] = True
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return DEM, mask, bounds


def DEMGenerate(npa, dst_filename, mask=None):
    """this function is used to generate digital elevation model (DEM file) of
    the 1st layer (band = 1) using a numpy array"""
    profile = DefaultGTiffProfile(count=1)
    profile['nodata'] = -999
    profile['width'] = npa.shape[0]
    profile['height'] = npa.shape[1]
    profile['dtype'] = 'float32'
    profile['blockxsize'] = 128
    profile['blockysize'] = 128
    # profile['transform'] = rio.Affine(1, 0, 0, 0, 1, 0)

    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    if mask is not None:
        npa = np.ma.masked_array(npa, mask)
        
    with rio.open(dst_filename, 'w', **profile) as dst:
        dst.write(npa, 1)


def InverseWeightedDistance(x, y, v, grid, power):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distance = np.sqrt((x-i)**2+(y-j)**2)
            if (distance**power).min() == 0:
                grid[i, j] = v[(distance**power).argmin()]
            else:
                total = np.sum(1/(distance**power))
                grid[i, j] = np.sum(v/(distance**power)/total)
    return grid
