from __future__ import division
import rasterio as rio
from rasterio.profiles import DefaultGTiffProfile
import numpy as np
import warnings


def ArrayToRaster(arr, filename, sample_raster, mask=None):
    # Ignore the warning for not having a georeference
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # Load sample raster metadata
    with rio.open(sample_raster, 'r') as src:
        meta = src.meta.copy()
        if mask is None:
            mask = src.read_masks(1).astype(bool)

    # Update metadata for output raster
    meta.update({
        'count': 1,
        'dtype': arr.dtype,
        'nodata': -9999,  # set nodata value here
    })

    # Create output raster file
    with rio.open(filename, 'w', **meta) as dst:
        # Mask array and write to raster file
        arr_masked = np.where(mask, arr, meta['nodata'])
        dst.write(arr_masked, 1)


def VelocitytoRasterIO(velx, vely, existingraster, dst_filename):
    with rio.open(existingraster) as src:
        naip_meta = src.profile
        mask = src.dataset_mask()

    naip_meta['count'] = 2
    naip_meta['nodata'] = -999
    naip_meta['dtype'] = 'float32'
    naip_meta['nodata'] = -999
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        dst.write(np.ma.masked_array(velx, mask), 1)
        dst.write(np.ma.masked_array(vely, mask), 2)


def RasterToArray(dem_file):
    # Ignore the warning for not having a georeference
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # Open the DEM file
    with rio.open(dem_file) as src:
        DEM = src.read(1)
        mask = src.read_masks(1)
        bounds = [src.bounds.left, src.bounds.top,
                  src.bounds.right, src.bounds.bottom]
        geotransform = src.transform
        cell_size = geotransform[0]

    DEM = DEM.astype(np.double)
    bounds = np.array(bounds)
    mask = ~mask.astype(bool)

    # Create a wall all around the domain
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return DEM, mask, bounds, cell_size


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
