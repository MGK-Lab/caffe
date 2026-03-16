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

from __future__ import division
import rasterio as rio
from rasterio.profiles import DefaultGTiffProfile
import numpy as np
import warnings
import pandas as pd


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

def RainResample(file_name, target_name, target_intervals):

    df = pd.read_csv(file_name, header=None, names=['time', 'height'])

    df['time'] = pd.to_datetime(df['time'], format='%H:%M')

    df = df.resample(
        f'{target_intervals}T', on='time', origin='start').sum().reset_index()

    # Adjust the time column to start from zero height
    first_row = df.iloc[[0]]
    first_row.iloc[0, 1] = 0
    if df['height'].iloc[0] != 0:
        df['time'] = df['time'] + \
            pd.to_timedelta(f'{target_intervals} minutes')
        df = pd.concat([first_row, df], ignore_index=True)

    df['time'] = df['time'].dt.strftime('%H:%M')

    df.to_csv(target_name, index=False, header=False)
