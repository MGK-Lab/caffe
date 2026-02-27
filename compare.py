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

import numpy as np
import rasterio

def compare_dems(dem1_path, dem2_path):

    # --- Load DEM 1 ---
    with rasterio.open(dem1_path) as src1:
        dem1 = src1.read(1).astype(np.float64)
        nodata1 = src1.nodata

    # --- Load DEM 2 ---
    with rasterio.open(dem2_path) as src2:
        dem2 = src2.read(1).astype(np.float64)
        nodata2 = src2.nodata

    # --- Replace NoData with 0 ---
    if nodata1 is not None:
        dem1[dem1 == nodata1] = 0.0

    if nodata2 is not None:
        dem2[dem2 == nodata2] = 0.0

    # --- Check dimensions ---
    if dem1.shape != dem2.shape:
        raise ValueError("DEM files have different shapes")

    # --- Relative difference ---
    # (dem1 - dem2) / dem2
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = (dem1 - dem2) / dem2

    # Replace inf and nan (caused by division by zero) with 0
    relative_diff[~np.isfinite(relative_diff)] = 0.0

    # --- Statistics ---
    min_diff = np.min(relative_diff)
    max_diff = np.max(relative_diff)
    mean_diff = np.mean(relative_diff)

    print("Relative Difference Statistics")
    print(f"Min  : {min_diff:.6e}")
    print(f"Max  : {max_diff:.6e}")
    print(f"Mean : {mean_diff:.6e}")

    return relative_diff, min_diff, max_diff, mean_diff


dem1 = "1_max_all_mwd.tif"
dem2 = "./maz/_max_all_mwd.tif"
compare_dems(dem1,dem2)