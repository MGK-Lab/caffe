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

def compare_dems(dem1_path, dem2_path, output_path):

    # --- Load DEM 1 ---
    with rasterio.open(dem1_path) as src1:
        dem1 = src1.read(1).astype(np.float64)
        nodata1 = src1.nodata
        profile = src1.profile

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

    # --- Compute difference ---
    relative_diff = np.zeros_like(dem1)

    # Where DEM2 is not zero → relative difference
    mask_rel = dem2 != 0
    relative_diff[mask_rel] = (dem1[mask_rel] - dem2[mask_rel]) / dem2[mask_rel]

    # Where DEM2 is zero → absolute difference
    mask_abs = dem2 == 0
    relative_diff[mask_abs] = np.abs(dem1[mask_abs] - dem2[mask_abs])

    # --- Statistics ---
    min_diff = np.min(relative_diff)
    max_diff = np.max(relative_diff)
    mean_diff = np.mean(relative_diff)

    print("Difference Statistics")
    print(f"Min  : {min_diff:.6e}")
    print(f"Max  : {max_diff:.6e}")
    print(f"Mean : {mean_diff:.6e}")
    
    tolerance = 1e-6  # adjust if needed

    diff = np.abs(dem1 - dem2)

    mismatch_mask = diff > tolerance
    num_mismatch = np.sum(mismatch_mask)

    total_cells = dem1.size
    match_cells = total_cells - num_mismatch

    print("\nCell Comparison")
    print(f"Total cells      : {total_cells}")
    print(f"Matching cells   : {match_cells}")
    print(f"Mismatching cells: {num_mismatch}")
    print(f"Mismatch percent : {100*num_mismatch/total_cells:.6f}%")

    # --- Save raster ---
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(relative_diff.astype(np.float32), 1)

    print(f"\nOutput DEM saved to: {output_path}")

    return relative_diff


dem1 = "./tests/parallel_mwd.tif"
dem2 = "./tests/serial_mwd.tif"
output = "./tests/relative_difference.tif"

compare_dems(dem1, dem2, output)