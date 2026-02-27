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
import sys
sys.path.append("./src")
from src.caffe_swmm_coupled import csc  # NOQA


if __name__ == "__main__":

    DEM_file = './tests/csc_dem.tif'
    hf = 0.09
    increment_constant = 0.0001
    EV_threshold = 0.000002

    SWMM_inp = './tests/csc_test.inp'

    csc_obj = csc()
    csc_obj.LoadCaffe(DEM_file, hf, increment_constant, EV_threshold)
    # an example of calling caffe class members
    csc_obj.caffe.setDEMCellSize(1)
    csc_obj.caffe.setOutputPath("./maz/")

    csc_obj.LoadSwmm(SWMM_inp)

    csc_obj.NodeElvCoordChecks()
    csc_obj.InteractionInterval(600)
    # csc_obj.ManholeProp(0.5, 1)
    csc_obj.Run_Caffe_BD_SWMM()
    # csc_obj.RunMulti_SWMMtoCaffe()
    # csc_obj.Run_Caffe_BD_SWMM()
