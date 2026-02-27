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
from caffe import caffe  # NOQA


if __name__ == "__main__":

    input_DEM_file = './tests/caffe_test.tif'
    hf = 0.09
    increment_constant = 1e-4
    EV_threshold = 1e-5

    # to make an instance of the caffe class model
    sim = caffe(input_DEM_file)
    sim.setConstants(hf, increment_constant, EV_threshold)
    sim.ExcessVolumeArray(np.array([[499, 499, 2000]]))
    sim.OpenBCArray(np.array([[300, 300]]))
    sim.EnableParallelRun(4)
    sim.RunSimulation()
    sim.setOutputPath("./tests/")
    sim.setOutputName("parallel")
    sim.CloseSimulation()
