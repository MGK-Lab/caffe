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
from cswmm import swmm  # NOQA

if __name__ == "__main__":

    swmm_obj = swmm('./tests/swmm_test.inp')
    swmm_obj.LoadNodes()
    for step in swmm_obj.sim:
        print("Node heads: " + str(swmm_obj.getNodesHead()))
        swmm_obj.setNodesInflow(np.zeros(len(swmm_obj.node_list)))
    swmm_obj.CloseSimulation()
