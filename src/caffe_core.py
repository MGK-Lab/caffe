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
from numba import njit

@njit()
def CAffe_engine(
    water_levels,
    mask,
    extra_volume_map,
    max_f,
    DEMshape,
    total_volume,
    cell_area,
    increment_constant,
    hf,
    EV_threshold,
):

    friction_head_loss = 0.0
    terminate = 0
    iteration = 1

    loop_beg = 0
    loop_end = water_levels.size - 1
    row_len = DEMshape[1]

    volume_spread = 0.0
    volume_spread_old = 0.0

    while terminate == 0:
        terminate = 1

        for i in range(loop_beg, loop_end):

            if extra_volume_map[i] > EV_threshold and mask[i] == 0:
                terminate = 0

                u = water_levels[i - row_len] - water_levels[i]
                d = water_levels[i + row_len] - water_levels[i]
                r = water_levels[i + 1] - water_levels[i]
                l = water_levels[i - 1] - water_levels[i]

                minlevels = min(u, d, r, l)

                # ---------------- Rule 4 : Partitioning ----------------
                if minlevels < 0.0:

                    friction_head_loss = hf * extra_volume_map[i] ** 0.25

                    if water_levels[i] + friction_head_loss > max_f[i]:
                        max_f[i] = water_levels[i] + friction_head_loss

                    if extra_volume_map[i] < 0.01:

                        if u <= minlevels:
                            extra_volume_map[i - row_len] += extra_volume_map[i]
                        elif r <= minlevels:
                            extra_volume_map[i + 1] += extra_volume_map[i]
                        elif d <= minlevels:
                            extra_volume_map[i + row_len] += extra_volume_map[i]
                        else:
                            extra_volume_map[i - 1] += extra_volume_map[i]

                        extra_volume_map[i] = 0.0

                    else:
                        levels_u = max(friction_head_loss - u, 0.0)
                        levels_r = max(friction_head_loss - r, 0.0)
                        levels_d = max(friction_head_loss - d, 0.0)
                        levels_l = max(friction_head_loss - l, 0.0)

                        deltav_total = levels_u + levels_r + levels_d + levels_l

                        if deltav_total > 0.0:
                            extra_volume_map[i - row_len] += (
                                levels_u / deltav_total * extra_volume_map[i]
                            )
                            extra_volume_map[i + 1] += (
                                levels_r / deltav_total * extra_volume_map[i]
                            )
                            extra_volume_map[i + row_len] += (
                                levels_d / deltav_total * extra_volume_map[i]
                            )
                            extra_volume_map[i - 1] += (
                                levels_l / deltav_total * extra_volume_map[i]
                            )

                        extra_volume_map[i] = 0.0

                # ---------------- Rule 1 : Ponding ----------------
                elif minlevels > 0.0:

                    level_change = min(minlevels, extra_volume_map[i])
                    water_levels[i] += level_change
                    extra_volume_map[i] -= level_change
                    volume_spread += level_change

                else:

                    maxlevels = max(u, d, r, l)

                    # --------------- Rule 2 : Spreading ---------------
                    if maxlevels == 0.0:

                        increase = extra_volume_map[i] / 4.0

                        extra_volume_map[i + 1] += increase
                        extra_volume_map[i - 1] += increase
                        extra_volume_map[i + row_len] += increase
                        extra_volume_map[i - row_len] += increase

                        extra_volume_map[i] = 0.0

                    # --------------- Rule 3 : Level Increase ----------
                    else:

                        increase = min(increment_constant, extra_volume_map[i])
                        water_levels[i] += increase
                        extra_volume_map[i] -= increase
                        volume_spread += increase

                        if extra_volume_map[i] > EV_threshold:

                            n_smaller = 0.0
                            w_u = 0.0
                            w_d = 0.0
                            w_r = 0.0
                            w_l = 0.0

                            if u <= 0.0:
                                n_smaller += 1.0
                                w_u = 1.0
                            if d <= 0.0:
                                n_smaller += 1.0
                                w_d = 1.0
                            if r <= 0.0:
                                n_smaller += 1.0
                                w_r = 1.0
                            if l <= 0.0:
                                n_smaller += 1.0
                                w_l = 1.0

                            if n_smaller > 0.0:
                                ev_increase = extra_volume_map[i] / n_smaller

                                extra_volume_map[i - row_len] += ev_increase * w_u
                                extra_volume_map[i + 1] += ev_increase * w_r
                                extra_volume_map[i + row_len] += ev_increase * w_d
                                extra_volume_map[i - 1] += ev_increase * w_l

                                extra_volume_map[i] = 0.0

        if (
            (volume_spread * cell_area - volume_spread_old < increment_constant)
            and terminate == 0
        ):
            terminate = 1

        if (
            (total_volume - volume_spread * cell_area < 10.0 * increment_constant)
            and terminate == 0
        ):
            terminate = 1

        volume_spread_old = volume_spread * cell_area
        iteration += 1

    return iteration - 1, volume_spread * cell_area