# cython: cdivision = True
# cython: boundscheck= False
# cython: wraparound = False

import numpy as np
import time
from time import strftime, localtime
cimport numpy as np
from libc.math cimport pow
from libc.stdio cimport printf

cpdef CAffe_engine(np.ndarray[np.double_t, ndim = 1] water_levels,
                   np.ndarray[np.uint8_t, ndim = 1, cast=True] mask,
                   np.ndarray[np.double_t, ndim = 1] extra_volume_map,
                   np.ndarray[np.double_t, ndim = 1] max_f,
                   np.ndarray[np.int64_t, ndim = 1] DEMshape,
                   double total_volume,
                   double cell_area,
                   double increment_constant,
                   double hf,
                   double EV_threshold):

    cdef double friction_head_loss = 0
    cdef double minlevels, maxlevels, level_change,increase, deltav_total,\
                levels_u, levels_d, levels_r, levels_l, \
                u, r, d, l, w_u, w_r, w_d, w_l, ev_increase, n_smaller
    cdef int terminate, iteration, i
    terminate = 0
    iteration = 1

    cdef int loop_beg = 0
    cdef int loop_end = water_levels.size - 1
    cdef int row_len = DEMshape[1]

    cdef double volume_spread = 0.
    cdef double volume_spread_old = 0.

    with nogil:
        while terminate == 0:
            terminate = 1 # exit loop when terminate == 1 (default)
            for i in range(loop_beg, loop_end):
                # Rule 0 : do nothing
                # if the central cell has no excess volume, continue the loop
                if (extra_volume_map[i] > EV_threshold and mask[i]==False):
                    terminate = 0
                    # estimate water level difference between central and
                    # neighbouring cells to find the coresponbding rule
                    u = water_levels[i - row_len] - water_levels[i] # North (up)
                    d = water_levels[i + row_len] - water_levels[i] # South (down)
                    r = water_levels[i + 1] - water_levels[i] # east (right)
                    l = water_levels[i - 1] - water_levels[i] # west (left)
                    minlevels = min(u, d, r, l)
                    if minlevels < 0.:
                        # Rule 4 : partitioning
                        # this means that at least one of the neghbouring cells
                        # has water level lower than the central cell
                        # spread water to downstream neighbors
                        # estimate friction head and update max water level
                        # max_f keeps track of the maximum water level that
                        # a cell has during the course of simulation
                        friction_head_loss = hf * pow(extra_volume_map[i], 0.25) # calculates hf value
                        if water_levels[i] + friction_head_loss > max_f[i]:
                            max_f[i] = water_levels[i] + friction_head_loss
                        # if the water to be spread is small, we do not divide
                        # it between dowsntream neighbors and simply transfer it
                        # to the lowest neighbour.
                        if extra_volume_map[i] < .01:
                            # find the lowest neighbour
                            if    u < minlevels + 0.000000001:
                                extra_volume_map[i - row_len] += extra_volume_map[i]
                            elif  r < minlevels + 0.000000001:
                                extra_volume_map[i + 1] += extra_volume_map[i]
                            elif  d < minlevels + 0.000000001:
                                extra_volume_map[i + row_len] += extra_volume_map[i]
                            elif  l < minlevels + 0.000000001:
                                extra_volume_map[i - 1] += extra_volume_map[i]
                            extra_volume_map[i] = 0
                        else:
                            # devide excess water using a weighted method,
                            # the higher the level difference, the more
                            # excess water will be received.
                            levels_u = friction_head_loss - u
                            levels_r = friction_head_loss - r
                            levels_d = friction_head_loss - d
                            levels_l = friction_head_loss - l
                            if levels_u < 0:
                                levels_u = 0.
                            if levels_d < 0 :
                                levels_d = 0.
                            if levels_r < 0:
                                levels_r = 0.
                            if levels_l < 0:
                                levels_l = 0.
                            deltav_total = levels_u + levels_r + levels_d + levels_l
                            extra_volume_map[i - row_len] += levels_u / deltav_total * extra_volume_map[i]
                            extra_volume_map[i + 1] += levels_r / deltav_total * extra_volume_map[i]
                            extra_volume_map[i + row_len] += levels_d / deltav_total * extra_volume_map[i]
                            extra_volume_map[i - 1] += levels_l / deltav_total * extra_volume_map[i]
                            extra_volume_map[i] = 0.
                    elif minlevels > 0.:
                        # Rule 1 : ponding
                        # the central cell is in a depression; fill the depression
                        level_change = min(minlevels, extra_volume_map[i])
                        water_levels[i] += level_change # the depression is filled
                        extra_volume_map[i] -= level_change # deduct that from extra_volume_map
                        volume_spread += level_change
                    else:
                        maxlevels = max(u,d,r,l)
                        if maxlevels == 0.:
                            # Rule 2 : spreading
                            # the excess water is equally splited between cells.
                            increase = extra_volume_map[i] / 4.
                            extra_volume_map[i + 1] += increase
                            extra_volume_map[i - 1] += increase
                            extra_volume_map[i + row_len] += increase
                            extra_volume_map[i - row_len] += increase
                            extra_volume_map[i] = 0.
                        else:
                            # Rule 3 : increasing level
                            # same level; level rises by the increment constant
                            increase = min(increment_constant, extra_volume_map[i])
                            water_levels[i] += increase
                            extra_volume_map[i] -= increase
                            volume_spread += increase
                            if extra_volume_map[i] > EV_threshold:
                                n_smaller = 0.
                                w_u = 0.
                                w_d = 0.
                                w_r = 0.
                                w_l = 0.
                                if u <= 0.:
                                    n_smaller += 1.
                                    w_u = 1.
                                if d <= 0.:
                                    n_smaller += 1.
                                    w_d = 1.
                                if r <= 0.:
                                    n_smaller += 1.
                                    w_r = 1.
                                if l <= 0.:
                                    n_smaller += 1.
                                    w_l = 1.
                                ev_increase = extra_volume_map[i] / n_smaller
                                extra_volume_map[i - row_len] += ev_increase * w_u
                                extra_volume_map[i + 1] += ev_increase * w_r
                                extra_volume_map[i + row_len] += ev_increase * w_d
                                extra_volume_map[i - 1] += ev_increase * w_l
                                extra_volume_map[i] = 0.
            if iteration % 2000== 0:
                with gil:
                  print("\niteration", iteration)
                  print("spreaded volume [m3] =", "{:.3f}".format(volume_spread * cell_area))

            if ((volume_spread * cell_area - volume_spread_old < increment_constant) and terminate == 0):
                terminate = 1

            if ((total_volume - volume_spread * cell_area < 10. * increment_constant) and terminate == 0):
                terminate = 1

            volume_spread_old = volume_spread * cell_area

            iteration += 1

        with gil:
            print("\niteration", iteration-1)
            print("spreaded volume [m3] =", "{:.3f}".format(volume_spread * cell_area))
