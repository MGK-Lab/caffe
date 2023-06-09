# cython: cdivision = True
# cython: boundscheck= False
# cython: wraparound = False

import numpy as np
# import time
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport pow
from cython.parallel import parallel, prange, threadid
from cython import nogil


def CAffe_engine(_water_levels, _mask, _extra_volume_map, _max_f, DEMshape, 
                    _cell_area, _increment_constant, _hf, _EV_threshold,
                    _vol_cutoff, _threads):

    cdef int threads = _threads
    cdef double EV_threshold = _EV_threshold
    cdef double increment_constant = _increment_constant
    cdef double hf = _hf
    cdef double cell_area = _cell_area
    cdef double vol_cutoff = _vol_cutoff
    cdef double[::1] extra_volume_map = _extra_volume_map
    cdef double[::1] water_levels = _water_levels
    cdef double[::1] max_f = _max_f
    cdef long[::1] mask = _mask.astype(int)


    cdef double u, r, d, l, minlevels, friction_head_loss, levels_u, levels_d,\
                levels_r, levels_l, deltav_total, level_change, maxlevels,\
                increase, w_u, w_r, w_d, w_l


    cdef int iteration
    cdef Py_ssize_t i
    terminate = 0
    iteration = 1
    cdef int loop_beg = 0
    cdef int loop_end = water_levels.size
    cdef int row_len = DEMshape[1]

    tmp = np.zeros(threads, dtype=np.float64)
    cdef double[::1] volume_spread = np.copy(tmp)
    cdef double[::1] ev_increase = np.copy(tmp)
    cdef double[::1] n_smaller = np.copy(tmp)
    tmp1 = np.ones(loop_end, dtype=int)
    cdef long[::1] terminate_local = tmp1
    cdef double total_vol = np.sum(extra_volume_map) * cell_area

    while terminate == 0:
        terminate = 1 # exit loop when terminate == 1 (default)

        with nogil, parallel(num_threads=threads):
            for i in prange(loop_beg, loop_end):
                terminate_local[i] = 1
                # Rule 0 : do nothing
                # if the central cell has no excess volume, continue the loop
                if (extra_volume_map[i] > EV_threshold and mask[i]==False):
                    terminate_local[i] = 0
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
                            if u < minlevels + 0.00000000001:
                                extra_volume_map[i - row_len] += extra_volume_map[i]
                            elif r < minlevels + 0.000000001:
                                extra_volume_map[i + 1] += extra_volume_map[i]
                            elif d < minlevels + 0.0000000001:
                                extra_volume_map[i + row_len] += extra_volume_map[i]
                            elif l < minlevels + 0.000000001:
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
                        volume_spread[threadid()] += level_change

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

                            volume_spread[threadid()] += increase

                            if extra_volume_map[i] > EV_threshold:
                                n_smaller[threadid()] = 0.
                                w_u = 0.
                                w_d = 0.
                                w_r = 0.
                                w_l = 0.

                                if u <= 0.:
                                    n_smaller[threadid()] += 1.
                                    w_u = 1.

                                if d <= 0.:
                                    n_smaller[threadid()] += 1.
                                    w_d = 1.

                                if r <= 0.:
                                    n_smaller[threadid()] += 1.
                                    w_r = 1.

                                if l <= 0.:
                                    n_smaller[threadid()] += 1.
                                    w_l = 1.

                                ev_increase[threadid()] = extra_volume_map[i] / n_smaller[threadid()]

                                extra_volume_map[i - row_len] += ev_increase[threadid()] * w_u
                                extra_volume_map[i + 1] += ev_increase[threadid()] * w_r
                                extra_volume_map[i + row_len] += ev_increase[threadid()] * w_d
                                extra_volume_map[i - 1] += ev_increase[threadid()] * w_l
                                extra_volume_map[i] = 0.


        for j in range(1,threads):
            volume_spread[0] += volume_spread[j]
            volume_spread[j] = 0
        
        if np.min(terminate_local) == 0:
            terminate = 0
        else:
            terminate = 1

        if iteration % 2000== 0:
            printf("\niteration %i\n", iteration)
            printf("spreaded volume [m3] = %.3f\n", volume_spread[0] * cell_area)

        if total_vol - volume_spread[0] * cell_area < vol_cutoff:
            printf("\niteration %i\n", iteration)
            printf("spreaded volume [m3] = %.3f\n", volume_spread[0] * cell_area)
            terminate = 1

        iteration += 1

    printf("\nlast iteration %i\n", iteration-1)
    printf("spreaded volume [m3] = %.3f\n", volume_spread[0] * cell_area)
