# cython: cdivision = True
# cython: boundscheck= False
# cython: wraparound = False

import numpy as np
import time
from time import strftime, localtime
cimport numpy as np
from libc.math cimport pow
from libc.stdio cimport printf


cpdef CAffe_engine(np.ndarray[np.float32_t, ndim = 1] water_levels,
                   np.ndarray[np.float32_t, ndim = 1] extra_volume_map,
                   np.ndarray[np.float32_t, ndim = 1] max_f,
                   int DEMshape0,
                   int DEMshape1,
                   double pixel_area,
                   double total_vol,
                   double water_raise,
                   double hf,
                   double EV_threshold,
                   int length):

    cdef double friction_head_loss = 0
    cdef double minlevels, maxlevels, level_change,increase, deltav_total,\
                levels_u, levels_d, levels_r, levels_l, sum_extra_volume_map, \
                u, r, d, l, w_u, w_r, w_d, w_l, ev_increase, n_smaller
    cdef int terminate, iteration, i
    terminate = 0
    iteration = 1
    cdef double start_t = time.time()

    cdef double volume_spread = 0. #to measure how much of extra_volume_map is spread

    with nogil:
        while terminate == 0:
            terminate = 1 # exit loop when terminate == 1 (default)

            # looping over extra_volume_map array items
            for i in range(DEMshape1 + 1, length):
                # if there are at least one cell with extra_volume_map[i] > EV_threshold,
                # then continue the loop
                if extra_volume_map[i] > EV_threshold:
                    terminate = 0
                    # estimate water level difference between central and neighbouring cells
                    u = water_levels[i - DEMshape1] - water_levels[i] # North cell (up)
                    d = water_levels[i + DEMshape1] - water_levels[i] # South cell (down)
                    r = water_levels[i + 1] - water_levels[i] # west (right)
                    l = water_levels[i - 1] - water_levels[i] # east (left)

                    minlevels = min(u, d, r, l)

                    if minlevels < 0.:
                        # this means that at least one of the neghbouring cells
                        # has water level lower than the central cell (conditon for Rule 4)
                        ############### " Rule  4 is activated " ##################################
                        # estimate friction head
                        friction_head_loss = hf * pow(extra_volume_map[i], 0.25) # calculates hf value
                        # update max water level
                        if water_levels[i] + friction_head_loss > max_f[i]:
                            max_f[i] = water_levels[i] + friction_head_loss # max_f keeps track of the maximum water level that a cell has during the course of sumulation.

                        ## spread water to downstream neighbors
                        # if the water to be spread is small, we do not divide it between dowsntream neighbors.
                        # simply transfer it to the lowest neighbour.
                        if extra_volume_map[i] < .01:
                            # find the lowest neighbour
                            if u < minlevels + 0.00000000001:
                                extra_volume_map[i - DEMshape1] += extra_volume_map[i]
                                extra_volume_map[i] = 0
                            elif r < minlevels + 0.000000001:
                                extra_volume_map[i + 1] += extra_volume_map[i]
                                extra_volume_map[i] = 0
                            elif d < minlevels + 0.0000000001:
                                extra_volume_map[i + DEMshape1] += extra_volume_map[i]
                                extra_volume_map[i] = 0
                            elif l < minlevels + 0.000000001:
                                extra_volume_map[i - 1] += extra_volume_map[i]
                                extra_volume_map[i] = 0
                        else:
                            # devide excess water using a weighted method, the higher the level difference, the more
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

                            extra_volume_map[i - DEMshape1] += levels_u / deltav_total * extra_volume_map[i]
                            extra_volume_map[i + 1] += levels_r / deltav_total * extra_volume_map[i]
                            extra_volume_map[i + DEMshape1] += levels_d / deltav_total * extra_volume_map[i]
                            extra_volume_map[i - 1] += levels_l / deltav_total * extra_volume_map[i]
                            extra_volume_map[i] = 0

                    elif minlevels > 0.:
                        ###############" Rule 1 (central cell is in a depression) -> fill that depression" ###########
                        level_change = min(minlevels, extra_volume_map[i])
                        water_levels[i] += level_change # the depression is filled
                        extra_volume_map[i] -= level_change # deduct that from extra_volume_map
                        volume_spread += level_change

                    else:
                        maxlevels = max(u,d,r,l)
                        if maxlevels == 0.:
                            ############ "Rule 2: same level, split" ###########################
                            # excess water is equally split between cells.
                            increase = extra_volume_map[i] / 4.
                            extra_volume_map[i + 1] += increase
                            extra_volume_map[i - 1] += increase
                            extra_volume_map[i + DEMshape1] += increase
                            extra_volume_map[i - DEMshape1] += increase
                            extra_volume_map[i] = 0

                        else:
                            ######### Rule 3 same level, level rise" ###############################
                            increase = min(water_raise, extra_volume_map[i])
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

                                extra_volume_map[i - DEMshape1] += ev_increase * w_u
                                extra_volume_map[i + 1] += ev_increase * w_r
                                extra_volume_map[i + DEMshape1] += ev_increase * w_d
                                extra_volume_map[i - 1] += ev_increase * w_l
                                extra_volume_map[i] = 0

            iteration += 1
            if iteration % 2000== 0:
                printf("iteration %f\n", iteration*1.)
                printf("\tvolume spread [m3] = %f\n", volume_spread * pixel_area)
                if total_vol - volume_spread * pixel_area < 5:
                    terminate = 1


    print("\n\n {} | Simulation finished at iteration = {} | duration = {} seconds").format(strftime('%H:%M:%S', localtime()),iteration, round(time.time() - start_t,1))
