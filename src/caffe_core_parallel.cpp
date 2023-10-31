#include <iostream>
#include <cmath>
#include <omp.h>

extern "C" {

inline void cell_rules(double* water_levels,
                    double* extra_volume_map,
                    double* max_f,
                    double increment_constant,
                    double hf,
                    double EV_threshold,
                    double & volume_spread,
                    int row_len,
                    int i) {

    // estimate water level difference between central and
    // neighboring cells to find the corresponding rule
    double u = water_levels[i - row_len] - water_levels[i]; // North (up)
    double d = water_levels[i + row_len] - water_levels[i]; // South (down)
    double r = water_levels[i + 1] - water_levels[i]; // east (right)
    double l = water_levels[i - 1] - water_levels[i]; // west (left)

    double minlevels = std::min(std::min(u, d), std::min(r, l));

    if (minlevels < 0.) {
        // Rule 4 : partitioning
        // this means that at least one of the neighboring cells
        // has a water level lower than the central cell
        // spread water to downstream neighbors

        // estimate friction head and update max water level
        // max_f keeps track of the maximum water level that
        // a cell has during the course of simulation
        double friction_head_loss = hf * std::pow(extra_volume_map[i], 0.25); // calculates hf value
        if (water_levels[i] + friction_head_loss > max_f[i]) 
            max_f[i] = water_levels[i] + friction_head_loss;

        // if the water to be spread is small, we do not divide
        // it between downstream neighbors and simply transfer it
        // to the lowest neighbor.
        if (extra_volume_map[i] < 0.01) {
            // find the lowest neighbor
            if      (u < minlevels + 0.000000001)
                extra_volume_map[i - row_len] += extra_volume_map[i];
            else if (r < minlevels + 0.000000001)
                extra_volume_map[i + 1] += extra_volume_map[i];
            else if (d < minlevels + 0.000000001)
                extra_volume_map[i + row_len] += extra_volume_map[i];
            else if (l < minlevels + 0.000000001)
                extra_volume_map[i - 1] += extra_volume_map[i];
            

            extra_volume_map[i] = 0;
        } 
        else {
            // divide excess water using a weighted method,
            // the higher the level difference, the more
            // excess water will be received.

            double levels_u = friction_head_loss - u;
            double levels_r = friction_head_loss - r;
            double levels_d = friction_head_loss - d;
            double levels_l = friction_head_loss - l;

            if (levels_u < 0) levels_u = 0.;
            if (levels_d < 0) levels_d = 0.;
            if (levels_r < 0) levels_r = 0.;
            if (levels_l < 0) levels_l = 0.;

            double deltav_total = levels_u + levels_r + levels_d + levels_l;

            extra_volume_map[i - row_len] += levels_u / deltav_total * extra_volume_map[i];
            extra_volume_map[i + 1] += levels_r / deltav_total * extra_volume_map[i];
            extra_volume_map[i + row_len] += levels_d / deltav_total * extra_volume_map[i];
            extra_volume_map[i - 1] += levels_l / deltav_total * extra_volume_map[i];

            extra_volume_map[i] = 0.;
        }
    } 
    else if (minlevels > 0.) {
        // Rule 1 : ponding
        // the central cell is in a depression; fill the depression
        double level_change = std::min(minlevels, extra_volume_map[i]);
        water_levels[i] += level_change; // the depression is filled
        extra_volume_map[i] -= level_change; // deduct that from extra_volume_map
        volume_spread += level_change;

    } 
    else {
        double maxlevels = std::max(std::max(u, d), std::max(r, l));
        double increase; 
        if (maxlevels == 0.) {
            // Rule 2 : spreading
            // the excess water is equally split between cells.
            increase = extra_volume_map[i] / 4.;
            extra_volume_map[i + 1] += increase;
            extra_volume_map[i - 1] += increase;
            extra_volume_map[i + row_len] += increase;
            extra_volume_map[i - row_len] += increase;
            extra_volume_map[i] = 0.;
        } 
        else {
            // Rule 3 : increasing level
            // same level; level rises by the increment constant
            increase = std::min(increment_constant, extra_volume_map[i]);
            water_levels[i] += increase;
            extra_volume_map[i] -= increase;
            volume_spread += increase;
            if (extra_volume_map[i] > EV_threshold) {
                double n_smaller = 0.;
                double w_u = 0.;
                double w_d = 0.;
                double w_r = 0.;
                double w_l = 0.;

                if (u <= 0.) {n_smaller += 1.; w_u = 1.;}
                if (d <= 0.) {n_smaller += 1.; w_d = 1.;}
                if (r <= 0.) {n_smaller += 1.; w_r = 1.;}
                if (l <= 0.) {n_smaller += 1.; w_l = 1.;}

                double ev_increase = extra_volume_map[i] / n_smaller;

                extra_volume_map[i - row_len] += ev_increase * w_u;
                extra_volume_map[i + 1] += ev_increase * w_r;
                extra_volume_map[i + row_len] += ev_increase * w_d;
                extra_volume_map[i - 1] += ev_increase * w_l;
                extra_volume_map[i] = 0.;
            }
        }
    }
}



void CAffe_engine(double* water_levels,
                    bool* mask,
                    double* extra_volume_map,
                    double* max_f,
                    long* DEMshape,
                    double cell_area,
                    double increment_constant,
                    double hf,
                    double EV_threshold,
                    double vol_cutoff,
                    int threads) {

    int terminate, iteration;
    terminate = 0;
    iteration = 1;
 
    int loop_end = DEMshape[0] * DEMshape[1] - 1;
    int row_len = DEMshape[1];

    double volume_spread = 0.;
    double volume_spread_old = 0.;

    int thread_num;
    double tmp = DEMshape[0] / 2.0;
    if (tmp < threads)
        thread_num = int(std::ceil(tmp));
    else
        thread_num = threads;
    
    omp_set_num_threads(thread_num);

    double volume_spread_loc[thread_num];
    for (int i = 0; i < thread_num; i++)
        volume_spread_loc[i] = 0.;

    int i_val[2 * thread_num + 1];
    i_val[2 * thread_num] = loop_end;
    for (int i = 0; i < 2 * thread_num; i++)
        i_val[i] = int(i * row_len * DEMshape[0] / thread_num / 2.0);

    omp_lock_t lock;
    omp_init_lock(&lock);

    while (terminate == 0) {
        terminate = 1; // exit loop when terminate == 1 (default)

        #pragma omp parallel for schedule(static, 1)
        for (int j = 0; j < thread_num; j++)
            for (int i = i_val[2*j]; i < i_val[2*j+1]; i++)
                if (extra_volume_map[i] > EV_threshold && mask[i] == 0) { 
                    if (terminate == 1){
                        omp_set_lock(&lock);  
                        terminate = 0;
                        omp_unset_lock(&lock);
                    }
                    cell_rules(water_levels, extra_volume_map, max_f,
                                increment_constant, hf, EV_threshold, 
                                volume_spread_loc[j], row_len, i);
                }

        #pragma omp parallel for schedule(static, 1)
        for (int j = 1; j < thread_num + 1; j++)
            for (int i = i_val[2*j-1]; i < i_val[2*j]; i++)
                if (extra_volume_map[i] > EV_threshold && mask[i] == 0) { 
                    if (terminate == 1){
                        omp_set_lock(&lock);  
                        terminate = 0;
                        omp_unset_lock(&lock);
                    }
                    cell_rules(water_levels, extra_volume_map, max_f,
                                increment_constant, hf, EV_threshold, 
                                volume_spread_loc[j], row_len, i);
                }

        for (int i = 0; i < thread_num; i++){
            volume_spread += volume_spread_loc[i];
            volume_spread_loc[i] = 0.;
        }


        if (iteration % 2000 == 0) {
            std::cout << "\niteration " << iteration << std::endl;
            std::cout << "spreaded volume [m3] = " << volume_spread * cell_area << std::endl;
        }

        if (terminate == 0 && (volume_spread * cell_area == volume_spread_old))
            terminate = 1;
        volume_spread_old = volume_spread * cell_area;

        iteration++;
    }
    std::cout << "\niteration " << iteration-1 << std::endl;
    std::cout << "spreaded volume [m3] = " << volume_spread * cell_area << std::endl;

    omp_destroy_lock(&lock);

}
}
int main() {
    return 0;
}
