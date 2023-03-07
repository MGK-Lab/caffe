import numpy as np
import util
import os
import matplotlib.pyplot as plt


def Delta(a):
    return 1 if a > 0 else 0


class VelocityCalculator:
    def __init__(self, cell_length, timestep, dem_file, swwmm_raster_file):
        self.cell_length = cell_length
        self.timestep = timestep
        self.Vx_proj = None
        self.Vy_proj = None
        self.dem, self.mask, tmp = util.DEMRead(dem_file)
        self.dem_file = dem_file
        self.theta = np.array([1, 1, -1, -1])
        self.swmm_ras, a, b = util.DEMRead(swwmm_raster_file)

    def read_water_depth(self, path, keyword):
        files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames)
                 in os.walk(path) for f in filenames if keyword in f]
        files.sort()

        arrays = []
        for file in files:
            wd, mask, bounds = util.DEMRead(file)
            arrays.append(wd)

        self.water_depth = np.stack(arrays, axis=0)

    def calculate_velocities_average(self):
        n, nx, ny = self.water_depth.shape

        # Initialize arrays for velocity magnitude and components
        self.Vx_proj = np.zeros((n-1, nx, ny))
        self.Vy_proj = np.zeros((n-1, nx, ny))
        self.V = np.zeros((n-1, nx, ny))

        # Loop through each cell to calculate velocities
        for n in range(0, n-1):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    dh_dt = (
                        self.water_depth[n+1, i, j] - self.water_depth[n, i, j]) / self.timestep
                    if (dh_dt != 0 and self.swmm_ras[i, j] != 1):
                        # Calculate slopes using central differences
                        dh_dx = -(self.water_depth[n, i+1, j] + self.dem[i+1, j] -
                                  self.water_depth[n, i-1, j] - self.dem[i-1, j]) / (2.0 * self.cell_length)
                        dh_dy = -(self.water_depth[n, i, j+1] + self.dem[i, j+1] -
                                  self.water_depth[n, i, j-1] - self.dem[i, j-1]) / (2.0 * self.cell_length)

                        if (dh_dx == 0 and dh_dy == 0):
                            thetha = 0
                        else:
                            # Calculate angle to x direction
                            theta = np.arctan2(dh_dx, dh_dy)

                        # Calculate mean water depth for each cell
                        mean_h = (self.water_depth[n, i-1, j] + self.water_depth[n, i+1, j] + self.water_depth[n,
                                  i, j-1] + self.water_depth[n, i, j+1] + 4*self.water_depth[n, i, j]) / 8.0
                        if mean_h == 0:
                            mean_h = (self.water_depth[n+1, i-1, j] + self.water_depth[n+1, i+1, j] + self.water_depth[n +
                                      1, i, j-1] + self.water_depth[n+1, i, j+1] + 4*self.water_depth[n, i, j]) / 8.0

                        # Calculate velocity magnitude and components for each cell
                        # change in water depth over time (m/s)
                        if mean_h == 0:
                            self.V[n, i, j] = 0
                        else:
                            self.V[n, i, j] = self.cell_length * dh_dt / mean_h
                        self.Vx_proj[n, i, j] = self.V[n, i, j] * np.cos(theta)
                        self.Vy_proj[n, i, j] = self.V[n, i, j] * np.sin(theta)
                    else:
                        self.V[n, i, j] = 0
                        self.Vx_proj[n, i, j] = 0
                        self.Vy_proj[n, i, j] = 0

    def calculate_velocities_localQ(self):
        n, nx, ny = self.water_depth.shape

        # Initialize arrays for velocity magnitude and components
        self.Vx_proj = np.zeros((n-1, nx, ny))
        self.Vy_proj = np.zeros((n-1, nx, ny))
        self.V = np.zeros((n-1, nx, ny))

        # Loop through each cell to calculate velocities
        for n in range(0, n-1):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    dh_dt = (
                        self.water_depth[n+1, i, j] - self.water_depth[n, i, j]) / self.timestep
                    if (dh_dt != 0 and self.swmm_ras[i, j] != 1):

                        dh_dt *= self.cell_length

                        z = np.array([self.dem[i, j], self.dem[i+1, j],
                                      self.dem[i, j+1], self.dem[i-1, j],
                                      self.dem[i, j-1]])
                        d = np.array([self.water_depth[n, i, j], self.water_depth[n, i+1, j],
                                      self.water_depth[n, i, j +
                                                       1], self.water_depth[n, i-1, j],
                                      self.water_depth[n, i, j-1]])
                        delta_H = d + z
                        delta_H = delta_H[1:5]-delta_H[0]

                        dir = np.zeros(4, dtype=np.int)
                        vloc = np.zeros(4, dtype=np.double)

                        # Q come in
                        if (dh_dt > 0):
                            for k in range(0, 4):
                                if (delta_H[k] > 0 and d[k+1] > 0):
                                    dir[k] = 1
                        # Q goes out
                        else:
                            for k in range(0, 4):
                                if (delta_H[k] < 0 and d[0] > 0):
                                    dir[k] = 1

                        d_bar_i = (d[0] + d[1:5]) / 2
                        sum = np.sum(delta_H*dir)
                        if sum > 0:
                            vloc = delta_H*dir / sum
                        for k in range(0, 4):
                            if d_bar_i[k] > 0:
                                vloc[k] = vloc[k] * abs(dh_dt) / d_bar_i[k]
                            else:
                                vloc[k] = 0

                        self.Vx_proj[n, i, j] = vloc[0] - \
                            vloc[2] if dh_dt < 0 else vloc[2]-vloc[0]
                        self.Vy_proj[n, i, j] = vloc[1] - \
                            vloc[3] if dh_dt < 0 else vloc[3]-vloc[1]
                        self.V[n, i, j] = (
                            self.Vx_proj[n, i, j]**2 + self.Vy_proj[n, i, j]**2)**0.5
                    # Q is zero
                    else:
                        self.V[n, i, j] = 0
                        self.Vx_proj[n, i, j] = 0
                        self.Vy_proj[n, i, j] = 0

    def VisualiseVec(self, num):
        x_coords = np.arange(0, self.water_depth.shape[2])
        y_coords = np.arange(0, self.water_depth.shape[1])
        X, Y = np.meshgrid(x_coords, y_coords)

        # Create a quiver plot of the velocity vectors
        fig, ax = plt.subplots()
        u_masked = np.ma.masked_where(
            self.Vx_proj[num, :, :] == 0, self.Vy_proj[num, :, :])
        v_masked = np.ma.masked_where(
            self.Vy_proj[num, :, :] == 0, self.Vx_proj[num, :, :])

        ax.imshow(self.dem, cmap='terrain')
        ax.quiver(X[::-1], Y, u_masked, v_masked, scale=None)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        plt.show()

    def VisualiseSca(self, num):

        arr = self.V[num, :, :]
        im = plt.imshow(arr, cmap='binary')
        cbar = plt.colorbar(im)
        arr = arr[arr > 0]
        im.set_clim(vmax=np.median(arr))
        im.set_clim(vmin=0)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()

    def CalcQuantile(self, values, name):

        stats = np.zeros((self.V.shape[0], len(values)))
        time = np.arange(self.timestep, (self.V.shape[0]+1)*self.timestep,
                         self.timestep)/60

        for i in range(0, self.V.shape[0]):
            arr = self.V[i, :, :]
            arr = arr.flatten()
            arr = arr[arr > 0]
            for j in range(0, len(values)):
                stats[i, j] = np.quantile(arr, values[j])

        for i in range(stats.shape[1]):
            plt.plot(time, stats[:, i]*100,
                     label='Quantile {}%'.format(values[(i)]*100))

        # Add labels and legend
        plt.xlabel('Time (min)')
        plt.ylabel('Velocity (cm/s)')
        plt.legend()
        plt.ylim([0, 25])
        plt.savefig(name)
        plt.show()

    def SaveTiff(self, path_name):

        for i in range(0, self.water_depth.shape[0]-1):
            string = f"{path_name}_{i:04d}.tif"
            util.arraytoRasterIO(self.V[i, :, :], self.dem_file,  string)
