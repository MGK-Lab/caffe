import numpy as np
import src.util as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # test with random numbers
    # np.random.seed(123433789)
    # grid = np.zeros((100, 100), dtype='float32')
    # x, y = np.random.randint(0, 100, 10), np.random.randint(0, 100, 10)
    # v = np.random.randint(0, 10, 10)

    dx = 1000
    dy = 1000
    grid = np.zeros((dx, dy), dtype='float32')
    x = np.repeat([0, dx/2, dx], 3)
    y = np.tile([0, dy/2, dy], 3)
    v = np.repeat([100], x.shape[0])
    v[4] = 95
    x = np.append(x, [250, 250, 750, 750])
    y = np.append(y, [250, 750, 250, 750])
    v = np.append(v, [93, 94, 92, 93])
    grid = ut.InverseWeightedDistance(x, y, v, grid, 2)

    file = 'gen_dem_test.tif'
    ut.DEMGenerate(grid, file)
    ut.PlotDEM(file, 10)

    # plt.figure(2)
    # plt.imshow(grid.T, origin='lower', interpolation='nearest', cmap='jet')
    # plt.scatter(x, y, c=None, cmap='jet', s=120)
    # plt.xlim(0, grid.shape[0])
    # plt.ylim(0, grid.shape[1])
    # plt.grid()
    plt.show()
