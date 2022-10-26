import numpy as np
import src.util as ut
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # test with random numbers
    # np.random.seed(123433789)
    # grid = np.zeros((100, 100), dtype='float32')
    # x, y = np.random.randint(0, 100, 10), np.random.randint(0, 100, 10)
    # v = np.random.randint(0, 10, 10)

    dx = 100
    dy = 100
    file = 'csc_dem.tif'

    x = np.repeat([0, dx/2, dx], 3)
    y = np.tile([0, dy/2, dy], 3)
    v = np.repeat([95.1], x.shape[0])
    x = np.append(x, [25, 25, 75, 75])
    y = np.append(y, [25, 75, 25, 75])
    v = np.append(v, [94.7, 94.75, 94.6, 94.7])

    grid = np.zeros((dx+1, dy+1), dtype='float32')
    grid = ut.InverseWeightedDistance(x, y, v, grid, 2)
    grid[0, :] = 95.1
    grid[-1, :] = 95.1
    grid[:, 0] = 95.1
    grid[:, -1] = 95.1
    ut.DEMGenerate(grid, file)
