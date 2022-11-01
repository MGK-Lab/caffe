from __future__ import division
import rasterio as rio
from rasterio.profiles import DefaultGTiffProfile
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os


def arraytoRasterIO(array, existingraster, dst_filename):
    """this function is used to save a 2D numpy array as a GIS raster map"""
    with rio.open(existingraster) as src:
        naip_meta = src.profile

    naip_meta['count'] = 1
    naip_meta['nodata'] = -999
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    # write your the ndvi raster object
    with rio.open(dst_filename, 'w', **naip_meta) as dst:
        dst.write(array, 1)


def DEMRead(dem_file):
    """this function is used to read digital elevation model (DEM file) of
    the 1st layer (band = 1)"""
    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    src = rio.open(dem_file)
    band = 1
    DEM = src.read(band)
    msk = src.read_masks(band)
    DEM = DEM.astype(np.double)

    mask = np.zeros_like(DEM, dtype=bool)
    mask[msk == 0] = True
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return DEM, mask


def DEMGenerate(npa, dst_filename):
    """this function is used to generate digital elevation model (DEM file) of
    the 1st layer (band = 1) using a numpy array"""
    profile = DefaultGTiffProfile(count=1)
    profile['nodata'] = -999
    profile['width'] = npa.shape[0]
    profile['height'] = npa.shape[1]
    profile['dtype'] = 'float32'
    profile['blockxsize'] = 128
    profile['blockysize'] = 128
    # profile['transform'] = rio.Affine(1, 0, 0, 0, 1, 0)

    warnings.filterwarnings(
        "ignore", category=rio.errors.NotGeoreferencedWarning)

    with rio.open(dst_filename, 'w', **profile) as dst:
        dst.write(npa, 1)


def InverseWeightedDistance(x, y, v, grid, power):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distance = np.sqrt((x-i)**2+(y-j)**2)
            if (distance**power).min() == 0:
                grid[i, j] = v[(distance**power).argmin()]
            else:
                total = np.sum(1/(distance**power))
                grid[i, j] = np.sum(v/(distance**power)/total)
    return grid


def PlotDEM3d(dem_file, n=10, azdeg=290, altdeg=80, cmp_name='gist_earth'):
    dem, mask = DEMRead(dem_file)

    x = np.linspace(0, dem.shape[0], dem.shape[1])
    y = np.linspace(0, dem.shape[1], dem.shape[0])
    x, y = np.meshgrid(x, y)

    region = np.s_[0:dem.shape[0]:n, 0:dem.shape[1]:n]
    x, y, z = x[region], y[region], dem[region]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.view_init(elev=altdeg, azim=azdeg)

    ls = LightSource(azdeg, altdeg)
    rgb = ls.shade(z, cmap=cm.get_cmap(cmp_name),
                   vert_exag=0.1, blend_mode='soft')

# I do not know why I should swap x and y to correctly illustrate the plot
    ax.plot_surface(y, x, z, rstride=1, cstride=1, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False)

    plt.xlim([0, dem.shape[1]])
    plt.ylim([0, dem.shape[0]])
    ax.set_box_aspect([np.amax(y), np.amax(x), np.amax(z)])
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    # plt.show()


def PlotDEM2d(dem_file, cmp_name='gist_earth'):
    dem, mask = DEMRead(dem_file)

    plt.imshow(dem.T, origin='lower', interpolation='nearest', cmap=cmp_name)
    plt.xlim(0, dem.shape[0])
    plt.ylim(0, dem.shape[1])
    plt.grid()

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    # plt.show()


def AnimateDEMs(path, name, fps=5, n=10, azdeg=290, altdeg=80, cmp_name='gist_earth'):

    files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames)
             in os.walk(path) for f in filenames]
    files.sort()

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.view_init(elev=altdeg, azim=azdeg)
    ls = LightSource(azdeg, altdeg)

    images = []

    for file in files:
        dem, mask = DEMRead(file)

        x = np.linspace(0, dem.shape[0], dem.shape[1])
        y = np.linspace(0, dem.shape[1], dem.shape[0])
        x, y = np.meshgrid(x, y)

        region = np.s_[0:dem.shape[0]:n, 0:dem.shape[1]:n]
        x, y, z = x[region], y[region], dem[region]

        rgb = ls.shade(z, cmap=cm.get_cmap(cmp_name),
                       vert_exag=0.1, blend_mode='soft')

        image = ax.plot_surface(y, x, z, rstride=1, cstride=1, facecolors=rgb,
                                linewidth=0, antialiased=False, shade=False)
        images.append([image])

    ani = animation.ArtistAnimation(fig, images)
    writergif = animation.PillowWriter(fps=fps)
    ani.save(name, writer=writergif)
