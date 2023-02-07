from __future__ import division
import src.util as ut
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def PlotDEM3d(dem_file, n=10, azdeg=290, altdeg=80, cmp_name='gist_earth'):
    dem, mask, bounds = ut.DEMRead(dem_file)

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
    dem, mask, bounds = ut.DEMRead(dem_file)
    mask[dem<=0]=True
    dem[mask==True] = np.amax(dem)
 
    plt.imshow(np.flipud(dem), origin='lower', interpolation='nearest', cmap=cmp_name)
    plt.xlim(0, dem.shape[1])
    plt.ylim(0, dem.shape[0])
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
        dem, mask, bounds = ut.DEMRead(file)

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
