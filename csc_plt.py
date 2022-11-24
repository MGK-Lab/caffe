import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import matplotlib.animation as animation

if __name__ == "__main__":

    file = './maz.npz'

    data = np.load(file, mmap_mode='r', allow_pickle=True)
    t = data['arr_0']
    exchangeamount = data['arr_1']
    nodeinfo = np.transpose(data['arr_2'])

    plt.plot(t, exchangeamount[:, 0], 'r.', label='Surcharged')
    plt.plot(t, exchangeamount[:, 1], 'b+', label='drained')
    plt.xticks(rotation=90)
    plt.xlabel('time')
    plt.ylabel('Volume ($m^3$)')
    plt.legend()
    formatter = dates.DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.savefig('./exchange.png')
    plt.close()

    fig, ax = plt.subplots()
    images = []
    counter = 0
    for x in nodeinfo:
        image, = plt.plot(x[0, :], 'r.')
        txt = plt.text(0.5, 1.01, str(
            t[counter]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        plt.xlabel('Joint No')
        plt.ylabel('Head (m)')
        images.append([image, txt])
        counter += 1
    ani = animation.ArtistAnimation(fig, images)
    writergif = animation.PillowWriter(fps=5)
    ani.save('head.gif', writer=writergif)

    fig, ax = plt.subplots()
    images = []
    counter = 0
    for x in nodeinfo:
        image, = plt.plot(x[1, :], 'r.')
        txt = plt.text(0.5, 1.01, str(
            t[counter]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        plt.xlabel('Joint No')
        plt.ylabel('Inflow ($m^3/s$)')
        images.append([image, txt])
        counter += 1
    ani = animation.ArtistAnimation(fig, images)
    writergif = animation.PillowWriter(fps=5)
    ani.save('inflow.gif', writer=writergif)

    fig, ax = plt.subplots()
    images = []
    counter = 0
    for x in nodeinfo:
        image, = plt.plot(x[2, :], 'r.')
        txt = plt.text(0.5, 1.01, str(
            t[counter]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        plt.xlabel('Joint No')
        plt.ylabel('Outflow ($m^3/s$)')
        images.append([image, txt])
        counter += 1
    ani = animation.ArtistAnimation(fig, images)
    writergif = animation.PillowWriter(fps=5)
    ani.save('outflow.gif', writer=writergif)

    fig, ax = plt.subplots()
    images = []
    counter = 0
    for x in nodeinfo:
        image, = plt.plot(x[2, :]-x[1, :], 'r.')
        txt = plt.text(0.5, 1.01, str(
            t[counter]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        plt.xlabel('Joint No')
        plt.ylabel('difference ($m^3/s$)')
        images.append([image, txt])
        counter += 1
    ani = animation.ArtistAnimation(fig, images)
    writergif = animation.PillowWriter(fps=5)
    ani.save('diff.gif', writer=writergif)
