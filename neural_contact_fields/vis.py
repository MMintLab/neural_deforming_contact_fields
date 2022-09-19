import matplotlib.pyplot as plt
import numpy as np


def plot_points(points: np.ndarray, colors=None, size=0.04, vis=True):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.grid()

    if vis:
        plt.show()
