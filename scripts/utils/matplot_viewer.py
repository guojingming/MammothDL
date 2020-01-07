import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Input pointcloud: (?, 3)
def show_pointcloud_fromarray(pointcloud, winname='PointCloud'):
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    minZ = 1000000
    maxZ = -1000000
    for i in range(pointcloud.shape[0]):
        point = pointcloud[i]
        if point[2] < minZ:
            minZ = point[2]
        elif point[2] > maxZ:
            maxZ = point[2]
    cm = plt.get_cmap("viridis")
    colors = []
    for i in range(pointcloud.shape[0]):
        rate = math.fabs(pointcloud[i, 1] - minZ) / (maxZ - minZ)
        colors.append(cm(rate))
    ax.scatter(
        pointcloud[:, 2],
        pointcloud[:, 0],
        pointcloud[:, 1],
        cmap='spectral',
        c=colors,
        s=3.0,
        linewidths=0,
        alpha=1,
        marker='.'
    )
    plt.title(winname)
    # ax.axis('scaled')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()