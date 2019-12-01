import numpy as np
from matplotlib import pyplot as plt


def data_visualize(path):
    tmp = np.load(path)
    voxels = tmp['voxel']
    segs = tmp['seg']

    for i in range(37, 60, 2):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(segs[i])
        plt.subplot(1, 2, 2)
        plt.imshow(voxels[i])
        plt.show()


if __name__ == '__main__':
    root = '/media/songkai/E4824F12824EE91E/3d_voxel/data/data_test/'
    path = 'candidate493.npz'
    data_visualize(root+path)