import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from bubble_finder.masks import apply_elliptical_mask

class voxel_masking:
    """
    voxel_masking
    ===========

    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.voxel_masking import voxel_masking



    Input Parameters
    ----------------
    axes_array:

    binary_array:

    index:

        

    Optional Keywords
    -----------------
    plot: boolean, optional
        set ''plot=True'' to display an image in the current graphic window
        showing the pixels used in the computation of the moments.

    vmin:

    vmax:


    Output Parameters
    -----------------
    Stored as attributes of the ''voxel_masking'' class:

    .masked_binary_array

    """

    def __init__(self, axes_array, binary_array, index, plot = True, vmin = None, vmax = None):

        self.binary_array = binary_array
        self.index = index
        self.plot = plot

        self.axes, self.voxel_size = axes_array[0], axes_array[1]

        the_map = np.sum(self.binary_array, axis = 2) 
        self.masked_binary_array = apply_elliptical_mask(self.binary_array, axes = self.axes)

        for i in range(int(self.axes[0]/self.voxel_size-1)):
            for j in range(int(self.axes[1]/self.voxel_size-1)):
                idx = the_map[i, j]  # Assuming `the_map` is your projected density map
                if idx >= self.index:
                    continue
                else:
                    self.masked_binary_array[i, j, :] = 0  # Set all z-values for this (x, y) coordinate to 0


        if self.plot:
            self._plot(self.masked_binary_array, vmin, vmax)


    def _plot(self, masked_binary_array, vmin, vmax):

        the_map = np.sum(masked_binary_array, axis = 2)

        plt.imshow(the_map.T, origin='lower', cmap='bone_r', aspect='auto', vmin = vmin, vmax = vmax)
        plt.colorbar(label='Density of Voxels')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('2D Density Map (Void Voxels Only) with Masks')





