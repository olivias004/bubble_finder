import numpy as np
import matplotlib.pyplot as plt

class thresholding:
    """
    thresholding
    ===========

    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.thresholding import thresholding






    Input Parameters
    ----------------

        

    Optional Keywords
    -----------------


        

    Output Parameters
    -----------------
    Stored as attributes of the ''density_mapping'' class:

    .threshold:

    .binary_array:



    """

    def __init__(self, density_grid, percentile=10, plot = True, 
        vmin = None, vmax = None):

        self.density_grid = density_grid
        self.percentile = percentile
        self.plot = plot

        self.threshold = np.percentile(self.density_grid[self.density_grid > 0], self.percentile)

        self.binary_array = np.where((self.density_grid < self.threshold), 1, 0)

        if self.plot:
            self._plot(self.binary_array, vmin, vmax)



    def _plot(self, binary_array, vmin, vmax):

        the_map = np.sum(binary_array, axis = 2)

        plt.imshow(the_map.T, origin='lower', cmap='bone_r', aspect='auto', vmin = vmin, vmax = vmax)
        plt.colorbar(label='Density of Voxels')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('2D Density Map (Void Voxels Only)')
        plt.show()
















