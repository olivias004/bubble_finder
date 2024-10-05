import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from bubble_finder.masks import clipping_data
from bubble_finder.find_axes import find_axes

class density_grid:
    """
    density_mapping
    ===========

    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.density_mapping import density_grid

        density_grid, grid_size = density_grid(densities, positions,
            axes = None, plot = True, voxel_size = 0.1, vmin = None, 
            vmax = None)




    Input Parameters
    ----------------
    densities:

    positions:
        

    Optional Keywords
    -----------------
    axes:

    voxel_size:

    vmin:

    vmax:

    plot: boolean, optional
        set ''plot=True'' to display an image in the current graphic window
        showing the pixels used in the computation of the moments.

        

    Output Parameters
    -----------------
    Stored as attributes of the ''density_mapping'' class:

    .data:

    .density_grid: 

    .grid_size: 


    """

    def __init__(self, densities, positions, axes = None, plot = False, 
        voxel_size = 0.1, vmin = None, vmax = None):
        self.axes = axes
        self.densities = densities
        self.positions = positions
        self.voxel_size = voxel_size

        self.data = [self.axes, self.voxel_size]

        if axes == None:
            self.axes = find_axes(positions)
        else:
            if type(axes) == list and len(axes) == 3:
                self.axes = axes
            else:
                raise ValueError('axes must be a list with exactly 3 elements.')


        data_mask = clipping_data(self.axes, self.positions)
        self.density_grid, self.grid_size = self._density_grid(self.positions[data_mask], 
            self.densities[data_mask], self.axes, self.voxel_size)

        if plot == True:
            self._plot(self.density_grid, vmin, vmax)


  
    def _density_grid(self, positions, densities, axes, voxel_size):

        grid_size = np.ceil(np.array(axes) / voxel_size).astype(int)

        density_grid = np.zeros(grid_size)

        shifted_positions = positions + [axes[0]/2, axes[1]/2, axes[2]/2]
        voxel_indices = np.floor(shifted_positions / voxel_size).astype(int)

        np.add.at(density_grid, tuple(voxel_indices.T), densities)

        return density_grid, grid_size



    def _plot(self, density_grid, vmin, vmax):
        xy_density_map = np.sum(density_grid, axis=2)

        # Plot the 2D density map
        plt.figure(figsize=(8, 6))
        plt.imshow(xy_density_map.T, origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Density $(M_\odot \, \mathrm{kpc}^{-3})$')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('2D Density Map (Summed Over Z-Axis)')
        plt.show()





