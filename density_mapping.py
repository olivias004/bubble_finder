import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from bubble_finder.find_axes import find_axes  # For axis alignment

class density_grid:
    """
    density_mapping
    ===========

    Purpose
    -------
    Generate a density grid from input particle densities and positions. 
    The grid is voxelized, and the density of particles is accumulated within each voxel.

    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.density_mapping import density_grid

        density_grid_instance = density_grid(densities, positions,
            axes=None, plot=True, voxel_size=0.1, vmin=None, vmax=None)

    Input Parameters
    ----------------
    densities : numpy.ndarray
        A 1D array of particle densities.
    positions : numpy.ndarray
        A 2D array of particle positions with shape (N, 3).
    axes : list, optional
        The dimensions of the grid in [x, y, z]. If not provided, axes are computed automatically.
    voxel_size : float, optional
        The size of each voxel in the grid. Default is 0.1.
    vmin : float, optional
        Minimum density value for the plot. Default is None.
    vmax : float, optional
        Maximum density value for the plot. Default is None.
    plot : bool, optional
        Set to True to generate a 2D density plot. Default is False.

    Output Parameters
    -----------------
    Stored as attributes of the `density_grid` class:

    .density_grid : numpy.ndarray
        A 3D grid representing the density of particles in each voxel.
    .grid_size : numpy.ndarray
        The size of the density grid in [x, y, z].
    """

    def __init__(self, densities, positions, axes=None, plot=False, 
                 voxel_size=0.1, vmin=None, vmax=None):

        # Define variables
        self.axes = axes
        self.densities = densities
        self.positions = positions
        self.voxel_size = voxel_size

        # Ensure axes are defined
        if self.axes is None:
            self.axes = find_axes(positions)  # Automatically determine axes
        else:
            if isinstance(self.axes, list) and len(axes) == 3:
                self.axes = axes
            else:
                raise ValueError('Axes must be a list with exactly 3 elements.')

        # Compute the density grid
        self.density_grid, self.grid_size = self._density_grid(
            self.positions, self.densities, self.axes, self.voxel_size
        )

        if plot:
            self._plot(self.density_grid, vmin, vmax)

    def _density_grid(self, positions, densities, axes, voxel_size):
        """
        Private method to generate a voxelized density grid.

        Parameters
        ----------
        positions : numpy.ndarray
            A 2D array of particle positions with shape (N, 3).
        densities : numpy.ndarray
            A 1D array of particle densities.
        axes : list
            The dimensions of the grid in [x, y, z].
        voxel_size : float
            The size of each voxel in the grid.

        Returns
        -------
        density_grid : numpy.ndarray
            A 3D density grid.
        grid_size : numpy.ndarray
            The size of the grid in [x, y, z].
        """
        # Compute the grid size based on the axes and voxel size
        grid_size = np.ceil(np.array(axes) / voxel_size).astype(int)

        # Initialize an empty density grid with the calculated size
        density_grid = np.zeros(grid_size)

        # Shift positions to align with the center of the grid
        shifted_positions = positions + np.array(axes) / 2

        # Convert positions to voxel indices within the grid
        voxel_indices = np.floor(shifted_positions / voxel_size).astype(int)

        # Filter out invalid indices (ensure they lie within the grid bounds)
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_size[2])
        )

        # Apply the valid mask to positions and densities
        valid_voxel_indices = voxel_indices[valid_mask]
        valid_densities = densities[valid_mask]

        # Accumulate densities into the corresponding voxels of the grid
        np.add.at(density_grid, tuple(valid_voxel_indices.T), valid_densities)

        return density_grid, grid_size

    def _plot(self, density_grid, vmin, vmax):
        """
        Private method to generate a 2D density map summed over the Z-axis.

        Parameters
        ----------
        density_grid : numpy.ndarray
            A 3D density grid.
        vmin : float, optional
            Minimum density value for the plot. Default is None.
        vmax : float, optional
            Maximum density value for the plot. Default is None.
        """
        # Sum the density grid along the Z-axis to create a 2D map
        xy_density_map = np.sum(density_grid, axis=2)

        # Generate the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(xy_density_map.T, origin='lower', cmap='viridis', aspect='auto', 
                   vmin=vmin, vmax=vmax)
        plt.colorbar(label='Density $(M_\odot \, \mathrm{kpc}^{-3})$')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('2D Density Map (Summed Over Z-Axis)')
