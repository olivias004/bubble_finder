import numpy as np
import matplotlib.pyplot as plt

class thresholding:
    """
    thresholding
    ===========

    Purpose
    -------
    Generate a binary mask for a given 3D density grid by applying a percentile-based threshold.
    The binary mask highlights regions of the density grid below the threshold value.

    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.thresholding import thresholding

        threshold_instance = thresholding(density_grid, percentile=10, plot=True)

    Input Parameters
    ----------------
    density_grid : numpy.ndarray
        A 3D array representing the density grid.
    percentile : float, optional
        The percentile threshold to determine the binary mask. Default is 10.
    plot : bool, optional
        Set to True to visualize the binary mask. Default is True.
    vmin : float, optional
        Minimum value for the color scale in the plot. Default is None.
    vmax : float, optional
        Maximum value for the color scale in the plot. Default is None.

    Output Parameters
    -----------------
    Stored as attributes of the `thresholding` class:

    .threshold : float
        The calculated threshold value based on the given percentile.
    .binary_array : numpy.ndarray
        A binary mask where values below the threshold are set to 1, and others are set to 0.
    """

    def __init__(self, density_grid, percentile=10, plot=True, 
                 vmin=None, vmax=None):
        """
        Initialize the `thresholding` class, compute the threshold, and generate the binary mask.

        Parameters
        ----------
        density_grid : numpy.ndarray
            A 3D array representing the density grid.
        percentile : float, optional
            The percentile threshold to determine the binary mask. Default is 10.
        plot : bool, optional
            Set to True to visualize the binary mask. Default is True.
        vmin : float, optional
            Minimum value for the color scale in the plot. Default is None.
        vmax : float, optional
            Maximum value for the color scale in the plot. Default is None.
        """
        self.density_grid = density_grid
        self.percentile = percentile
        self.plot = plot

        # Calculate the threshold value based on the given percentile
        self.threshold = np.percentile(self.density_grid[self.density_grid > 0], self.percentile)

        # Generate the binary mask
        self.binary_array = np.where((self.density_grid < self.threshold), 1, 0)

        # Optionally plot the binary mask
        if self.plot:
            self._plot(self.binary_array, vmin, vmax)

    def _plot(self, binary_array, vmin, vmax):
        """
        Generate a 2D visualization of the binary mask.

        Parameters
        ----------
        binary_array : numpy.ndarray
            A binary array to be visualized.
        vmin : float, optional
            Minimum value for the color scale. Default is None.
        vmax : float, optional
            Maximum value for the color scale. Default is None.
        """
        # Project the binary array onto the XY plane by summing over the Z-axis
        the_map = np.sum(binary_array, axis=2)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(the_map.T, origin='lower', cmap='bone_r', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Number of Void Voxels (Summed Over Z)')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('Projected Binary Mask (Summed Over Z-Axis)')
