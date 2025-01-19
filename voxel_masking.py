import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from bubble_finder.masks import apply_elliptical_mask, apply_data_mask

class voxel_masking:
    """
    voxel_masking
    ===========

    Purpose
    -------
    Apply a series of masks to a 3D binary array to filter out voxels 
    based on an elliptical region and a threshold index.

    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.voxel_masking import voxel_masking

        voxel_mask_instance = voxel_masking(
            axes=[20, 20, 3], 
            voxel_size=0.04, 
            binary_array=thresholds.binary_array, 
            index=75, 
            plot=True
        )

        masked_binary_array = voxel_mask_instance.get_masked_array()
    Input Parameters
    ----------------
    axes : list
        The dimensions of the grid [x_extent, y_extent, z_extent].
    voxel_size : float
        The size of each voxel in the grid.
    binary_array : numpy.ndarray
        A 3D binary array where 1 indicates a selected voxel, and 0 indicates a non-selected voxel.
    index : int
        The threshold index for masking. Only voxels with summed values >= index will be retained.
    plot : bool, optional
        Set to True to display an image in the current graphic window 
        showing the masked binary array. Default is True.
    vmin : float, optional
        Minimum value for the color scale in the plot. Default is None.
    vmax : float, optional
        Maximum value for the color scale in the plot. Default is None.

    Returns
    -------
    masked_binary_array : numpy.ndarray
        The resulting masked binary array after applying both the elliptical and data masks.
    """

    def __init__(self, axes, voxel_size, binary_array, index, plot=False, vmin=None, vmax=None):
        """
        Initialize the `voxel_masking` class, apply the masks, and optionally plot the results.

        Parameters
        ----------
        axes : list
            The dimensions of the grid [x_extent, y_extent, z_extent].
        voxel_size : float
            The size of each voxel in the grid.
        binary_array : numpy.ndarray
            A 3D binary array where 1 indicates a selected voxel, and 0 indicates a non-selected voxel.
        index : int
            The threshold index for masking. Only voxels with summed values >= index will be retained.
        plot : bool, optional
            Set to True to display the results visually. Default is True.
        vmin : float, optional
            Minimum value for the color scale in the plot. Default is None.
        vmax : float, optional
            Maximum value for the color scale in the plot. Default is None.
        """
        # Store input parameters
        self.binary_array = binary_array
        self.index = index
        self.plot = plot
        self.axes = axes
        self.voxel_size = voxel_size

        # Apply the elliptical mask to filter outside the defined elliptical region
        self.masked_binary_array = apply_elliptical_mask(self.binary_array, axes=self.axes)

        # Apply the data mask to filter regions below the threshold index
        self.masked_binary_array = apply_data_mask(
            self.masked_binary_array, axes=self.axes, voxel_size=self.voxel_size, threshold_index=self.index
        )

        # Optionally plot the result
        if self.plot:
            self._plot(self.masked_binary_array, vmin, vmax)

    def _plot(self, masked_binary_array, vmin, vmax):
        """
        Generate a 2D visualization of the masked binary array.

        Parameters
        ----------
        masked_binary_array : numpy.ndarray
            The 3D binary array after applying the masks.
        vmin : float, optional
            Minimum value for the color scale. Default is None.
        vmax : float, optional
            Maximum value for the color scale. Default is None.
        """
        # Project the 3D binary array into 2D by summing along the Z-axis
        the_map = np.sum(masked_binary_array, axis=2)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(the_map.T, origin='lower', cmap='bone_r', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Number of Retained Voxels (Summed Over Z)')
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('Masked Binary Array: Thresholded and Elliptical Filter Applied')

    def get_masked_array(self):
        """
        Get the masked binary array after applying the masks.

        Returns
        -------
        numpy.ndarray
            The resulting masked binary array.
        """
        return self.masked_binary_array
