import numpy as np

def ellipse(x, y, a, b):
    """
    Compute the value of the elliptical equation for given x, y, semi-major axis (a), and semi-minor axis (b).

    Parameters
    ----------
    x : float or numpy.ndarray
        The x-coordinate(s).
    y : float or numpy.ndarray
        The y-coordinate(s).
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.

    Returns
    -------
    float or numpy.ndarray
        The computed value(s) of the ellipse equation.
    """
    return (x / a)**2 + (y / b)**2


def apply_elliptical_mask(binary_array, axes):
    """
    Apply an elliptical mask to a 3D binary array, setting voxels outside the ellipse to zero.

    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.masks import apply_elliptical_mask
        masked_binary_array = apply_elliptical_mask(binary_array, axes)

    Parameters
    ----------
    binary_array : numpy.ndarray
        A 3D binary array where 1 indicates a selected voxel, and 0 indicates a non-selected voxel.
    axes : list or numpy.ndarray
        The dimensions of the grid [x_extent, y_extent, z_extent].

    Returns
    -------
    binary_array : numpy.ndarray
        The modified binary array with masked values set to zero.
    """
    x_dim, y_dim, z_dim = binary_array.shape

    # Create voxel grid positions for x and y
    x = np.linspace(-axes[0] / 2, axes[0] / 2, x_dim)
    y = np.linspace(-axes[1] / 2, axes[1] / 2, y_dim)

    # Loop through every voxel in the array
    for i in range(x_dim):
        for j in range(y_dim):
            # Check if the voxel lies outside the ellipse
            if ellipse(x[i], y[j], axes[0] / 2, axes[1] / 2) > 1:
                # Set all z-values for this (x, y) coordinate to 0
                binary_array[i, j, :] = 0 

    return binary_array


def apply_data_mask(binary_array, axes, voxel_size, threshold_index):
    """
    Apply a data mask to a binary array, retaining only regions with summed voxel values 
    along the Z-axis above a threshold.

    Parameters
    ----------
    binary_array : numpy.ndarray
        A 3D binary array where 1 indicates a selected voxel, and 0 indicates a non-selected voxel.
    axes : list or numpy.ndarray
        The dimensions of the grid [x_extent, y_extent, z_extent].
    voxel_size : float
        The size of each voxel in the grid.
    threshold_index : int
        The threshold index for masking. Only voxels with values >= threshold_index will be retained.

    Returns
    -------
    masked_binary_array : numpy.ndarray
        A masked version of the binary array with regions below the threshold set to zero.
    """
    # Project the 3D binary array into a 2D map (sum along the Z-axis)
    projected_map = np.sum(binary_array, axis=2)

    # Initialize the masked binary array
    masked_binary_array = binary_array.copy()

    # Loop through the 2D grid to apply the mask
    for i in range(int(axes[0] / voxel_size)):
        for j in range(int(axes[1] / voxel_size)):
            # Check the value in the projected map
            idx = projected_map[i, j]
            if idx < threshold_index:
                # If below the threshold, set the entire column (along Z-axis) to 0
                masked_binary_array[i, j, :] = 0

    return masked_binary_array
