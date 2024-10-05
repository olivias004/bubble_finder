import numpy as np



def clipping_data(axes, positions):
    """
    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.masks import clipping_data

        mask = clipping_data(axes, positions)


    Input Parameters
    ----------------

    axes:

    positions:



    Output Parameters
    -----------------
    final_mask:



    """

    # Extract axes limits
    x_radius = axes[0] / 2  # Radius for x (semi-major axis for ellipse)
    y_radius = axes[1] / 2  # Radius for y (semi-minor axis for ellipse)
    z_min, z_max = -axes[2] / 2, axes[2] / 2  # Limits for z-axis

    # Extract positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Create a circular (or elliptical) mask in the x-y plane
    # For a circle, x_radius = y_radius
    mask_xy = (x**2 / x_radius**2) + (y**2 / y_radius**2) <= 1

    # Create a z-axis mask
    mask_z = (z >= z_min) & (z <= z_max)

    # Combine the masks to get the final mask
    final_mask = mask_xy & mask_z

    return final_mask


def ellipse(x, y, a, b):
    """
    Purpose
    -------

    Calling Sequence
    ----------------
    x, y:

    a, b:

    Output Parameters
    -----------------
    elliptical function:
    
    """
    return (x/a)**2 + (y/b)**2


def apply_elliptical_mask(binary_array, axes):
    """
    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.masks import apply_elliptical_mask
        masked_binary_array = apply_elliptical_mask(binary_array, axes)




    Input Parameters
    ----------------
    binary_array:

    axes:



    Output Parameters
    -----------------
    binary_array:


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






