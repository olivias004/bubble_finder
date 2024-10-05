import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt

class BubbleFinder:
    """
    BubbleFinder
    ===========

    Purpose
    -------


    Calling Sequence
    ----------------
    


    Input Parameters
    ----------------
    density_map: 

    axes: 
    

    Optional Keywords
    -----------------
    
        

    Output Parameters
    -----------------

    """

    def __init__(self, density_map, axes, percentile = 25, plot = False):
        self.density_map = density_map
        self.axes = axes
        self.percentile = percentile

        #Find the bubbles
        threshold = np.percentile(density_grid, self.percentile)
        underdense_mask = density_grid < threshold

        struct = generate_binary_structure(2, 1)  # 2D connectivity, single connection
        cleaned_mask = binary_opening(underdense_mask, structure=struct)

        labeled_underdense, num_features = label(cleaned_mask)

        centers = center_of_mass(cleaned_mask, labels=labeled_underdense, index=range(1, num_features + 1))

        if plot:
            self._find_plot(axes)


    def _find_plot():
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        ax.set_title('Contours of Underdense Regions')

        # Plot the density grid
        #img = ax.imshow(density_grid, origin='lower', cmap='viridis')
        img = ax.imshow(smoothed_density_grid, extent=(-self.axes[0]/2, self.axes[0]/2, -self.axes[1]/2, self.axes[1]/2), origin='lower', norm=LogNorm(), cmap='viridis')

        # Add contours at the threshold level
        contours = ax.contour(density_grid, levels=[threshold], colors='white', extent=(-self.axes[0]/2, self.axes[0]/2, -self.axes[1]/2, self.axes[1]/2))
        plt.clabel(contours, inline=True, fontsize=8, fmt=f'Threshold: {threshold:.2f}')

        # Colorbar for reference
        fig.colorbar(img, ax=ax)
















        threshold = np.percentile(density_grid, self.percentile)

        # Create a binary mask where the density is below the threshold
        underdense_mask = density_grid < threshold

        # Clean up the mask to ensure each region is distinct using morphological opening
        struct = generate_binary_structure(2, 1)  # 2D connectivity, single connection
        cleaned_mask = binary_opening(underdense_mask, structure=struct)

        # Label the connected underdense regions
        labeled_underdense, num_features = label(cleaned_mask)

        # Calculate the centers of mass for each labeled region
        centers = center_of_mass(cleaned_mask, labels=labeled_underdense, index=range(1, num_features + 1))

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        ax.set_title('Contours and Centers of Underdense Regions')

        # Display the density map
        #img = ax.imshow(smoothed_density_grid, extent=(-10, 10, -10, 10), origin='lower', norm=LogNorm(), cmap='viridis')        
        img = ax.imshow(density_grid, origin='lower', cmap='viridis', norm=LogNorm())

        # Contour the underdense regions and plot their centers
        contours = ax.contour(cleaned_mask, levels=[0.5], colors='white')
        #contours = ax.contour(density_grid, levels=[threshold], colors='white', extent=(-10,10,-10,10))
        for center in centers:
            if not np.isnan(center[0]):  # Check if center is computed (not NaN)
                ax.plot(center[1], center[0], 'ro')  # Plot the center as a red dot

        # Add a colorbar for reference
        fig.colorbar(img, ax=ax)

        plt.show()




