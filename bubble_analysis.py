import numpy as np
from scipy.spatial import KDTree, ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns


def plot_nearest_neighbor_distribution(centroids, voxel_size=0.04):
    """
    Plots the nearest neighbor distance distribution for bubble centroids.

    Parameters
    ----------
    centroids : numpy.ndarray
        Array of bubble centroids.
    voxel_size : float, optional
        The size of each voxel in kpc. Default is 0.04.
    """
    # KDTree for nearest neighbor distances
    kd_tree = KDTree(centroids)
    distances, _ = kd_tree.query(centroids, k=2)  # k=2 to exclude self
    nearest_distances = distances[:, 1]  # Second nearest is the actual nearest neighbor

    # Plot the nearest neighbor distance distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(nearest_distances * voxel_size, kde=True, bins=100, color='blue', label='Nearest Neighbor Distances')
    plt.title('Nearest Neighbor Distance Distribution')
    plt.xlabel('Distance (kpc)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def plot_centroids_on_density_map(density_grid, centroids, threshold_percentile=30, vmin=65, vmax=75):
    """
    Visualize the centroids of bubbles over the density map.

    Parameters
    ----------
    density_grid : numpy.ndarray
        The 3D density grid.
    centroids : numpy.ndarray
        Array of bubble centroids.
    threshold_percentile : float, optional
        The percentile threshold to generate the binary array for the density map. Default is 30.
    vmin : float, optional
        Minimum value for the color scale in the plot. Default is 65.
    vmax : float, optional
        Maximum value for the color scale in the plot. Default is 75.
    """
    # Generate a binary array using the density threshold
    threshold = np.percentile(density_grid[density_grid > 0], threshold_percentile)
    binary_array = np.where((density_grid < threshold), 1, 0)
    density_projection = np.sum(binary_array, axis=2)

    # Plot the density map with centroids
    plt.figure(figsize=(20, 20))
    plt.imshow(density_projection.T, origin='lower', cmap='bone_r', aspect='auto', vmin=vmin, vmax=vmax)

    # Overlay the centroids
    if centroids.size > 0:  # Ensure centers exist
        plt.scatter(
            centroids[:, 0],  # Transposed: y-coordinates become x
            centroids[:, 1],  # Transposed: x-coordinates become y
            c='red',
            s=50,
            edgecolors='white',
            label='Bubble Centers'
        )

    plt.xlabel("X-axis (grid units)")
    plt.ylabel("Y-axis (grid units)")
    plt.title("Bubble Centers Overlaid on Density Map")
    plt.legend()
    plt.tight_layout()
    plt.show()


class BubbleAnalysis:
    """
    BubbleAnalysis
    ==============
    
    Purpose
    -------
    Provides analysis of bubbles, including volume, semi-major/minor axes, positions, 
    galactocentric distances, and centroid visualizations. Results can be saved to an ASCII file.

    Features
    --------
    - Tabulated properties for bubbles (ASCII output).
    - Nearest neighbor distance plot.
    - Visualization of centroids over the density distribution.
    """

    def __init__(self, bubbles, density_grid, output_file="bubble_analysis.txt", delimiter='|'):
        """
        Initialize the BubbleAnalysis class.

        Parameters
        ----------
        bubbles : list of numpy.ndarray
            List of valid bubble clusters, where each cluster is a numpy array of voxel coordinates.
        density_grid : numpy.ndarray
            The 3D density grid used for analysis.
        output_file : str, optional
            Name of the output ASCII file for saving analysis. Default is "bubble_analysis.txt".
        delimiter : str, optional
            Delimiter for the ASCII output. Default is '|'.
        """
        self.bubbles = bubbles
        self.density_grid = density_grid
        self.output_file = output_file
        self.delimiter = delimiter

        # Perform analysis and save results
        self.bubble_analysis = self.analyze_bubbles()
        self.save_analysis()

    def analyze_bubbles(self):
        """
        Perform bubble analysis and compute properties for each bubble.

        Returns
        -------
        list of dict
            A list containing properties for each bubble.
        """
        analysis_results = []
        for bubble_id, bubble in enumerate(self.bubbles, start=1):
            # Calculate bubble properties
            volume = len(bubble)  # Number of voxels
            center = np.mean(bubble, axis=0)  # Geometric center
            galactocentric_distance = np.linalg.norm(center)  # Distance from origin
            bounding_box_min = np.min(bubble, axis=0)
            bounding_box_max = np.max(bubble, axis=0)
            semi_axes = (bounding_box_max - bounding_box_min) / 2  # Semi-major/minor axes

            # Append results
            analysis_results.append({
                'ID': bubble_id,
                'Position': center,
                'Semi-Major Axes': semi_axes,
                'Galactocentric Distance': galactocentric_distance,
                'Volume': volume
            })

        return analysis_results

    def save_analysis(self):
        """
        Save the analysis results to an ASCII file.
        """
        with open(self.output_file, 'w') as f:
            # Write the header
            headers = ['ID', 'Position', 'Semi-Major Axes', 'Galactocentric Distance', 'Volume']
            f.write(self.delimiter.join(headers) + '\n')

            # Write the data
            for bubble in self.bubble_analysis:
                f.write(f"{bubble['ID']}{self.delimiter}"
                        f"{bubble['Position'][0]:.4f},{bubble['Position'][1]:.4f},{bubble['Position'][2]:.4f}{self.delimiter}"
                        f"{bubble['Semi-Major Axes'][0]:.4f},{bubble['Semi-Major Axes'][1]:.4f},{bubble['Semi-Major Axes'][2]:.4f}{self.delimiter}"
                        f"{bubble['Galactocentric Distance']:.4f}{self.delimiter}"
                        f"{bubble['Volume']}\n")

        print(f"Bubble analysis saved to {self.output_file}")
