import numpy as np
from scipy.spatial import ConvexHull

def analyze_bubbles_and_save(void_clusters, densities, output_file="bubble_analysis.txt", delimiter='|'):
    """
    Analyzes the bubbles (voids) in terms of volume, density, and other features, and saves the results in an ASCII file.

    Parameters:
    - void_clusters: List of 3D arrays containing voxel coordinates for each void
    - densities: A list of arrays where each entry represents the density values for the corresponding void
    - output_file: Name of the output ASCII file
    - delimiter: Delimiter for the output file (default: '|')
    """
    bubble_analysis = []

    for bubble_id, (void, density_values) in enumerate(zip(void_clusters, densities), start=1):
        # Volume: Number of voxels in the bubble
        volume = len(void)
        
        # Average Density: Mean of the density values for this bubble
        avg_density = np.mean(density_values)
        
        # Center Location: Geometric centroid of the bubble
        center = np.mean(void, axis=0)
        
        # Lowest Density Point Location: Find the point with the lowest density
        min_density_idx = np.argmin(density_values)
        lowest_density_point = void[min_density_idx]
        
        # Bounding Box Dimensions (min/max along each axis)
        bounding_box_min = np.min(void, axis=0)
        bounding_box_max = np.max(void, axis=0)
        bounding_box_dims = bounding_box_max - bounding_box_min
        
        # Check if the void is degenerate (flat or near-coplanar)
        if np.ptp(void, axis=0).min() < 1e-6:  # If the range of points in any dimension is very small
            surface_area = 0  # Assign 0 surface area to flat bubbles
            aspect_ratio = 0  # No aspect ratio for flat bubbles
            sphericity = 0  # No sphericity for flat bubbles
            compactness = 0  # No compactness for flat bubbles
            print(f"Bubble {bubble_id} is near-coplanar or flat, skipping convex hull.")
        else:
            # Surface Area: Use convex hull approximation (with jitter to avoid coplanarity issues)
            try:
                hull = ConvexHull(void)
                surface_area = hull.area
            except Exception as e:
                print(f"Bubble {bubble_id}: ConvexHull failed with error: {e}")
                surface_area = 0

            # Elongation (aspect ratio): Ratio of largest to smallest principal component
            cov_matrix = np.cov(void, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            aspect_ratio = np.sqrt(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else 0
            
            # Sphericity: Based on volume and surface area
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area if surface_area > 0 else 0
            
            # Compactness: Ratio of volume to surface area cubed
            compactness = (volume / surface_area ** (3/2)) if surface_area > 0 else 0
        
        # Create the analysis record for this bubble
        bubble_record = {
            'ID': bubble_id,
            'Volume': volume,
            'Average Density': avg_density,
            'Center': center,
            'Lowest Density Point': lowest_density_point,
            'Bounding Box Dimensions': bounding_box_dims,
            'Surface Area': surface_area,
            'Aspect Ratio': aspect_ratio,
            'Sphericity': sphericity,
            'Compactness': compactness,
        }
        
        # Append the analysis to the list
        bubble_analysis.append(bubble_record)

    # Save the analysis to an ASCII file
    with open(output_file, 'w') as f:
        # Write the header
        headers = ['ID', 'Volume', 'Average Density', 'Center', 'Lowest Density Point',
                   'Bounding Box Dimensions', 'Surface Area', 'Aspect Ratio', 'Sphericity', 'Compactness']
        f.write(delimiter.join(headers) + '\n')
        
        # Write the data
        for bubble in bubble_analysis:
            f.write(f"{bubble['ID']}{delimiter}{bubble['Volume']}{delimiter}"
                    f"{bubble['Average Density']:.4f}{delimiter}"
                    f"{bubble['Center'][0]:.4f},{bubble['Center'][1]:.4f},{bubble['Center'][2]:.4f}{delimiter}"
                    f"{bubble['Lowest Density Point'][0]:.4f},{bubble['Lowest Density Point'][1]:.4f},{bubble['Lowest Density Point'][2]:.4f}{delimiter}"
                    f"{bubble['Bounding Box Dimensions'][0]:.4f},{bubble['Bounding Box Dimensions'][1]:.4f},{bubble['Bounding Box Dimensions'][2]:.4f}{delimiter}"
                    f"{bubble['Surface Area']:.4f}{delimiter}{bubble['Aspect Ratio']:.4f}{delimiter}"
                    f"{bubble['Sphericity']:.4f}{delimiter}{bubble['Compactness']:.4f}\n")
    
    print(f"Analysis saved to {output_file}")

