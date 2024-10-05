import h5py
import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

class find_bubbles:
    """
    find_bubbles
    ===========

    Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.find_bubbles import find_bubbles

		Calling sequence of the function then varies based upon variable 'local'




    Input Parameters
    ----------------
    binary_array:

    local: 
    	

    chunk_size:
    	set chunk_size = None when running the module on a local computer. If 
    	running on a remote computer, set chunk_size to a value. If in doubt,
    	set chunk_size = 100.


    Optional Keywords
    -----------------
	min_voxel_count:


    Output Parameters
    -----------------
    Stored as attributes of the ''density_mapping'' class:

    .data:

    .density_grid: 

    .grid_size: 


    """
    def __init__(self, binary_array, local=True, chunk_size=None, min_voxel_count=100):
        self.binary_array = binary_array
        self.min_voxel_count = min_voxel_count
        self.chunk_size = chunk_size

        if local:
            # Complete by brute force (local processing)
            if chunk_size is not None:
                raise ValueError("Can't have chunk_size when running locally; it is only for remote programming.")
            self.bubbles, self.projection = self.find_local_bubbles(self.binary_array, min_voxel_count)    
            self.visualize_local_projection(self.projection)

        else:
            # Remote processing (small voxel size with multiprocessing)
            if chunk_size is None:
                raise ValueError("chunk_size is required for remote processing.")
            self.process_remote_in_chunks('binary_array.h5', 'dataset', chunk_size)

    # Function to label bubbles and project onto x-y plane
    def find_local_bubbles(self, binary_array, min_voxel_count=10):
        start_time = time.time()

        # Step 1: Label connected components in the binary array
        labeled_array, num_features = label(binary_array, connectivity=1, return_num=True)

        # Step 2: Get unique cluster IDs and their corresponding voxel counts
        unique_ids, voxel_counts = np.unique(labeled_array, return_counts=True)

        # Step 3: Filter clusters by minimum voxel count
        valid_clusters = unique_ids[voxel_counts >= min_voxel_count]

        # Step 4: Find the voxels for valid clusters (no multiprocessing)
        bubble_clusters = []
        
        # Add tqdm progress bar for processing bubble clusters
        for cluster_id in tqdm(valid_clusters, desc="Processing clusters", unit="cluster"):
            if cluster_id != 0:  # Skip background
                cluster = np.argwhere(labeled_array == cluster_id)
                bubble_clusters.append(cluster)

        # Step 5: Create a 2D projection (x-y plane) with different colors for each bubble
        x_size, y_size = binary_array.shape[0], binary_array.shape[1]  # Assuming x and y are the first two dimensions
        projection = np.zeros((x_size, y_size), dtype=int)  # Initialize projection array

        # Add tqdm progress bar for projection creation
        for i, cluster in enumerate(tqdm(bubble_clusters, desc="Projecting clusters", unit="cluster"), start=1):
            for voxel in cluster:
                x, y = voxel[0], voxel[1]  # Project to x-y plane
                projection[x, y] = i  # Assign a unique ID for each bubble (i)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        return bubble_clusters, projection

    def visualize_local_projection(self, projection):
        """
        Visualizes the 2D projection of bubbles with a unique random color for each bubble.
        """
        # Get the number of unique bubbles in the projection (ignoring 0 for background)
        unique_bubbles = np.unique(projection)
        unique_bubbles = unique_bubbles[unique_bubbles != 0]  # Remove background (bubble ID 0)
        
        num_bubbles = len(unique_bubbles)

        # Create a random color map for each bubble
        np.random.seed(42)  # Set seed for reproducibility
        colors = np.random.rand(num_bubbles, 3)  # Generate random RGB colors for each bubble
        
        # Create an empty RGB image to map the projection with colors
        rgb_projection = np.zeros((*projection.shape, 3), dtype=float)

        # Assign each bubble its corresponding random color
        for i, bubble_id in enumerate(unique_bubbles):
            rgb_projection[projection == bubble_id] = colors[i]

        # Plot the RGB projection
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_projection, interpolation='nearest')

        # Add labels and title
        plt.title('2D Projection of Bubbles (Random Color Coded)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Show the plot
        plt.show()

    def find_remote_bubbles(self, binary_array_chunk):
        """
        Finds connected bubble clusters in a chunk of the binary array.
        """
        start_time = time.time()  # Start timing
        labeled_array, num_features = label(binary_array_chunk, connectivity=1, return_num=True)

        # Get unique cluster IDs and their voxel counts
        unique_ids, voxel_counts = np.unique(labeled_array, return_counts=True)

        # Filter out clusters below the minimum voxel count
        valid_clusters = unique_ids[voxel_counts >= self.min_voxel_count]

        # Extract clusters as arrays of voxel coordinates
        bubble_clusters = [np.argwhere(labeled_array == cluster_id) for cluster_id in valid_clusters if cluster_id != 0]

        end_time = time.time()  # End timing
        print(f"Chunk processing took: {end_time - start_time:.2f} seconds")
        print(f"Found {len(bubble_clusters)} valid clusters in this chunk.")
        return bubble_clusters

    def visualise_remote_projection(self, bubble_clusters, binary_array_shape):
        """
        Projects bubble clusters onto the x-y plane and saves the visualization as an image.
        """
        # Create an empty projection for the x-y plane
        x_size, y_size = binary_array_shape[0], binary_array_shape[1]
        projection = np.zeros((x_size, y_size), dtype=int)

        # Project each cluster onto the x-y plane with distinct colors
        for i, cluster in enumerate(bubble_clusters, start=1):
            for voxel in cluster:
                x, y = voxel[0], voxel[1]  # x-y projection
                projection[x, y] = i  # Assign a unique color index

        # Visualize the projection
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.get_cmap('tab20', np.max(projection))
        plt.imshow(projection, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.title("2D Projection of Bubble Clusters onto x-y Plane")
        plt.xlabel('x')
        plt.ylabel('y')

        # Save the plot as an image
        plt.savefig('bubble_projection.png')
        print("Projection image saved as 'bubble_projection.png'.")

    def process_remote_in_chunks(self, filename, dataset_name, chunk_size):
        """
        Process the binary array in chunks to find bubble clusters and save results.
        """
        start_time = time.time()  # Start timing the whole process

        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            dataset = f[dataset_name]
            dataset_shape = dataset.shape
            print(f"Dataset shape: {dataset_shape}")

            all_bubble_clusters = []

            # Process the dataset in chunks (along the first dimension)
            for i in range(0, dataset_shape[0], chunk_size):
                chunk_start_time = time.time()  # Start timing for this chunk

                end_index = min(i + chunk_size, dataset_shape[0])
                print(f"Processing chunk: {i} to {end_index}")

                # Load the chunk of data
                data_chunk = dataset[i:end_index]

                # Process this chunk to find bubble clusters
                bubble_clusters = self.find_remote_bubbles(data_chunk)
                all_bubble_clusters.extend(bubble_clusters)  # Add to the list of all clusters

                chunk_end_time = time.time()  # End timing for this chunk
                print(f"Chunk {i} to {end_index} took: {chunk_end_time - chunk_start_time:.2f} seconds")




