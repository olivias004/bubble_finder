import h5py
import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import time
from tqdm import tqdm

class find_bubbles:
    """
    find_bubbles
    ===========

    Purpose
    -------
    Identify connected clusters (bubbles) in a binary voxel array and remove all invalid voxels.

    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.find_bubbles import find_bubbles

    Input Parameters
    ----------------
    binary_array: 3D binary numpy array
        Array with 1s marking potential voxels of interest and 0s elsewhere.

    local: boolean, optional
        If True, process locally without chunking. Default is True.

    chunk_size: int, optional
        Chunk size to use for remote processing. Default is None.

    min_voxel_count: int, optional
        Minimum number of voxels for a cluster to be considered valid. Default is 100.

    Output Parameters
    -----------------
    Stored as attributes of the ''find_bubbles'' class:

    .bubbles: list of numpy arrays
        List of valid clusters, where each cluster is a numpy array of voxel coordinates.

    .cleaned_array: numpy array
        Binary array with only valid cluster voxels retained.

    """
    def __init__(self, binary_array, local=True, chunk_size=None, min_voxel_count=100):
        self.binary_array = binary_array
        self.min_voxel_count = min_voxel_count
        self.chunk_size = chunk_size

        if local:
            # Complete by brute force (local processing)
            if chunk_size is not None:
                raise ValueError("Can't have chunk_size when running locally; it is only for remote programming.")
            self.bubbles, self.cleaned_array = self.find_local_bubbles(self.binary_array, min_voxel_count)
            self.visualize_local_projection(self.cleaned_array, self.bubbles)

        else:
            # Remote processing (small voxel size with multiprocessing)
            if chunk_size is None:
                raise ValueError("chunk_size is required for remote processing.")
            self.process_remote_in_chunks('binary_array.h5', 'dataset', chunk_size)

    def find_local_bubbles(self, binary_array, min_voxel_count=10):
        start_time = time.time()

        # Step 1: Label connected components in the binary array
        labeled_array, num_features = label(binary_array, connectivity=1, return_num=True)

        # Step 2: Get unique cluster IDs and their corresponding voxel counts
        unique_ids, voxel_counts = np.unique(labeled_array, return_counts=True)

        # Step 3: Filter clusters by minimum voxel count
        valid_clusters = unique_ids[voxel_counts >= min_voxel_count]

        # Step 4: Find the voxels for valid clusters and clean the array
        bubble_clusters = []
        cleaned_array = np.zeros_like(binary_array, dtype=np.int32)  # Initialize cleaned array

        for cluster_id in tqdm(valid_clusters, desc="Processing clusters", unit="cluster"):
            if cluster_id != 0:  # Skip background
                cluster = np.argwhere(labeled_array == cluster_id)
                bubble_clusters.append(cluster)

                # Retain valid voxels in the cleaned array
                cleaned_array[labeled_array == cluster_id] = cluster_id

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        return bubble_clusters, cleaned_array

    def visualize_local_projection(self, cleaned_array, bubbles):
        """
        Visualizes the 2D projection of the cleaned binary array with distinct colors for each cluster.

        Parameters
        ----------
        cleaned_array : numpy.ndarray
            The binary array containing valid clusters.
        bubbles : list of numpy.ndarray
            List of valid clusters as numpy arrays of voxel coordinates.
        """
        # Project onto the x-y plane by summing along the z-axis
        projection = np.sum(cleaned_array, axis=2)

        # Create a discrete colormap with unique colors for each cluster
        num_clusters = len(bubbles)
        colors = plt.cm.get_cmap('tab20', num_clusters)  # Use a colormap with discrete colors

        # Remap cluster IDs to indices within the range of the colormap
        projection_normalized = (projection % num_clusters).astype(int)

        # Create the plot
        plt.figure(figsize=(10, 10))

        # Set the background color to white
        plt.imshow(projection_normalized.T, origin='lower', cmap=colors, aspect='auto', alpha=1)
        plt.gca().set_facecolor('white')

        # Add labels and title
        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc)')
        plt.title('2D Projection of Clusters with Unique Colors')

        # Show the plot without a colorbar
        plt.show()



    def process_remote_in_chunks(self, filename, dataset_name, chunk_size):
        """
        Process the binary array in chunks to find bubble clusters and clean the array.
        """
        start_time = time.time()  # Start timing the whole process

        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            dataset = f[dataset_name]
            dataset_shape = dataset.shape
            print(f"Dataset shape: {dataset_shape}")

            all_bubble_clusters = []
            cleaned_array = np.zeros(dataset_shape, dtype=np.int32)  # Initialize cleaned array

            # Process the dataset in chunks (along the first dimension)
            for i in range(0, dataset_shape[0], chunk_size):
                chunk_start_time = time.time()  # Start timing for this chunk

                end_index = min(i + chunk_size, dataset_shape[0])
                print(f"Processing chunk: {i} to {end_index}")

                # Load the chunk of data
                data_chunk = dataset[i:end_index]

                # Process this chunk to find bubble clusters
                bubble_clusters, cleaned_chunk = self.find_local_bubbles(data_chunk)
                all_bubble_clusters.extend(bubble_clusters)  # Add to the list of all clusters

                # Update the cleaned array
                cleaned_array[i:end_index] = cleaned_chunk

                chunk_end_time = time.time()  # End timing for this chunk
                print(f"Chunk {i} to {end_index} took: {chunk_end_time - chunk_start_time:.2f} seconds")

            end_time = time.time()  # End timing the whole process
            print(f"Total processing time: {end_time - start_time:.2f} seconds")

        self.bubbles = all_bubble_clusters
        self.cleaned_array = cleaned_array

