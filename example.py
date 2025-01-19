#EXEMPLAR

#IMPORTS=====================
import numpy as np
import sys
import glob
import os
import re
import struct

#TIPSY simulation
import pynbody
import pynbody.plot.sph as sph
import pynbody.plot as pp

#bubble_finder functions
from bubble_finder.density_mapping import density_grid
from bubble_finder.thresholding import thresholding
from bubble_finder.voxel_masking import voxel_masking
from bubble_finder.find_bubbles import find_bubbles


#Data analysis 
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree

#INPUTS======================
#Values
params = {"font.family":"serif","mathtext.fontset":"stix"}
matplotlib.rcParams.update(params)

kpc_cgs= 3.08567758e21
G_cgs  = 6.67e-8
Mo_cgs = 1.99e33
umass_GizToGas = 1.  #1e9Mo
umass = 1.0 #* umass_GizToGas
udist = 1.0  #kpc
uvel  = np.sqrt( G_cgs * umass * Mo_cgs / (udist * kpc_cgs) )/1e5
udens = umass * Mo_cgs / (udist * kpc_cgs)**3.
utime = np.sqrt(1./(udens * G_cgs))
sec2myr = 60.*60.*24.*365.*1e6

#inputs
filepth = '/Users/livisilcock/Documents/PROJECTS/VOIDS/data/iso_b/og_data/GLX.0'
testname = filepth.split('/')[-2]
timestep = (['1000'])

#access inputs
i=0
filenom = (filepth + timestep[i])
dno = timestep[i]
simulation = pynbody.load(filenom)
simulation.physical_units()

t_now =  simulation.properties['time'].in_units('Myr')
timestr = str( np.round(float(t_now), 1) )
pynbody.analysis.angmom.faceon(simulation)

s = simulation

#FINDING BUBBLES=============
#Density Map
sol = density_grid(s.gas['rho'], s.gas['pos'], axes = [20,20,3], 
	vmin = 0, vmax = 5e9, voxel_size = 0.04, plot =True)
plt.show()

#Name the variables obtained
density_grid = sol.density_grid
grid_size = sol.grid_size


#Masking non-bubble voxels
thresholds = thresholding(density_grid, percentile = 30, plot = True, vmin = 65, vmax = 75)
plt.show()

# Apply voxel masking
voxel_mask_instance = voxel_masking(
    axes=[20, 20, 3], 
    voxel_size=0.04, 
    binary_array=thresholds.binary_array, 
    index=75, 
    plot=True
)
plt.show()

# Retrieve the masked binary array
masked_binary_array = voxel_mask_instance.get_masked_array()

# Find bubbles using the masked binary array
bubbles_instance = find_bubbles(masked_binary_array, min_voxel_count=1000)

# Access the results
bubbles = bubbles_instance.bubbles          # List of valid clusters
cleaned_array = bubbles_instance.cleaned_array  # Cleaned binary array with valid clusters







# Assuming `bubbles` is an instance of the `find_bubbles` class
# The `bubbles.bubbles` attribute contains the list of valid bubble clusters as voxel arrays

bubble_centers = []

# Extract bubble centers from valid clusters
for bubble in bubbles:  # `bubbles.bubbles` is a list of voxel arrays
    # Calculate the center of each bubble
    center_coords = np.mean(bubble, axis=0)
    bubble_centers.append(center_coords)

# Convert bubble centers to a numpy array
bubble_centers = np.array(bubble_centers)




threshold = np.percentile(density_grid[density_grid > 0], 30)
binary_array = np.where((density_grid < threshold), 1, 0)

the_map = np.sum(binary_array, axis=2)

plt.figure(figsize=(20, 20))
plt.imshow(the_map.T, origin='lower', cmap='bone_r', aspect='auto', vmin=65, vmax=75)
plt.colorbar(label='Number of Voxels')

# Adjust the bubble center coordinates to match the transposed map
if bubble_centers.size > 0:  # Ensure centers exist
    plt.scatter(
        bubble_centers[:, 0],  # Transposed: y-coordinates become x
        bubble_centers[:, 1],  # Transposed: x-coordinates become y
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
# plt.savefig("/Users/livisilcock/Desktop/important_bubbles/centroids.png", bbox_inches = "tight")
# plt.close()












def plot_nearest_neighbor_distribution(centroids):
    """
    Plots the distribution of nearest neighbor distances.
    
    Parameters:
    - centroids: Numpy array of bubble centroids.
    """
    # Step 1: Build KD-Tree
    kd_tree = KDTree(centroids)
    
    # Step 2: Find the nearest neighbor for each bubble (excluding itself)
    distances, _ = kd_tree.query(centroids, k=2)  # k=2 to get the nearest neighbor excluding self
    nearest_distances = distances[:, 1]  # The second nearest is the actual nearest neighbor (excluding self)

    # Step 3: Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(nearest_distances * 0.04, kde=True, bins=100, color='blue', label='Nearest Neighbor Distances')
    plt.title('Nearest Neighbor Distance Distribution')
    plt.xlabel('Distance (kpc)')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.savefig("/Users/livisilcock/Desktop/important_bubbles/nearest_neighbour.png", bbox_inches = "tight")
    # plt.close()


# Extract the centroids for each bubble
manual_centroids = []

# Loop through each bubble cluster
for bubble in bubbles:  # `bubbles.bubbles` is a list of voxel arrays
    # Calculate the centroid of each bubble
    centroid = np.mean(bubble, axis=0)
    manual_centroids.append(centroid)

# Convert to a numpy array for further processing
manual_centroids = np.array(manual_centroids)

# Now use the manual_centroids in your function
plot_nearest_neighbor_distribution(manual_centroids)
plt.show()



