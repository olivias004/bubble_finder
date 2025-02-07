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
from bubble_finder.bubble_analysis import BubbleAnalysis, plot_nearest_neighbor_distribution, plot_centroids_on_density_map



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


# Analyse the bubbles
bubble_analysis = BubbleAnalysis(
    bubbles=bubbles, 
    density_grid=density_grid, 
    output_file="/Users/livisilcock/Desktop/important_bubbles/bubble_analysis.txt"
)

centroids = np.array([bubble['Position'] for bubble in bubble_analysis.bubble_analysis])
plot_nearest_neighbor_distribution(centroids)
plot_centroids_on_density_map(density_grid=density_grid, centroids=centroids)










