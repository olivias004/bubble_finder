#EXEMPLAR

#IMPORTS=====================
import pynbody
import pynbody.plot.sph as sph
import pynbody.plot as pp
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import matplotlib
import os
import pynbody.plot.sph as sph
import scipy.signal
import scipy.interpolate
import re
import struct

#bubble_finder functions
from bubble_finder.density_mapping import density_grid
from bubble_finder.thresholding import thresholding
from bubble_finder.voxel_masking import voxel_masking
from bubble_finder.find_bubbles import find_bubbles

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
filepth = 'GLX.0'
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

density_grid = sol.density_grid
grid_size = sol.grid_size
axes_array = sol.data

#Masking non-bubble voxels
thresholds = thresholding(density_grid, percentile = 10, plot = True, vmin = 65, vmax = 75)
voxel_stuff = voxel_masking(axes_array, thresholds.binary_array, 75)

#Finding bubbles
sol = find_bubbles(thresholds.binary_array)



