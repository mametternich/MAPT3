# Package importation if needed
import numpy as np
import os
 
# ----------------------------------------
# Structure of a MAPT3 parameter file.
# ----------------------------------------

# --- General

# Path to the project (type: str):
# Better to put absolute path here
path = '/'.join(str(os.path.realpath(__file__)).split('/')[0:-1])+'/'   # directory where this file located

# Expected number of point at the surface
# of the convection model (type: float):
# nop = 368120
nop = 100000 # A.J. Largely enough for Marla model

# Radius of the convection model surface
# (type: float):
modelRadius = 2.2 # A.J. if you adim the mesh as I did, you need to update this here.
# modelRadius = 6356731.0      # Earth radius in meters

# Planetary model (type: MAPT3.planetaryModels):
from MAPT3.planetaryModels import Earth
planetaryModel = Earth

# Minimum polygon area (in number of points):
# below the inversion of the velocity field
# is not computed and the polygon is not
# considered as a plate (type: int):
polyminsize = 20 # A.J. From what I see of your horizontal resolution, you may want to lower this parameter 

# --- Tessellation

# Minimum persistence thresholds tested during
# the tessellation (type: np.ndarray)
pmin = np.array([100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,25000,30000,35000,40000,45000,50000])

# --- Optimization

# Critical (minimal) plateness of plates to be
# defined as rigid (type: float).
#   -> Definition of plateness after Janin et
#      al., 2025, Alisic et al., 2012
#   -> Values defined in Janin et al., 2025
P1c = 0.90
P2c = 0.80

# Random seed used during the optimization if
# the randomization is activated (type: int >0)
rseed = 5730

# Area of the smallest spatial fragment for
# which the plateness have to be good to be
# considered as rigid.
# (area in number of surface point (here
# by default, 0.8% total surface))
fragment_size = int(nop*0.8/100)



