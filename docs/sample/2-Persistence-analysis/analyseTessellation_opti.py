# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
# import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
# from paraview.simple import *
# from paraview import servermanager
# from vtkmodules.util.numpy_support import vtk_to_numpy
# from scipy.interpolate import griddata
# from scipy.interpolate import RBFInterpolator

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
# import MAPT3.tessellation
from MAPT3.rigidity import rigid
from MAPT3.project import Project

# print(MAPT3.tessellation.__file__, file=sys.stderr) # checking which file actually is accessed

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================

# Quick and simple visualisation of a tessellation
plotPlateSizeDistribution = True 

# ---- 1. Tessellation description

path  = './OPTIMIZED/'
file = 'fDys30-sc_01045_optimized.h5'

# ---- 2. Creat the PlateGather object

pg = PlateGather()
pg.load_from_h5(path+file)

# ---- 3. Simple figure

# Visualisation of the plate boundaries and non-rigid polygons (where the
# explored values of pmin do not allow to find a rigid plate definition).

fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(111, projection=ccrs.Robinson())
ax.scatter(pg.lonnr,pg.latnr,s=1,transform=ccrs.PlateCarree(),label='non-rigid')
ax.scatter(pg.lonb,pg.latb,s=1,transform=ccrs.PlateCarree(),label='plate boundaries')
#fig.savefig('AGE467-555_llsvp_vp00073_opti.png',dpi=200)
ax.legend(loc='upper left')
plt.show()

# Visualisation of the plate edges
fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(111, projection=ccrs.Robinson())
ax.scatter(pg.lonnr,pg.latnr,s=1,transform=ccrs.PlateCarree(),label='non-rigid')
ax.scatter(pg.lone,pg.late,s=0.5,transform=ccrs.PlateCarree(),c='k',label='plate edges')
ax.scatter(pg.lone,pg.late,s=0.5,transform=ccrs.PlateCarree(),c='k')
ax.legend(loc='upper left')
plt.show()
# save figure with transparent background
savefigpath= '../../figs/'+file+'-edges.png'
fig.savefig(savefigpath,dpi=200,transparent=True)

# ---- 4. Exploration (example of use of some MAPT3 functions)

# Try to find why the plate covering the point (0°E,0°N) is not rigid.

# Find the ID of the plate
pID = pg.plateID[pg.get_plateID(0,0)]

# Find the plate with pID and plot
plot_pid=18
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
ax.set_global()
cmap = ax.scatter(pg.lon,pg.lat,c=pg.plateID,s=1,cmap=plt.cm.magma,transform=ccrs.PlateCarree())
mask = pg.plateID == plot_pid
ax.scatter(pg.lon[mask], pg.lat[mask], s=6, c='cyan', transform=ccrs.PlateCarree(), label=f'Plate {plot_pid}')
cbar = fig.colorbar(cmap,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8)
ax.legend(loc='lower left')
plt.show()  

# Inverse the velocity field of the plate and get its local plateness P1 and P2
wx,wy,wz = pg.get_rotation(pID,r=50,plot=False)
P1 = pg.P11 
P2 = pg.P12

# Test if the plate if rigid or not and why
is_rigid = rigid(pg,pID,wx,wy,wz,P1,P2,plot=True)

# Find its adjacent plates
pg.get_neighborhood()