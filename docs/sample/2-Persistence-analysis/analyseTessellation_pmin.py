# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project

# Scientific colour map importation
import sys
sys.path.append('/Users/marlametternich/Documents/ScientificColourMaps7/lajolla')
from lajolla import lajolla_map

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================

# Quick and simple visualisation of a tessellation

# ---- 1. Tessellation description

pmin = 1000

path  = '../1-Tessellation/TTK_outputs/'+'p'+intstringer(round(pmin),5)+'/'
# file  = 'AGE467-555_llsvp_vp00071.h5'
# file  = 'freeDys30-sid_00670.h5'
# file  = 'freeDys30-sid_01045.h5'
# file = 'fDys20_eta20_00500.h5'
file  = 'fDys30_01045.h5'
# file  = 'Dys30_00900.h5'
# file  = 'freeCys50_00500.h5'

# ---- 2. Creat the PlateGather object

pg = PlateGather()
pg.load_from_h5(path+file)

# ---- 3. Simple figure

# Visualisation of the plate boundaries

fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(111, projection=ccrs.Robinson())
mgsv = ax.scatter(pg.lon,pg.lat,s=0.6,c=pg.magGSV,transform=ccrs.PlateCarree(),cmap=lajolla_map
                   ,vmin=0,vmax=1e5)
ax.scatter(pg.lonb,pg.latb,s=0.01,c='green',transform=ccrs.PlateCarree())
noa = 2000
aid = list(range(pg.lon.shape[0]))
mask_arrows = random.sample(aid,noa)
xa = pg.lon[mask_arrows]
ya = pg.lat[mask_arrows]
vx = pg.vphi[mask_arrows]
vy = -pg.vtheta[mask_arrows]
Q  = ax.quiver(xa,ya,vx,vy,scale=6e5,width=1.5e-3,color='red',transform=ccrs.PlateCarree())
# ax.legend()
# ax.gridlines(draw_labels=True)
#fig.savefig('AGE467-555_llsvp_vp00073_p02000.png',dpi=200)
# Add colorbar
cb = fig.colorbar(mgsv, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
cb.set_label('magGSV')   # optional label   
plt.show()
