import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project
from MAPT3.optimize import optimize, resampling_param
from MAPT3.rigidity import rigid as rigid_rigidity  # Use alias to avoid conflict

# Scientific colour map importation
import sys
sys.path.append('/Users/marlametternich/Documents/ScientificColourMaps7/lajolla')
from lajolla import lajolla_map

# Load the project parameters
Project.set('../myparameters.py')

# Define the list of pmin values that will be consider  during the optimization
pthreshold = Project.pmin # here, all the values used for the tessellation

# Models to plot
models = ['fDys30-sc']
frames = [1045]

for model,frame in zip(models,frames):
    print(f'Optimizing tessellation for model {model} at frame {frame}')

    # Tessellation that will be optimized
    myfile = f'{model}_{frame:05d}'

    # Here, we can define a function to help locate the tessellation file
    def path2h5file(pmin,file=myfile):
        path  = os.path.abspath('../1-Tessellation/TTK_outputs/'+'p'+intstringer(round(pmin),5))
        file  = myfile+'.h5'
        return path+'/'+file

    # Output file name
    myfile  = myfile
    savefigpath= '../../figs/'+model+'/'
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)
    
    # Test rigidity for the optimized tessellation (optional, added by Marla)
    # Load the optimized tessellation
    pg_optimized = PlateGather()
    pg_optimized.load_from_h5(f'./OPTIMIZED/{myfile}_optimized.h5')
    
    # Initialize rigidity array for all points
    real_rigidity = np.zeros(len(pg_optimized.x), dtype=int)  # 0 = non-rigid, 1 = rigid (or make it bool?)
    
    # Unique plateIDs from optimized.plateID
    unique_plateIDs = np.unique(pg_optimized.plateID)

    # Loop through each plate and test rigidity
    for pID in unique_plateIDs:
        # Compute rotation for this plate
        # automatic calculation of r based on nr points in the plate
        surf = np.count_nonzero(pg_optimized.plateID == pID)
        r = resampling_param(surf)
        wx, wy, wz = pg_optimized.get_rotation(pID, r=r, plot=False)
        P1 = pg_optimized.P11
        P2 = pg_optimized.P12
        
        # Test rigidity
        is_rigid = rigid_rigidity(pg_optimized, pID, wx, wy, wz, P1, P2, plot=False)
        print(f'  Plate {pID}: {"Rigid" if is_rigid else "Non-rigid"}')
        
        # Fill the rigidity array for all points belonging to this plate
        plate_mask = pg_optimized.plateID == pID
        real_rigidity[plate_mask] = 1 if is_rigid else 0

    # Prepare data for plotting
    lon = pg_optimized.lon
    lat = pg_optimized.lat
    persistence = pg_optimized.pmin

    plates = pg_optimized.plateID
    unique_ids = np.unique(plates)
    shuffled_ids = unique_ids.copy()
    np.random.shuffle(shuffled_ids)
    id_map = dict(zip(unique_ids, shuffled_ids))
    plates_randomized = np.array([id_map[i] for i in plates])

    # Plot the overview map with rigidity based on rigid.rigidity()
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(1,3,2, projection=ccrs.Robinson())
    ax3 = fig.add_subplot(1,3,3, projection=ccrs.Robinson())
    
    ax1.set_title('Tessellation',fontweight='bold',fontname='Georgia',fontsize=14)
    ax1.set_global()
    cmap1  = ax1.scatter(lon,lat,c=plates_randomized,s=1,cmap=plt.cm.magma,transform=ccrs.PlateCarree())
    cbar1 = fig.colorbar(cmap1,ax=ax1,orientation='horizontal', pad=0.05, shrink=0.8)

    # ---
    ax2.set_title('Persistence',fontweight='bold',fontname='Georgia',fontsize=14)
    ax2.set_global()
    cmap2 = ax2.scatter(lon,lat,c=persistence,s=1,vmin=pthreshold[0],vmax=5000,cmap=plt.cm.magma,transform=ccrs.PlateCarree())
    cbar2 = fig.colorbar(cmap2,ax=ax2,orientation='horizontal', pad=0.05, shrink=0.8)

    # ---
    ax3.set_title('Rigidity',fontweight='bold',fontname='Georgia',fontsize=14)
    ax3.set_global()
    cmap3 = ax3.scatter(lon,lat,c=real_rigidity,s=1,cmap=plt.cm.magma,vmin=0,vmax=1,transform=ccrs.PlateCarree())
    
    cbar3 = plt.colorbar(cmap3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
    # cbar3 = fig.colorbar(cmap3, ax=ax3, orientation='horizontal')
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(['Non-rigid', 'Rigid'])
    
    plt.tight_layout()
    # Save the figure
    fig_filename = f'{savefigpath}{myfile}_rigidity_map.png'
    fig.savefig(fig_filename, dpi=200, bbox_inches='tight')
    print(f'  Rigidity map saved to: {fig_filename}')
    plt.show()
    plt.close()