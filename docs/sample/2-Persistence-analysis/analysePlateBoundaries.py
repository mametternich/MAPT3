# -*- coding: utf-8 -*-
"""
@summary: calculate iCDF and PDF of plate size distribution for multiple frames and models
          and plot them together with the Bird 2003 reference distribution
@author:  Marla Metternich
"""

# Packages importation
# import sys
import glob
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
from paraview.simple import *
from vtkmodules.util.numpy_support import vtk_to_numpy
from scipy.io import savemat
from scipy.interpolate import griddata
from matplotlib.colors import SymLogNorm
# from plateSizeDistr import prepare_frame_distributions

# MAPT3 importation
from MAPT3.tessellation import PlateGather

# ==================================================

path  = './OPTIMIZED/'
drive = '/Volumes/MarsBook'
models = ['fDys50-sc']
# models = ['fDys20-sc', 'Dys20', 'fDys30-sc','Dys30', 'fDys50-sc','Dys50']

# Set allframes to True to automatically detect all available frames,
# or False to use the manually specified frames list below
allframes = False
# frames = [720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
frames = [1020]

# AUTO-DETECT FRAMES IF REQUESTED
frames_by_model = {}
if allframes:
    for model_name in models:
        h5_pattern = os.path.join(path, f'{model_name}_*_optimized.h5')
        h5_files = glob.glob(h5_pattern)
        model_frames = []
        for h5_file_path in h5_files:
            match = re.search(rf'{model_name}_(\d+)_optimized\.h5', h5_file_path)
            if match:
                model_frames.append(int(match.group(1)))
        model_frames = sorted(list(set(model_frames)))
        frames_by_model[model_name] = model_frames
        print(f'Auto-detected {len(model_frames)} frames for {model_name}: {model_frames}')

# BUILD MODEL-FRAME PAIRS
if allframes:
    called_pairs = [(model_name, f) for model_name in models for f in frames_by_model.get(model_name, [])]
else:
    called_pairs = list(zip(models, frames))

# n_called = len(called_pairs)
# called_models = np.array([m for m, _ in called_pairs], dtype=object)
# called_frames = np.array([f for _, f in called_pairs], dtype=np.int32)
# valid_mask = np.zeros(n_called, dtype=bool)
# frame_data = {}
# for idx, (model_name, frame) in enumerate(called_pairs):
#     frame_info = prepare_frame_distributions(path, model_name, frame)
#     if frame_info is not None:
#         frame_data[(model_name, frame)] = frame_info
#         valid_mask[idx] = True

# if not np.any(valid_mask):
#     print('WARNING: No valid frames with usable plates were found. Nothing to plot.')

output_dir = os.path.expanduser('~/Documents/Earth/Figures/PlateBoundaries/')

print('MARLA iteration = ', called_pairs)
# ---- Calculate types of boundaries for each frame and model
model_to_idx = {m: j for j, m in enumerate(models)}
frac_sum_by_model = np.zeros((4, len(models)))
frac_valid_count_by_model = np.zeros(len(models), dtype=int)
for i, (model_name, frame) in enumerate(called_pairs):
    file = f'{model_name}_{frame:05d}_optimized.h5'
    pg = PlateGather()
    pg.load_from_h5(path+file)

    # Fetch the VTK dataset from the stagpyviz pipeline
    divvor_file = f'{drive}/Earth/Models/{model_name}/+op/earthDmods_div_vor_{frame:05d}.vtu'

    input_data = XMLUnstructuredGridReader(FileName=[divvor_file])
    pointArray = ['div_v','vor_v']
    input_data.PointArrayStatus = pointArray
    input_data.UpdatePipeline()
    vtk_data = servermanager.Fetch(input_data)
    coords = vtk_to_numpy(vtk_data.GetPoints().GetData())
    div_v = vtk_to_numpy(vtk_data.GetPointData().GetArray('div_v'))
    vor_v = vtk_to_numpy(vtk_data.GetPointData().GetArray('vor_v'))
    # pg.T = vtk_to_numpy(vtk_data.GetPointData().GetArray('temperature'))

    # Calculate divergence & vorticity on the plate boundary points (interpolate in Cartesian coordinates)
    pg.hdivb = griddata(coords, div_v, (pg.xb,pg.yb,pg.zb), method='nearest')
    pg.hvorb  = griddata(coords, vor_v, (pg.xb,pg.yb,pg.zb), method='nearest')
    # pg.Tb  = griddata(coords, pg.T, (pg.xb,pg.yb,pg.zb), method='nearest')

    # Load radial velocities at the boundaries
    if len(pg.vrb) == len(pg.lonb) and np.count_nonzero(pg.vrb) == 0:
        points = np.column_stack((pg.lon, pg.lat))
        pg.vrb = griddata(points, pg.vr, (pg.lonb, pg.latb), method='nearest')

    # Testing only
    # print('   min, max divergence: ', np.min(pg.hdivb), np.max(pg.hdivb))
    # print('   min, max vorticity: ', np.min(pg.hvorb), np.max(pg.hvorb))
    # print('   min, max radial velocity: ', np.min(pg.vr), np.max(pg.vr))
    # print('   min, max radial velocity at boundaries: ', np.min(pg.vrb), np.max(pg.vrb))

    if len(pg.vrb) > 0:
        # Plotting the divergence and vorticity on the plate boundaries
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_subplot(1,3,1, projection=ccrs.Robinson())
        ax.set_global()
        norm=SymLogNorm(linthresh=10, vmin=np.min(-1e6), vmax=np.max(1e6), base=10)
        cmap = ax.scatter(pg.lonb,pg.latb,c=pg.hdivb,s=1,cmap=plt.cm.PiYG_r,transform=ccrs.PlateCarree(),norm=norm)
        cbar = fig.colorbar(cmap,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Divergence', fontweight='bold', fontname='Georgia', fontsize=14)
        ax = fig.add_subplot(1,3,2, projection=ccrs.Robinson())
        ax.set_global()
        norm=SymLogNorm(linthresh=10, vmin=np.min(-1e6), vmax=np.max(1e6), base=10)
        cmap = ax.scatter(pg.lonb,pg.latb,c=pg.hvorb,s=1,cmap=plt.cm.PiYG_r,transform=ccrs.PlateCarree(),norm=norm)
        cbar = fig.colorbar(cmap,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)   
        cbar.set_label('Vorticity', fontweight='bold', fontname='Georgia', fontsize=14)
        ax = fig.add_subplot(1,3,3, projection=ccrs.Robinson())
        ax.set_global()
        norm=SymLogNorm(linthresh=10, vmin=np.min(-1e5), vmax=np.max(1e5), base=10)
        cmap = ax.scatter(pg.lonb,pg.latb,c=pg.vrb,s=1,cmap=plt.cm.BrBG,transform=ccrs.PlateCarree(),norm=norm)
        cbar = fig.colorbar(cmap,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Radial velocity', fontweight='bold', fontname='Georgia', fontsize=14)
        # ax = fig.add_subplot(1,3,4, projection=ccrs.Robinson())
        # ax.set_global()
        # norm=SymLogNorm(linthresh=10, vmin=np.min(-1e5), vmax=np.max(1e5), base=10)
        # cmap = ax.scatter(pg.lon,pg.lat,c=pg.vr,s=1,cmap=plt.cm.BrBG,transform=ccrs.PlateCarree(),norm=norm)
        # cbar = fig.colorbar(cmap,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8)

        outpath = os.path.join(output_dir,model_name)
        os.makedirs(outpath, exist_ok=True)
        figname = f'div-vor-vr_{frame:05d}.png'
        plt.savefig(os.path.join(outpath, figname), dpi=300, transparent=True, bbox_inches='tight')

        # Compute the type of boundary based on the sign of the divergence
        # pg = pg.boundariesDiagnotic(plot=True)
        [convf,divf,traf,othf] = pg.boundariesDiagnostic_MM(plateID=None,plot=True)

        # Fraction of each plate time (global)
        # number of points detected as X / total number of points (for all boundaries) 
        # NOT corrected for internal plate boundaries like within function itself
        # convf = np.count_nonzero(pg.pbtype == 1)/len(pg.pbtype) 
        # divf  = np.count_nonzero(pg.pbtype == 2)/len(pg.pbtype)
        # traf  = np.count_nonzero(pg.pbtype == 3)/len(pg.pbtype)
        # othf  = np.count_nonzero(pg.pbtype == 4)/len(pg.pbtype)

        # print(f"Fraction of convergent boundaries: {convf:.3f}")
        # print(f"Fraction of divergent boundaries: {divf:.3f}")
        # print(f"Fraction of transform boundaries: {traf:.3f}")
        # print(f"Fraction of other boundaries: {othf:.3f}")

        # Compute the perimeter for all plates
        pg.compute_dim_perimeter_area()
        totperi = np.sum(pg.peridim) / 2  # as each plate boundary is counted twice
        print(f"Total perimeter of plate boundaries: {totperi:.0f} km")
        print('-'*30)

        # Length of each plate boundary type 
        convl = convf * totperi
        divl  = divf  * totperi
        trafl = traf  * totperi
        othl  = othf  * totperi
        print(f"Length of convergent boundaries: {convl:.0f} km")
        print(f"Length of divergent boundaries: {divl:.0f} km")
        print(f"Length of transform boundaries: {trafl:.0f} km")
        print(f"Length of other boundaries: {othl:.0f} km")
    else:
        print("No plate boundaries detected, skipping perimeter and length calculations. Entering NaNs.")
        convf = np.nan
        divf  = np.nan
        traf  = np.nan
        othf  = np.nan
    
    # Accumulate [convf, divf, traf, othf] per model across frames
    j = model_to_idx[model_name]
    if np.isfinite(convf):
        frac_sum_by_model[:, j] += np.array([convf, divf, traf, othf], dtype=float)
        frac_valid_count_by_model[j] += 1
    print('-'*30)
    print('-'*30)

# Build 4 x number_of_models average array
# row 1: convf, row 2: divf, row 3: traf, row 4: othf
frac_ave_by_model = np.full((4, len(models)), np.nan)
np.divide(
    frac_sum_by_model,
    frac_valid_count_by_model,
    out=frac_ave_by_model,
    where=frac_valid_count_by_model[None, :] > 0
)

# Save average boundary fractions per model to MATLAB files
output_matlab_dir = '/Users/marlametternich/Documents/MATLAB/data/PBratios'
os.makedirs(output_matlab_dir, exist_ok=True)
for j, m in enumerate(models):
    print(f"Average fractions for {m}: convf={frac_ave_by_model[0, j]:.4f}, divf={frac_ave_by_model[1, j]:.4f}, traf={frac_ave_by_model[2, j]:.4f}, othf={frac_ave_by_model[3, j]:.4f} (valid frames={frac_valid_count_by_model[j]})")

    if allframes:
        # if model_name doesn't contain eta**, add it for clarity
        if not re.search(r'_eta\d{2}', m):
            print('Adding eta21 to model name for MATLAB output filename')
            m = re.sub(r'(ys\d{2,})', r'\1_eta21', m, count=1)
        matlab_filename = f'{m}.mat'
        matlab_filepath = os.path.join(output_matlab_dir, matlab_filename)
        savemat(matlab_filepath, {
            'ave_convf': float(frac_ave_by_model[0, j]),
            'ave_divf': float(frac_ave_by_model[1, j]),
            'ave_traf': float(frac_ave_by_model[2, j]),
            'ave_othf': float(frac_ave_by_model[3, j])
        })