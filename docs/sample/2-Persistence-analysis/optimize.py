# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
import re

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project
from MAPT3.optimize import optimize, resampling_param
from MAPT3.rigidity import rigid as rigid_rigidity  # Use alias to avoid conflict

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================


# Define the list of pmin values that will be consider  during the optimization
pthreshold = Project.pmin # here, all the values used for the tessellation

# Tessellation that will be optimized   
# models = ['fDys30-ysg-sc', 'fDys50-ysg-sc']
# models = ['fDys20-ysg-sc', 'fDys20_eta20-ysg-sc']
# models = ['Dys30-sc','Dys30_eta20-sc','Dys40_eta20-sc','Dys50-sc','Dys50_eta20-sc']
models = ['fDys20','fDys50']

# Set allframes to True to automatically detect all available frames,
# or False to use the manually specified frames list below
allframes = True

frames = [920, 940, 960, 980, 1000, 1020, 1040, 1053]
# frames = [1032,1066]

plotOneModel = True
if not allframes and len(models)<len(frames): plotOneModel = True  # plots multiple time steps for 1 model only

# AUTO-DETECT FRAMES IF REQUESTED
frames_by_model = {}
if allframes:
    xdmf_folder = os.path.join(os.path.dirname(__file__), '../1-Tessellation/TTK_outputs')
    for model_name in models:
        model_frames = []
        # Search in all pmin subdirectories for this model's frames
        for pmin_dir in os.listdir(xdmf_folder):
            if pmin_dir.startswith('p'):
                h5_file = os.path.join(xdmf_folder, pmin_dir, f'{model_name}_*.h5')
                h5_files = glob.glob(h5_file)
                for h5_file_path in h5_files:
                    match = re.search(rf'{model_name}_(\d+)\.h5', h5_file_path)
                    if match:
                        model_frames.append(int(match.group(1)))
        model_frames = sorted(list(set(model_frames)))
        frames_by_model[model_name] = model_frames
        print(f'Auto-detected {len(model_frames)} frames for {model_name}: {model_frames}')

# BUILD MODEL-FRAME PAIRS
if allframes:
    model_frame_pairs = [(model_name, frame) for model_name in models for frame in frames_by_model.get(model_name, [])]
else:
    model_frame_pairs = ([(models[0], f) for f in frames] if plotOneModel else list(zip(models, frames)))

for model,frame in model_frame_pairs:
    print(f'Optimizing tessellation for model {model} at frame {frame}')

    # Tessellation that will be optimized
    myfile = f'{model}_{frame:05d}'

    # Here, we can define a function to help locate the tessellation file
    def path2h5file(pmin,file=myfile):
        path  = os.path.abspath('../1-Tessellation/TTK_outputs/'+'p'+intstringer(round(pmin),5))
        file  = myfile+'.h5'
        return path+'/'+file

    # Output file name
    ofilename  = myfile
    savefigpath= '../../figs/'+model+'/'
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)

    # Optimization function
    optimize(path2h5file, pthreshold, ofilename, output_path='./OPTIMIZED',\
            plot_missed = False, add_missedPlates = True, overlap_order = 0, geographic_search = 1, \
            edges_detection_method='paraview',verbose_edgesExtraction=False,\
            simplex_threshold = 0.01,hidden=True,fig_path=savefigpath)


