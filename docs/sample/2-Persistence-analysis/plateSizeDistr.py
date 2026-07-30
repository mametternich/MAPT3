# -*- coding: utf-8 -*-
"""
@summary: calculate iCDF and PDF of plate size distribution for multiple frames and models
          and plot them together with the Bird 2003 reference distribution
@author:  Marla Metternich
"""

# Packages importation
# import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from scipy.io import savemat
import re
import glob

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
# import MAPT3.tessellation
from MAPT3.rigidity import rigid
from MAPT3.project import Project

# Scientific colour map importation
import sys
sys.path.append('/Users/marlametternich/Documents/ScientificColourMaps7/imola')
from imola import imola_map

# Load the project parameters
Project.set('../myparameters.py')


# ===FUNCTIONS ===

def filter_zero_area_plates(pg, model_name, frame):
    """
    Filter out plates with zero surface area from a PlateGather object.
    This is a workaround for micro-plates that have no triangles in the convex hull.
    
    Args:
        pg (PlateGather): Plate data object
        model_name (str): Model name for warning messages
        frame (int): Frame number for warning messages
        
    Returns:
        PlateGather: Modified pg object (filtered in place)
    """
    valid_mask = pg.surfdim > 0
    n_invalid = np.sum(~valid_mask)
    
    if n_invalid > 0:
        print(f'WARNING: Found {n_invalid} plate(s) with zero surface area in {model_name} frame {frame}. Filtering them out.')
        # Create a mapping from old plate IDs to new ones
        valid_plate_ids = np.where(valid_mask)[0]
        plateID_remap = np.full(pg.nop, -1, dtype=np.int32)
        for new_idx, old_idx in enumerate(valid_plate_ids):
            plateID_remap[old_idx] = new_idx
        
        # Filter surface data
        surface_mask = np.isin(pg.plateID, valid_plate_ids)
        pg.x = pg.x[surface_mask]
        pg.y = pg.y[surface_mask]
        pg.z = pg.z[surface_mask]
        pg.vx = pg.vx[surface_mask]
        pg.vy = pg.vy[surface_mask]
        pg.vz = pg.vz[surface_mask]
        pg.vr = pg.vr[surface_mask]
        pg.vtheta = pg.vtheta[surface_mask]
        pg.vphi = pg.vphi[surface_mask]
        pg.pressure = pg.pressure[surface_mask]
        pg.magGSV = pg.magGSV[surface_mask]
        pg.lon = pg.lon[surface_mask]
        pg.lat = pg.lat[surface_mask]
        pg.r = pg.r[surface_mask]
        pg.plateID = plateID_remap[pg.plateID[surface_mask].astype(np.int32)]
        pg.pointID = pg.pointID[surface_mask]
        
        # Filter edge data if present
        if len(pg.xe) > 0:
            edge_mask = np.isin(pg.plateIDe, valid_plate_ids)
            pg.xe = pg.xe[edge_mask]
            pg.ye = pg.ye[edge_mask]
            pg.ze = pg.ze[edge_mask]
            pg.vxe = pg.vxe[edge_mask]
            pg.vye = pg.vye[edge_mask]
            pg.vze = pg.vze[edge_mask]
            pg.vre = pg.vre[edge_mask]
            pg.vthetae = pg.vthetae[edge_mask]
            pg.vphie = pg.vphie[edge_mask]
            pg.pressuree = pg.pressuree[edge_mask]
            pg.magGSVe = pg.magGSVe[edge_mask]
            pg.lone = pg.lone[edge_mask]
            pg.late = pg.late[edge_mask]
            pg.re = pg.re[edge_mask]
            pg.plateIDe = plateID_remap[pg.plateIDe[edge_mask].astype(np.int32)]
            pg.pointIDe = pg.pointIDe[edge_mask]
        
        # Filter and update metadata
        pg.surfdim = pg.surfdim[valid_mask]
        pg.peridim = pg.peridim[valid_mask]
        pg.surf = pg.surf[valid_mask]
        pg.peri = pg.peri[valid_mask]
        pg.nop = len(valid_plate_ids)
    
    return pg

def remove_edge_less_plates(pg):
    """Remove plates that have no edge points (which crash the perimeter routine)."""
    print(f'Before removing edge-less plates: {pg.nop} plates, {len(pg.xe)} edge points')

    # plates that actually have edges
    plates_with_edges = np.unique(pg.plateIDe.astype(int))

    # keep only surface points belonging to those plates
    surface_mask = np.isin(pg.plateID, plates_with_edges)

    pg.x = pg.x[surface_mask]
    pg.y = pg.y[surface_mask]
    pg.z = pg.z[surface_mask]
    pg.vx = pg.vx[surface_mask]
    pg.vy = pg.vy[surface_mask]
    pg.vz = pg.vz[surface_mask]
    pg.lon = pg.lon[surface_mask]
    pg.lat = pg.lat[surface_mask]
    pg.r = pg.r[surface_mask]
    pg.plateID = pg.plateID[surface_mask]

    return pg

def prepare_frame_data(path, model_name, frame):
    """Load a frame and return a PlateGather only if it contains usable plates."""
    file = f'{model_name}_{frame:05d}_optimized.h5'
    pg = PlateGather()
    pg.load_from_h5(path + file)

    # print(f'Model {model_name} frame {frame}: lone {len(pg.lone)} late {len(pg.late)} surf {len(pg.surf)} peri {len(pg.peri)}')

    if pg.nop == 0 or len(pg.surf) == 0 or np.sum(pg.surf) == 0:
        print(f'Skipping {model_name} frame {frame}: no plates found.')
        return None

    if len(pg.lone) == 0 or len(pg.late) == 0 or len(pg.xe) == 0:
        print(f'Skipping {model_name} frame {frame}: no edge points available for perimeter calculation.')
        return None

    try:
        pg.compute_dim_perimeter_area()
    except IndexError as exc:
        print(f'Skipping {model_name} frame {frame}: unable to compute plate dimensions ({exc}).')
        return None

    pg = filter_zero_area_plates(pg, model_name, frame)
    if pg.nop == 0 or len(pg.surfdim) == 0:
        print(f'Skipping {model_name} frame {frame}: no valid plates remain after filtering.')
        return None

    return pg

def has_valid_distribution(distribution_values):
    """Return True when a distribution array contains finite positive values."""
    values = np.asarray(distribution_values)
    if values.size == 0:
        return False
    mask = np.isfinite(values) & (values > 0)
    return np.any(mask)

def prepare_frame_distributions(path, model_name, frame, path_to_sizedistrData=None):
    """Load a frame and precompute plot-ready distributions when available."""
    pg = prepare_frame_data(path, model_name, frame)
    if pg is None:
        return None

    try:
        dist_log = pg.get_distribution(
            earthSizeDistriFile=path_to_sizedistrData, nbins=20, interval='log')
        dist_raw = pg.get_distribution(
            earthSizeDistriFile=path_to_sizedistrData, nbins=20, interval='raw')
    except Exception as exc:
        print(f'Skipping {model_name} frame {frame}: unable to compute distributions ({exc}).')
        return None

    bins_log, cumul_log, pdf_log, bins_Bird, pdfBird, cumul_bins_Bird, cumul_Bird = dist_log
    bins_raw, cumul_raw, pdf_raw, _, _, _, _ = dist_raw

    if not has_valid_distribution(pdf_log):
        print(f'Skipping {model_name} frame {frame}: no valid PDF values available.')
        return None

    if not has_valid_distribution(cumul_raw):
        print(f'Skipping {model_name} frame {frame}: no valid cumulative distribution available.')
        return None

    if len(bins_log) < 2 or len(bins_raw) < 2:
        print(f'Skipping {model_name} frame {frame}: insufficient bins for plotting.')
        return None

    if len(cumul_bins_Bird) == 0 or len(cumul_Bird) == 0 or not np.isfinite(cumul_Bird[0]) or cumul_Bird[0] == 0:
        raise RuntimeError('Bird 2003 reference distribution is invalid.')

    return {
        'pg': pg,
        'log': dist_log,
        'raw': dist_raw,
    }

def format_frame_list(frames_to_format, width=5):
    """Format frame numbers for filenames."""
    return '-'.join(f'{frame:0{width}d}' for frame in frames_to_format)
# ==================================================

path  = './OPTIMIZED/'
# models = ['fDys20-sc','fDys30-sc','fDys50-sc','fDys20_eta20-sc','fDys30_eta20-sc','fDys50_eta20-sc']
models = ['fDys50']

# Set allframes to True to automatically detect all available frames,
# or False to use the manually specified frames list below
allframes = False
plotSpread = False
WSD_to_imposed_models = False
plot_CCDF_PDF_together = False
# frames = [720, 740, 760, 780, 800, 820, 840, 860, 880, 900]
frames = [860]

# Some checks
if not allframes and len(models)<len(frames): allframes = True  # plots multiple time steps for 1 model only
if not allframes: plotSpread = False
if plot_CCDF_PDF_together and (allframes or plotSpread):
    raise ValueError('plot_CCDF_PDF_together can only be used when allframes is False and plotSpread is False.')
if WSD_to_imposed_models and not allframes: print('WARNING: WSD_to_imposed_models is only meaningful when allframes is True. Ignoring it.')
if WSD_to_imposed_models:
    imposed_models = [m for m in models if not m.startswith('f')]
    if len(imposed_models) == 0:
        raise ValueError(
            'WSD_to_imposed_models is True but no imposed models were found. ')
    if len(models) % 2 != 0:
        raise ValueError(
            'WSD_to_imposed_models is True supposedly there are not as many self-consistent as imposed models. ')

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

# BUILD CALLED MODEL-FRAME PAIRS
if allframes:
    called_pairs = [(model_name, frame) for model_name in models for frame in frames_by_model.get(model_name, [])]
else:
    called_pairs = list(zip(models, frames))

path_to_sizedistrData = '/Users/marlametternich/Documents/Code/MAPT3/MAPT3/Bird_2003_Table1_SurfaceSteradian.npy'

n_called = len(called_pairs)
called_models = np.array([m for m, _ in called_pairs], dtype=object)
called_frames = np.array([f for _, f in called_pairs], dtype=np.int32)
valid_mask = np.zeros(n_called, dtype=bool)
frame_data = {}
for idx, (model_name, frame) in enumerate(called_pairs):
    frame_info = prepare_frame_distributions(path, model_name, frame, path_to_sizedistrData)
    if frame_info is not None:
        frame_data[(model_name, frame)] = frame_info
        valid_mask[idx] = True

if not np.any(valid_mask):
    print('WARNING: No valid frames with usable plates were found. Nothing to plot.')

if allframes:
    output_dir = os.path.expanduser('~/Documents/Earth/Figures/PlateSizeDistr/OneModel/')
else:
    output_dir = os.path.expanduser('~/Documents/Earth/Figures/PlateSizeDistr/')

if plot_CCDF_PDF_together:
    combined_output_dir = '/Users/marlametternich/Documents/Earth/Figures/PlateSizeDistr/CCDF+PDF/'
    os.makedirs(combined_output_dir, exist_ok=True)

# Color palette (converted from the provided Matlab-style matrix)
cmap = [
    (0.1216, 0.4667, 0.7059),  # blue 1
    (0.6824, 0.7804, 0.9098),  # light blue 2
    (1.0000, 0.4980, 0.0549),  # orange 3
    (1.0000, 0.7333, 0.4706),  # light orange 4
    (0.1725, 0.6275, 0.1725),  # green 5
    (0.5961, 0.8745, 0.5412),  # light green 6
    # (0.8392, 0.1529, 0.1569),  # red 7
    (1.0000, 0.5961, 0.5882),  # light red 8
    (0.5804, 0.4039, 0.7412),  # purple 9
    (0.7725, 0.6902, 0.8353),  # light purple 10
    (0.5490, 0.3373, 0.2941),  # brown 11
    (0.7686, 0.6118, 0.5804),  # light brown 12
    (0.8902, 0.4667, 0.7608),  # pink 13
    (0.9686, 0.7137, 0.8235),  # light pink 14
    (0.4980, 0.4980, 0.4980),  # gray 15
    (0.7804, 0.7804, 0.7804),  # light gray 16
    (0.7373, 0.7412, 0.1333),  # yellow 17
    (0.8588, 0.8588, 0.5529),  # light yellow 18
    (0.0902, 0.7451, 0.8118),  # cyan 19
    (0.6196, 0.8549, 0.8980),  # light cyan 20
]
imola_cmap = imola_map.reversed()

# Build plotting groups from index masks.
if allframes:
    group_defs = [(model_name, np.where(called_models == model_name)[0]) for model_name in models]
else:
    group_defs = [('combined', np.arange(n_called, dtype=np.int32))]

# Precompute W per called pair, leaving NaN for rejected frames.
W_by_pair = np.full(n_called, np.nan, dtype=float)
valid_indices = np.where(valid_mask)[0]
stored_distribution_by_frame = {}
for idx in valid_indices:
    model_name = called_models[idx]
    frame = int(called_frames[idx])
    # print('Processing WSD for model ', model_name, ' frame ', frame)
    bins, cumul, _, _, _, cumul_bins_Bird, cumul_Bird = frame_data[(model_name, frame)]['raw']

    if WSD_to_imposed_models:
        if model_name.startswith('f'):
            stored_distribution_by_frame[frame] = (bins, cumul, idx)
            print(f'Storing self-consistent distribution for frame {frame} from model {model_name}')
            continue

        if frame not in stored_distribution_by_frame:
            print(f'Skipping WSD for {model_name} frame {frame}: no matching self-consistent distribution was stored.')
            continue

        model_bins, model_cumul, model_idx = stored_distribution_by_frame[frame]
        x_model = np.log10(model_bins)
        x_imposed = np.log10(bins)
        ccdf_model = model_cumul / model_cumul[0]   # normalise so it's dominated by shape, not scale
        ccdf_imposed = cumul / cumul[0]
        imposed_on_model = np.interp(x_model, x_imposed, ccdf_imposed)
        W_value = np.sum(np.abs(imposed_on_model - ccdf_model) * np.diff(x_model, prepend=x_model[0]))
        W_by_pair[model_idx] = W_value
        W_by_pair[idx] = np.nan
        print('Wasserstein distance for model ', called_models[model_idx], ' frame ', frame, ' against imposed reference is ', W_by_pair[model_idx])
    else:
        x_model = np.log10(bins)
        x_Bird = np.log10(cumul_bins_Bird)
        ccdf_model = cumul / cumul[0]   # normalise so it's dominated by shape, not scale (THIS USED TO NOT BE DONE, bug?)
        ccdf_earth = cumul_Bird / cumul_Bird[0]
        F_earth_on_model = np.interp(x_model, x_Bird, ccdf_earth)
        W_by_pair[idx] = np.sum(np.abs(F_earth_on_model - ccdf_model) * np.diff(x_model, prepend=x_model[0]))
        print('Wasserstein distance for model ', model_name, ' frame ', frame, ' is ', W_by_pair[idx])

os.makedirs(output_dir, exist_ok=True)
for group_name, group_indices in group_defs:
    group_indices = np.asarray(group_indices, dtype=np.int32)
    group_valid_indices = group_indices[valid_mask[group_indices]]

    if group_valid_indices.size == 0:
        print(f'Skipping plots for {group_name}: no valid frames in this group.')
        continue

    frames_used = called_frames[group_valid_indices].tolist()
    min_frame = int(np.min(called_frames[group_valid_indices]))
    max_frame = int(np.max(called_frames[group_valid_indices]))
    norm_idx = plt.Normalize(vmin=0, vmax=max(1, len(group_valid_indices) - 1))

    if plot_CCDF_PDF_together:
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        ax1_pdf = ax1.twinx()
        ax1.set_xlabel('Plate area (km'+r'$^2$'+')', fontweight='bold', fontname='Georgia', fontsize=14)
        ax1.set_ylabel('Plate count', fontweight='bold', fontname='Georgia', fontsize=14)
        ax1_pdf.set_ylabel('Probability Density', fontweight='bold', fontname='Georgia', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1_pdf.set_yscale('log')
        ax1_pdf.set_ylim(1e-12, 1e-5)

        bird_plotted_ccdf = False
        bird_plotted_pdf = False
        model_handles = []
        model_labels = []

        for i, idx in enumerate(group_valid_indices):
            model_name = called_models[idx]
            frame = int(called_frames[idx])
            bins_log, cumul_log, pdf_log, bins_Bird, pdfBird, cumul_bins_Bird, cumul_Bird = frame_data[(model_name, frame)]['log']
            bins_raw, cumul_raw, pdf_raw, _, _, _, _ = frame_data[(model_name, frame)]['raw']

            mask_cumul = ~np.isnan(cumul_raw) & (cumul_raw > 0)
            mask_pdf = ~np.isnan(pdf_log) & (pdf_log > 0)
            midpoints = (bins_log[:-1] + bins_log[1:]) / 2
            bins_Bird_mid = (bins_Bird[:-1] + bins_Bird[1:]) / 2.0

            color = cmap[i % len(cmap)]

            if not bird_plotted_ccdf:
                mask_bird_cumul = ~np.isnan(cumul_Bird) & (cumul_Bird > 0)
                if np.any(mask_bird_cumul):
                    line = ax1.plot(cumul_bins_Bird[mask_bird_cumul], cumul_Bird[mask_bird_cumul], linestyle='-', linewidth=2, c='k', label='Bird (2003)')
                    model_handles.extend(line)
                    model_labels.append('Bird (2003)')
                bird_plotted_ccdf = True

            if not bird_plotted_pdf:
                mask_bird_pdf = ~np.isnan(pdfBird) & (pdfBird > 0)
                if np.any(mask_bird_pdf):
                    ax1_pdf.plot(bins_Bird_mid[mask_bird_pdf], pdfBird[mask_bird_pdf], linestyle='--', linewidth=2, c='k')
                bird_plotted_pdf = True

            line_ccdf = ax1.plot(bins_raw[mask_cumul], cumul_raw[mask_cumul], linestyle='-', linewidth=1.5, color=color, label=model_name)
            ax1_pdf.plot(midpoints[mask_pdf], pdf_log[mask_pdf], linestyle='--', linewidth=1.5, color=color)
            model_handles.extend(line_ccdf)
            model_labels.extend([model_name])

        style_handles = [
            Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='CCDF'),
            Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='PDF'),
        ]
        ax1.legend(model_handles + style_handles, model_labels + ['CCDF', 'PDF'], loc='best')
        

        outpath = os.path.join(combined_output_dir, group_name + '_' + '-'.join(models) + '_CCDF+PDF.png')
        fig1.savefig(outpath, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved combined CCDF+PDF figure to {outpath}")
        continue

    # === Probability density plot (PDF) ===
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Plate area (km'+r'$^2$'+')', fontweight='bold', fontname='Georgia', fontsize=14)
    ax1.set_ylabel('Probability Density', fontweight='bold', fontname='Georgia', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    bird_plotted = False
    for i, idx in enumerate(group_valid_indices):
        model_name = called_models[idx]
        frame = int(called_frames[idx])
        bins, _, pdf, bins_Bird, pdfBird, _, _ = frame_data[(model_name, frame)]['log']
        midpoints = (bins[:-1] + bins[1:]) / 2
        bins_Bird_mid = (bins_Bird[:-1] + bins_Bird[1:]) / 2.0

        if not bird_plotted:
            mask_bird = ~np.isnan(pdfBird) & (pdfBird > 0)
            if np.any(mask_bird):
                ax1.plot(bins_Bird_mid[mask_bird], pdfBird[mask_bird], linestyle='-', linewidth=2, c='k', label='Bird (2003)')
            bird_plotted = True

        mask_pdf = ~np.isnan(pdf) & (pdf > 0)
        if allframes:
            color = imola_cmap(norm_idx(i))
            ax1.plot(midpoints[mask_pdf], pdf[mask_pdf], linestyle='-', linewidth=1.5, color=color)
        else:
            ax1.plot(midpoints[mask_pdf], pdf[mask_pdf], linestyle='-', linewidth=1.5,
                     color=cmap[i % len(cmap)], label=model_name)

    ax1.legend(loc='best')
    if allframes:
        sm = plt.cm.ScalarMappable(cmap=imola_cmap, norm=norm_idx)
        sm.set_array([])
        cbar = fig1.colorbar(sm, ax=ax1, pad=0.02, fraction=0.05)
        cbar.set_label('Frame', fontweight='bold', fontname='Georgia', fontsize=12)
        cbar.set_ticks([0, max(1, len(group_valid_indices) - 1)])
        cbar.set_ticklabels([str(min_frame), str(max_frame)])
        fname = group_name + '_' + format_frame_list(frames_used, width=5) + '-pdf.png'
    else:
        fname = '-'.join(models) + '-pdf.png'

    outpath = os.path.join(output_dir, fname)
    fig1.savefig(outpath, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved PDF figure to {outpath}")

    # === Inverse cumulative plot (iCDF) ===
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Plate area (km'+r'$^2$'+')', fontweight='bold', fontname='Georgia', fontsize=14)
    ax2.set_ylabel('Plate count', fontweight='bold', fontname='Georgia', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    x_models_all = []
    for i, idx in enumerate(group_valid_indices):
        model_name = called_models[idx]
        frame = int(called_frames[idx])
        bins, cumul, _, _, _, cumul_bins_Bird, cumul_Bird = frame_data[(model_name, frame)]['raw']

        if allframes and plotSpread:
            x_models_all.append(np.log10(bins))

        mask_cumul = ~np.isnan(cumul) & (cumul > 0)
        if allframes:
            color = imola_cmap(norm_idx(i))
            ax2.plot(bins[mask_cumul], cumul[mask_cumul], linestyle='-', linewidth=1.5, color=color)
        else:
            ax2.plot(bins[mask_cumul], cumul[mask_cumul], linestyle='-', linewidth=1.5,
                     color=cmap[i % len(cmap)], label=model_name)

    # Bird curve (same for each frame, so reusing last loaded values is fine)
    ax2.plot(cumul_bins_Bird, cumul_Bird, linestyle='-', linewidth=2, c='k', label='Bird (2003)')

    if allframes and plotSpread and len(x_models_all) > 0:
        xmin = min(x.min() for x in x_models_all)
        xmax = max(x.max() for x in x_models_all)
        nref = max(len(x) for x in x_models_all)
        spread_x_ref = np.linspace(xmin, xmax, nref)
        spread_c_models = []

        for idx in group_valid_indices:
            model_name = called_models[idx]
            frame = int(called_frames[idx])
            bins, cumul, _, _, _, _, _ = frame_data[(model_name, frame)]['raw']
            x_model = np.log10(bins)
            C_interp = np.interp(spread_x_ref, x_model, cumul, left=np.nan, right=np.nan)
            spread_c_models.append(C_interp)

        spread_c_models = np.vstack(spread_c_models)
        spread_counts = np.nanmax(spread_c_models, axis=0) - np.nanmin(spread_c_models, axis=0)
        Cframes_mean = np.nanmean(spread_c_models, axis=0)
        ax2.errorbar(10**spread_x_ref, Cframes_mean, yerr=spread_counts / 2, color='red', fmt='o',
                     markersize=3, elinewidth=0.7, label='Temporal variability')

        max_spread_id = int(np.argmax(spread_counts))
        ax2.errorbar(10**spread_x_ref[max_spread_id], Cframes_mean[max_spread_id],
                     yerr=spread_counts[max_spread_id] / 2, color='magenta', fmt='o',
                     markersize=3, elinewidth=0.7,
                     label=f'Max spread: {spread_counts[max_spread_id]:.2f} at {10**spread_x_ref[max_spread_id]:.2e} $\\mathrm{{km^2}}$')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(0.5, 2*10**2)

    ax2.legend(loc='best')
    if allframes:
        sm = plt.cm.ScalarMappable(cmap=imola_cmap, norm=norm_idx)
        sm.set_array([])
        cbar = fig2.colorbar(sm, ax=ax2, pad=0.02, fraction=0.05)
        cbar.set_label('Frame', fontweight='bold', fontname='Georgia', fontsize=12)
        cbar.set_ticks([0, max(1, len(group_valid_indices) - 1)])
        cbar.set_ticklabels([str(min_frame), str(max_frame)])
        if plotSpread:
            fname2 = group_name + '_' + format_frame_list(frames_used, width=4) + '_cumulative+spread.png'
        else:
            fname2 = group_name + '_' + format_frame_list(frames_used, width=4) + '_cumulative.png'
    else:
        fname2 = '-'.join(models) + '_cumulative.png'

    outpath2 = os.path.join(output_dir, fname2)
    fig2.savefig(outpath2, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved iCDF figure to {outpath2}")

# Plot Wasserstein Distance (WSD) for models over time and average
if allframes:
    if WSD_to_imposed_models:
        output_matlab_dir = '/Users/marlametternich/Documents/MATLAB/data/WSD_to_imposed_models'
    else:
        output_matlab_dir = '/Users/marlametternich/Documents/MATLAB/data/WSD_iCDF'
    os.makedirs(output_matlab_dir, exist_ok=True)
    
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel('Frame', fontweight='bold', fontname='Georgia', fontsize=14)
    ax3.set_ylabel('Total plate count', fontweight='bold', fontname='Georgia', fontsize=14)

    for model_name in models:   # STILL TO DO: make sure correct value for correct model is saved
        model_mask = called_models == model_name
        model_valid_mask = model_mask & np.isfinite(W_by_pair)
        model_valid_frames = np.sort(called_frames[model_valid_mask]) if np.any(model_valid_mask) else np.array([], dtype=np.int32)

        if np.any(model_valid_mask):
            W_ave = float(np.nanmean(W_by_pair[model_valid_mask]))
            print(f'Average Wasserstein distance for {model_name} W_ave = {W_ave:4.2f}')
        else:
            print(f'No valid frames for {model_name}, skipping Wasserstein export and plot entry.')
            continue

        if model_valid_frames.size > 0:
            model_min_frame = int(np.min(model_valid_frames))
            model_max_frame = int(np.max(model_valid_frames))
        else:
            model_min_frame = 0
            model_max_frame = 0

        matlab_model_name = model_name
        if not re.search(r'_eta\d{2}', matlab_model_name):
            print('Adding eta21 to model name for MATLAB output filename')
            matlab_model_name = re.sub(r'(ys\d{2,})', r'\1_eta21', matlab_model_name, count=1)

        matlab_filename = f'{matlab_model_name}_{model_min_frame:04d}-{model_max_frame:04d}.mat'
        matlab_filepath = os.path.join(output_matlab_dir, matlab_filename)
        savemat(matlab_filepath, {
            'W_ave': W_ave,
            'model_name': matlab_model_name,
            'frame': model_valid_frames.astype(np.int32),
        })
        print(f"Saved average WSD distance to {matlab_filepath}")

        # Also plot total number of plates vs time for all models
        model_valid_indices = np.where(model_valid_mask)[0]
        model_total_plate_count = []
        model_frames_for_plot = []

        if model_valid_indices.size > 0:
            # Keep frame/count pairs ordered in time for plotting.
            order = np.argsort(called_frames[model_valid_indices])
            for idx in model_valid_indices[order]:
                frame = int(called_frames[idx])
                _, cumul, _, _, _, _, _ = frame_data[(model_name, frame)]['raw']
                if len(cumul) == 0 or not np.isfinite(cumul[0]) or cumul[0] <= 0:
                    continue
                model_frames_for_plot.append(frame)
                model_total_plate_count.append(float(cumul[0]))

        if len(model_frames_for_plot) > 0:
            model_color = cmap[models.index(model_name) % len(cmap)]
            ax3.plot(model_frames_for_plot, model_total_plate_count,
                     linestyle='-', marker='o', linewidth=1.5, markersize=4,
                     color=model_color, label=model_name)
        else:
            print(f'No valid cumulative[0] values to plot for {model_name}.')

    if len(ax3.lines) > 0:
        ax3.legend(loc='best')
        # ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        fig3.tight_layout()
        outpath3 = os.path.join('/Users/marlametternich/Documents/Earth/Figures/PlateSizeDistr/', 'allmodels_total_plates_vs_frame.png')
        fig3.savefig(outpath3, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved total-plates-vs-frame figure to {outpath3}")
    else:
        print('No valid total-plate-count data available. Fig3 was not saved.')
        