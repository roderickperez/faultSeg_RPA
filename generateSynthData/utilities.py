from networkx import nodes
import numpy as np
import os
import matplotlib.pyplot as plt
import math # For radians in rose plot
from matplotlib.colors import ListedColormap
from collections import Counter
import json
import pandas as pd
import random
import plotly.graph_objects as go
import cigvis.plotlyplot as cgp
from cigvis import colormap, config
from ipywidgets import IntSlider, Play, Button, HBox, VBox, interactive_output, jslink, Dropdown, interact
from IPython.display import display
from scipy.ndimage import map_coordinates, binary_dilation, generate_binary_structure
from IPython.display import display, clear_output  # add this import

from ipywidgets import IntSlider, Dropdown, VBox, HBox, interactive_output, Output
from IPython.display import display, clear_output
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"   # or "notebook_connected" — pick ONE

# One persistent anchor so we only ever show ONE viewer
_VIEW_ANCHOR = None

def parse_windows(s):
    out = []
    for rng in s.split(','):
        lo, hi = map(lambda x: float(x.strip()), rng.split('-'))
        out.append((lo, hi))
    return out

def in_windows(a, windows):
    a = a % 360.0
    for lo, hi in windows:
        lo, hi = lo % 360.0, hi % 360.0
        span = (hi - lo) % 360.0 or 360.0
        if (a - lo) % 360.0 <= span:
            return True
    return False

def which_window(a, wins):
    for i, w in enumerate(wins):
        if in_windows(a, [w]):
            return i
    return None

def append_param_to_cmd(cmd_list, arg_name, value):
    if value is not None:
        if isinstance(value, tuple):
            cmd_list += [f"--{arg_name}", f"{value[0]},{value[1]}"]
        else:
            cmd_list += [f"--{arg_name}", str(value)]

def choose_idx_with_faults(split, STATS_DIR, BASE_OUT, FMT):
    stats_f = os.path.join(STATS_DIR, f"statistics_{split}.json")
    with open(stats_f) as f: s = json.load(f)

    by_cube = {}
    for fp in s["all_fault_params"]:
        by_cube.setdefault(fp["cube_id"], []).append(fp)

    ext = ".npy" if FMT == "npy" else ".dat"
    mdir = os.path.join(BASE_OUT, split, "fault")
    existing = {int(os.path.splitext(fn)[0]) for fn in os.listdir(mdir) if fn.endswith(ext)}

    cands = [cid for cid in by_cube if by_cube[cid] and cid in existing]
    if not cands:
        raise RuntimeError("No matching cube ids between stats and files.")
    return random.choice(cands)

def load_volumes(split, idx, BASE_OUT, CUBE_SHAPE, FMT):
    fn = f"{idx}.{FMT}"
    seis_p = os.path.join(BASE_OUT, split, "seis", fn)
    mask_p = os.path.join(BASE_OUT, split, "fault", fn)
    if FMT == "npy":
        seismic = np.load(seis_p).astype(np.float32)
        mask    = np.load(mask_p).astype(np.int32)
    else:
        seismic = np.fromfile(seis_p, np.float32).reshape(CUBE_SHAPE)
        mask    = np.fromfile(mask_p,  np.uint8).reshape(CUBE_SHAPE).astype(np.int32)
    return seismic, mask

# --- One-shot 3D viewer: grayscale seismic + mask overlay (CIGVis style) ---

def view_cube_overlay(
    split,
    idx,
    STATS_DIR,
    BASE_OUT,
    FMT="npy",
    CUBE_SHAPE=None,
    mask_mode=0,
    bg_pct=(1, 99),
    pos=None,
    show_cbar=False,
):
    """Show ONE 3D CIGVis figure for a specific cube id."""
    seismic, mask = load_volumes(split, idx, BASE_OUT, CUBE_SHAPE, FMT)

    vmin, vmax = np.percentile(seismic, bg_pct)
    mask_vis   = mask.astype(np.int32)
    max_label  = int(mask_vis.max())

    # transparent background, colored faults
    if mask_mode == 0:
        values = [0, 1]
        cols_overlay = [
            (1.0, 1.0, 1.0, 0.0),  # bg transparent
            (1.0, 0.0, 0.0, 0.95), # red
        ]
    else:
        values = [0, 1, 2] + ([3] if max_label >= 3 else [])
        cols_overlay = [
            (1.0, 1.0, 1.0, 0.0),   # bg transparent
            (0.00, 0.80, 0.00, 0.95), # class 1 → green
            (0.50, 0.00, 0.50, 0.95), # class 2 → purple
        ] + ([(1.00, 0.55, 0.00, 0.95)] if max_label >= 3 else [])  # class 3 → orange

    fg_cmap = colormap.custom_disc_cmap(values, cols_overlay)
    fg_cmap = colormap.set_alpha_except_min(fg_cmap, 0.95)

    ni, nx, nt = seismic.shape
    if pos is None:
        pos = [ni // 2, nx // 2, nt // 2]

    nodes = []
    nodes += cgp.create_overlay(
        bg_volume=seismic,
        fg_volume=mask_vis,
        pos=pos,
        bg_cmap="gray",
        bg_clim=[float(vmin), float(vmax)],
        fg_cmap=fg_cmap,
        fg_clim=[0.0, float(max_label)],
        interpolation="nearest",
        show_cbar=show_cbar,
    )

    cgp.plot3D(
        nodes,
        aspectratio=dict(x=ni / nx, y=1.0, z=nt / nx),
        aspectmode="manual",
    )
    print(f"Viewing split='{split}', cube id={idx}, pos={pos}")


def view_one_random_cube_overlay(
    split,
    STATS_DIR,
    BASE_OUT,
    FMT="npy",
    CUBE_SHAPE=None,
    mask_mode=0,
    bg_pct=(1, 99),
    pos=None,
    show_cbar=False,
):
    """Pick a random cube with faults and show ONE 3D CIGVis figure."""
    idx = choose_idx_with_faults(split, STATS_DIR, BASE_OUT, FMT)
    view_cube_overlay(
        split=split, idx=idx,
        STATS_DIR=STATS_DIR, BASE_OUT=BASE_OUT,
        FMT=FMT, CUBE_SHAPE=CUBE_SHAPE,
        mask_mode=mask_mode, bg_pct=bg_pct, pos=pos, show_cbar=show_cbar,
    )
    return idx

def plot_histogram(data, ax, title, xlabel, bins='auto', xlim=None, hist_range=None, use_percentile=True):
    if not data:
        return
    
    if use_percentile:
        low, high = np.percentile(data, [2, 98])
    else:
        low, high = xlim if xlim else (min(data), max(data))

    ax.hist(data, bins=bins, range=hist_range if hist_range else (low, high), edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    if xlim:
        ax.set_xlim(xlim)

def plot_rose_diagram(ax, strikes, mask_mode, all_fault_params, normal_colour, inverse_colour):
    if not strikes:
        return

    if mask_mode == 1:
        strikes_normal = np.radians([f['strike'] for f in all_fault_params if f['fault_type'] == 'Normal'])
        strikes_inverse = np.radians([f['strike'] for f in all_fault_params if f['fault_type'] == 'Inverse'])

        num_bins = 18
        bins_rad = np.linspace(0, 2 * np.pi, num_bins + 1)
        cnt_norm, _ = np.histogram(strikes_normal, bins=bins_rad)
        cnt_inv, _ = np.histogram(strikes_inverse, bins=bins_rad)

        centres = bins_rad[:-1] + np.diff(bins_rad) / 2
        width = np.diff(bins_rad)[0]

        ax.bar(centres, cnt_norm, width=width, bottom=0.0, color=normal_colour, edgecolor='black', alpha=0.8, label='Normal')
        ax.bar(centres, cnt_inv, width=width, bottom=cnt_norm, color=inverse_colour, edgecolor='black', alpha=0.8, label='Inverse')

        ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0))
        ax.set_title('Fault Strike Angles', va='bottom', y=1.1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    else:
        angles = np.deg2rad(np.mod(strikes, 360.0))
        num_bins = 18
        bins_rad = np.linspace(0, 2 * np.pi, num_bins + 1)
        cnt, _ = np.histogram(angles, bins=bins_rad)

        centres = bins_rad[:-1] + np.diff(bins_rad) / 2
        width = np.diff(bins_rad)[0]

        ax.bar(centres, cnt, width=width, bottom=0.0, edgecolor='black', alpha=0.9)
        ax.set_title('Fault Strike Angles', va='bottom', y=1.1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

def plot_fault_counts(ax, all_fault_params, mask_mode, normal_colour, inverse_colour):
    if mask_mode == 0:
        ax.bar(['Total'], [len(all_fault_params)], color=['skyblue'])
    else:
        normal_count = sum(1 for f in all_fault_params if f['fault_type'] == 'Normal')
        inverse_count = sum(1 for f in all_fault_params if f['fault_type'] == 'Inverse')
        ax.bar(['Normal', 'Inverse'], [normal_count, inverse_count], color=[normal_colour, inverse_colour])
    ax.set_title('Fault Type Distribution')
    ax.set_ylabel('Number of Faults')

def count_pixels(mask_cubes, mask_mode):
    total_voxels = 0
    fault_voxels = 0
    normal_voxels = 0
    inverse_voxels = 0
    overlap_voxels = 0

    per_cube_fault_pct = []
    per_cube_normal_pct = []
    per_cube_inverse_pct = []
    per_cube_overlap_pct = []

    for mask_cube in mask_cubes:
        cube_total = mask_cube.size
        total_voxels += cube_total

        if mask_mode == 0:
            cube_fault = np.sum(mask_cube == 1)
            fault_voxels += cube_fault
            per_cube_fault_pct.append((cube_fault / cube_total) * 100)
        else:
            cube_normal = np.sum(mask_cube == 1)
            cube_inverse = np.sum(mask_cube == 2)
            cube_overlap = np.sum(mask_cube == 3)
            
            normal_voxels += cube_normal
            inverse_voxels += cube_inverse
            overlap_voxels += cube_overlap

            per_cube_normal_pct.append((cube_normal / cube_total) * 100)
            per_cube_inverse_pct.append((cube_inverse / cube_total) * 100)
            per_cube_overlap_pct.append((cube_overlap / cube_total) * 100)

    overall_pct = {}
    mean_pct = {}

    if mask_mode == 0:
        overall_pct['fault'] = (fault_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        overall_pct['no_fault'] = 100 - overall_pct['fault']
        mean_pct['fault'] = np.mean(per_cube_fault_pct) if per_cube_fault_pct else 0
        mean_pct['no_fault'] = 100 - mean_pct['fault']
    else:
        overall_pct['normal'] = (normal_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        overall_pct['inverse'] = (inverse_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        overall_pct['overlap'] = (overlap_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        overall_pct['no_fault'] = 100 - (overall_pct['normal'] + overall_pct['inverse'] + overall_pct['overlap'])

        mean_pct['normal'] = np.mean(per_cube_normal_pct) if per_cube_normal_pct else 0
        mean_pct['inverse'] = np.mean(per_cube_inverse_pct) if per_cube_inverse_pct else 0
        mean_pct['overlap'] = np.mean(per_cube_overlap_pct) if per_cube_overlap_pct else 0
        mean_pct['no_fault'] = 100 - (mean_pct['normal'] + mean_pct['inverse'] + mean_pct['overlap'])

    return overall_pct, mean_pct

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_fault_points():
    pass