import os
import shutil
import argparse
import numpy as np
from scipy.ndimage import map_coordinates
from datetime import datetime
import json # Keep json for saving stats
from tqdm import tqdm
import random
import time
from math import fmod

# Helper to parse "int" or "int,int" ranges from CLI
def parse_int_or_range(s):
    parts = s.split(',')
    if len(parts) == 1:
        try:
            return int(parts[0])
        except ValueError:
             raise argparse.ArgumentTypeError(f"Invalid int value: {s}")
    elif len(parts) == 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
             raise argparse.ArgumentTypeError(f"Invalid int range format: {s}. Use int or int,int.")
    else:
        raise argparse.ArgumentTypeError("Must be int or min,max")

def parse_float_or_range(s):
    parts = s.split(',')
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
             raise argparse.ArgumentTypeError(f"Invalid float value: {s}")
    elif len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
             raise argparse.ArgumentTypeError(f"Invalid float range format: {s}. Use float or float,float.")
    else:
        raise argparse.ArgumentTypeError("Must be float or min,max")

# -- Data-generation functions --

def generate_folding_shift(nx, ny, nz, num_gaussians=5):
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    shift2d = np.zeros((nx, ny), dtype=np.float32)

    num_gaussians = int(num_gaussians)
    for _ in range(num_gaussians):
        bk    = np.random.uniform(10, 25)
        ck    = np.random.uniform(0, nx)
        dk    = np.random.uniform(0, ny)
        sigma = np.random.uniform(5, 15)
        gauss = np.exp(-(((X - ck)**2 + (Y - dk)**2) / (2.0 * sigma**2))).astype(np.float32)
        shift2d += (bk * gauss).astype(np.float32)

    a0 = np.random.uniform(-5, 5)
    depth_scale = (1.5 * np.arange(nz, dtype=np.float32)[None, None, :] / nz)
    return a0 + shift2d[:, :, None] * depth_scale

def parse_ranges_list(s: str, *, dtype=float):
    """
    Convert '40-50,220-230'  →  [(40, 50), (220, 230)]
    Leading / trailing white-space inside the range list is ignored.
    """
    ranges = []
    for rng in s.split(','):
        lo, hi = map(str.strip, rng.split('-'))     # <─ NEW strip()
        ranges.append((dtype(lo), dtype(hi)))
    return ranges

def ricker_wavelet(f, length, dt):
    # Ensure f, length, dt are float
    f = float(f)
    length = float(length)
    dt = float(dt)
    t_ = np.arange(-length/2, (length-dt)/2, dt)
    a_ = (np.pi*f*t_)**2
    return t_, (1-2*a_)*np.exp(-a_)

# -- Main CLI --

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic seismic/mask pairs with statistics.")
    p.add_argument("--num-pairs", type=int, default=10,
                   help="How many (seismic,mask) cubes to make.")
    p.add_argument("--format", choices=["npy","npz", "dat"], default="npy",
                   help="File format for output: npy, compressed npz, or dat.")
    p.add_argument("--mask-mode", type=int, choices=[0,1], default=1,
                   help="0=boolean BW mask, 1=uint8 colored mask (1=normal,2=reverse).")
    p.add_argument("--size", type=int, default=128,
                   help="Cube dimension NX=NY=NZ (default 128).")
    p.add_argument("--freq", type=parse_float_or_range, default="50.0",
                   help="Wavelet center frequency Hz, or min,max for random.")
    p.add_argument("--dt", type=float, default=0.002,
                   help="Sampling interval (s) for the wavelet (default 0.002).")
    p.add_argument("--length", type=float, default=0.2,
                   help="Total time length (s) of the wavelet (default 0.2).")
    p.add_argument("--num-gaussians", type=parse_int_or_range, default="5",
                   help="How many Gaussians (or min,max) to sum for folding shift.")
    p.add_argument("--faults", type=parse_int_or_range, default="0,6",
                   help="Number of faults per cube: int or min,max (default random 0–6).")
    p.add_argument("--max-disp", type=parse_int_or_range, default="5,30", # Assuming integer displacement in grid units
                   help="Fault slip magnitude per cube: int or min,max (default random 5–30).")
    p.add_argument("--strike", type=parse_ranges_list, default="0-360",                  #  same idea, different syntax
                   help="One or many strike ranges, e.g. 40-50,220-230,135-145,315-325")
    p.add_argument("--dip", type=parse_float_or_range, default="10,90",
                   help="Fault dip angle (deg) or min,max (default 10–90). Applied PER FAULT.")
    p.add_argument("--noise", type=parse_float_or_range, default=None,
                   help="Noise sigma: None=no noise, or value, or min,max.")
    p.add_argument("--zip", action="store_true",
                   help="Also zip up the seismic/ and mask/ folders.")
    # action="store_true" means the flag's presence sets it to True.
    # default=False means if the flag is absent, it's False.
    # No value like True/False should be passed on the command line.
    p.add_argument("--plot", action="store_true", default=False,
                   help="Generate statistics plots (only when run directly by script).")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Base directory for output train, and validation folders.")
    p.add_argument("--train-split", type=float, default=0.7,
                   help="Percentage of data to use for training (e.g., 0.7 for 70%).")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Percentage of data to use for validation (e.g., 0.15 for 15%).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.add_argument("--workers", type=int, default=1, help="CPU workers for per-cube parallelism.")
    p.add_argument("--mask-thickness", type=float, default=1.5,
               help="Half-thickness (in voxels) around the fault plane to label.")
    p.add_argument("--strike-sampling", choices=["span","equal"], default="span",
               help="How to sample strike windows: 'span' (proportional to angular width) or 'equal' (equal per window).")

    return p.parse_args()


def prepare_dirs(base):
    # New directory structure
    splits = ['train', 'validation']
    subfolders = ['seis', 'fault']
    
    # Clean and create directories
    for split in splits:
        split_path = os.path.join(base, split)
        if os.path.exists(split_path):
            try:
                shutil.rmtree(split_path)
                print(f"Cleaned directory: {split_path}")
            except OSError as e:
                print(f"Error clearing directory {split_path}: {e}")
                exit(1)
        for subfolder in subfolders:
            dir_path = os.path.join(base, split, subfolder)
            os.makedirs(dir_path, exist_ok=True)
            
    return {split: {sub: os.path.join(base, split, sub) for sub in subfolders} for split in splits}


def save_array(arr, path, fmt):
    if fmt=="npy": np.save(path, arr)
    elif fmt=="dat": arr.tofile(path)
    else:        np.savez_compressed(path, arr=arr) # Save with a keyword 'arr' for npz

def warp_along_z(volume: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Apply a vertical shift field to a 3-D volume.
    - If `shift` is (nx, ny): constant offset along z.
    - If `shift` is (nx, ny, nz): depth-varying offset (what folding uses).
    """
    nx, ny, nz = volume.shape
    X = np.arange(nx, dtype=np.float32)[:, None, None]
    Y = np.arange(ny, dtype=np.float32)[None, :, None]
    Z = np.arange(nz, dtype=np.float32)[None, None, :]

    if shift.ndim == 2:
        Znew = Z + shift[:, :, None].astype(np.float32)
    else:
        Znew = Z + shift.astype(np.float32)

    # coords order must match array axes: [x, y, z]
    coords = [
        np.broadcast_to(X, (nx, ny, nz)),
        np.broadcast_to(Y, (nx, ny, nz)),
        Znew,
    ]
    out = map_coordinates(volume.astype(np.float32), coords, order=1, mode='reflect')
    return out

def apply_vertical_shift(volume, shift):
    # Use the full 3-D shift field (keeps depth-dependent folding)
    return warp_along_z(volume, shift)

def add_planar_shear(volume):
    nx, ny, _ = volume.shape
    e0 = np.random.uniform(-5, 5)
    f  = np.random.uniform(-0.05, 0.05)
    g  = np.random.uniform(-0.05, 0.05)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    s2 = (e0 + f*x + g*y).astype(np.float32)      # (nx, ny)
    return warp_along_z(volume, s2)                # broadcast along z

def convolve_volume_with_wavelet(volume, wavelet):
    """
    Linear convolution along z for all (x,y) traces at once using FFT.
    Matches 'same' length like np.convolve(..., mode='same').
    """
    vol = volume.astype(np.float32)
    nz  = vol.shape[2]
    w   = np.asarray(wavelet, dtype=np.float32)

    # length for linear convolution, then trim back to 'same'
    L = nz + w.size - 1

    VolF = np.fft.rfft(vol, n=L, axis=2)              # (nx, ny, Lr)
    WF   = np.fft.rfft(w,   n=L)                       # (Lr,)
    out  = np.fft.irfft(VolF * WF[None, None, :], n=L, axis=2)

    start = (w.size - 1) // 2
    end   = start + nz
    return out[:, :, start:end].astype(np.float32)


def sample_strike_equal_windows(windows):
    k = np.random.randint(len(windows))
    lo, hi = windows[k]
    lo %= 360.0; hi %= 360.0
    span = (hi - lo) % 360.0 or 360.0
    return (lo + np.random.uniform(0, span)) % 360.0

def angle_in_windows(a, windows):
    a = a % 360.0
    for lo, hi in windows:
        lo, hi = lo % 360.0, hi % 360.0
        span = (hi - lo) % 360.0 or 360.0
        if (a - lo) % 360.0 <= span:
            return True
    return False


def add_faults_and_plane_mask(
    volume, *, cube_index, mask_mode=1,
    num_faults_this_cube=0, max_disp_this_cube=(-50, 50),
    strike_range=(0, 360), dip_range=(10, 90),
    mask_thickness=0.5,
    strike_sampler=None):

    nx, ny, nz = volume.shape
    vol = volume.astype(np.float32)

    # --- parse dip
    if isinstance(dip_range, tuple):
        dip_lo, dip_hi = dip_range
    else:
        raise ValueError("dip_range must be a tuple (lo, hi)")

    # --- parse strike windows
    if isinstance(strike_range, list):
        strike_windows = strike_range
    elif isinstance(strike_range, tuple):
        strike_windows = [strike_range]
    else:
        raise ValueError("strike_range must be tuple or list[tuple]")

    # --- outputs
    plane_mask = np.zeros_like(vol, dtype=(bool if mask_mode == 0 else np.uint8))
    fault_params_list = []

    # --- coordinate grids (build ONCE)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    Xf = X.astype(np.float32); Yf = Y.astype(np.float32); Zf = Z.astype(np.float32)

    faulted = vol  # will be updated per fault
    
    if strike_sampler is None:
        strike_sampler = sample_strike_union

    for _ in range(int(num_faults_this_cube)):
        # random plane location
        x0 = np.random.randint(0, nx)
        y0 = np.random.randint(0, ny)
        z0 = np.random.randint(0, nz)

        strike_deg = strike_sampler(strike_windows)
        assert angle_in_windows(strike_deg, strike_windows)

        dip_deg = np.random.uniform(dip_lo, dip_hi)
        dip = np.deg2rad(dip_deg)

        # --- choose slip magnitude directly from the full range
        low, high = max_disp_this_cube
        d_applied = np.random.uniform(low, high)
        dz_norm = -np.cos(dip)
        if abs(dz_norm) < 1e-9:  # Avoid division by near-zero
            d_vert = d_applied  # Default to applied displacement if dip is near vertical
        else:
            d_vert = d_applied * dz_norm
        slip_sign = 1 if d_vert >= 0 else -1

        # --- compute plane normal
        theta = np.deg2rad((90.0 - strike_deg) % 360.0)
        a = np.cos(theta) * np.sin(dip)
        b = np.sin(theta) * np.sin(dip)
        c = -np.cos(dip)
        nrm = np.sqrt(a*a + b*b + c*c)
        a, b, c = a/nrm, b/nrm, c/nrm

        # signed distances to plane
        D = a*(X - x0) + b*(Y - y0) + c*(Z - z0)

        # --- improved voxelization with adjustable thickness
        base_th = 1.0 * (abs(a) + abs(b) + abs(c))
        th = float(mask_thickness) * base_th
        on_plane = np.abs(D) <= th

        # --- handle mask updates and fault type
        if mask_mode == 0:
            plane_mask |= on_plane
            fault_type = None
        else:
            label = 1 if slip_sign == -1 else 2
            fault_type = "Normal" if slip_sign == -1 else "Inverse"
            tgt = on_plane
            if np.any(tgt):
                pm = plane_mask[tgt]
                overlap = (pm != 0) & (pm != label)
                pm[overlap] = 3
                pm[(pm == 0) & ~overlap] = label
                plane_mask[tgt] = pm

        # --- apply shift to faulted volume
        pos = (D > 0)
        if np.any(pos):
            Xi = Xf[pos] - d_applied * a
            Yi = Yf[pos] - d_applied * b
            Zi = Zf[pos] - d_applied * c
            coords = np.vstack([Xi, Yi, Zi])
            samples = map_coordinates(faulted, coords, order=1, mode="reflect")
            faulted[pos] = samples

        # --- stats
        mask_voxels = int(on_plane.sum())
        fault_params_list.append({
            "cube_id": int(cube_index),
            "disp_range_signed": (float(low), float(high)),
            "applied_disp_signed": float(d_applied),
            "vertical_disp_component": float(d_vert),
            "strike": float(strike_deg),
            "dip": float(dip_deg),
            "fault_type": fault_type,
            "mask_voxels": mask_voxels
        })

    return faulted, plane_mask, fault_params_list

# def add_faults_and_plane_mask(
#     volume, *, cube_index, mask_mode=1,
#     num_faults_this_cube=0, max_disp_this_cube=(-50, 50),
#     strike_range=(0, 360), dip_range=(10, 90),
#     mask_thickness=0.5,
#     strike_sampler=None):

#     nx, ny, nz = volume.shape
#     vol = volume.astype(np.float32)

#     # --- parse dip
#     if isinstance(dip_range, tuple):
#         dip_lo, dip_hi = dip_range
#     else:
#         raise ValueError("dip_range must be a tuple (lo, hi)")

#     # --- parse strike windows
#     if isinstance(strike_range, list):
#         strike_windows = strike_range
#     elif isinstance(strike_range, tuple):
#         strike_windows = [strike_range]
#     else:
#         raise ValueError("strike_range must be tuple or list[tuple]")

#     # --- outputs
#     plane_mask = np.zeros_like(vol, dtype=(bool if mask_mode == 0 else np.uint8))
#     fault_params_list = []

#     # --- coordinate grids (build ONCE)
#     X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
#     Xf = X.astype(np.float32); Yf = Y.astype(np.float32); Zf = Z.astype(np.float32)

#     faulted = vol  # will be updated per fault
    
#     if strike_sampler is None:                # default behavior
#         strike_sampler = sample_strike_union

#     for _ in range(int(num_faults_this_cube)):
#         # random plane location
#         x0 = np.random.randint(0, nx)
#         y0 = np.random.randint(0, ny)
#         z0 = np.random.randint(0, nz)


#         # strike (equal probability per window)
#         # strike_deg = sample_strike_equal_windows(strike_windows)
#         # assert angle_in_windows(strike_deg, strike_windows), (strike_deg, strike_windows)
        
#         strike_deg = strike_sampler(strike_windows)   # <-- use whichever sampler you chose
#         assert angle_in_windows(strike_deg, strike_windows)


#         # start with a dip candidate; may be resampled in the loop
#         dip_deg = np.random.uniform(dip_lo, dip_hi)
#         dip = np.deg2rad(dip_deg)

#         # --- choose slip magnitude; may resample dip
#         low, high = max_disp_this_cube
#         low_abs, high_abs = abs(low), abs(high)
#         max_tries = 50
#         slip_sign = 1
#         for _try in range(max_tries):
#             dz_norm = -np.cos(dip)
#             if abs(dz_norm) < 1e-3:
#                 dip_deg = np.random.uniform(dip_lo, dip_hi)
#                 dip = np.deg2rad(dip_deg)
#                 continue
#             max_vert_allowed = 1.5 * high_abs * abs(dz_norm)
#             if max_vert_allowed < low_abs:
#                 dip_deg = np.random.uniform(dip_lo, dip_hi)
#                 dip = np.deg2rad(dip_deg)
#                 continue
#             slip_sign  = np.random.choice([-1, 1])
#             upper      = min(high_abs, max_vert_allowed)
#             d_vert_mag = np.random.uniform(low_abs, upper)
#             d_vert     = slip_sign * d_vert_mag
#             d_applied  = d_vert / dz_norm
#             break
#         else:
#             dz_norm = -np.cos(dip) if abs(np.cos(dip)) >= 1e-9 else -1.0
#             d_applied = np.sign(dz_norm) * 1.5 * high_abs
#             d_vert    = d_applied * dz_norm
#             slip_sign = 1 if d_vert >= 0 else -1

#         # NOW compute the plane normal with the final dip
#         theta = np.deg2rad((90.0 - strike_deg) % 360.0)
#         a = np.cos(theta) * np.sin(dip)
#         b = np.sin(theta) * np.sin(dip)
#         c = -np.cos(dip)
#         nrm = np.sqrt(a*a + b*b + c*c)
#         a, b, c = a/nrm, b/nrm, c/nrm

#         # signed distances to plane
#         D = a*(X - x0) + b*(Y - y0) + c*(Z - z0)

#         # --- conservative voxelization of a plane ---
#         # mark a voxel if the infinite plane crosses the voxel cell
#         base_th = 0.5 * (abs(a) + abs(b) + abs(c))
#         th = float(mask_thickness) * base_th
#         on_plane = np.abs(D) <= th

#         # label/type and write into the mask
#         if mask_mode == 0:
#             plane_mask |= on_plane          # boolean union
#             fault_type = None
#         else:
#             label = 1 if slip_sign == -1 else 2
#             fault_type = "Normal" if slip_sign == -1 else "Inverse"

#             tgt = on_plane
#             if np.any(tgt):
#                 pm = plane_mask[tgt]
#                 overlap = (pm != 0) & (pm != label)
#                 pm[overlap] = 3
#                 pm[(pm == 0) & ~overlap] = label
#                 plane_mask[tgt] = pm

#         # hanging wall boolean mask (for shifting the “upthrown” side)
#         pos = (D > 0)

#         # shift only where pos==True
#         if np.any(pos):
#             Xi = Xf[pos] - d_applied * a
#             Yi = Yf[pos] - d_applied * b
#             Zi = Zf[pos] - d_applied * c
#             coords = np.vstack([Xi, Yi, Zi])
#             samples = map_coordinates(faulted, coords, order=1, mode="reflect")
#             faulted[pos] = samples

#         # stats
#         mask_voxels = int(on_plane.sum())
#         fault_params_list.append({
#             "cube_id": int(cube_index),
#             "disp_range_signed": (float(low), float(high)),
#             "applied_disp_signed": float(d_applied),
#             "vertical_disp_component": float(d_vert),
#             "strike": float(strike_deg),
#             "dip": float(dip_deg),
#             "fault_type": fault_type,
#             "mask_voxels": mask_voxels
#         })

        


#     return faulted, plane_mask, fault_params_list

def merge_windows(windows):
    # windows: list of (lo, hi) deg, can wrap
    pts = []
    for lo, hi in windows:
        lo %= 360.0; hi %= 360.0
        if (hi - lo) % 360.0 == 0:  # full circle
            return [(0.0, 360.0)]
        if lo <= hi:
            pts.append((lo, hi))
        else:  # wrap
            pts.append((lo, 360.0))
            pts.append((0.0, hi))
    pts.sort()
    merged = []
    cur_lo, cur_hi = pts[0]
    for lo, hi in pts[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi)); cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged

def sample_strike_union(windows):
    merged = merge_windows(windows)
    spans = [hi - lo for lo, hi in merged]
    total = sum(spans)
    r = np.random.uniform(0, total)
    for (lo, hi), span in zip(merged, spans):
        if r < span:
            return (lo + r) % 360.0
        r -= span
    return merged[-1][1] % 360.0

#######################################
#######################################
#######################################
#######################################

def main():
    args = parse_args()
    
    strike_sampler = (sample_strike_equal_windows 
                  if args.strike_sampling == "equal" 
                  else sample_strike_union)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Validate splits
    # If a value ≤ 1 → treat as percentage, otherwise as absolute count
    if args.train_split < 1.0:
        num_train = int(round(args.num_pairs * args.train_split))
    else:
        num_train = int(round(args.train_split))

    if args.val_split < 1.0:
        num_val = int(round(args.num_pairs * args.val_split))
    else:
        num_val = int(round(args.val_split))

    # sanity-check
    if num_train + num_val != args.num_pairs:
        raise ValueError(
            f"ERROR: requested train + val ({num_train + num_val}) "
            f"≠ num_pairs ({args.num_pairs})."
        )

    # Use the output path determined by argparse or default
    base_out = os.path.expanduser(args.output_dir)

    # Prepare the new directory structure
    dirs = prepare_dirs(base_out)

    nx = ny = nz = args.size

    # Create a shuffled list of assignments
    assignments = ['train'] * num_train + ['validation'] * num_val
    #np.random.shuffle(assignments)

    print(f"\nData split: {num_train} train, {num_val} validation.")

    # Dictionaries to collect parameters for statistics for each split
    stats = {
        'all': {'cube_level_params': [], 'all_fault_params': []},
        'train': {'cube_level_params': [], 'all_fault_params': []},
        'validation': {'cube_level_params': [], 'all_fault_params': []}
    }

    # Determine *once* if ranges are used for cube-level params from the parsed args
    is_freq_range = isinstance(args.freq, tuple)
    is_gauss_range = isinstance(args.num_gaussians, tuple)
    is_noise_range = isinstance(args.noise, tuple)
    is_faults_range = isinstance(args.faults, tuple)
    is_max_disp_range = isinstance(args.max_disp, tuple)

    strike_range_arg = args.strike
    dip_range_arg = args.dip

    print(f"Output format: {args.format}")
    print(f"\nStarting data generation for {args.num_pairs} pairs...")
    first_timing_msg = None
    for i in tqdm(range(args.num_pairs), desc="Generating Cubes"):
        assignment = assignments[i]

        # ---- sample cube-level params (keep your existing code above) ----
        ng_this_cube = np.random.randint(args.num_gaussians[0], args.num_gaussians[1] + 1) if is_gauss_range else args.num_gaussians
        freq_this_cube = np.random.uniform(args.freq[0], args.freq[1]) if is_freq_range else args.freq

        noise_sigma_this_cube = None
        if args.noise is not None:
            noise_sigma_this_cube = np.random.uniform(args.noise[0], args.noise[1]) if is_noise_range else args.noise

        nf_this_cube = np.random.randint(args.faults[0], args.faults[1] + 1) if is_faults_range else args.faults

        if isinstance(args.max_disp, tuple):
            disp_range_this_cube = args.max_disp
        else:
            disp_range_this_cube = (-args.max_disp, args.max_disp)

        cube_params = {
            "freq": float(freq_this_cube),
            "noise_sigma": (float(noise_sigma_this_cube) if noise_sigma_this_cube is not None else None),
            "num_gaussians": int(ng_this_cube),
            "num_faults_generated": int(nf_this_cube),
            "disp_range_used": [float(disp_range_this_cube[0]), float(disp_range_this_cube[1])],
        }
        stats[assignment]["cube_level_params"].append(cube_params)
        stats["all"]["cube_level_params"].append(cube_params)

        # ---- base reflectivity
        r1d  = np.random.uniform(-1, 1, nz).astype(np.float32)
        refl = np.tile(r1d, (nx, ny, 1))

        # ── timings per stage ────────────────────────────────────────────
        t0 = time.perf_counter()
        s1 = generate_folding_shift(nx, ny, nz, num_gaussians=ng_this_cube)
        folded = apply_vertical_shift(refl, s1)
        t1 = time.perf_counter()

        sheared = add_planar_shear(folded)
        t2 = time.perf_counter()

        faulted, mask, fault_list_this_cube = add_faults_and_plane_mask(
            sheared,
            cube_index=i,
            mask_mode=args.mask_mode,
            num_faults_this_cube=nf_this_cube,
            max_disp_this_cube=disp_range_this_cube,
            strike_range=strike_range_arg,
            dip_range=dip_range_arg,
            mask_thickness=args.mask_thickness,
            strike_sampler=strike_sampler          # <-- NEW
        )
        # update stats immediately (you had this earlier but commented it)
        stats[assignment]['all_fault_params'].extend(fault_list_this_cube)
        stats['all']['all_fault_params'].extend(fault_list_this_cube)
        t3 = time.perf_counter()

        _, wavelet = ricker_wavelet(freq_this_cube, args.length, args.dt)
        seismic = convolve_volume_with_wavelet(faulted, wavelet)
        t4 = time.perf_counter()

        # ---- noise & scaling
        if noise_sigma_this_cube is not None:
            noisy = seismic + np.random.normal(0, noise_sigma_this_cube, seismic.shape).astype(np.float32)
        else:
            noisy = seismic

        data_max_abs = float(np.max(np.abs(noisy)))
        scaled_noisy = noisy / data_max_abs if data_max_abs > 1e-9 else noisy
        noisy_to_save = scaled_noisy.astype(np.float32)
        mask_to_save  = mask.astype(np.uint8)
        t5 = time.perf_counter()

        # ---- save
        fname = f"{i}.{args.format}"
        seismic_path = os.path.join(dirs[assignment]['seis'], fname)
        mask_path    = os.path.join(dirs[assignment]['fault'], fname)
        save_array(noisy_to_save, seismic_path, args.format)
        save_array(mask_to_save,  mask_path,    args.format)
        t6 = time.perf_counter()

        if first_timing_msg is None:
            first_timing_msg = (
                f"[timings cube 0] fold {t1-t0:.3f}s | shear {t2-t1:.3f}s | "
                f"faults {t3-t2:.3f}s | conv {t4-t3:.3f}s | "
                f"post(noise+scale) {t5-t4:.3f}s | save {t6-t5:.3f}s"
            )
            
    if first_timing_msg:
        print(first_timing_msg)
        
    # --- Save Statistics Data ---
    print("\n--- Saving Statistics Files ---")
    try:
        # Save full statistics
        full_stats_path = os.path.join(base_out, "statistics_full.json")
        with open(full_stats_path, 'w') as f:
            json.dump(stats['all'], f, indent=4)
        print(f"Saved full statistics to {full_stats_path}")

        # Save individual split statistics
        for split in ['train', 'validation']:
            if stats[split]['cube_level_params']: # Only save if there's data for the split
                split_stats_path = os.path.join(base_out, f"statistics_{split}.json")
                with open(split_stats_path, 'w') as f:
                    json.dump(stats[split], f, indent=4)
                print(f"Saved {split} statistics to {split_stats_path}")

    except Exception as e:
        print(f"Error saving statistics data: {e}")

    if args.plot:
        print("\n--- Generating Statistics Plots (Standalone mode) ---")
        print("Standalone plotting is enabled but not fully implemented in this script version.")
        print("Please run from the notebook to see plots.")


    # zip if requested
    if args.zip:
        print("\nCreating zip archives...")
        for split in ['train', 'validation']:
            split_dir = os.path.join(base_out, split)
            if os.path.exists(split_dir):
                 print(f"Archiving {split_dir}...")
                 shutil.make_archive(split_dir, 'zip', root_dir=base_out, base_dir=split)
                 print(f"Created {split_dir}.zip")
            else:
                print(f"Warning: Directory not found for zipping: {split_dir}")


    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Generation complete.")
    for split in ['train', 'validation']:
        print(f" • {split}/seis/ → {dirs[split]['seis']}")
        print(f" • {split}/fault/ → {dirs[split]['fault']}")
    if args.zip:
        print(f" • Archives created in {base_out}")
    print("Done ✅")


if __name__=="__main__":
    main()