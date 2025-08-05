import os
import shutil
import argparse
import numpy as np
from scipy.ndimage import map_coordinates
from datetime import datetime
import json # Keep json for saving stats
from tqdm import tqdm

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
    """
    Produces a smooth vertical shift field by summing `num_gaussians` 2D Gaussians.
    """
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    shift2d = np.zeros((nx, ny))
    # Ensure num_gaussians is an integer
    num_gaussians = int(num_gaussians)
    for _ in range(num_gaussians):
        bk = np.random.uniform(10, 25)
        ck = np.random.uniform(0, nx)
        dk = np.random.uniform(0, ny)
        sigma = np.random.uniform(5, 15)
        shift2d += bk * np.exp(-((x[:,:,0]-ck)**2 + (y[:,:,0]-dk)**2)/(2*sigma**2))
    a0 = np.random.uniform(-5, 5)
    depth_scale = (1.5 * np.arange(nz)[None,None,:] / nz)
    return a0 + shift2d[:, :, None] * depth_scale

def apply_vertical_shift(volume, shift):
    """Applies the vertical warp `shift` to each trace along z."""
    nx, ny, nz = volume.shape
    out = np.zeros_like(volume)
    for i in range(nx):
        for j in range(ny):
            # Ensure new_z coordinates are within bounds [0, nz-1] before mapping
            # Use order=1 for linear interpolation
            new_z = np.clip(np.arange(nz).astype(np.float32) + shift[i,j], 0.0, float(nz-1)) # Use float for interpolation
            out[i,j,:] = map_coordinates(volume[i,j,:].astype(np.float32), [new_z], order=1, mode='reflect') # Use reflect mode for boundaries
    return out

def add_planar_shear(volume):
    """Applies a simple planar shear defined by random e0, f, g."""
    nx, ny, nz = volume.shape
    e0 = np.random.uniform(-5,5)
    f  = np.random.uniform(-0.05,0.05) # Reduced range for less extreme shear
    g  = np.random.uniform(-0.05,0.05) # Reduced range
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    s2 = e0 + f*x + g*y
    out = np.zeros_like(volume)
    for i in range(nx):
        for j in range(ny):
            new_z = np.clip(np.arange(nz).astype(np.float32) + s2[i,j,0], 0.0, float(nz-1)) # Use float for interpolation
            out[i,j,:] = map_coordinates(volume[i,j,:].astype(np.float32), [new_z], order=1, mode='reflect') # Use reflect mode
    return out

def add_faults_and_plane_mask(volume,
                              *,
                              cube_index,
                              mask_mode       = 1,
                              num_faults_this_cube = 0,
                              max_disp_this_cube   = 0,
                              strike_range   = (0, 360),
                              dip_range      = (10, 90)):

    nx, ny, nz = volume.shape

    # ---------- helpers for single-value vs. range -------------------------
    sr = strike_range if isinstance(strike_range, tuple) else (strike_range, strike_range)
    dr = dip_range    if isinstance(dip_range,    tuple) else (dip_range,    dip_range)

    # ---------- allocate outputs ------------------------------------------
    faulted     = volume.copy()
    plane_mask  = np.zeros_like(volume,
                                dtype = (bool if mask_mode == 0 else np.uint8))
    fault_params_list = []

    nf_actual = int(num_faults_this_cube)

    # ========================  main loop  ==================================
    for _ in range(nf_actual):

        # -- random plane location & orientation ---------------------------
        x0 = np.random.randint(0, nx)
        y0 = np.random.randint(0, ny)
        z0 = np.random.randint(0, nz)
        strike_deg = np.random.uniform(*sr)
        dip_deg    = np.random.uniform(*dr)
        dip        = np.radians(dip_deg)
        strike_rad = np.radians(strike_deg)

        # -- signed vertical-displacement range for this cube -------------
        if isinstance(max_disp_this_cube, tuple):
            low, high = max_disp_this_cube          # already signed
        else:
            low, high = -max_disp_this_cube, max_disp_this_cube

        max_abs_disp = max(abs(low), abs(high))     # for sanity caps

        # -- draw vertical displacement, rejecting only pathological dips --
        while True:
            d_vert  = np.random.uniform(low, high)
            dz_norm = -np.cos(dip)                  # Z-component of plane normal
            if abs(dz_norm) < 1e-6:                 # almost vertical plane
                continue                            # resample d_vert
            d_applied = d_vert / dz_norm            # total slip along plane
            # keep total slip reasonable (≤ 1.5 × allowed vertical component)
            if abs(d_applied) <= max_abs_disp * 1.5:
                break

        # -- fault type & label -------------------------------------------
        slip_sign  = -1 if d_vert < 0 else 1        # Normal (-), Reverse (+)
        if mask_mode == 0:                          # binary mask
            label, fault_type = True, None
        else:                                       # multi-class mask
            label      = 1 if slip_sign == -1 else 2
            fault_type = 'Normal' if slip_sign == -1 else 'Reverse'

        # -- plane coefficients (a, b, c) ---------------------------------
        a = np.cos(strike_rad) * np.sin(dip)
        b = np.sin(strike_rad) * np.sin(dip)
        c = dz_norm                                     # = -cos(dip)
        norm = np.linalg.norm([a, b, c])
        a, b, c = a / norm, b / norm, c / norm          # unit normal

        # -- signed distance of every voxel to the plane ------------------
        X, Y, Z = np.meshgrid(np.arange(nx),
                              np.arange(ny),
                              np.arange(nz),
                              indexing='ij')
        D = a * (X - x0) + b * (Y - y0) + c * (Z - z0)

        # mask voxels close to the plane
        mask_loc = np.abs(D) < 0.5
        plane_mask[mask_loc] = label

        # voxels on the hanging-wall side (D > 0)
        pos = D > 0

        # -- shift those voxels by (d_applied * normal) --------------------
        I_s = X[pos].astype(np.float32) - d_applied * a
        J_s = Y[pos].astype(np.float32) - d_applied * b
        K_s = Z[pos].astype(np.float32) - d_applied * c

        coords = np.vstack([I_s, J_s, K_s])
        if coords.shape[1] > 0:                      # safety
            samples = map_coordinates(volume.astype(np.float32),
                                       coords, order=1, mode='reflect')
            faulted[pos] = samples

        # -- log parameters ------------------------------------------------
        fault_params_list.append({
            'cube_id'               : int(cube_index),
            'disp_range_signed'     : (float(low), float(high)),
            'applied_disp_signed'   : float(d_applied),
            'vertical_disp_component': float(d_vert),
            'strike'                : float(strike_deg),
            'dip'                   : float(dip_deg),
            'fault_type'            : fault_type
        })

    return faulted, plane_mask, fault_params_list

def ricker_wavelet(f, length, dt):
    # Ensure f, length, dt are float
    f = float(f)
    length = float(length)
    dt = float(dt)
    t_ = np.arange(-length/2, (length-dt)/2, dt)
    a_ = (np.pi*f*t_)**2
    return t_, (1-2*a_)*np.exp(-a_)

def convolve_volume_with_wavelet(volume, wavelet):
    conv_vol = np.zeros_like(volume, dtype=np.float32) # Ensure output is float32
    # Volume should be float32 from previous steps (apply_vertical_shift etc.)
    vol_float = volume.astype(np.float32)
    wavelet_float = wavelet.astype(np.float32) # Ensure wavelet is float32
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            # Use scipy.signal.convolve if performance is critical, but np.convolve is fine
            conv_vol[i,j,:] = np.convolve(vol_float[i,j,:], wavelet_float, mode='same')
    return conv_vol

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
    p.add_argument("--strike", type=parse_float_or_range, default="0,360",
                   help="Fault strike angle (deg) or min,max (default 0–360). Applied PER FAULT.")
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

def main():
    args = parse_args()

    # Validate splits
    # If a value ≤ 1 → treat as percentage, otherwise as absolute count
    if args.train_split <= 1.0:
        num_train = int(round(args.num_pairs * args.train_split))
    else:
        num_train = int(round(args.train_split))

    if args.val_split <= 1.0:
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

    # Determine *once* if ranges are used for fault-level params (strike, dip) from the parsed args
    # These ranges are passed to add_faults_and_plane_mask for per-fault sampling
    strike_range_arg = args.strike
    dip_range_arg = args.dip

    print(f"Output format: {args.format}")
    print(f"\nStarting data generation for {args.num_pairs} pairs...")
    for i in tqdm(range(args.num_pairs), desc="Generating Cubes"):
        assignment = assignments[i]
        #print(f"Generating cube {i+1}/{args.num_pairs} (Assignment: {assignment})...")

        # Determine actual cube-level parameters for this cube by sampling from ranges in args
        # Number of Gaussians for folding
        ng_this_cube = np.random.randint(args.num_gaussians[0], args.num_gaussians[1] + 1) if is_gauss_range else args.num_gaussians

        # Frequency
        freq_this_cube = np.random.uniform(args.freq[0], args.freq[1]) if is_freq_range else args.freq

        # Noise Sigma
        noise_sigma_this_cube = None
        if args.noise is not None:
            noise_sigma_this_cube = np.random.uniform(args.noise[0], args.noise[1]) if is_noise_range else args.noise

        # Number of faults for this cube
        nf_this_cube = np.random.randint(args.faults[0], args.faults[1] + 1) if is_faults_range else args.faults

        # ------------------------------------------------------------------
        # Signed vertical-displacement range to use in this cube
        # ------------------------------------------------------------------
        if isinstance(args.max_disp, tuple):              # e.g. (-50, 50)
            disp_range_this_cube = args.max_disp
        else:                                             # single positive → symmetric
            disp_range_this_cube = (-args.max_disp, args.max_disp)

        # Record it for statistics
        # ------------------------------------------------------------------
        # Cube-level parameters  – build the dict *once*
        # ------------------------------------------------------------------
        cube_params = {
            "freq"              : float(freq_this_cube),
            "noise_sigma"       : (float(noise_sigma_this_cube)
                                   if noise_sigma_this_cube is not None else None),
            "num_gaussians"     : int(ng_this_cube),
            "num_faults_generated": int(nf_this_cube),
            # NEW: signed range actually used in this cube
            "disp_range_used"   : [float(disp_range_this_cube[0]),
                                   float(disp_range_this_cube[1])]
        }

        # add to in-memory statistics
        stats[assignment]["cube_level_params"].append(cube_params)
        stats["all"]["cube_level_params"].append(cube_params)

        # base reflectivity
        r1d = np.random.uniform(-1,1,nz)
        refl = np.tile(r1d,(nx,ny,1)).astype(np.float32) # Start with float32

        # fold + shear
        s1 = generate_folding_shift(nx, ny, nz, num_gaussians=ng_this_cube) # Use determined ng
        folded = apply_vertical_shift(refl, s1)
        sheared = add_planar_shear(folded)

        # faults + mask
        faulted, mask, fault_list_this_cube = add_faults_and_plane_mask(
                sheared,
                cube_index = i,
                mask_mode  = args.mask_mode,
                num_faults_this_cube = nf_this_cube,
                max_disp_this_cube   = disp_range_this_cube,   # ← was md_this_cube
                strike_range = strike_range_arg,
                dip_range    = dip_range_arg
        )
        # Collect fault parameters generated in this cube
        stats[assignment]['all_fault_params'].extend(fault_list_this_cube)
        stats['all']['all_fault_params'].extend(fault_list_this_cube)

        # wavelet
        _, wavelet = ricker_wavelet(freq_this_cube, args.length, args.dt) # Use determined freq
        seismic = convolve_volume_with_wavelet(faulted, wavelet)

        # add noise if requested
        if noise_sigma_this_cube is not None: # Use determined noise_sigma
            noisy = seismic + np.random.normal(0, noise_sigma_this_cube, seismic.shape).astype(np.float32)
        else:
            noisy = seismic # Already float32 from convolve

        # --- Data Value Scaling ---
        # Scale seismic data to roughly [-1, 1] range to prevent very large/small values
        data_max_abs = np.max(np.abs(noisy))
        if data_max_abs > 1e-9: # Use a small epsilon to avoid division by zero/near-zero
             scaled_noisy = noisy / data_max_abs
        else:
             scaled_noisy = noisy # Volume is effectively zero or constant

        # Save as float32 is typical for seismic data
        noisy_to_save = scaled_noisy.astype(np.float32)
        mask_to_save = mask.astype(np.uint8) # Mask should always be uint8


        # save to the correct split directory
        fname = f"{i}.{args.format}"
        seismic_path = os.path.join(dirs[assignment]['seis'], fname)
        mask_path = os.path.join(dirs[assignment]['fault'], fname)
        save_array(noisy_to_save, seismic_path, args.format)
        save_array(mask_to_save, mask_path, args.format)

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


    # --- Plotting (ONLY if --plot is True) ---
    # This block will NOT run when called from the notebook with --plot False
    if args.plot:
        print("\n--- Generating Statistics Plots (Standalone mode) ---")
        # Load data from the saved JSON to ensure consistency, or use in-memory lists
        # Need matplotlib and other plotting deps if this is enabled.
        # import matplotlib.pyplot as plt
        # import math
        # from matplotlib.colors import ListedColormap
        # if os.path.exists(stats_file_path):
        #     try:
        #         with open(stats_file_path, 'r') as f:
        #              loaded_stats = json.load(f)
        #         # Now loaded_stats is a dict with keys 'all', 'train', etc.
        #         cube_level_params_loaded = loaded_stats.get('all', {}).get('cube_level_params', [])
        #         all_fault_params_loaded = loaded_stats.get('all', {}).get('all_fault_params', [])
        #         # Call plotting functions using loaded_stats
        #         # plot_fault_counts(all_fault_params_loaded)
        #         # ... etc ...
        #         # plt.show()
        #     except Exception as e:
        #         print(f"Error loading saved stats for standalone plot: {e}")
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