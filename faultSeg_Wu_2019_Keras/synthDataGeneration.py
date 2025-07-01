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
                               mask_mode=1,
                               num_faults_this_cube=0, # Specific number of faults for this call
                               max_disp_this_cube=0, # Specific max displacement magnitude for this call
                               strike_range=(0,360), # Range for random picking PER FAULT
                               dip_range=(10,90)):   # Range for random picking PER FAULT
    """
    Inserts dip-slip faults and returns (faulted_volume, mask, list_of_fault_params):
      - mask_mode=0 → boolean mask (True on any fault-plane)
      - mask_mode=1 → uint8 mask: 1=normal (hanging wall down), 2=reverse (hanging wall up)

    num_faults_this_cube : int (the number of faults to add in this specific call)
    max_disp_this_cube   : float or int (the max displacement magnitude for faults in this specific call)
                           Assumed to be integer grid units based on parsing.
    strike_range         : float or (min,max) deg - range used *per fault*
    dip_range            : float or (min,max) deg - range used *per fault*
    """
    nx, ny, nz = volume.shape

    sr = strike_range if isinstance(strike_range, tuple) else (strike_range, strike_range)
    dr = dip_range    if isinstance(dip_range, tuple)    else (dip_range, dip_range)

    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    faulted = volume.copy()
    plane_mask = np.zeros_like(volume, dtype=(bool if mask_mode==0 else np.uint8))

    fault_params_list = [] # Collect parameters for each generated fault

    nf_actual = int(num_faults_this_cube) # Ensure int

    for _ in range(nf_actual):
        x0 = np.random.randint(0, nx)
        y0 = np.random.randint(0, ny)
        z0 = np.random.randint(0, nz)

        # Pick random strike and dip PER FAULT within the provided ranges
        strike_deg = np.random.uniform(*sr)
        dip_deg    = np.random.uniform(*dr)

        # Ensure dip is not exactly 0 or 90 for numerical stability if needed
        # dip_deg = np.clip(dip_deg, 1e-6, 90.0 - 1e-6) # Example clipping if needed

        strike = np.radians(strike_deg)
        dip    = np.radians(dip_deg)

        # Pick slip sign PER FAULT if mask_mode is 1
        slip_sign = 1 # Default for mask_mode 0 (no separate labels)
        label     = True # Default for mask_mode 0 (boolean)
        fault_type = None # Default for mask_mode 0
        if mask_mode==1:
            # Ensure ~50/50 normal/inverse distribution over many faults
            # As determined previously: slip_sign = -1 for Normal (HW down, Z increase), slip_sign = +1 for Inverse (HW up, Z decrease)
            slip_sign = np.random.choice([+1, -1]) # +1 for Inverse, -1 for Normal
            label = 1 if slip_sign == -1 else 2 # Label 1 for Normal, Label 2 for Inverse
            fault_type = 'Normal' if slip_sign == -1 else 'Inverse'

        # max_disp_this_cube is the magnitude parameter used for *this* cube
        # The actual displacement applied along the normal is signed by slip_sign
        d_magnitude = float(max_disp_this_cube) # Ensure float for calculations
        d_applied = slip_sign * d_magnitude # The signed displacement magnitude applied along the normal

        # Calculate fault plane normal vector (a,b,c)
        # This normal is perpendicular to the plane. Displacement d_applied is applied along this normal.
        # Assuming Z is depth (downwards), normal vector has dz_norm = -cos(dip)
        dx_norm = np.cos(strike)*np.sin(dip)
        dy_norm = np.sin(strike)*np.sin(dip)
        dz_norm = -np.cos(dip)

        # Normalize the normal vector
        norm_vec = np.array([dx_norm, dy_norm, dz_norm])
        norm_norm = np.linalg.norm(norm_vec)
        # Handle potential division by zero if dip is exactly 0 or 180 (very unlikely with ranges)
        a, b, c = norm_vec / (norm_norm if norm_norm > 1e-6 else 1.0)


        # Plane equation: a*(X-x0) + b*(Y-y0) + c*(Z-z0) = 0
        D = a*(X-x0) + b*(Y-y0) + c*(Z-z0)

        # Mask location: points close to the plane (|D| < 0.5 grid units)
        mask_loc = np.abs(D) < 0.5
        plane_mask[mask_loc] = label # Assign label (True, 1, or 2)

        # Hanging wall: points on one side of the plane (defined by sign of D)
        # Assume D > 0 is the hanging wall side for consistent displacement application.
        pos = D > 0
        # Apply displacement along the normal vector (a,b,c) with magnitude d_applied
        # A point (X_out, Y_out, Z_out) in the *output* volume at the hanging wall side
        # was originally at (X_out - d_applied*a, Y_out - d_applied*b, Z_out - d_applied*c)
        # in the *input* volume (volume). We need to sample the input volume at this original location.
        # Use float coordinates for sampling and then clip/round for integer indexing if needed,
        # but map_coordinates takes float indices and handles interpolation.
        I_sample = X[pos].astype(np.float32) - d_applied * a
        J_sample = Y[pos].astype(np.float32) - d_applied * b
        K_sample = Z[pos].astype(np.float32) - d_applied * c

        # Collect coordinates for map_coordinates
        coords = np.vstack([I_sample, J_sample, K_sample])

        # Sample the input volume at the displaced coordinates
        # Use order=1 for linear interpolation
        # Mode 'reflect' handles boundary conditions if sample points fall outside volume bounds [0, nx/ny/nz-1]
        if coords.shape[1] > 0: # Only attempt sampling if there are points in 'pos'
             sampled_values = map_coordinates(volume.astype(np.float32), coords, order=1, mode='reflect')
             # Assign sampled values to the hanging wall positions in the faulted volume
             faulted[pos] = sampled_values


        # Record parameters for this fault
        # Cast to standard Python types for JSON serialization
        fault_params_list.append({
            'disp_magnitude_per_cube': float(max_disp_this_cube), # The magnitude value used for this cube
            'applied_disp_signed': float(d_applied), # The signed displacement applied along normal
            'vertical_disp_component': float(d_applied * c), # The Z component of the displacement vector
            'strike': float(strike_deg),
            'dip': float(dip_deg),
            'slip_sign': int(slip_sign), # Ensure JSON serializable
            'fault_type': fault_type # 'Normal' or 'Inverse' (string is JSON serializable)
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
                   help="Base directory for output train, validation, and prediction folders.")
    p.add_argument("--train-split", type=float, default=0.7,
                   help="Percentage of data to use for training (e.g., 0.7 for 70%).")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Percentage of data to use for validation (e.g., 0.15 for 15%).")
    return p.parse_args()


def prepare_dirs(base):
    # New directory structure
    splits = ['train', 'validation', 'prediction']
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
    if not (0 <= args.train_split <= 1 and 0 <= args.val_split <= 1):
        raise ValueError("Train and validation splits must be between 0 and 1.")
    pred_split = 1.0 - args.train_split - args.val_split
    if pred_split < 0:
        raise ValueError("Sum of train and validation splits cannot exceed 1.0.")

    # Use the output path determined by argparse or default
    base_out = os.path.expanduser(args.output_dir)

    # Prepare the new directory structure
    dirs = prepare_dirs(base_out)

    nx = ny = nz = args.size

    # --- Data assignment ---
    num_train = int(np.floor(args.num_pairs * args.train_split))
    num_val = int(np.floor(args.num_pairs * args.val_split))
    num_pred = args.num_pairs - num_train - num_val

    # Create a shuffled list of assignments
    assignments = ['train'] * num_train + ['validation'] * num_val + ['prediction'] * num_pred
    np.random.shuffle(assignments)

    print(f"\nData split: {num_train} train, {num_val} validation, {num_pred} prediction.")

    # Dictionaries to collect parameters for statistics for each split
    stats = {
        'all': {'cube_level_params': [], 'all_fault_params': []},
        'train': {'cube_level_params': [], 'all_fault_params': []},
        'validation': {'cube_level_params': [], 'all_fault_params': []},
        'prediction': {'cube_level_params': [], 'all_fault_params': []}
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

        # Max displacement magnitude for this cube's faults
        # Use randint if the input was int,int range, uniform if float,float range
        if is_max_disp_range:
             # Assuming parse_int_or_range made it an int tuple
             md_this_cube = np.random.randint(args.max_disp[0], args.max_disp[1] + 1)
        else:
            md_this_cube = args.max_disp # Use fixed value


        # Record cube-level parameters *before* generation
        # Cast to standard Python types for JSON serialization
        cube_params = {
            'freq': float(freq_this_cube),
            'noise_sigma': float(noise_sigma_this_cube) if noise_sigma_this_cube is not None else None,
            'num_gaussians': int(ng_this_cube),
            'num_faults_generated': int(nf_this_cube),
            'max_disp_used': float(md_this_cube) # The magnitude used for ALL faults in this cube
        }
        stats[assignment]['cube_level_params'].append(cube_params)
        stats['all']['cube_level_params'].append(cube_params)


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
            mask_mode   = args.mask_mode,
            num_faults_this_cube  = nf_this_cube,  # Pass the determined number for this cube
            max_disp_this_cube    = md_this_cube,  # Pass the determined max_disp for this cube
            strike_range= strike_range_arg, # Pass ranges/values from args
            dip_range   = dip_range_arg     # Pass ranges/values from args
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
        for split in ['train', 'validation', 'prediction']:
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
        for split in ['train', 'validation', 'prediction']:
            split_dir = os.path.join(base_out, split)
            if os.path.exists(split_dir):
                 print(f"Archiving {split_dir}...")
                 shutil.make_archive(split_dir, 'zip', root_dir=base_out, base_dir=split)
                 print(f"Created {split_dir}.zip")
            else:
                print(f"Warning: Directory not found for zipping: {split_dir}")


    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Generation complete.")
    for split in ['train', 'validation', 'prediction']:
        print(f" • {split}/seis/ → {dirs[split]['seis']}")
        print(f" • {split}/fault/ → {dirs[split]['fault']}")
    if args.zip:
        print(f" • Archives created in {base_out}")
    print("Done ✅")


if __name__=="__main__":
    main()