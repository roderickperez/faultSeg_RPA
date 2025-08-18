import os
import shutil
import numpy as np
from tqdm import tqdm
from constants import *
import cupy as cp

xp = cp  # Use cupy for GPU acceleration

# Set up output directories (under the current script directory)
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

DATA_DIR  = os.path.join(ROOT_DIR, DATA_DIR_NAME)
STATS_DIR = os.path.join(ROOT_DIR, STATS_DIR_NAME)
IMAGE_DIR = os.path.join(ROOT_DIR, IMAGE_DIR_NAME)

for _dir in (DATA_DIR, STATS_DIR, IMAGE_DIR):
    os.makedirs(_dir, exist_ok=True)

def _uniform_int_distribution(n, low, high):
    """Generate n integers uniformly distributed between low and high inclusive."""
    if n <= 0:
        return []
    k = high - low + 1
    if n < k:
        # If fewer samples than categories, choose values uniformly at random
        return list(cp.random.randint(low, high+1, size=n))
    base = n // k
    remainder = n % k
    counts = [base] * k
    for i in range(remainder):
        counts[i] += 1
    values = []
    for i, count in enumerate(counts):
        values += [low + i] * count
    cp.random.shuffle(values)
    return values

def generate_dataset(train_count, val_count, mask_mode=1):
    """
    Generate synthetic seismic volumes and corresponding fault masks for training and validation splits.
    Returns a dictionary containing statistics for 'train', 'validation', and 'full' datasets.
    """
    global classic_polarity
    # Use the provided mask_mode for labeling
    globals()['mask_mode'] = mask_mode

    # Prepare output folder structure
    for split in ("train", "validation"):
        shutil.rmtree(os.path.join(DATA_DIR, split), ignore_errors=True)
        os.makedirs(os.path.join(DATA_DIR, split, "seismic"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "fault"), exist_ok=True)

    # Determine fault count distribution (uniform 1–8 across dataset)
    train_fault_counts = _uniform_int_distribution(train_count, 1, 8)
    val_fault_counts   = _uniform_int_distribution(val_count, 1, 8)

    # Containers for generation parameters and fault info
    train_params = []; val_params = []
    train_faults_all = []; val_faults_all = []

    # Pixel-class distribution accumulators
    train_total_vox = 0; val_total_vox = 0
    train_class_counts = cp.zeros(3 if mask_mode==1 else 2, dtype=np.int64)
    val_class_counts   = cp.zeros(3 if mask_mode==1 else 2, dtype=np.int64)
    train_sum_class_pct = cp.zeros_like(train_class_counts, dtype=float)
    val_sum_class_pct   = cp.zeros_like(val_class_counts, dtype=float)

    # Compute final cube size after cropping
    cube_size = NX - 2*PAD

    # --- Generate training set cubes ---
    for i in tqdm(range(train_count), desc="Generating training cubes"):
        num_faults = train_fault_counts[i] if i < len(train_fault_counts) else cp.random.randint(1, 9)
        # Step 1: Base horizontal reflectivity volume
        volume = cp.zeros((NZ, NY, NX), dtype=float)
        trace = cp.random.uniform(-1.0, 1.0, size=NZ)
        for z in range(NZ):
            volume[z, :, :] = trace[z]

        # Step 2: Vertical folding (dome or sag)
        if apply_deformation:
            # Randomly choose dome polarity ('up' or 'down') with equal probability
            classic_polarity = 'up' if cp.random.rand() < 0.5 else 'down'
            # Create lateral 2D Gaussian bumps
            X = np.arange(NX); Y = np.arange(NY)
            xx, yy = np.meshgrid(X, Y, indexing='xy')
            L_xy = np.full((NY, NX), cp.random.uniform(*classic_a0_range), dtype=float)
            # Define safe region for bump centers (avoid edges that will be cropped)
            if classic_keep_within_crop and PAD > 0:
                safe_margin = int(classic_safe_margin_frac * PAD)
                cx_range = (safe_margin, NX - safe_margin)
                cy_range = (safe_margin, NY - safe_margin)
            else:
                cx_range, cy_range = (0, NX), (0, NY)
            # Sum a number of Gaussian bumps
            for _ in range(int(classic_num_bumps)):
                sign = 1.0 if cp.random.rand() < classic_pos_amp_prob else -1.0
                b_k = sign * cp.random.uniform(*classic_bk_range)
                c_k = cp.random.uniform(*cx_range)
                d_k = cp.random.uniform(*cy_range)
                s_k = cp.random.uniform(*classic_sigma_range)
                L_xy += b_k * np.exp(-(((xx - c_k)**2 + (yy - d_k)**2) / (2.0 * s_k**2)))
            # Flip for 'up' polarity (so positive bumps become upward domes)
            if str(classic_polarity).lower().startswith('up'):
                L_xy *= -1.0
            # Depth-dependent scaling of the lateral shift map
            z_idx = np.arange(NZ, dtype=float)
            depth_weight = (z_idx / max(NZ - 1, 1.0)) ** classic_depth_power
            # Apply vertical shift via interpolation
            try:
                #from scipy.ndimage import map_coordinates
                from cupyx.scipy.ndimage import map_coordinates  # GPU version
                Zmap = z_idx[:, None, None] + (classic_depth_scale * depth_weight)[:, None, None] * L_xy[None, :, :]
                Zmap = np.clip(Zmap, 0.0, NZ - 1.0)
                coords = np.vstack([
                    Zmap.reshape(1, -1),
                    np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1),
                    np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
                ])
                volume = map_coordinates(volume, coords, order=3, mode='nearest').reshape(NZ, NY, NX)
            except ImportError:
                folded = np.empty_like(volume)
                for j in range(NY):
                    for k in range(NX):
                        shift = classic_depth_scale * depth_weight * L_xy[j, k]
                        src_positions = np.clip(z_idx + shift, 0.0, NZ - 1.0)
                        folded[:, j, k] = np.interp(src_positions, z_idx, volume[:, j, k])
                volume = folded

        # Step 3: Planar shearing (if enabled)
        if apply_shear:
            e0 = cp.random.uniform(*e0_range)
            f  = cp.random.uniform(*f_range)
            g  = cp.random.uniform(*g_range)
            Zcoord = np.arange(NZ)[:, None, None].astype(float)
            Xcoord = np.arange(NX)[None, None, :]
            Ycoord = np.arange(NY)[None, :, None]
            shear_map = e0 + f * Xcoord + g * Ycoord
            Zmap = np.clip(Zcoord + shear_map, 0.0, NZ - 1.0)
            try:
                #from scipy.ndimage import map_coordinates
                from cupyx.scipy.ndimage import map_coordinates  # GPU version
                coords = np.vstack([
                    Zmap.reshape(1, -1),
                    np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1),
                    np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
                ])
                volume = map_coordinates(volume, coords, order=3, mode='nearest').reshape(NZ, NY, NX)
            except ImportError:
                sheared = np.empty_like(volume)
                z_idx = np.arange(NZ)
                for j in range(NY):
                    for k in range(NX):
                        s_val = float(e0 + f * k + g * j)
                        src = np.clip(z_idx + s_val, 0.0, NZ - 1.0)
                        sheared[:, j, k] = np.interp(src, z_idx, volume[:, j, k])
                volume = sheared

        # Step 4: Faulting
        fault_params_list = []
        if apply_faulting:
            # Precompute grid arrays for plane calculations
            Z = np.arange(NZ, dtype=float)
            X2d, Y2d = np.meshgrid(np.arange(NX), np.arange(NY), indexing='xy')
            # Initialize mask volumes
            fault_mask = cp.zeros((NZ, NY, NX), dtype=np.uint8)       # 2-voxel thick training mask
            fault_mask_display = cp.zeros((NZ, NY, NX), dtype=np.uint8)  # 1-voxel break display
            fault_id_display   = cp.zeros((NZ, NY, NX), dtype=np.int16)  # fault ID at break (for plane reconstruction)
            faulted_volume = volume.copy()

            # Helper functions for fault orientation sampling
            def sample_strike_deg(strike_range, mode='random'):
                """Sample a strike angle in degrees given mode."""
                if mode == 'random':
                    return cp.random.uniform(*strike_range)
                # mode == 'two_sets'
                p = np.array(strike_two_set_weights, dtype=float)
                p /= p.sum()
                set_idx = cp.random.choice([0, 1], p=p)
                base_angle = strike_two_set_means[set_idx]
                if cp.random.rand() < 0.5:
                    base_angle += 180.0  # include the opposite direction (same fault line)
                return (base_angle + cp.random.uniform(-strike_two_set_spread, strike_two_set_spread)) % 360.0

            def sample_orientation():
                """Sample dip (with uniform normal distribution) and strike (according to mode)."""
                dip_min, dip_max = dip_range
                # Sample dip angle via uniform cos(dip) distribution for unbiased normal orientations
                dmin, dmax = np.deg2rad([dip_min, dip_max])
                cos_dip = cp.random.uniform(np.cos(dmax), np.cos(dmin))
                dip_deg = np.rad2deg(np.arccos(np.clip(cos_dip, -1.0, 1.0)))
                # Sample strike angle
                strike_deg = sample_strike_deg((0, 360), mode=strike_sampling_mode)
                return dip_deg, strike_deg

            # Propose fault planes until we have the desired number (or run out of attempts)
            accepted_faults = []
            attempts = 0
            while len(accepted_faults) < num_faults and attempts < fault_max_proposals:
                attempts += 1
                dip_deg, strike_deg = sample_orientation()
                fault_type_choice = cp.random.choice(fault_types, p=np.array(fault_type_weights)/np.sum(fault_type_weights))
                fault_type_label = 'Normal' if fault_type_choice == 'normal' else 'Inverse'
                dist_mode = cp.random.choice(fault_distribution_modes)
                max_slip = cp.random.uniform(*max_slip_range)
                # Calculate fault plane parameters (A*x + B*y + C*z + D = 0)
                phi = np.deg2rad(strike_deg)
                theta = np.deg2rad(dip_deg)
                dip_dir = phi + np.pi/2.0
                A = np.sin(theta) * np.cos(dip_dir)
                B = np.sin(theta) * np.sin(dip_dir)
                C = -np.cos(theta)
                if abs(C) < 1e-3:
                    continue  # skip near-horizontal planes
                # Choose a random point near volume center for the plane
                x0 = NX/2 + cp.random.uniform(-0.1*NX, 0.1*NX)
                y0 = NY/2 + cp.random.uniform(-0.1*NY, 0.1*NY)
                z0 = NZ/2 + cp.random.uniform(-0.1*NZ, 0.1*NZ)
                D = -(A*x0 + B*y0 + C*z0)
                # Strike and dip unit vectors
                strike_vec = np.array([np.sin(phi), -np.cos(phi), 0.0])
                strike_vec /= np.linalg.norm(strike_vec)
                dip_vec = np.cross([A, B, C], strike_vec)
                dip_vec /= np.linalg.norm(dip_vec)
                if dip_vec[2] < 0:
                    dip_vec = -dip_vec  # ensure dip_vec points downward in Z
                # Compute plane intersection depth for each (x,y) in the grid
                z_plane = -(A * X2d + B * Y2d + D) / C
                valid = np.isfinite(z_plane) & (z_plane >= 0) & (z_plane <= NZ - 1)
                if valid.mean() < fault_min_cut_fraction:
                    continue  # plane doesn't cut through enough of the volume
                # Check separation from already accepted faults
                too_close = False
                for prev in accepted_faults:
                    prev_plane = prev['z_plane']
                    overlap_mask = valid & np.isfinite(prev_plane)
                    if not np.any(overlap_mask):
                        continue
                    # Compute overlap where two planes are both present
                    separation = np.abs(z_plane[overlap_mask] - prev_plane[overlap_mask])
                    if (separation < fault_min_sep_z).mean() > fault_max_overlap_frac:
                        too_close = True
                        break
                if too_close:
                    continue
                # Compute fault slip distribution parameters
                # Project relative coordinates onto strike and dip directions
                relx = X2d - x0
                rely = Y2d - y0
                relz = z_plane - z0
                u_map = relx * strike_vec[0] + rely * strike_vec[1] + relz * strike_vec[2]
                v_map = relx * dip_vec[0] + rely * dip_vec[1] + relz * dip_vec[2]
                # Determine slip distribution along dip direction
                # Compute dip-axis span using volume corners
                corners = np.array([[0,0,0],[NX,0,0],[0,NY,0],[NX,NY,0],[0,0,NZ],[NX,0,NZ],[0,NY,NZ],[NX,NY,NZ]], float)
                rel_corners = corners - np.array([x0, y0, z0])
                v_along_dip = rel_corners @ dip_vec
                v_min, v_max = v_along_dip.min(), v_along_dip.max()
                v_span = max(v_max - v_min, 1e-6)
                # Accept the fault
                accepted_faults.append({
                    'A': A, 'B': B, 'C': C, 'D': D,
                    'strike': strike_deg, 'dip': dip_deg,
                    'fault_type': fault_type_label,
                    'dist_mode': dist_mode, 'max_slip': max_slip,
                    'z_plane': z_plane,
                    'u_map': u_map, 'v_map': v_map,
                    'v_min': v_min, 'v_max': v_max, 'v_span': v_span
                })
            # Apply each accepted fault sequentially
            for fid, fault in enumerate(accepted_faults, start=1):
                z_plane = fault['z_plane']
                fault_type_label = fault['fault_type']
                dist_mode = fault['dist_mode']; max_slip = fault['max_slip']
                # Compute slip distribution on this fault plane
                if dist_mode == 'gaussian':
                    sigma_u = (fault['u_map'].max() - fault['u_map'].min()) / 3.0
                    sigma_v = fault['v_span'] / 3.0
                    slip_values = max_slip * np.exp(-(fault['u_map']**2)/(2*sigma_u**2) - (fault['v_map']**2)/(2*sigma_v**2))
                else:
                    if fault_type_label == 'Normal':  # normal fault: slip increases with depth (v_map increasing)
                        slip_values = max_slip * (fault['v_map'] - fault['v_min']) / fault['v_span']
                    else:  # reverse fault: slip increases upward (v_map decreasing)
                        slip_values = max_slip * (fault['v_max'] - fault['v_map']) / fault['v_span']
                    slip_values = np.clip(slip_values, 0.0, max_slip)
                # Determine hanging wall shift: positive (down) for normal, negative (up) for reverse
                offset_values = slip_values if fault_type_label == 'Normal' else -slip_values
                # Mask label for this fault
                label_val = 1 if (mask_mode == 0 or fault_type_label == 'Normal') else 2

                # Prepare new volumes for applying this fault
                new_volume = np.empty_like(faulted_volume)
                new_mask = cp.zeros_like(fault_mask, dtype=np.uint8)
                new_mask_disp = cp.zeros_like(fault_mask_display, dtype=np.uint8)
                new_id_disp = cp.zeros_like(fault_id_display, dtype=np.int16)

                # Apply fault offset to each column (y,x)
                for y in range(NY):
                    z_plane_line = z_plane[y]       # array of plane depths at this y for all x
                    offset_line = offset_values[y]  # array of offsets at this y for all x
                    for x in range(NX):
                        z0_line = float(z_plane_line[x])
                        off = float(offset_line[x])
                        # Compute source indices for this (y,x) trace after applying offset
                        src_z = Z.copy()
                        if np.isfinite(z0_line):
                            hanging = src_z < z0_line
                            src_z[hanging] += off  # shift hanging-wall portion
                        src_z = np.clip(src_z, 0.0, NZ - 1.0)
                        # 1) Apply warp to seismic volume (linear interp)
                        trace = faulted_volume[:, y, x]
                        new_volume[:, y, x] = np.interp(src_z, Z, trace)
                        # 2) Warp existing masks (nearest-neighbor)
                        src_idx = np.clip(np.rint(src_z).astype(int), 0, NZ-1)
                        new_mask[:, y, x]      = fault_mask[src_idx, y, x]
                        new_mask_disp[:, y, x] = fault_mask_display[src_idx, y, x]
                        new_id_disp[:, y, x]   = fault_id_display[src_idx, y, x]
                        # 3) Mark this fault's break in mask volumes
                        if np.isfinite(z0_line) and 0 <= z0_line < NZ - 1:
                            z_low = int(np.floor(z0_line))
                            z_high = z_low + 1
                            # Training mask: mark two voxels around the fault plane
                            new_mask[z_low, y, x]  = label_val
                            new_mask[z_high, y, x] = label_val
                            # Display mask: mark the break at the floor depth
                            z_break = z_low
                            new_mask_disp[z_break, y, x] = label_val
                            new_id_disp[z_break, y, x]   = fid
                # Commit updated volumes for the next fault
                faulted_volume = new_volume
                fault_mask = new_mask
                fault_mask_display = new_mask_disp
                fault_id_display = new_id_disp

                # Record fault parameters for statistics
                applied_disp = max_slip if fault_type_label == 'Normal' else -max_slip
                fault_params_list.append({
                    'fault_type': fault_type_label,
                    'strike': fault['strike'],
                    'dip': fault['dip'],
                    'max_slip': max_slip,
                    'applied_disp_signed': applied_disp,
                    'A': fault['A'], 'B': fault['B'], 'C': fault['C'], 'D': fault['D']
                })
        else:
            # No faulting: create empty mask
            faulted_volume = volume
            fault_mask = cp.zeros((NZ, NY, NX), dtype=np.uint8)

        # Step 5: Band-limit with wavelet and add noise
        peak_freq = cp.random.uniform(*wavelet_peak_freq_range)
        # Create Ricker wavelet
        t = cp.linspace(-1, 1, wavelet_length, dtype=cp.float32)
        pf = cp.float(peak_freq)
        pi2f2t2 = (cp.pi**2) * (pf**2) * (t**2)
        ricker = (1 - 2*pi2f2t2) * cp.exp(-pi2f2t2)
        ricker /= cp.max(cp.abs(ricker))
        
        # 1D convolution along depth (Z) axis using FFT (same length output)
        n = volume.shape[0]
        m = ricker.size
        L = int(2 ** cp.ceil(cp.log2(n + m - 1)))
        V = cp.fft.rfft(volume, n=L, axis=0)
        W = cp.fft.rfft(ricker, n=L)
        V *= W[:, None, None]
        conv_full = cp.fft.irfft(V, n=L, axis=0)
        start = (m - 1) // 2; end = start + n
        seismic = conv_full[start:end, :, :]  # band-limited seismic volume

        # Add noise if enabled
        noise_sigma_val = None
        if apply_noise:
            if noise_type == 'gaussian':
                data_range = seismic.max() - seismic.min()
                noise_std = noise_intensity * data_range
                seismic += cp.random.normal(0.0, noise_std, size=seismic.shape)
                noise_sigma_val = noise_std
            elif noise_type == 'uniform':
                rng = noise_intensity * (seismic.max() - seismic.min())
                seismic += cp.random.uniform(-rng, rng, size=seismic.shape)
            elif noise_type == 'speckle':
                seismic *= cp.random.normal(1.0, noise_intensity, size=seismic.shape)
            elif noise_type == 'salt_pepper':
                frac = noise_intensity
                total_vox = seismic.size
                n_noisy = int(frac * total_vox)
                if n_noisy > 0:
                    coords = np.unravel_index(cp.random.choice(total_vox, size=n_noisy, replace=False), seismic.shape)
                    half = n_noisy // 2
                    seismic[coords][:half] = seismic.min()
                    seismic[coords][half:] = seismic.max()

        # Step 6: Crop padded edges and save volume and mask
        cropped_volume = seismic[PAD:-PAD, PAD:-PAD, PAD:-PAD] if PAD > 0 else seismic
        cropped_mask   = fault_mask[PAD:-PAD, PAD:-PAD, PAD:-PAD] if PAD > 0 else fault_mask
        vol_path = os.path.join(DATA_DIR, "train", "seismic", f"{i:03d}.npy")
        mask_path = os.path.join(DATA_DIR, "train", "fault", f"{i:03d}.npy")
        np.save(vol_path, cropped_volume.astype(np.float32))
        np.save(mask_path, cropped_mask.astype(np.uint8))

        # Update pixel class distribution statistics
        total_vox = cropped_mask.size
        train_total_vox += total_vox
        if mask_mode == 0:
            fault_count = np.sum(cropped_mask != 0)
            train_class_counts[0] += (total_vox - fault_count)  # no-fault voxels
            train_class_counts[1] += fault_count               # fault voxels
            train_sum_class_pct[1] += (fault_count / total_vox * 100.0)
            train_sum_class_pct[0] += ((total_vox - fault_count) / total_vox * 100.0)
        else:
            normal_count = np.sum(cropped_mask == 1)
            inverse_count = np.sum(cropped_mask == 2)
            no_fault_count = total_vox - (normal_count + inverse_count)
            train_class_counts[0] += no_fault_count
            train_class_counts[1] += normal_count
            train_class_counts[2] += inverse_count
            train_sum_class_pct[0] += (no_fault_count / total_vox * 100.0)
            train_sum_class_pct[1] += (normal_count / total_vox * 100.0)
            train_sum_class_pct[2] += (inverse_count / total_vox * 100.0)

        # Record cube-level parameters
        cube_params = {
            'num_faults': len(fault_params_list),
            'noise_sigma': noise_sigma_val
        }
        cube_params['faults'] = fault_params_list  # list of fault parameter dicts for this cube
        train_params.append(cube_params)
        train_faults_all.extend(fault_params_list)
    # end for each training cube

    # --- Generate validation set cubes ---
    for j in tqdm(range(val_count), desc="Generating validation cubes"):
        num_faults = val_fault_counts[j] if j < len(val_fault_counts) else cp.random.randint(1, 9)
        # (The generation steps for validation cubes mirror those for training cubes, with output paths adjusted)
        # Step 1: Base reflectivity volume
        volume = cp.zeros((NZ, NY, NX), dtype=float)
        trace = cp.random.uniform(-1.0, 1.0, size=NZ)
        for z in range(NZ):
            volume[z, :, :] = trace[z]
        # Step 2: Vertical folding
        if apply_deformation:
            classic_polarity = 'up' if cp.random.rand() < 0.5 else 'down'
            X = np.arange(NX); Y = np.arange(NY)
            xx, yy = np.meshgrid(X, Y, indexing='xy')
            L_xy = np.full((NY, NX), cp.random.uniform(*classic_a0_range), dtype=float)
            if classic_keep_within_crop and PAD > 0:
                safe_margin = int(classic_safe_margin_frac * PAD)
                cx_range = (safe_margin, NX - safe_margin)
                cy_range = (safe_margin, NY - safe_margin)
            else:
                cx_range, cy_range = (0, NX), (0, NY)
            for _ in range(int(classic_num_bumps)):
                sign = 1.0 if cp.random.rand() < classic_pos_amp_prob else -1.0
                b_k = sign * cp.random.uniform(*classic_bk_range)
                c_k = cp.random.uniform(*cx_range)
                d_k = cp.random.uniform(*cy_range)
                s_k = cp.random.uniform(*classic_sigma_range)
                L_xy += b_k * np.exp(-(((xx - c_k)**2 + (yy - d_k)**2) / (2.0 * s_k**2)))
            if str(classic_polarity).lower().startswith('up'):
                L_xy *= -1.0
            z_idx = np.arange(NZ, dtype=float)
            depth_weight = (z_idx / max(NZ - 1, 1.0)) ** classic_depth_power
            try:
                #from scipy.ndimage import map_coordinates
                from cupyx.scipy.ndimage import map_coordinates  # GPU version
                Zmap = z_idx[:, None, None] + (classic_depth_scale * depth_weight)[:, None, None] * L_xy[None, :, :]
                Zmap = np.clip(Zmap, 0.0, NZ - 1.0)
                coords = np.vstack([
                    Zmap.reshape(1, -1),
                    np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1),
                    np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
                ])
                volume = map_coordinates(volume, coords, order=3, mode='nearest').reshape(NZ, NY, NX)
            except ImportError:
                folded = np.empty_like(volume)
                for j_y in range(NY):
                    for j_x in range(NX):
                        shift = classic_depth_scale * depth_weight * L_xy[j_y, j_x]
                        src_positions = np.clip(z_idx + shift, 0.0, NZ - 1.0)
                        folded[:, j_y, j_x] = np.interp(src_positions, z_idx, volume[:, j_y, j_x])
                volume = folded
        if apply_shear:
            e0 = cp.random.uniform(*e0_range)
            f  = cp.random.uniform(*f_range)
            g  = cp.random.uniform(*g_range)
            Zcoord = np.arange(NZ)[:, None, None].astype(float)
            Xcoord = np.arange(NX)[None, None, :]
            Ycoord = np.arange(NY)[None, :, None]
            shear_map = e0 + f * Xcoord + g * Ycoord
            Zmap = np.clip(Zcoord + shear_map, 0.0, NZ - 1.0)
            try:
                #from scipy.ndimage import map_coordinates
                from cupyx.scipy.ndimage import map_coordinates  # GPU version
                coords = np.vstack([
                    Zmap.reshape(1, -1),
                    np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1),
                    np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
                ])
                volume = map_coordinates(volume, coords, order=3, mode='nearest').reshape(NZ, NY, NX)
            except ImportError:
                sheared = np.empty_like(volume)
                z_idx = np.arange(NZ)
                for j_y in range(NY):
                    for j_x in range(NX):
                        s_val = float(e0 + f * j_x + g * j_y)
                        src = np.clip(z_idx + s_val, 0.0, NZ - 1.0)
                        sheared[:, j_y, j_x] = np.interp(src, z_idx, volume[:, j_y, j_x])
                volume = sheared
        fault_params_list = []
        if apply_faulting:
            Z = np.arange(NZ, dtype=float)
            X2d, Y2d = np.meshgrid(np.arange(NX), np.arange(NY), indexing='xy')
            fault_mask = cp.zeros((NZ, NY, NX), dtype=np.uint8)
            fault_mask_display = cp.zeros((NZ, NY, NX), dtype=np.uint8)
            fault_id_display = cp.zeros((NZ, NY, NX), dtype=np.int16)
            faulted_volume = volume.copy()
            accepted_faults = []
            attempts = 0
            # (Use the same fault plane selection procedure as above)
            while len(accepted_faults) < num_faults and attempts < fault_max_proposals:
                attempts += 1
                dip_deg, strike_deg = sample_orientation()
                fault_type_choice = cp.random.choice(fault_types, p=np.array(fault_type_weights)/np.sum(fault_type_weights))
                fault_type_label = 'Normal' if fault_type_choice == 'normal' else 'Inverse'
                dist_mode = cp.random.choice(fault_distribution_modes)
                max_slip = cp.random.uniform(*max_slip_range)
                phi = np.deg2rad(strike_deg)
                theta = np.deg2rad(dip_deg)
                dip_dir = phi + np.pi/2.0
                A = np.sin(theta) * np.cos(dip_dir); B = np.sin(theta) * np.sin(dip_dir); C = -np.cos(theta)
                if abs(C) < 1e-3:
                    continue
                x0 = NX/2 + cp.random.uniform(-0.1*NX, 0.1*NX)
                y0 = NY/2 + cp.random.uniform(-0.1*NY, 0.1*NY)
                z0 = NZ/2 + cp.random.uniform(-0.1*NZ, 0.1*NZ)
                D = -(A*x0 + B*y0 + C*z0)
                strike_vec = np.array([np.sin(phi), -np.cos(phi), 0.0]); strike_vec /= np.linalg.norm(strike_vec)
                dip_vec = np.cross([A, B, C], strike_vec); dip_vec /= np.linalg.norm(dip_vec)
                if dip_vec[2] < 0: dip_vec = -dip_vec
                z_plane = -(A*X2d + B*Y2d + D) / C
                valid = np.isfinite(z_plane) & (z_plane >= 0) & (z_plane <= NZ - 1)
                if valid.mean() < fault_min_cut_fraction:
                    continue
                too_close = False
                for prev in accepted_faults:
                    prev_plane = prev['z_plane']
                    overlap_mask = valid & np.isfinite(prev_plane)
                    if not np.any(overlap_mask): 
                        continue
                    separation = np.abs(z_plane[overlap_mask] - prev_plane[overlap_mask])
                    if (separation < fault_min_sep_z).mean() > fault_max_overlap_frac:
                        too_close = True; break
                if too_close:
                    continue
                relx = X2d - x0; rely = Y2d - y0; relz = z_plane - z0
                u_map = relx * strike_vec[0] + rely * strike_vec[1] + relz * strike_vec[2]
                v_map = relx * dip_vec[0] + rely * dip_vec[1] + relz * dip_vec[2]
                corners = np.array([[0,0,0],[NX,0,0],[0,NY,0],[NX,NY,0],[0,0,NZ],[NX,0,NZ],[0,NY,NZ],[NX,NY,NZ]], float)
                rel_corners = corners - np.array([x0, y0, z0])
                v_vals = rel_corners @ dip_vec
                v_min, v_max = v_vals.min(), v_vals.max()
                v_span = max(v_max - v_min, 1e-6)
                accepted_faults.append({
                    'A':A,'B':B,'C':C,'D':D,
                    'strike': strike_deg, 'dip': dip_deg,
                    'fault_type': fault_type_label,
                    'dist_mode': dist_mode, 'max_slip': max_slip,
                    'z_plane': z_plane, 'u_map': u_map, 'v_map': v_map,
                    'v_min': v_min, 'v_max': v_max, 'v_span': v_span
                })
            for fid, fault in enumerate(accepted_faults, start=1):
                z_plane = fault['z_plane']; fault_type_label = fault['fault_type']
                dist_mode = fault['dist_mode']; max_slip = fault['max_slip']
                if dist_mode == 'gaussian':
                    sigma_u = (fault['u_map'].max() - fault['u_map'].min()) / 3.0
                    sigma_v = fault['v_span'] / 3.0
                    slip_values = max_slip * np.exp(-(fault['u_map']**2)/(2*sigma_u**2) - (fault['v_map']**2)/(2*sigma_v**2))
                else:
                    if fault_type_label == 'Normal':
                        slip_values = max_slip * (fault['v_map'] - fault['v_min']) / fault['v_span']
                    else:
                        slip_values = max_slip * (fault['v_max'] - fault['v_map']) / fault['v_span']
                    slip_values = np.clip(slip_values, 0.0, max_slip)
                offset_values = slip_values if fault_type_label == 'Normal' else -slip_values
                label_val = 1 if (mask_mode == 0 or fault_type_label == 'Normal') else 2
                new_volume = np.empty_like(faulted_volume)
                new_mask = cp.zeros_like(fault_mask, dtype=np.uint8)
                new_mask_disp = cp.zeros_like(fault_mask_display, dtype=np.uint8)
                new_id_disp = cp.zeros_like(fault_id_display, dtype=np.int16)
                for y in range(NY):
                    z_plane_line = z_plane[y]; offset_line = offset_values[y]
                    for x in range(NX):
                        z0_line = float(z_plane_line[x]); off = float(offset_line[x])
                        src_z = Z.copy()
                        if np.isfinite(z0_line):
                            hanging = src_z < z0_line
                            src_z[hanging] += off
                        src_z = np.clip(src_z, 0.0, NZ-1.0)
                        trace = faulted_volume[:, y, x]
                        new_volume[:, y, x] = np.interp(src_z, Z, trace)
                        src_idx = np.clip(np.rint(src_z).astype(int), 0, NZ-1)
                        new_mask[:, y, x]      = fault_mask[src_idx, y, x]
                        new_mask_disp[:, y, x] = fault_mask_display[src_idx, y, x]
                        new_id_disp[:, y, x]   = fault_id_display[src_idx, y, x]
                        if np.isfinite(z0_line) and 0 <= z0_line < NZ-1:
                            z_low = int(np.floor(z0_line)); z_high = z_low + 1
                            new_mask[z_low, y, x]  = label_val
                            new_mask[z_high, y, x] = label_val
                            z_break = z_low
                            new_mask_disp[z_break, y, x] = label_val
                            new_id_disp[z_break, y, x]   = fid
                faulted_volume = new_volume; fault_mask = new_mask
                fault_mask_display = new_mask_disp; fault_id_display = new_id_disp
                applied_disp = max_slip if fault['fault_type'] == 'Normal' else -max_slip
                fault_params_list.append({
                    'fault_type': fault['fault_type'],
                    'strike': fault['strike'],
                    'dip': fault['dip'],
                    'max_slip': max_slip,
                    'applied_disp_signed': applied_disp,
                    'A': fault['A'], 'B': fault['B'], 'C': fault['C'], 'D': fault['D']
                })
        else:
            faulted_volume = volume
            fault_mask = cp.zeros((NZ, NY, NX), dtype=np.uint8)
        # Wavelet & noise
        peak_freq = cp.random.uniform(*wavelet_peak_freq_range)
        t = cp.linspace(-1, 1, wavelet_length, dtype=cp.float32)
        pf = float(peak_freq); t2 = t**2; pi2f2t2 = (cp.pi**2)*(pf**2)*t2
        ricker = (1 - 2*pi2f2t2) * cp.exp(-pi2f2t2)
        ricker /= cp.max(cp.abs(ricker))
        n = faulted_volume.shape[0]; m = ricker.size
        L = int(2 ** np.ceil(np.log2(n + m - 1)))
        V = cp.fft.rfft(faulted_volume, n=L, axis=0)
        W = cp.fft.rfft(ricker, n=L)[:, None, None]
        conv_full = cp.fft.irfft(V * W, n=L, axis=0)
        start = (m-1)//2; end = start + n
        seismic = conv_full[start:end, :, :]
        noise_sigma_val = None
        if apply_noise:
            if noise_type == 'gaussian':
                data_range = seismic.max() - seismic.min()
                noise_std = noise_intensity * data_range
                seismic += cp.random.normal(0.0, noise_std, size=seismic.shape)
                noise_sigma_val = noise_std
            elif noise_type == 'uniform':
                rng = noise_intensity * (seismic.max() - seismic.min())
                seismic += cp.random.uniform(-rng, rng, size=seismic.shape)
            elif noise_type == 'speckle':
                seismic *= cp.random.normal(1.0, noise_intensity, size=seismic.shape)
            elif noise_type == 'salt_pepper':
                frac = noise_intensity; total_vox = seismic.size; n_noisy = int(frac * total_vox)
                if n_noisy > 0:
                    coords = np.unravel_index(cp.random.choice(total_vox, size=n_noisy, replace=False), seismic.shape)
                    half = n_noisy // 2
                    seismic[coords][:half] = seismic.min()
                    seismic[coords][half:] = seismic.max()
        cropped_volume = seismic[PAD:-PAD, PAD:-PAD, PAD:-PAD] if PAD > 0 else seismic
        cropped_mask   = fault_mask[PAD:-PAD, PAD:-PAD, PAD:-PAD] if PAD > 0 else fault_mask
        vol_path = os.path.join(DATA_DIR, "validation", "seismic", f"{j:03d}.npy")
        mask_path = os.path.join(DATA_DIR, "validation", "fault", f"{j:03d}.npy")
        np.save(vol_path, cropped_volume.astype(np.float32))
        np.save(mask_path, cropped_mask.astype(np.uint8))
        total_vox = cropped_mask.size
        val_total_vox += total_vox
        if mask_mode == 0:
            fault_count = np.sum(cropped_mask != 0)
            val_class_counts[0] += (total_vox - fault_count)
            val_class_counts[1] += fault_count
            val_sum_class_pct[1] += (fault_count / total_vox * 100.0)
            val_sum_class_pct[0] += ((total_vox - fault_count) / total_vox * 100.0)
        else:
            normal_count = np.sum(cropped_mask == 1)
            inverse_count = np.sum(cropped_mask == 2)
            no_fault_count = total_vox - (normal_count + inverse_count)
            val_class_counts[0] += no_fault_count
            val_class_counts[1] += normal_count
            val_class_counts[2] += inverse_count
            val_sum_class_pct[0] += (no_fault_count / total_vox * 100.0)
            val_sum_class_pct[1] += (normal_count / total_vox * 100.0)
            val_sum_class_pct[2] += (inverse_count / total_vox * 100.0)
        cube_params = {
            'num_faults': len(fault_params_list),
            'noise_sigma': noise_sigma_val
        }
        cube_params['faults'] = fault_params_list
        val_params.append(cube_params)
        val_faults_all.extend(fault_params_list)
    # end for each validation cube

    # --- Compute pixel-class distribution statistics ---
    train_overall_pct = {}; train_mean_pct = {}
    val_overall_pct   = {}; val_mean_pct   = {}
    full_overall_pct  = {}; full_mean_pct  = {}

    # Training split stats
    if train_total_vox > 0:
        if mask_mode == 0:
            fault_pct = (train_class_counts[1] / train_total_vox) * 100.0
            train_overall_pct['no_fault'] = 100.0 - fault_pct
            train_overall_pct['fault']    = fault_pct
            mean_fault_pct = train_sum_class_pct[1] / train_count
            train_mean_pct['fault']    = mean_fault_pct
            train_mean_pct['no_fault'] = 100.0 - mean_fault_pct
        else:
            normal_pct = (train_class_counts[1] / train_total_vox) * 100.0
            inverse_pct = (train_class_counts[2] / train_total_vox) * 100.0
            train_overall_pct['normal']  = normal_pct
            train_overall_pct['inverse'] = inverse_pct
            train_overall_pct['no_fault'] = 100.0 - (normal_pct + inverse_pct)
            mean_normal_pct = train_sum_class_pct[1] / train_count
            mean_inverse_pct = train_sum_class_pct[2] / train_count
            train_mean_pct['normal']  = mean_normal_pct
            train_mean_pct['inverse'] = mean_inverse_pct
            train_mean_pct['no_fault'] = 100.0 - (mean_normal_pct + mean_inverse_pct)
    # Validation split stats
    if val_total_vox > 0:
        if mask_mode == 0:
            fault_pct = (val_class_counts[1] / val_total_vox) * 100.0
            val_overall_pct['no_fault'] = 100.0 - fault_pct
            val_overall_pct['fault']    = fault_pct
            mean_fault_pct = val_sum_class_pct[1] / max(val_count, 1)
            val_mean_pct['fault']    = mean_fault_pct
            val_mean_pct['no_fault'] = 100.0 - mean_fault_pct
        else:
            normal_pct = (val_class_counts[1] / val_total_vox) * 100.0
            inverse_pct = (val_class_counts[2] / val_total_vox) * 100.0
            val_overall_pct['normal']  = normal_pct
            val_overall_pct['inverse'] = inverse_pct
            val_overall_pct['no_fault'] = 100.0 - (normal_pct + inverse_pct)
            mean_normal_pct = val_sum_class_pct[1] / max(val_count, 1)
            mean_inverse_pct = val_sum_class_pct[2] / max(val_count, 1)
            val_mean_pct['normal']  = mean_normal_pct
            val_mean_pct['inverse'] = mean_inverse_pct
            val_mean_pct['no_fault'] = 100.0 - (mean_normal_pct + mean_inverse_pct)
    # Full dataset stats (train + validation combined)
    total_vox_all = train_total_vox + val_total_vox
    total_cubes = max(train_count + val_count, 1)
    if mask_mode == 0:
        total_fault_vox = train_class_counts[1] + val_class_counts[1]
        fault_pct_all = (total_fault_vox / total_vox_all * 100.0) if total_vox_all > 0 else 0.0
        full_overall_pct['no_fault'] = 100.0 - fault_pct_all
        full_overall_pct['fault']    = fault_pct_all
        mean_fault_pct_all = (train_sum_class_pct[1] + val_sum_class_pct[1]) / total_cubes
        full_mean_pct['fault']    = mean_fault_pct_all
        full_mean_pct['no_fault'] = 100.0 - mean_fault_pct_all
    else:
        total_normal_vox = train_class_counts[1] + val_class_counts[1]
        total_inverse_vox = train_class_counts[2] + val_class_counts[2]
        normal_pct_all = (total_normal_vox / total_vox_all * 100.0) if total_vox_all > 0 else 0.0
        inverse_pct_all = (total_inverse_vox / total_vox_all * 100.0) if total_vox_all > 0 else 0.0
        full_overall_pct['normal']  = normal_pct_all
        full_overall_pct['inverse'] = inverse_pct_all
        full_overall_pct['no_fault'] = 100.0 - (normal_pct_all + inverse_pct_all)
        mean_normal_pct_all = (train_sum_class_pct[1] + val_sum_class_pct[1]) / total_cubes
        mean_inverse_pct_all = (train_sum_class_pct[2] + val_sum_class_pct[2]) / total_cubes
        full_mean_pct['normal']  = mean_normal_pct_all
        full_mean_pct['inverse'] = mean_inverse_pct_all
        full_mean_pct['no_fault'] = 100.0 - (mean_normal_pct_all + mean_inverse_pct_all)

    # Assemble results into a dictionary
    all_stats_data = {
        'train': {
            'cube_level_params': train_params,
            'all_fault_params': train_faults_all,
            'pixel_dist': {'overall_pct': train_overall_pct, 'mean_pct': train_mean_pct}
        },
        'validation': {
            'cube_level_params': val_params,
            'all_fault_params': val_faults_all,
            'pixel_dist': {'overall_pct': val_overall_pct, 'mean_pct': val_mean_pct}
        }
    }
    full_cube_params = train_params + val_params
    full_fault_params = train_faults_all + val_faults_all
    all_stats_data['full'] = {
        'cube_level_params': full_cube_params,
        'all_fault_params': full_fault_params,
        'pixel_dist': {'overall_pct': full_overall_pct, 'mean_pct': full_mean_pct}
    }

    # Save full dataset statistics to a JSON file for reference
    stats_file = os.path.join(STATS_DIR, "statistics_full.json")
    try:
        import json
        with open(stats_file, "w") as sf:
            json.dump(all_stats_data['full'], sf, indent=4)
    except Exception as e:
        print(f"Warning: could not save statistics JSON: {e}")

    return all_stats_data