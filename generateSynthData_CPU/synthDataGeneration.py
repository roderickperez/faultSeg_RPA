import os
import numpy as np
from constants import *
# Where to read/write datasets (set from notebook)
BASE_OUT = None
def set_base_out(path: str):
    """Sets the root output directory for reading/writing cubes."""
    global BASE_OUT
    BASE_OUT = path
    
# --------------------------
def _uniform_int_distribution(n, low, high):
    """Deterministically spread counts uniform across [low, high] for n items."""
    if n <= 0:
        return []
    k = high - low + 1
    if n < k:
        return [int(v) for v in np.random.randint(low, high+1, size=n)]
    base = n // k
    rem  = n %  k
    counts = [base] * k
    for i in range(rem):
        counts[i] += 1
    vals = []
    for i, c in enumerate(counts):
        vals += [low + i] * c
    np.random.shuffle(vals)
    return vals

def ricker_wavelet(length=41, peak_freq=28.0):
    t = np.linspace(-1, 1, length)
    pf = float(peak_freq)
    pi2f2t2 = (np.pi**2) * (pf**2) * (t**2)
    w = (1 - 2*pi2f2t2) * np.exp(-pi2f2t2)
    return w / np.max(np.abs(w))

def conv1d_along_axis(vol, kernel, axis=0):
    vol = np.asarray(vol); kern = np.asarray(kernel)
    n = vol.shape[axis]; m = kern.size
    L = int(2 ** np.ceil(np.log2(n + m - 1)))
    V = np.fft.rfft(vol,  n=L, axis=axis)
    K = np.fft.rfft(kern, n=L)
    shape = [1]*vol.ndim; shape[axis] = K.shape[0]
    Kb = K.reshape(shape)
    Y = np.fft.irfft(V * Kb, n=L, axis=axis)
    start = (m - 1)//2; end = start + n
    sl = [slice(None)]*vol.ndim; sl[axis] = slice(start, end)
    return Y[tuple(sl)]

def sample_strike_deg(strike_range_deg, mode='random',
                      means=(45.0, 135.0), spread=12.0, weights=(0.5, 0.5)):
    if mode == 'random':
        return np.random.uniform(*strike_range_deg)
    p = np.array(weights, dtype=float); p /= p.sum()
    set_idx = np.random.choice([0, 1], p=p)
    base = means[set_idx]
    base += 180.0 if (np.random.rand() < 0.5) else 0.0
    return (base + np.random.uniform(-spread, spread)) % 360.0

def sample_orientation(dip_range_deg, strike_range_deg,
                       mode='uniform_normal',
                       strike_mode='random',
                       strike_means=(45.0, 135.0),
                       strike_spread=12.0,
                       strike_weights=(0.5, 0.5)):
    dip_min, dip_max = dip_range_deg
    dmin, dmax = np.deg2rad([dip_min, dip_max])
    cd = np.random.uniform(np.cos(dmax), np.cos(dmin))
    dip_deg = np.rad2deg(np.arccos(np.clip(cd, -1.0, 1.0)))
    strike_deg = sample_strike_deg(strike_range_deg,
                                   mode=strike_mode,
                                   means=strike_means,
                                   spread=strike_spread,
                                   weights=strike_weights)
    return dip_deg, strike_deg

# =========================
def generate_one_cube(num_faults):
    """Generate one cube and return cropped seismic, 2-px mask (ONLY), per-fault params, noise sigma."""
    # Step 1 — base reflectivity
    volume = np.zeros((NZ, NY, NX), dtype=float)
    reflectivity_trace = np.random.uniform(-1.0, 1.0, size=NZ)
    for z in range(NZ):
        volume[z, :, :] = reflectivity_trace[z]

    # Step 2 — folding (classic bumps)
    if apply_deformation:
        X = np.arange(NX); Y = np.arange(NY)
        xx, yy = np.meshgrid(X, Y, indexing='xy')
        L_xy = np.full((NY, NX), np.random.uniform(*classic_a0_range), dtype=float)
        if classic_keep_within_crop and PAD > 0:
            safe = int(classic_safe_margin_frac * PAD)
            cx_rng = (safe, NX - safe); cy_rng = (safe, NY - safe)
        else:
            cx_rng, cy_rng = (0, NX), (0, NY)
        for _ in range(int(classic_num_bumps)):
            sign = 1.0 if (np.random.rand() < classic_pos_amp_prob) else -1.0
            b_k  = sign * np.random.uniform(*classic_bk_range)
            c_k  = np.random.uniform(*cx_rng)
            d_k  = np.random.uniform(*cy_rng)
            s_k  = np.random.uniform(*classic_sigma_range)
            L_xy += b_k * np.exp(-(((xx - c_k)**2 + (yy - d_k)**2) / (2.0 * s_k**2)))
        # Polarity per cube
        if np.random.rand() < dome_up_probability:
            L_xy *= -1.0

        z_indices   = np.arange(NZ, dtype=float)
        depth_weight = (z_indices / max(NZ - 1, 1.0)) ** classic_depth_power
        try:
            from scipy.ndimage import map_coordinates
            Zmap = (z_indices[:, None, None]
                    + (classic_depth_scale * depth_weight)[:, None, None] * L_xy[None, :, :])
            Zmap = np.clip(Zmap, 0.0, NZ - 1.0)
            Zc = Zmap.reshape(1, -1)
            Yc = np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            Xc = np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            folded_flat = map_coordinates(volume, np.vstack([Zc, Yc, Xc]),
                                          order=3, mode='nearest')
            volume = folded_flat.reshape(NZ, NY, NX)
        except Exception:
            folded = np.empty_like(volume)
            for j in range(NY):
                for i in range(NX):
                    shift_values     = classic_depth_scale * depth_weight * L_xy[j, i]
                    source_positions = np.clip(z_indices + shift_values, 0.0, NZ - 1.0)
                    folded[:, j, i]  = np.interp(source_positions, z_indices, volume[:, j, i])
            volume = folded

    # Step 3 — shear (optional)
    if apply_shear:
        e0 = np.random.uniform(*e0_range)
        f  = np.random.uniform(*f_range)
        g  = np.random.uniform(*g_range)
        Z  = np.arange(NZ)[:, None, None].astype(float)
        Xg = np.arange(NX)[None, None, :]
        Yg = np.arange(NY)[None, :, None]
        S  = e0 + f * Xg + g * Yg
        Zmap = np.clip(Z + S, 0.0, NZ - 1.0)
        try:
            from scipy.ndimage import map_coordinates
            Zc = Zmap.reshape(1, -1)
            Yc = np.repeat(np.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            Xc = np.tile(np.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            sheared_flat = map_coordinates(volume, np.vstack([Zc, Yc, Xc]),
                                           order=3, mode='nearest')
            volume = sheared_flat.reshape(NZ, NY, NX)
        except Exception:
            sheared = np.empty_like(volume)
            z_idx = np.arange(NZ)
            for j in range(NY):
                for i in range(NX):
                    s2   = float(e0 + f * i + g * j)
                    src  = np.clip(z_idx + s2, 0.0, NZ - 1.0)
                    sheared[:, j, i] = np.interp(src, z_idx, volume[:, j, i])
            volume = sheared

    # Step 4 — faulting (ONLY 2-px training mask kept)
    fault_params_list = []
    fault_mask = np.zeros((NZ, NY, NX), dtype=np.uint8)    # 2-px mask
    if apply_faulting:
        Z = np.arange(NZ).astype(float)
        X2d, Y2d = np.meshgrid(np.arange(NX), np.arange(NY), indexing='xy')
        faulted_volume = volume.copy()

        accepted_faults, planes_z_maps = [], []
        attempts = 0
        orientation_sampling = 'uniform_normal'

        while len(accepted_faults) < num_faults and attempts < fault_max_proposals:
            attempts += 1
            dip_deg, strike_deg = sample_orientation(
                dip_range, strike_range, mode=orientation_sampling,
                strike_mode=strike_sampling_mode,
                strike_means=strike_two_set_means,
                strike_spread=strike_two_set_spread,
                strike_weights=strike_two_set_weights
            )
            fault_type_choice = np.random.choice(fault_types, p=np.array(fault_type_weights)/np.sum(fault_type_weights))
            fault_type_label  = 'Normal' if fault_type_choice == 'normal' else 'Reverse'
            dist_mode  = np.random.choice(fault_distribution_modes)
            max_slip   = np.random.uniform(*max_slip_range)

            phi     = np.deg2rad(strike_deg)
            theta   = np.deg2rad(dip_deg)
            dip_dir = phi + np.pi/2.0

            A = np.sin(theta)*np.cos(dip_dir)
            B = np.sin(theta)*np.sin(dip_dir)
            C = -np.cos(theta)
            if abs(C) < 1e-3:
                continue

            x0 = NX/2 + np.random.uniform(-0.1*NX, 0.1*NX)
            y0 = NY/2 + np.random.uniform(-0.1*NY, 0.1*NY)
            z0 = NZ/2 + np.random.uniform(-0.1*NZ, 0.1*NZ)
            D  = -(A*x0 + B*y0 + C*z0)

            strike_vec = np.array([np.sin(phi), -np.cos(phi), 0.0]); strike_vec /= np.linalg.norm(strike_vec)
            dip_vec    = np.cross([A,B,C], strike_vec);               dip_vec   /= np.linalg.norm(dip_vec)
            if dip_vec[2] < 0: dip_vec = -dip_vec

            z_plane_2d = -(A*X2d + B*Y2d + D) / C
            valid_mask_plane = np.isfinite(z_plane_2d) & (z_plane_2d >= 0) & (z_plane_2d <= NZ-1)
            if valid_mask_plane.mean() < fault_min_cut_fraction:
                continue

            ok = True
            for z_prev in planes_z_maps:
                both = valid_mask_plane & np.isfinite(z_prev)
                if not np.any(both):
                    continue
                too_close = (np.abs(z_plane_2d[both] - z_prev[both]) < fault_min_sep_z).mean()
                if too_close > fault_max_overlap_frac:
                    ok = False; break
            if not ok:
                continue

            relx = X2d - x0
            rely = Y2d - y0
            relz = z_plane_2d - z0
            u_map = relx*strike_vec[0] + rely*strike_vec[1] + relz*strike_vec[2]
            v_map = relx*dip_vec[0]    + rely*dip_vec[1]    + relz*dip_vec[2]

            # dip span using corners
            corners = np.array([[0,0,0],[NX,0,0],[0,NY,0],[NX,NY,0],
                                [0,0,NZ],[NX,0,NZ],[0,NY,NZ],[NX,NY,NZ]], float)
            relc = corners - np.array([x0,y0,z0])
            v_vals = relc @ dip_vec
            v_min, v_max = v_vals.min(), v_vals.max()
            v_span = max(v_max - v_min, 1e-6)

            accepted_faults.append(dict(
                A=A,B=B,C=C,D=D, strike_deg=strike_deg, dip_deg=dip_deg,
                fault_type=fault_type_label, dist_mode=dist_mode, max_slip=max_slip,
                z_plane_2d=z_plane_2d, u_map=u_map, v_map=v_map,
                v_min=v_min, v_max=v_max, v_span=v_span
            ))
            planes_z_maps.append(z_plane_2d)

        # apply accepted faults (per-trace)
        for fid, f in enumerate(accepted_faults, start=1):
            z_plane_2d = f['z_plane_2d']
            u_map, v_map = f['u_map'], f['v_map']
            v_min, v_max, v_span = f['v_min'], f['v_max'], f['v_span']
            fault_type_label, dist_mode, max_slip = f['fault_type'], f['dist_mode'], f['max_slip']

            if dist_mode == 'gaussian':
                Sigma_u = (u_map.max() - u_map.min()) / 3.0
                Sigma_v = v_span / 3.0
                slip_2d = max_slip * np.exp(-(u_map**2)/(2*Sigma_u**2) - (v_map**2)/(2*Sigma_v**2))
            else:
                if fault_type_label == 'Normal':
                    slip_2d = max_slip * (v_map - v_min) / v_span
                else:
                    slip_2d = max_slip * (v_max - v_map) / v_span
                slip_2d = np.clip(slip_2d, 0.0, max_slip)

            signed_offset_2d = slip_2d if fault_type_label == 'Normal' else -slip_2d
            label_this_fault = 1 if (mask_mode == 0 or fault_type_label == 'Normal') else 2

            new_vol        = np.empty_like(faulted_volume)
            new_fault_mask = np.zeros_like(fault_mask, dtype=np.uint8)

            for j in range(NY):
                z_plane_row = z_plane_2d[j]
                offset_row  = signed_offset_2d[j]
                for i in range(NX):
                    z_plane = float(z_plane_row[i])
                    off     = float(offset_row[i])

                    src = Z.copy()
                    if np.isfinite(z_plane):
                        hanging = src < z_plane
                        src[hanging] = src[hanging] + off
                    src = np.clip(src, 0.0, NZ - 1.0)

                    trace = faulted_volume[:, j, i]
                    new_vol[:, j, i] = np.interp(src, Z, trace)

                    # carry previous mask labels through the warp
                    src_nn = np.clip(np.round(src).astype(int), 0, NZ-1)
                    new_fault_mask[:, j, i] = fault_mask[src_nn, j, i]

                    # paint a 2-voxel-thick break at the fault plane (training mask)
                    if np.isfinite(z_plane) and (0.0 <= z_plane < NZ - 1):
                        z_low  = int(np.floor(z_plane))
                        z_high = z_low + 1
                        new_fault_mask[z_low,  j, i] = label_this_fault
                        new_fault_mask[z_high, j, i] = label_this_fault

            faulted_volume = new_vol
            fault_mask     = new_fault_mask

            applied_disp = max_slip if fault_type_label == 'Normal' else -max_slip
            fault_params_list.append({
                'fault_type': fault_type_label,
                'strike': f['strike_deg'],
                'dip': f['dip_deg'],
                'max_slip': max_slip,
                'applied_disp_signed': applied_disp,
                'A': f['A'], 'B': f['B'], 'C': f['C'], 'D': f['D']
            })

        volume = faulted_volume

    # Step 5 — band-limit + noise
    peak_freq = np.random.uniform(*wavelet_peak_freq_range)
    w = ricker_wavelet(length=wavelet_length, peak_freq=peak_freq)
    seismic = conv1d_along_axis(volume, w, axis=0)
    noise_sigma_val = None
    if apply_noise:
        if noise_type == 'gaussian':
            data_std  = seismic.max() - seismic.min()
            noise_std = noise_intensity * data_std
            seismic   = seismic + np.random.normal(0.0, noise_std, size=seismic.shape)
            noise_sigma_val = noise_std
        elif noise_type == 'uniform':
            rng = noise_intensity * (seismic.max() - seismic.min())
            seismic = seismic + np.random.uniform(-rng, +rng, size=seismic.shape)
        elif noise_type == 'speckle':
            seismic = seismic * np.random.normal(1.0, noise_intensity, size=seismic.shape)
        elif noise_type == 'salt_pepper':
            frac = noise_intensity
            nvox = seismic.size
            nnoisy = int(frac * nvox)
            if nnoisy > 0:
                coords = np.unravel_index(np.random.choice(nvox, size=nnoisy, replace=False), seismic.shape)
                half = nnoisy//2
                seismic[coords][:half] = seismic.min()
                seismic[coords][half:] = seismic.max()

    # Step 6 — crop & return (NO display mask)
    crop = PAD
    cropped_volume = seismic[crop:-crop, crop:-crop, crop:-crop] if crop > 0 else seismic
    cropped_mask   = fault_mask[crop:-crop, crop:-crop, crop:-crop] if crop > 0 else fault_mask

    return (cropped_volume.astype(np.float32),
            cropped_mask.astype(np.uint8),
            fault_params_list,
            noise_sigma_val)
    

def _accumulate_pixel_stats(mask, accum_counts, accum_pct_sums):
    total = mask.size
    if mask_mode == 0:
        fault = np.count_nonzero(mask)
        nof   = total - fault
        accum_counts[0] += nof
        accum_counts[1] += fault
        accum_pct_sums[0] += (nof   / total * 100.0)
        accum_pct_sums[1] += (fault / total * 100.0)
    else:
        normal  = np.sum(mask == 1)
        inverse = np.sum(mask == 2)
        nof     = total - (normal + inverse)
        accum_counts[0] += nof
        accum_counts[1] += normal
        accum_counts[2] += inverse
        accum_pct_sums[0] += (nof     / total * 100.0)
        accum_pct_sums[1] += (normal  / total * 100.0)
        accum_pct_sums[2] += (inverse / total * 100.0)

def _pct_summary(total_vox, counts, pct_sums, n_cubes):
    overall = {}
    mean    = {}
    if mask_mode == 0:
        fault_pct = (counts[1] / total_vox * 100.0) if total_vox>0 else 0.0
        overall['no_fault'] = 100.0 - fault_pct
        overall['fault']    = fault_pct
        mean_fault = pct_sums[1] / max(n_cubes,1)
        mean['fault']    = mean_fault
        mean['no_fault'] = 100.0 - mean_fault
    else:
        normal_pct  = (counts[1] / total_vox * 100.0) if total_vox>0 else 0.0
        inverse_pct = (counts[2] / total_vox * 100.0) if total_vox>0 else 0.0
        overall['normal']   = normal_pct
        overall['inverse']  = inverse_pct
        overall['no_fault'] = 100.0 - (normal_pct + inverse_pct)
        mean['no_fault'] = pct_sums[0] / max(n_cubes,1)
        mean['normal']   = pct_sums[1] / max(n_cubes,1)
        mean['inverse']  = pct_sums[2] / max(n_cubes,1)
    return overall, mean

# ---------- utilities (inline) ----------
def plot_histogram(data, ax, title, xlabel, bins='auto', xlim=None, hist_range=None, use_percentile=True):
    if data is None or len(data) == 0:
        return
    if use_percentile:
        low, high = np.percentile(data, [2, 98])
    else:
        low, high = (min(data), max(data)) if not xlim else xlim
    ax.hist(data, bins=bins, range=(low, high) if hist_range is None else hist_range,
            edgecolor='black', alpha=0.7)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel('Frequency')
    if xlim:
        ax.set_xlim(xlim)

def count_pixels(mask_cubes, mask_mode):
    total_voxels = 0
    no_fault_vox = normal_vox = inverse_vox = overlap_vox = 0
    pct_no_fault_list = []; pct_normal_list = []; pct_inverse_list = []; pct_overlap_list = []
    for mask in mask_cubes:
        n_vox = mask.size
        total_voxels += n_vox
        if mask_mode == 0:
            fault_count = np.sum(mask != 0)
            no_fault = n_vox - fault_count
            no_fault_vox += no_fault
            normal_vox   += fault_count
            pct_no_fault_list.append(no_fault / n_vox * 100.0)
            pct_normal_list.append(fault_count / n_vox * 100.0)
        else:
            n_normal  = np.sum(mask == 1)
            n_inverse = np.sum(mask == 2)
            n_overlap = np.sum(mask == 3) if np.any(mask == 3) else 0
            no_fault = n_vox - (n_normal + n_inverse + n_overlap)
            no_fault_vox += no_fault
            normal_vox   += n_normal
            inverse_vox  += n_inverse
            overlap_vox  += n_overlap
            pct_no_fault_list.append(no_fault / n_vox * 100.0)
            pct_normal_list.append(n_normal / n_vox * 100.0)
            pct_inverse_list.append(n_inverse / n_vox * 100.0)
            if n_overlap > 0:
                pct_overlap_list.append(n_overlap / n_vox * 100.0)

    overall_pct = {}
    mean_pct = {}
    if mask_mode == 0:
        overall_pct['no_fault'] = (no_fault_vox / total_voxels * 100.0) if total_voxels > 0 else 0.0
        overall_pct['fault']    = (normal_vox    / total_voxels * 100.0) if total_voxels > 0 else 0.0
        mean_fault = np.mean(pct_normal_list) if pct_normal_list else 0.0
        mean_pct['fault']    = mean_fault
        mean_pct['no_fault'] = 100.0 - mean_fault
    else:
        overall_pct['no_fault'] = (no_fault_vox / total_voxels * 100.0) if total_voxels > 0 else 0.0
        overall_pct['normal']  = (normal_vox   / total_voxels * 100.0) if total_voxels > 0 else 0.0
        overall_pct['inverse'] = (inverse_vox  / total_voxels * 100.0) if total_voxels > 0 else 0.0
        if overlap_vox > 0:
            overall_pct['overlap'] = (overlap_vox / total_voxels * 100.0)
        mean_normal   = np.mean(pct_normal_list)  if pct_normal_list  else 0.0
        mean_inverse  = np.mean(pct_inverse_list) if pct_inverse_list else 0.0
        mean_no_fault = np.mean(pct_no_fault_list) if pct_no_fault_list else 0.0
        mean_pct['normal']  = mean_normal
        mean_pct['inverse'] = mean_inverse
        mean_pct['no_fault'] = mean_no_fault
        if pct_overlap_list:
            mean_pct['overlap'] = np.mean(pct_overlap_list)
    return overall_pct, mean_pct

# synthDataGeneration.py
def _save_cube(split, idx, vol, mask2px, base_out=None):
    root = base_out or BASE_OUT
    if root is None:
        raise ValueError("BASE_OUT is not set. Call utilities.set_base_out(...) first or pass base_out=...")
    out_seis  = os.path.join(root, split, "seis")
    out_fault = os.path.join(root, split, "fault")
    os.makedirs(out_seis,  exist_ok=True)
    os.makedirs(out_fault, exist_ok=True)
    np.save(os.path.join(out_seis,  f"{idx:03d}.npy"), vol)
    np.save(os.path.join(out_fault, f"{idx:03d}.npy"), mask2px)

def load_cube_and_masks(split, index, base_out=None):
    root = base_out or BASE_OUT
    if root is None:
        raise ValueError("BASE_OUT is not set. Call utilities.set_base_out(...) first or pass base_out=...")
    vol_path   = os.path.join(root, split, "seis",  f"{index:03d}.npy")
    mask2_path = os.path.join(root, split, "fault", f"{index:03d}.npy")
    vol   = np.load(vol_path)
    mask2 = np.load(mask2_path)
    return vol, mask2


def collapse_to_display_mask(mask2px):
    """
    Collapse a 2-px-thick training mask into a 1-px display mask that
    aligns with the final (post-warp) seismic. Works with mask_mode 0 or 1.
    """
    NZ, NY, NX = mask2px.shape
    disp = np.zeros((NZ, NY, NX), dtype=np.uint8)
    for y in range(NY):
        for x in range(NX):
            col = mask2px[:, y, x]
            nz = np.flatnonzero(col)
            if nz.size == 0:
                continue
            # split into contiguous runs
            splits = np.where(np.diff(nz) > 1)[0] + 1
            segments = np.split(nz, splits)
            for seg in segments:
                zc = int(np.round(seg.mean()))  # center of the run
                if mask_mode == 0:
                    lbl = 1
                else:
                    vals = col[seg]
                    lbl = 2 if np.any(vals == 2) else 1  # preserve class
                disp[zc, y, x] = lbl
    return disp


def reconstruct_fault_planes_from_params(cube_fault_params, mask_shape, pad=None):
    if pad is None:
        pad = PAD
    """
    Rebuild planar Z(x,y) inside the CROPPED cube coordinates (0..NZc-1 etc.).
    The original planes were defined in FULL coords; map (x',y') -> (x'+PAD, y'+PAD),
    then shift Z by -PAD (because we cropped PAD at top).
    """
    NZc, NYc, NXc = mask_shape
    Xc, Yc = np.meshgrid(np.arange(NXc), np.arange(NYc), indexing='xy')

    fault_surfaces = []
    for fid, f in enumerate(cube_fault_params, start=1):
        A, B, C, D = f['A'], f['B'], f['C'], f['D']
        if abs(C) < 1e-9:
            Zp = None
        else:
            # map cropped (x',y') to full (x'+pad, y'+pad), then subtract pad in z
            Z_full = -(A * (Xc + pad) + B * (Yc + pad) + D) / C
            Z_crop = Z_full - pad

            Zp = Z_crop.astype(float)
            Zp[(Zp < 0) | (Zp > NZc - 1)] = np.nan

        ftype = f.get('fault_type', 'Normal')
        label = 1 if (mask_mode == 0 or ftype == 'Normal') else 2
        if mask_mode == 0:
            color, name = RGBA_RED, f"Fault {fid}"
        else:
            if ftype == 'Normal':
                color, name = RGBA_GREEN, f"Fault {fid} (Normal)"
            else:
                color, name = RGBA_PURPLE, f"Fault {fid} (Reverse)"

        fault_surfaces.append({'X': Xc, 'Y': Yc, 'Z': Zp,
                               'color': color, 'name': name,
                               'fault_type': ftype, 'label': label})
    return fault_surfaces


def display_mask_from_planes(planes, shape):
    """Create a 1-px (Z) mask from reconstructed planes (for overlays only)."""
    NZc, NYc, NXc = shape
    disp = np.zeros((NZc, NYc, NXc), dtype=np.uint8)
    for p in planes:
        Zp = p['Z']
        lbl = int(p.get('label', 1))
        if Zp is None:
            continue
        ny, nx = Zp.shape
        ys, xs = np.where(np.isfinite(Zp))
        if ys.size == 0:
            continue
        zz = np.floor(Zp[ys, xs]).astype(int)
        keep = (zz >= 0) & (zz < NZc)
        disp[zz[keep], ys[keep], xs[keep]] = lbl
    return disp
