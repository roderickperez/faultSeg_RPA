import os
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi  # GPU SciPy-compatible ndimage
from constants import *

xp = cp  # Use cupy for GPU acceleration

# Where to read/write datasets (set from notebook)
BASE_OUT = None
def set_base_out(path: str):
    """Sets the root output directory for reading/writing cubes."""
    global BASE_OUT
    BASE_OUT = path

def to_cpu(a):
    """Return a NumPy array if `a` is a CuPy array; otherwise return `a` unchanged."""
    try:
        import cupy as _cp
        if isinstance(a, _cp.ndarray):
            return _cp.asnumpy(a)
    except Exception:
        pass
    return a


# --------------------------

def _scalar_to_float(x):
    import numpy as _np
    try:
        import cupy as _cp
        if isinstance(x, _cp.ndarray):
            return float(_cp.asnumpy(x))
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(_np.asarray(x).squeeze())

def _uniform_int_distribution(n, low, high):
    """Deterministically spread counts uniform across [low, high] for n items."""
    if n <= 0:
        return []
    k = high - low + 1
    if n < k:
        # draw n values uniformly and return a Python list
        return list(np.random.randint(low, high + 1, size=n))

    base, rem = divmod(n, k)
    counts = [base] * k
    for i in range(rem):
        counts[i] += 1

    vals = []
    for i, c in enumerate(counts):
        vals += [low + i] * c

    a = cp.asarray(vals)
    cp.random.shuffle(a)          # operates on a CuPy array
    vals = cp.asnumpy(a).tolist() # back to a Python list
    return vals

def ricker_wavelet(length=41, peak_freq=28.0):
    t  = cp.linspace(-1, 1, length, dtype=cp.float32)
    pf = cp.asarray(peak_freq, dtype=cp.float32)  # robust to Python/NumPy/CuPy scalars
    pi2f2t2 = (cp.pi**2) * (pf**2) * (t**2)
    w = (1 - 2*pi2f2t2) * cp.exp(-pi2f2t2)
    return w / cp.max(cp.abs(w))

def conv1d_along_axis(vol, kernel, axis=0):
    vol  = cp.asarray(vol)
    kern = cp.asarray(kernel)
    n = vol.shape[axis]
    m = kern.size
    L = int(2 ** cp.ceil(cp.log2(n + m - 1)))
    V = cp.fft.rfft(vol,  n=L, axis=axis)
    K = cp.fft.rfft(kern, n=L)
    shape = [1] * vol.ndim
    shape[axis] = K.shape[0]
    Kb = K.reshape(shape)
    Y = cp.fft.irfft(V * Kb, n=L, axis=axis)
    start = (m - 1) // 2
    end   = start + n
    sl = [slice(None)] * vol.ndim
    sl[axis] = slice(start, end)
    return Y[tuple(sl)]


def sample_strike_deg(strike_range_deg, mode='random',
                      means=(45.0, 135.0), spread=12.0, weights=(0.5, 0.5)):
    assert hasattr(strike_range_deg, "__len__") and len(strike_range_deg) == 2, \
        "strike_range_deg must be a (min, max) pair"
    if mode == 'random':
        # return a plain float (avoids later NumPy/CuPy mixing edge cases)
        return float(cp.asnumpy(cp.random.uniform(*strike_range_deg)))

    # p must be a CuPy array and length must match 'a'
    p = cp.asarray(weights, dtype=cp.float32)
    p /= p.sum()

    # RandomState.choice needs 'size'; sample index then convert to int
    set_idx = int(cp.random.choice(cp.array([0, 1], dtype=cp.int32),
                                   size=1, replace=True, p=p).item())

    base = float(means[set_idx])
    # Avoid truth-testing a CuPy 0-D; cast to Python float
    if float(cp.asnumpy(cp.random.rand())) < 0.5:
        base += 180.0
    return (base + float(cp.asnumpy(cp.random.uniform(-spread, spread)))) % 360.0


def sample_orientation(dip_range_deg, strike_range_deg,
                       mode='uniform_normal',
                       strike_mode='random',
                       strike_means=(45.0, 135.0),
                       strike_spread=12.0,
                       strike_weights=(0.5, 0.5)):
    assert hasattr(dip_range_deg, "__len__") and len(dip_range_deg) == 2, \
        "dip_range_deg must be a (min, max) pair"
    assert hasattr(strike_range_deg, "__len__") and len(strike_range_deg) == 2, \
        "strike_range_deg must be a (min, max) pair"
    dip_min, dip_max = dip_range_deg
    dmin, dmax = np.deg2rad([dip_min, dip_max])
    # ensure pure Python/NumPy scalar
    cd = float(cp.asnumpy(cp.random.uniform(np.cos(dmax), np.cos(dmin))))
    dip_deg = float(np.rad2deg(np.arccos(np.clip(cd, -1.0, 1.0))))
    strike_deg = sample_strike_deg(
        strike_range_deg,
        mode=strike_mode,
        means=strike_means,
        spread=strike_spread,
        weights=strike_weights,
    )
    return dip_deg, strike_deg

def _warp_volume_and_mask_along_z(vol_in, mask_in, z_plane_2d, offset_2d):
    """Vectorized Z-axis warp on GPU: linear interp for volume, NN for mask."""
    vol_in   = cp.asarray(vol_in)
    mask_in  = cp.asarray(mask_in)
    z0       = cp.asarray(z_plane_2d, dtype=cp.float32)[None, :, :]   # (1,NY,NX)
    off      = cp.asarray(offset_2d, dtype=cp.float32)[None, :, :]    # (1,NY,NX)

    NZ, NY, NX = vol_in.shape
    z = cp.arange(NZ, dtype=cp.float32)[:, None, None]                # (NZ,1,1)

    src = cp.where(cp.isfinite(z0) & (z < z0), z + off, z)
    src = cp.clip(src, 0.0, NZ - 1.0)                                 # (NZ,NY,NX)

    zlo = cp.floor(src).astype(cp.int32)
    zhi = cp.clip(zlo + 1, 0, NZ - 1)
    w   = src - zlo

    v0 = cp.take_along_axis(vol_in, zlo, axis=0)
    v1 = cp.take_along_axis(vol_in, zhi, axis=0)
    vol_out = (1.0 - w) * v0 + w * v1

    src_nn  = cp.clip(cp.rint(src).astype(cp.int32), 0, NZ - 1)
    mask_out = cp.take_along_axis(mask_in, src_nn, axis=0)

    return vol_out, mask_out


def _paint_break_2px(mask3d, z_plane_2d, label_val):
    """Paint the 2-voxel-thick training mask at the break."""
    NZ, NY, NX = mask3d.shape
    z0 = cp.asarray(z_plane_2d, dtype=cp.float32)
    valid = cp.isfinite(z0) & (z0 >= 0) & (z0 < NZ - 1)
    if not bool(valid.any()):
        return mask3d
    ys, xs = cp.nonzero(valid)
    zlow = cp.floor(z0[ys, xs]).astype(cp.int32)
    mask3d[zlow,     ys, xs] = cp.uint8(label_val)
    mask3d[zlow + 1, ys, xs] = cp.uint8(label_val)
    return mask3d


# =========================
def generate_one_cube(num_faults):
    """Generate one cube and return cropped seismic, 2-px mask (ONLY), per-fault params, noise sigma."""
    # Step 1 — base reflectivity
    volume = cp.zeros((NZ, NY, NX), dtype=float)
    reflectivity_trace = cp.random.uniform(-1.0, 1.0, size=NZ)
    for z in range(NZ):
        volume[z, :, :] = reflectivity_trace[z]

    # Step 2 — folding (classic bumps)
    if apply_deformation:
        X = cp.arange(NX, dtype=cp.float32)
        Y = cp.arange(NY, dtype=cp.float32)
        xx, yy = cp.meshgrid(X, Y, indexing='xy')

        # make the fill a Python float (not a 0-D CuPy array), then create on GPU
        a0 = float(cp.asnumpy(cp.random.uniform(*classic_a0_range)))
        L_xy = cp.full((NY, NX), a0, dtype=cp.float32)

        z_indices   = cp.arange(NZ, dtype=cp.float32)
        depth_weight = (z_indices / max(NZ - 1, 1.0)) ** classic_depth_power
        try:
            Zmap = (z_indices[:, None, None]
                    + (classic_depth_scale * depth_weight)[:, None, None] * L_xy[None, :, :])
            Zmap = cp.clip(Zmap, 0.0, NZ - 1.0)

            Zc = Zmap.reshape(1, -1)
            Yc = cp.repeat(cp.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            Xc = cp.tile(cp.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)

            folded_flat = cpx_ndi.map_coordinates(volume, cp.vstack([Zc, Yc, Xc]),
                                                order=3, mode='nearest')
            volume = folded_flat.reshape(NZ, NY, NX)
        
        except Exception:
            # CPU fallback (only if GPU ndimage interpolation is unavailable)
            vol_np = cp.asnumpy(volume)
            z_idx_np = np.arange(NZ, dtype=float)
            depth_weight_np = cp.asnumpy(depth_weight)
            L_xy_np = cp.asnumpy(L_xy)

            folded = np.empty_like(vol_np)
            for j in range(NY):
                for i in range(NX):
                    shift = float(classic_depth_scale) * depth_weight_np * float(L_xy_np[j, i])
                    src_positions = np.clip(z_idx_np + shift, 0.0, NZ - 1.0)
                    folded[:, j, i] = np.interp(src_positions, z_idx_np, vol_np[:, j, i])
            volume = cp.asarray(folded)

    # Step 3 — shear (optional)
    if apply_shear:
        e0 = cp.random.uniform(*e0_range)
        f  = cp.random.uniform(*f_range)
        g  = cp.random.uniform(*g_range)
        
        Z  = cp.arange(NZ, dtype=cp.float32)[:, None, None]
        Xg = cp.arange(NX, dtype=cp.float32)[None, None, :]
        Yg = cp.arange(NY, dtype=cp.float32)[None, :, None]
        shear_map = e0 + f * Xg + g * Yg
        Zmap = cp.clip(Z + shear_map, 0.0, NZ - 1.0)
        try:
            Zc = Zmap.reshape(1, -1)
            Yc = cp.repeat(cp.arange(NY), NX)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            Xc = cp.tile(cp.arange(NX), NY)[None, :].repeat(NZ, axis=1).reshape(1, -1)
            sheared_flat = cpx_ndi.map_coordinates(volume, cp.vstack([Zc, Yc, Xc]),
                                                order=3, mode='nearest')
            volume = sheared_flat.reshape(NZ, NY, NX)

        except Exception:
            # CPU fallback (only if GPU ndimage interpolation is unavailable)
            vol_np = cp.asnumpy(volume)
            z_idx_np = np.arange(NZ, dtype=float)

            sheared = np.empty_like(vol_np)
            for j in range(NY):
                for i in range(NX):
                    s2 = float(e0 + f * i + g * j)
                    src = np.clip(z_idx_np + s2, 0.0, NZ - 1.0)
                    sheared[:, j, i] = np.interp(src, z_idx_np, vol_np[:, j, i])
            volume = cp.asarray(sheared)


    # Step 4 — faulting (ONLY 2-px training mask kept)
    fault_params_list = []
    fault_mask = cp.zeros((NZ, NY, NX), dtype=np.uint8)    # 2-px mask
    if apply_faulting:
        Z = cp.arange(NZ, dtype=cp.float32)
        X2d, Y2d = cp.meshgrid(cp.arange(NX), cp.arange(NY), indexing='xy')

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
            # fault type
            p_ft = cp.asarray(fault_type_weights, dtype=cp.float32)
            p_ft /= p_ft.sum()
            ft_idx = int(cp.random.choice(cp.arange(len(fault_types)), size=1, replace=True, p=p_ft).item())
            fault_type_choice = fault_types[ft_idx]
            fault_type_label  = 'Normal' if fault_type_choice == 'normal' else 'Reverse'  # <-- add this

            # distribution mode
            dm_idx = int(cp.random.choice(cp.arange(len(fault_distribution_modes)), size=1, replace=True).item())
            dist_mode = fault_distribution_modes[dm_idx]

            max_slip   = cp.random.uniform(*max_slip_range)

            phi     = np.deg2rad(strike_deg)
            theta   = np.deg2rad(dip_deg)
            dip_dir = phi + np.pi/2.0

            A = np.sin(theta)*np.cos(dip_dir)
            B = np.sin(theta)*np.sin(dip_dir)
            C = -np.cos(theta)
            if abs(C) < 1e-3:
                continue

            x0 = NX/2 + cp.random.uniform(-0.1*NX, 0.1*NX)
            y0 = NY/2 + cp.random.uniform(-0.1*NY, 0.1*NY)
            z0 = NZ/2 + cp.random.uniform(-0.1*NZ, 0.1*NZ)
            D  = -(A*x0 + B*y0 + C*z0)

            # --- strike & dip vectors: do in NumPy first, then move to GPU
            strike_vec_np = np.array([np.sin(phi), -np.cos(phi), 0.0], dtype=np.float32)
            strike_vec_np /= (np.linalg.norm(strike_vec_np) + 1e-12)

            nvec_np = np.array([float(A), float(B), float(C)], dtype=np.float32)
            dip_vec_np = np.cross(nvec_np, strike_vec_np)
            dip_vec_np /= (np.linalg.norm(dip_vec_np) + 1e-12)
            if dip_vec_np[2] < 0:
                dip_vec_np = -dip_vec_np

            # move to GPU as float32 for downstream vectorized math
            strike_vec = cp.asarray(strike_vec_np, dtype=cp.float32)
            dip_vec    = cp.asarray(dip_vec_np,    dtype=cp.float32)

            z_plane_2d = -(A*X2d + B*Y2d + D) / C
            valid_mask_plane = cp.isfinite(z_plane_2d) & (z_plane_2d >= 0) & (z_plane_2d <= NZ-1)
            if valid_mask_plane.mean() < fault_min_cut_fraction:
                continue

            ok = True
            for z_prev in planes_z_maps:
                both = valid_mask_plane & cp.isfinite(z_prev)
                if not bool(both.any()):
                    continue
                too_close = (cp.abs(z_plane_2d[both] - z_prev[both]) < fault_min_sep_z).mean()

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
            corners = cp.asarray([[0,0,0],[NX,0,0],[0,NY,0],[NX,NY,0],[0,0,NZ],[NX,0,NZ],[0,NY,NZ],[NX,NY,NZ]], dtype=cp.float32)
            relc = corners - cp.asarray([x0, y0, z0], dtype=cp.float32)
            v_vals = relc @ dip_vec
            v_min = float(cp.asnumpy(v_vals.min()))
            v_max = float(cp.asnumpy(v_vals.max()))
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
                slip_2d = max_slip * cp.exp(-(u_map**2)/(2*Sigma_u**2) - (v_map**2)/(2*Sigma_v**2))

            else:
                if fault_type_label == 'Normal':
                    slip_2d = max_slip * (v_map - v_min) / v_span
                else:
                    slip_2d = max_slip * (v_max - v_map) / v_span
                slip_2d = cp.clip(slip_2d, 0.0, max_slip)

            signed_offset_2d = slip_2d if fault_type_label == 'Normal' else -slip_2d
            label_this_fault = 1 if (mask_mode == 0 or fault_type_label == 'Normal') else 2

            # Vectorized Z-warp (linear interp for volume, NN for mask)
            new_vol, new_fault_mask = _warp_volume_and_mask_along_z(
                faulted_volume, fault_mask, z_plane_2d, signed_offset_2d
            )
            # Paint 2-px break for this fault (label_this_fault)
            new_fault_mask = _paint_break_2px(new_fault_mask, z_plane_2d, label_this_fault)

            faulted_volume = new_vol
            fault_mask     = new_fault_mask

            applied_disp = max_slip if fault_type_label == 'Normal' else -max_slip
            fault_params_list.append({
                'fault_type': fault_type_label,
                'strike': float(f['strike_deg']),
                'dip': float(f['dip_deg']),
                'max_slip': float(max_slip),
                'applied_disp_signed': float(applied_disp),
                'A': float(f['A']), 'B': float(f['B']), 'C': float(f['C']), 'D': float(f['D']),
            })

        volume = faulted_volume

    # Step 5 — band-limit + noise
    peak_freq = float(cp.asnumpy(cp.random.uniform(*wavelet_peak_freq_range)))
    w = ricker_wavelet(length=wavelet_length, peak_freq=peak_freq)
    seismic = conv1d_along_axis(volume, w, axis=0)
    noise_sigma_val = None
    if apply_noise:
        if noise_type == 'gaussian':
            data_std  = seismic.max() - seismic.min()
            noise_std = noise_intensity * data_std
            seismic   = seismic + cp.random.normal(0.0, noise_std, size=seismic.shape)
            noise_sigma_val = noise_std
        elif noise_type == 'uniform':
            rng = noise_intensity * (seismic.max() - seismic.min())
            seismic = seismic + cp.random.uniform(-rng, +rng, size=seismic.shape)
        elif noise_type == 'speckle':
            seismic = seismic * cp.random.normal(1.0, noise_intensity, size=seismic.shape)
        elif noise_type == 'salt_pepper':
            frac = noise_intensity
            nvox = seismic.size
            nnoisy = int(frac * nvox)
            if nnoisy > 0:
                idx = cp.random.choice(nvox, size=nnoisy, replace=False)
                coords = cp.unravel_index(idx, seismic.shape)
                half = nnoisy//2
                seismic[coords][:half] = seismic.min()
                seismic[coords][half:] = seismic.max()

    # Step 6 — crop & return (NO display mask)
    crop = PAD
    cropped_volume = seismic[crop:-crop, crop:-crop, crop:-crop] if crop > 0 else seismic
    cropped_mask   = fault_mask[crop:-crop, crop:-crop, crop:-crop] if crop > 0 else fault_mask

    return (to_cpu(cropped_volume).astype(np.float32, copy=False),
        to_cpu(cropped_mask).astype(np.uint8,  copy=False),
        fault_params_list,
        noise_sigma_val)
    
def _save_cube(split, idx, vol, mask2px, base_out=None):
    root = base_out or BASE_OUT
    if root is None:
        raise ValueError("BASE_OUT is not set. Call utilities.set_base_out(...) first or pass base_out=...")
    v = to_cpu(vol).astype(np.float32, copy=False)
    m = to_cpu(mask2px).astype(np.uint8,  copy=False)
    np.save(os.path.join(root, split, "seismic", f"{idx:03d}.npy"), v)
    np.save(os.path.join(root, split, "fault",   f"{idx:03d}.npy"), m)

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

def load_cube_and_masks(split, index, base_out=None):
    root = base_out or BASE_OUT
    if root is None:
        raise ValueError("BASE_OUT is not set. Call utilities.set_base_out(...) first or pass base_out=...")
    vol_path   = os.path.join(root, split, "seismic", f"{index:03d}.npy")
    mask2_path = os.path.join(root, split, "fault",   f"{index:03d}.npy")
    vol   = np.load(vol_path)
    mask2 = np.load(mask2_path)
    return vol, mask2

def collapse_to_display_mask(mask2px):
    """
    Collapse a 2-px-thick training mask into a 1-px display mask (NumPy output).
    """
    mask = to_cpu(mask2px)
    NZ, NY, NX = mask.shape
    disp = np.zeros((NZ, NY, NX), dtype=np.uint8)
    for y in range(NY):
        for x in range(NX):
            col = mask[:, y, x]
            nz = np.flatnonzero(col)
            if nz.size == 0:
                continue
            splits = np.where(np.diff(nz) > 1)[0] + 1
            for seg in np.split(nz, splits):
                zc = int(round(seg.mean()))
                if mask_mode == 0:
                    lbl = 1
                else:
                    vals = col[seg]
                    lbl = 2 if np.any(vals == 2) else 1
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
        A = _scalar_to_float(f['A'])
        B = _scalar_to_float(f['B'])
        C = _scalar_to_float(f['C'])
        D = _scalar_to_float(f['D'])

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
        ys, xs = np.where(np.isfinite(Zp))
        if ys.size == 0:
            continue
        zz = np.floor(Zp[ys, xs]).astype(int)
        keep = (zz >= 0) & (zz < NZc)
        disp[zz[keep], ys[keep], xs[keep]] = lbl
    return disp


