import numpy as np
import cupy as cp
import plotly.graph_objects as go
from constants import EPS_X, EPS_Y, EPS_Z, RGBA_RED, RGBA_GREEN, RGBA_PURPLE, mask_mode
from plotly.subplots import make_subplots
from utilities import (
    load_cube_and_masks,
    reconstruct_fault_planes_from_params,
    display_mask_from_planes,
)
from ipywidgets import interact, IntSlider, Checkbox, FloatSlider

xp = cp  # Use cupy for GPU acceleration

def add_plane_slice_intersections(fig, planes, inline, xline, time):
    """
    Draw intersections between reconstructed planes and the three slice surfaces.
    Uses 0.5-center offsets so lines land on the same cell centers as the slice textures.
    """
    if not planes:
        return

    for p in planes:
        Zp = p.get("Z")
        if Zp is None:
            continue

        ny, nx = Zp.shape
        half = 0.5

        # ---------- Inline (Y = inline) ----------
        if 0 <= inline < ny:
            z_line = Zp[inline, :]
            valid = np.isfinite(z_line)
            if np.count_nonzero(valid) >= 2:
                xs = np.arange(nx, dtype=float)[valid] + half
                ys = np.full(xs.shape, float(inline) + half + EPS_Y)
                zs = z_line[valid]
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=p["color"], width=6),
                    name=f"{p['name']} ∩ Inline",
                    showlegend=False
                ))

        # ---------- Xline (X = xline) ----------
        if 0 <= xline < nx:
            z_col = Zp[:, xline]
            valid = np.isfinite(z_col)
            if np.count_nonzero(valid) >= 2:
                ys = np.arange(ny, dtype=float)[valid] + half
                xs = np.full(ys.shape, float(xline) + half + EPS_X)
                zs = z_col[valid]
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=p["color"], width=6),
                    name=f"{p['name']} ∩ Xline",
                    showlegend=False
                ))

        # ---------- Time (Z = time) ----------
        # Find a sub-voxel iso-line of Zp == time via 1D crossings per row.
        xs_all, ys_all = [], []
        for yy in range(ny):
            row = Zp[yy, :]
            valid = np.isfinite(row)
            if np.count_nonzero(valid) < 2:
                continue

            r = row - float(time)
            # indices k where segment [k, k+1] crosses zero (sign change or exact zero)
            sign = np.sign(r)
            cross = np.where((sign[:-1] * sign[1:] <= 0) & np.isfinite(r[:-1]) & np.isfinite(r[1:]))[0]
            if cross.size == 0:
                continue

            # Linear interp for sub-voxel X
            for k in cross:
                z0, z1 = r[k], r[k + 1]
                if z1 == z0:
                    t_param = 0.0
                else:
                    t_param = -z0 / (z1 - z0)
                x_interp = k + float(np.clip(t_param, 0.0, 1.0))
                xs_all.append(x_interp + half)
                ys_all.append(yy + half)

        if len(xs_all) >= 2:
            # Sort by Y to make a cleaner polyline; keep Z constant at 'time'
            order = np.argsort(ys_all)
            xs_sorted = np.asarray(xs_all, dtype=float)[order]
            ys_sorted = np.asarray(ys_all, dtype=float)[order]
            zs_sorted = np.full_like(xs_sorted, float(time) + EPS_Z)

            fig.add_trace(go.Scatter3d(
                x=xs_sorted, y=ys_sorted, z=zs_sorted,
                mode="lines",
                line=dict(color=p["color"], width=6),
                name=f"{p['name']} ∩ Time",
                showlegend=False
            ))

def mask_heatmap_overlay(mask_slice, mode=None):
    mode = mask_mode if mode is None else mode
    if mode == 0:
        z = (mask_slice != 0).astype(float)
        colorscale = [
            [0.0,  "rgba(0,0,0,0)"],
            [0.499,"rgba(0,0,0,0)"],
            [0.5,  "rgba(255,0,0,1)"],
            [1.0,  "rgba(255,0,0,1)"],
        ]
        return go.Heatmap(z=z, colorscale=colorscale, zmin=0, zmax=1, showscale=False, zsmooth=False)
    else:
        z = mask_slice.astype(float)/2.0
        colorscale = [
            [0.00,"rgba(0,0,0,0)"], [0.33,"rgba(0,0,0,0)"],
            [0.50, RGBA_GREEN], [0.66, RGBA_GREEN],
            [0.99, RGBA_PURPLE], [1.00, RGBA_PURPLE],
        ]
        return go.Heatmap(z=z, colorscale=colorscale, zmin=0, zmax=1, showscale=False, zsmooth=False)

def make_3d_fig_seismic(seis, planes, vmin, vmax, inline, xline, time,
                        show_intersections=False, show_planes=True, plane_opacity=0.55, title="3D Seismic"):
    NZ, NY, NX = seis.shape
    fig = go.Figure()
    # inline
    X_inl, Z_inl = np.meshgrid(np.arange(NX), np.arange(NZ))
    Y_inl = np.full_like(X_inl, float(inline))
    fig.add_trace(go.Surface(x=X_inl, y=Y_inl, z=Z_inl,
                             surfacecolor=seis[:, inline, :], colorscale="Gray",
                             cmin=vmin, cmax=vmax, showscale=False, name="Inline"))
    # xline
    Y_xl, Z_xl = np.meshgrid(np.arange(NY), np.arange(NZ))
    X_xl = np.full_like(Y_xl, float(xline))
    fig.add_trace(go.Surface(x=X_xl, y=Y_xl, z=Z_xl,
                             surfacecolor=seis[:, :, xline], colorscale="Gray",
                             cmin=vmin, cmax=vmax, showscale=False, name="Xline"))
    # time
    X_t, Y_t = np.meshgrid(np.arange(NX), np.arange(NY))
    Z_t = np.full_like(X_t, float(time))
    fig.add_trace(go.Surface(x=X_t, y=Y_t, z=Z_t,
                             surfacecolor=seis[time, :, :], colorscale="Gray",
                             cmin=vmin, cmax=vmax, showscale=False, name="Time"))
    if show_intersections:
        add_plane_slice_intersections(fig, planes, inline, xline, time)
    if show_planes and planes:
        plane_kw = dict(showscale=False, opacity=float(plane_opacity), hoverinfo="skip",
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))
        for p in planes:
            if p["Z"] is None: continue
            SC = np.ones_like(p["Z"]); cs = [[0.0, p["color"]], [1.0, p["color"]]]
            fig.add_trace(go.Surface(x=p["X"], y=p["Y"], z=p["Z"],
                                     surfacecolor=SC, colorscale=cs, name=p["name"], **plane_kw))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig

def _mask_surface(mask_2d, plane_coord, const_index, eps=0.0):
    if plane_coord == 'z':   # XY
        ny, nx = mask_2d.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny)); Z = np.full((ny, nx), float(const_index) + eps)
    elif plane_coord == 'y': # XZ
        nz, nx = mask_2d.shape
        X, Z = np.meshgrid(np.arange(nx), np.arange(nz)); Y = np.full((nz, nx), float(const_index) + eps)
    else:                    # YZ
        nz, ny = mask_2d.shape
        Y, Z = np.meshgrid(np.arange(ny), np.arange(nz)); X = np.full((nz, ny), float(const_index) + eps)
    if mask_mode == 0:
        sc = (mask_2d > 0).astype(float)
        colorscale = [[0.0,"rgb(0,0,0)"], [0.499,"rgb(0,0,0)"], [0.50, RGBA_RED], [1.0, RGBA_RED]]
        return X, Y, Z, sc, colorscale, 0.0, 1.0
    else:  # mask_mode == 1
        sc = (mask_2d.astype(float) / 2.0)   # 0 (bg), 0.5 (Normal), 1.0 (Reverse)

        # Hard steps to prevent green halos around purple:
        colorscale = [
            [0.00, "rgb(0,0,0)"],  [0.49, "rgb(0,0,0)"],   # background
            [0.50, RGBA_GREEN],    [0.51, RGBA_GREEN],     # Normal only at ~0.50
            [0.99, RGBA_PURPLE],   [1.00, RGBA_PURPLE],    # Reverse for >0.51
        ]
        return X, Y, Z, sc, colorscale, 0.0, 1.0

def make_3d_fig_mask(mask, planes, inline, xline, time,
                     show_planes=True, plane_opacity=0.55, show_intersections=False, title="3D Fault Mask"):
    NZ, NY, NX = mask.shape
    fig = go.Figure()
    common_kw = dict(showscale=False, opacity=1.0, hoverinfo="skip",
                     lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))
    mk_inl = mask[:, inline, :]
    X,Y,Z,SC,CS,cmin,cmax = _mask_surface(mk_inl, plane_coord='y', const_index=inline)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=SC, colorscale=CS, cmin=cmin, cmax=cmax, name="Inline mask", **common_kw))
    mk_xl = mask[:, :, xline]
    X,Y,Z,SC,CS,cmin,cmax = _mask_surface(mk_xl, plane_coord='x', const_index=xline)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=SC, colorscale=CS, cmin=cmin, cmax=cmax, name="Xline mask", **common_kw))
    mk_t = mask[time, :, :]
    X,Y,Z,SC,CS,cmin,cmax = _mask_surface(mk_t, plane_coord='z', const_index=time)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=SC, colorscale=CS, cmin=cmin, cmax=cmax, name="Time mask", **common_kw))
    if show_intersections:
        add_plane_slice_intersections(fig, planes, inline, xline, time)
    if show_planes and planes:
        plane_kw = dict(showscale=False, opacity=float(plane_opacity), hoverinfo="skip",
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))
        for p in planes:
            if p["Z"] is None: continue
            SC = np.ones_like(p["Z"]); cs = [[0.0, p["color"]], [1.0, p["color"]]]
            fig.add_trace(go.Surface(x=p["X"], y=p["Y"], z=p["Z"],
                                     surfacecolor=SC, colorscale=cs, name=p["name"], **plane_kw))
    fig.update_layout(title=title, width=900, height=700,
                      scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ------------------------------------------------------------
def make_three_panel_slice_figure(volume, mask2px, mask_display,
                                  axis='inline', title='', mode=None):
    """
    axis: 'inline' (Y), 'xline' (X), or 'time' (Z)
    Returns a Plotly Figure with frames & a slider.
    """
    import numpy as np
    import plotly.graph_objects as go
    from constants import mask_mode as _mm

    mode = _mm if mode is None else mode
    p1, p99 = np.percentile(volume, (1, 99))
    gmin, gmax = float(p1), float(p99)

    # Axis-specific extractors
    if axis == 'inline':        # vary j (Y)
        N = volume.shape[1]
        get = lambda idx: (volume[:, idx, :], mask2px[:, idx, :], mask_display[:, idx, :])
        x_title, y_title = 'X', 'Depth (Z)'
        slider_prefix = 'Inline index: '
    elif axis == 'xline':       # vary i (X)
        N = volume.shape[2]
        get = lambda idx: (volume[:, :, idx], mask2px[:, :, idx], mask_display[:, :, idx])
        x_title, y_title = 'Y', 'Depth (Z)'
        slider_prefix = 'Crossline index: '
    else:                       # 'time' — vary k (Z)
        N = volume.shape[0]
        get = lambda idx: (volume[idx, :, :], mask2px[idx, :, :], mask_display[idx, :, :])
        x_title, y_title = 'X', 'Y'
        slider_prefix = 'Slice Z: '

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Seismic", "Mask (opaque)", "Seismic + Mask"),
                        horizontal_spacing=0.08)

    # First frame
    z_seis, z_mask, z_disp = get(0)
    fig.add_trace(go.Heatmap(z=z_seis, colorscale="Gray",
                             zmin=gmin, zmax=gmax, showscale=False, zsmooth=False), row=1, col=1)
    if mode == 0:
        z = (z_mask != 0).astype(float); zmin, zmax = 0, 1
        cm = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
    else:
        z = z_mask.astype(float)/2.0; zmin, zmax = 0, 1
        cm = [
            [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
            [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
            [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
        ]
    fig.add_trace(go.Heatmap(z=z, colorscale=cm, zmin=zmin, zmax=zmax,
                             showscale=False, zsmooth=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=z_seis, colorscale="Gray",
                             zmin=gmin, zmax=gmax, showscale=False, zsmooth=False), row=1, col=3)
    fig.add_trace(mask_heatmap_overlay(z_disp, mode=mode), row=1, col=3)

    # Frames
    frames = []
    for idx in range(N):
        z_seis, z_mask, z_disp = get(idx)
        if mode == 0:
            z = (z_mask != 0).astype(float); zmin, zmax = 0, 1
            cm_now = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
        else:
            z = z_mask.astype(float)/2.0; zmin, zmax = 0, 1
            cm_now = [
                [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
                [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
                [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
            ]
        frames.append(go.Frame(
            name=str(idx),
            data=[
                go.Heatmap(z=z_seis, colorscale="Gray", zmin=gmin, zmax=gmax, showscale=False, zsmooth=False),
                go.Heatmap(z=z,      colorscale=cm_now, zmin=zmin, zmax=zmax, showscale=False, zsmooth=False),
                go.Heatmap(z=z_seis, colorscale="Gray", zmin=gmin, zmax=gmax, showscale=False, zsmooth=False),
                mask_heatmap_overlay(z_disp, mode=mode),
            ]
        ))
    fig.frames = frames
    steps = [dict(method="animate",
                  args=[[str(j)], {"mode":"immediate","frame":{"duration":0,"redraw":True}}],
                  label=str(j)) for j in range(N)]

    fig.update_layout(title=title, width=1200, height=520,
                      sliders=[dict(currentvalue={"prefix": slider_prefix}, pad={"t": 50}, steps=steps)])

    # Axis ranges
    nz, ny, nx = volume.shape
    if axis in ('inline', 'xline'):
        for c in (1,2,3):
            fig.update_yaxes(title_text="Depth (Z)", row=1, col=c, autorange="reversed", range=[0, nz])
    else:
        for c in (1,2,3):
            fig.update_yaxes(title_text="Y", row=1, col=c, range=[0, ny])

    if axis == 'inline':
        for c in (1,2,3):
            fig.update_xaxes(title_text="X", row=1, col=c, range=[0, nx])
    elif axis == 'xline':
        for c in (1,2,3):
            fig.update_xaxes(title_text="Y", row=1, col=c, range=[0, ny])
    else:  # time
        for c in (1,2,3):
            fig.update_xaxes(title_text="X", row=1, col=c, range=[0, nx])
    return fig

# ------------------------------------------------------------
def show_sample_viewers(all_stats_data, split='train', index=None, base_out=None, show=True):
    """
    Loads a cube + mask for the given split/index, reconstructs planes,
    builds 3 animated 2D viewers, and returns everything.
    """
    import numpy as np
    from constants import PAD, mask_mode

    # Choose index
    n_cubes = len(all_stats_data[split]['cube_level_params'])
    if n_cubes == 0:
        print(f"No {split} samples.")
        return None
    if index is None:
        index = int(cp.random.randint(0, n_cubes))

    # Load & reconstruct
    vol, mask2 = load_cube_and_masks(split, index, base_out=base_out)
    faults = all_stats_data[split]['cube_level_params'][index]['faults']
    planes = reconstruct_fault_planes_from_params(faults, mask2.shape, pad=PAD)
    from utilities import collapse_to_display_mask
    mask1 = collapse_to_display_mask(mask2)

    print(f"{split.capitalize()} sample #{index:03d} — volume shape: {vol.shape}, faults: {len(faults)}")

    # 2D animated viewers
    fig_inline = make_three_panel_slice_figure(vol, mask2, mask1, axis='inline',
                                               title=f"Inline (Y-axis) Slices — {split.upper()}",
                                               mode=mask_mode)
    fig_xline  = make_three_panel_slice_figure(vol, mask2, mask1, axis='xline',
                                               title=f"Crossline (X-axis) Slices — {split.upper()}",
                                               mode=mask_mode)
    fig_time   = make_three_panel_slice_figure(vol, mask2, mask1, axis='time',
                                               title=f"Time (Depth) Slices — {split.upper()}",
                                               mode=mask_mode)

    if show:
        fig_inline.show(); fig_xline.show(); fig_time.show()

    # Useful for 3D viewers
    vmin = float(np.percentile(vol, 1))
    vmax = float(np.percentile(vol, 99))

    return {
        "index": index,
        "volume": vol,
        "mask2px": mask2,
        "mask_display": mask1,
        "planes": planes,
        "fig_inline": fig_inline,
        "fig_xline": fig_xline,
        "fig_time": fig_time,
        "vmin": vmin,
        "vmax": vmax,
    }

def build_2d_slice_panels(cropped_volume, cropped_mask, cropped_mask_display, mode=None):
    """Return (fig_inline, fig_xline, fig_time) for the given volume/masks."""
    mode = mask_mode if mode is None else mode
    import plotly.graph_objects as go  # local alias

    p1, p99 = np.percentile(cropped_volume, (1, 99))
    global_min, global_max = float(p1), float(p99)

    # ---------- Inline viewer ----------
    fig_inline = make_subplots(rows=1, cols=3,
                               subplot_titles=("Seismic", "Mask (opaque)", "Seismic + Mask"),
                               horizontal_spacing=0.08)
    j0 = 0
    fig_inline.add_trace(go.Heatmap(z=cropped_volume[:, j0, :], colorscale="Gray",
                                    zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=1)
    center_mask = cropped_mask[:, j0, :]
    if mode == 0:
        z_mask = (center_mask != 0).astype(float); zmin, zmax = 0, 1
        cm_colorscale = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
    else:
        z_mask = center_mask.astype(float)/2.0; zmin, zmax = 0, 1
        cm_colorscale = [
            [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
            [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
            [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
        ]
    fig_inline.add_trace(go.Heatmap(z=z_mask, colorscale=cm_colorscale, zmin=zmin, zmax=zmax,
                                    showscale=False, zsmooth=False), row=1, col=2)
    fig_inline.add_trace(go.Heatmap(z=cropped_volume[:, j0, :], colorscale="Gray",
                                    zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=3)
    fig_inline.add_trace(mask_heatmap_overlay(cropped_mask_display[:, j0, :], mode=mode), row=1, col=3)

    frames = []
    for j in range(cropped_volume.shape[1]):
        data = [go.Heatmap(z=cropped_volume[:, j, :], colorscale="Gray",
                           zmin=global_min, zmax=global_max, showscale=False, zsmooth=False)]
        if mode == 0:
            z_mask = (cropped_mask[:, j, :] != 0).astype(float); zmin, zmax = 0, 1
            cm = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
        else:
            z_mask = cropped_mask[:, j, :].astype(float)/2.0; zmin, zmax = 0, 1
            cm = [
                [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
                [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
                [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
            ]
        data.append(go.Heatmap(z=z_mask, colorscale=cm, zmin=zmin, zmax=zmax, showscale=False, zsmooth=False))
        data.append(go.Heatmap(z=cropped_volume[:, j, :], colorscale="Gray",
                               zmin=global_min, zmax=global_max, showscale=False, zsmooth=False))
        data.append(mask_heatmap_overlay(cropped_mask_display[:, j, :], mode=mode))
        frames.append(go.Frame(name=str(j), data=data))
    fig_inline.frames = frames
    steps = [dict(method="animate",
                  args=[[str(j)], {"mode":"immediate","frame":{"duration":0,"redraw":True}}],
                  label=str(j)) for j in range(len(frames))]
    fig_inline.update_layout(title="Inline (Y-axis) Slices",
                             width=1200, height=520,
                             sliders=[dict(currentvalue={"prefix": "Inline index: "}, pad={"t": 50}, steps=steps)])
    for c in (1,2,3):
        fig_inline.update_xaxes(title_text="X", row=1, col=c, range=[0, cropped_volume.shape[2]])
        fig_inline.update_yaxes(title_text="Depth (Z)", row=1, col=c, autorange="reversed",
                                range=[0, cropped_volume.shape[0]])

    # ---------- Crossline viewer ----------
    fig_xline = make_subplots(rows=1, cols=3,
                              subplot_titles=("Seismic", "Mask (opaque)", "Seismic + Mask"),
                              horizontal_spacing=0.08)
    i0 = 0
    fig_xline.add_trace(go.Heatmap(z=cropped_volume[:, :, i0], colorscale="Gray",
                                   zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=1)
    center_mask_x = cropped_mask[:, :, i0]
    if mode == 0:
        z_mask = (center_mask_x != 0).astype(float); zmin, zmax = 0, 1
        cm_colorscale = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
    else:
        z_mask = center_mask_x.astype(float)/2.0; zmin, zmax = 0, 1
        cm_colorscale = [
            [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
            [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
            [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
        ]
    fig_xline.add_trace(go.Heatmap(z=z_mask, colorscale=cm_colorscale, zmin=zmin, zmax=zmax,
                                   showscale=False, zsmooth=False), row=1, col=2)
    fig_xline.add_trace(go.Heatmap(z=cropped_volume[:, :, i0], colorscale="Gray",
                                   zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=3)
    fig_xline.add_trace(mask_heatmap_overlay(cropped_mask_display[:, :, i0], mode=mode), row=1, col=3)

    frames_x = []
    for i in range(cropped_volume.shape[2]):
        data = [go.Heatmap(z=cropped_volume[:, :, i], colorscale="Gray",
                           zmin=global_min, zmax=global_max, showscale=False, zsmooth=False)]
        if mode == 0:
            z_mask = (cropped_mask[:, :, i] != 0).astype(float); zmin, zmax = 0, 1
            cm = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
        else:
            z_mask = cropped_mask[:, :, i].astype(float)/2.0; zmin, zmax = 0, 1
            cm = [
                [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
                [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
                [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
            ]
        data.append(go.Heatmap(z=z_mask, colorscale=cm, zmin=zmin, zmax=zmax, showscale=False, zsmooth=False))
        data.append(go.Heatmap(z=cropped_volume[:, :, i], colorscale="Gray",
                               zmin=global_min, zmax=global_max, showscale=False, zsmooth=False))
        data.append(mask_heatmap_overlay(cropped_mask_display[:, :, i], mode=mode))
        frames_x.append(go.Frame(name=str(i), data=data))
    fig_xline.frames = frames_x
    steps_x = [dict(args=[[f.name], {"frame":{"duration":0,"redraw":True}, "mode":"immediate"}],
                    label=str(idx), method="animate") for idx, f in enumerate(fig_xline.frames)]
    fig_xline.update_layout(title="Crossline (X-axis) Slices", width=1200, height=520,
                            sliders=[dict(currentvalue={"prefix": "Crossline index: "}, pad={"t": 50}, steps=steps_x)])
    for c in (1,2,3):
        fig_xline.update_xaxes(title_text="Y", row=1, col=c, range=[0, cropped_volume.shape[1]])
        fig_xline.update_yaxes(title_text="Depth (Z)", row=1, col=c, autorange="reversed",
                               range=[0, cropped_volume.shape[0]])

    # ---------- Time/Depth viewer ----------
    fig_time3 = make_subplots(rows=1, cols=3,
                              subplot_titles=("Seismic", "Mask (opaque)", "Seismic + Mask"),
                              horizontal_spacing=0.08)
    k0 = 0
    fig_time3.add_trace(go.Heatmap(z=cropped_volume[k0, :, :], colorscale="Gray",
                                   zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=1)
    center_mask_t = cropped_mask[k0, :, :]
    if mode == 0:
        z_mask = (center_mask_t != 0).astype(float); zmin, zmax = 0, 1
        cm_colorscale = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
    else:
        z_mask = center_mask_t.astype(float)/2.0; zmin, zmax = 0, 1
        cm_colorscale = [
            [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
            [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
            [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
        ]
    fig_time3.add_trace(go.Heatmap(z=z_mask, colorscale=cm_colorscale, zmin=zmin, zmax=zmax,
                                   showscale=False, zsmooth=False), row=1, col=2)
    fig_time3.add_trace(go.Heatmap(z=cropped_volume[k0, :, :], colorscale="Gray",
                                   zmin=global_min, zmax=global_max, showscale=False, zsmooth=False), row=1, col=3)
    fig_time3.add_trace(mask_heatmap_overlay(cropped_mask_display[k0, :, :], mode=mode), row=1, col=3)

    frames_t = []
    for k in range(cropped_volume.shape[0]):
        data = [go.Heatmap(z=cropped_volume[k, :, :], colorscale="Gray",
                           zmin=global_min, zmax=global_max, showscale=False, zsmooth=False)]
        if mode == 0:
            z_mask = (cropped_mask[k, :, :] != 0).astype(float); zmin, zmax = 0, 1
            cm = [[0,"rgb(0,0,0)"],[0.5,"rgb(0,0,0)"],[0.5,"rgb(255,0,0)"],[1,"rgb(255,0,0)"]]
        else:
            z_mask = cropped_mask[k, :, :].astype(float)/2.0; zmin, zmax = 0, 1
            cm = [
                [0.00,"rgb(0,0,0)"],[0.33,"rgb(0,0,0)"],
                [0.50,"rgb(0,204,0)"],[0.66,"rgb(0,204,0)"],
                [0.99,"rgb(128,0,128)"],[1.00,"rgb(128,0,128)"],
            ]
        data.append(go.Heatmap(z=z_mask, colorscale=cm, zmin=zmin, zmax=zmax, showscale=False, zsmooth=False))
        data.append(go.Heatmap(z=cropped_volume[k, :, :], colorscale="Gray",
                               zmin=global_min, zmax=global_max, showscale=False, zsmooth=False))
        data.append(mask_heatmap_overlay(cropped_mask_display[k, :, :], mode=mode))
        frames_t.append(go.Frame(name=str(k), data=data))
    fig_time3.frames = frames_t
    steps_t = [dict(args=[[f.name], {"frame":{"duration":0,"redraw":True}, "mode":"immediate"}],
                    label=str(idx), method="animate") for idx, f in enumerate(fig_time3.frames)]
    fig_time3.update_layout(title="Time (Depth) Slices", width=1200, height=520,
                            sliders=[dict(currentvalue={"prefix": "Slice Z: "}, pad={"t": 50}, steps=steps_t)])
    for c in (1,2,3):
        fig_time3.update_xaxes(title_text="X", row=1, col=c, range=[0, cropped_volume.shape[2]])
        fig_time3.update_yaxes(title_text="Y", row=1, col=c, range=[0, cropped_volume.shape[1]])

    return fig_inline, fig_xline, fig_time3


def random_train_sample_panels(all_stats_data, train_count, base_out, pad, mode=None):
    """
    Pick a random TRAIN sample, rebuild planes & masks, and return:
    {
      'index', 'fig_inline', 'fig_xline', 'fig_time',
      'fault_planes_3d', 'volume', 'mask', 'mask_display', 'vmin', 'vmax'
    }
    """
    if train_count <= 0:
        print("No training samples.")
        return None

    idx = int(cp.random.randint(0, train_count))
    vol, mask2 = load_cube_and_masks("train", idx, base_out=base_out)
    faults = all_stats_data['train']['cube_level_params'][idx]['faults']
    planes = reconstruct_fault_planes_from_params(faults, mask2.shape, pad=pad)
    from utilities import collapse_to_display_mask
    mask1 = collapse_to_display_mask(mask2)

    print(f"Training sample #{idx:03d} — volume shape: {vol.shape}, faults: {len(faults)}")

    f_inl, f_xl, f_time = build_2d_slice_panels(vol, mask2, mask1, mode=mode if mode is not None else mask_mode)

    vmin = float(np.percentile(vol, 1))
    vmax = float(np.percentile(vol, 99))

    return {
        'index': idx,
        'fig_inline': f_inl,
        'fig_xline': f_xl,
        'fig_time': f_time,
        'fault_planes_3d': planes,
        'volume': vol,
        'mask': mask2,
        'mask_display': mask1,
        'vmin': vmin,
        'vmax': vmax,
    }


# --- NEW in viewers.py ---

from ipywidgets import interact, IntSlider, Checkbox, FloatSlider

def interact_3d_seismic(seis, planes, vmin=None, vmax=None, title="3D Seismic"):
    import numpy as np
    NZ, NY, NX = seis.shape
    if vmin is None: vmin = float(np.percentile(seis, 1))
    if vmax is None: vmax = float(np.percentile(seis, 99))
    final_cube_size = min(NX, NY, NZ)

    def _cb(inline, xline, time, show_intersections, show_planes, plane_opacity):
        make_3d_fig_seismic(
            seis, planes, vmin, vmax,
            inline, xline, time,
            show_intersections=show_intersections,
            show_planes=show_planes,
            plane_opacity=plane_opacity,
            title=title
        ).show()

    return interact(
        _cb,
        inline=IntSlider(min=0, max=NY-1, step=1, value=min(NY-1, final_cube_size//2), description='Inline (y)'),
        xline =IntSlider(min=0, max=NX-1, step=1, value=min(NX-1, final_cube_size//2), description='Xline (x)'),
        time  =IntSlider(min=0, max=NZ-1, step=1, value=min(NZ-1, final_cube_size//2), description='Time (z)'),
        show_intersections=Checkbox(value=False, description='Show Plane Intersections'),
        show_planes=Checkbox(value=True,  description='Show Fault Planes'),
        plane_opacity=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.55, description='Plane Opacity')
    )

def interact_3d_mask(mask, planes, title="3D Fault Mask"):
    NZ, NY, NX = mask.shape
    final_cube_size = min(NX, NY, NZ)

    def _cb(inline, xline, time, show_planes, plane_opacity, show_intersections):
        make_3d_fig_mask(
            mask, planes,
            inline, xline, time,
            show_planes=show_planes,
            plane_opacity=plane_opacity,
            show_intersections=show_intersections,
            title=title
        ).show()

    return interact(
        _cb,
        inline=IntSlider(min=0, max=NY-1, step=1, value=min(NY-1, final_cube_size//2), description='Inline (y)'),
        xline =IntSlider(min=0, max=NX-1, step=1, value=min(NX-1, final_cube_size//2), description='Xline (x)'),
        time  =IntSlider(min=0, max=NZ-1, step=1, value=min(NZ-1, final_cube_size//2), description='Time (z)'),
        show_planes=Checkbox(value=True,  description='Show Fault Planes'),
        plane_opacity=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.55, description='Plane Opacity'),
        show_intersections=Checkbox(value=False, description='Show Plane Intersections')
    )
    

