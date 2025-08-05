import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_fault_counts(fault_params_list):
    """Plots bar chart of total, normal, and inverse fault counts."""
    if not fault_params_list:
        print("No fault data to plot counts.")
        return None

    normal_count = sum(1 for f in fault_params_list if f['fault_type'] == 'Normal')
    inverse_count = sum(1 for f in fault_params_list if f['fault_type'] == 'Inverse')
    total_count = len(fault_params_list)

    labels = ['Total', 'Normal', 'Inverse']
    counts = [total_count, normal_count, inverse_count]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, counts, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel('Number of Faults')
    ax.set_title('Fault Type Distribution Across All Cubes')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(counts):
        ax.text(i, count + max(counts) * 0.02, str(count), ha='center')
    plt.tight_layout()
    return fig

def plot_histogram(data, ax=None, title=None, xlabel=None, bins='auto'):
    """Plots a histogram on a given axes object."""
    if not data or (isinstance(data, list) and not any(x is not None for x in data)):
        print(f"No data to plot histogram for '{title}'.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    filtered_data = [x for x in data if x is not None]
    if not filtered_data:
         print(f"No valid (non-None) data to plot histogram for '{title}'.")
         return

    ax.hist(filtered_data, bins=bins, edgecolor='black')
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if 'fig' in locals() and ax.figure == fig:
        fig.tight_layout()

    return ax.figure

def plot_rose_diagram(strikes_deg, ax=None, title="Fault Strike Angle Distribution (Rose Diagram)"):
    """Plots a rose diagram for strike angles on a given axes object."""
    if not strikes_deg:
        print("No strike data to plot for rose diagram.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    strikes_rad = np.radians(strikes_deg)
    num_bins = 18
    bin_edges_deg = np.linspace(0, 360, num_bins + 1)
    bin_edges_rad = np.radians(bin_edges_deg)
    counts, _ = np.histogram(strikes_rad, bins=bin_edges_rad)
    bin_centers_rad = bin_edges_rad[:-1] + np.diff(bin_edges_rad)/2
    width = np.diff(bin_edges_rad)[0]
    ax.bar(bin_centers_rad, counts, width=width, bottom=0.0, align='center', alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title, va='bottom', y=1.1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f'{i}°' for i in np.arange(0, 360, 30)])
    ax.grid(True)

    if 'fig' in locals() and ax.figure == fig:
        fig.tight_layout()

    return ax.figure

def append_param_to_cmd(cmd_list, arg_name, value):
    if value is not None:
        if isinstance(value, tuple):
            cmd_list += [f"--{arg_name}", f"{value[0]},{value[1]}"]
        else:
            cmd_list += [f"--{arg_name}", str(value)]

def normalize(v):
    return ((v - v.min()) / (v.max() - v.min()) * 255).astype(np.uint8)

def plot_fault_points(fig, slice_mask, axis_index, slice_type, color, label, nx, ny, nz):
    if slice_type == "inline":
        ks, js = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=np.full_like(ks, axis_index),
            y=js,
            z=nz - 1 - ks,
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))
    elif slice_type == "crossline":
        ks, is_ = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=is_,
            y=np.full_like(is_, axis_index),
            z=nz - 1 - ks,
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))
    elif slice_type == "timeslice":
        is_, js = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=is_,
            y=js,
            z=np.full_like(is_, axis_index),
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))
        

def count_pixels(mask_cubes, mask_mode=0):

    # --- make sure we have an iterable ---------------------------------------
    if isinstance(mask_cubes, np.ndarray) and mask_cubes.ndim == 3:
        mask_cubes = [mask_cubes]
    elif mask_cubes is None:
        mask_cubes = []

    # nothing to count --------------------------------------------------------
    if len(mask_cubes) == 0:
        if mask_mode == 0:
            zero = {"no_fault": 0.0, "fault": 0.0}
        else:
            zero = {"no_fault": 0.0, "normal": 0.0, "inverse": 0.0}
        return zero, zero

    # choose labels -----------------------------------------------------------
    if mask_mode == 0:
        labels  = ["no_fault", "fault"]
        classes = (0, 1)
    else:
        labels  = ["no_fault", "normal", "inverse"]
        classes = (0, 1, 2)

    # initialise counters -----------------------------------------------------
    total_counts = dict.fromkeys(labels, 0)
    per_cube_pct = {lab: [] for lab in labels}

    # iterate cubes -----------------------------------------------------------
    for cube in mask_cubes:
        cube_size = cube.size
        for lab, cls in zip(labels, classes):
            cls_count = int((cube == cls).sum())
            total_counts[lab] += cls_count
            per_cube_pct[lab].append(100.0 * cls_count / cube_size)

    grand_total = float(sum(total_counts.values()))

    # avoid division-by-zero when all voxels are class 0 ----------------------
    if grand_total == 0:
        overall_pct = {lab: 0.0 for lab in labels}
    else:
        overall_pct = {lab: 100.0 * cnt / grand_total
                       for lab, cnt in total_counts.items()}

    mean_pct = {lab: float(np.mean(pcts)) for lab, pcts in per_cube_pct.items()}

    return overall_pct, mean_pct
