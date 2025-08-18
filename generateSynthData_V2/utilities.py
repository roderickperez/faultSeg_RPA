import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, ax, title, xlabel, bins='auto', xlim=None, hist_range=None, use_percentile=True):
    """Plot a histogram of the given data on Axes `ax` with specified settings."""
    if data is None or len(data) == 0:
        return
    # Determine histogram range
    if use_percentile:
        # Use 2nd–98th percentile to avoid outliers dominating scale
        low, high = np.percentile(data, [2, 98])
    else:
        low, high = (min(data), max(data)) if not xlim else xlim
    ax.hist(data, bins=bins, range=(low, high) if hist_range is None else hist_range,
            edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    if xlim:
        ax.set_xlim(xlim)

def count_pixels(mask_cubes, mask_mode):
    """
    Compute overall and mean per-cube percentage of each pixel class in a list of mask volumes.
    Returns (overall_pct, mean_pct) as dictionaries.
    """
    total_voxels = 0
    # Initialize counters
    no_fault_vox = normal_vox = inverse_vox = overlap_vox = 0
    # Lists for per-cube percentages
    pct_no_fault_list = []; pct_normal_list = []; pct_inverse_list = []; pct_overlap_list = []

    for mask in mask_cubes:
        n_vox = mask.size
        total_voxels += n_vox
        if mask_mode == 0:
            # Binary mask (fault or no_fault)
            fault_count = np.sum(mask != 0)
            no_fault = n_vox - fault_count
            no_fault_vox += no_fault
            normal_vox += fault_count  # treat all faults as "normal" category for binary case
            pct_no_fault_list.append(no_fault / n_vox * 100.0)
            pct_normal_list.append(fault_count / n_vox * 100.0)
        else:
            # Multi-class mask (0=no fault, 1=normal, 2=reverse[, 3=overlap])
            n_normal = np.sum(mask == 1)
            n_inverse = np.sum(mask == 2)
            n_overlap = np.sum(mask == 3) if np.any(mask == 3) else 0
            no_fault = n_vox - (n_normal + n_inverse + n_overlap)
            no_fault_vox += no_fault
            normal_vox += n_normal
            inverse_vox += n_inverse
            overlap_vox += n_overlap
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
        # (If an 'overlap' class were used, include it here. Overlap is 0 in this dataset.)
        if overlap_vox > 0:
            overall_pct['overlap'] = (overlap_vox / total_voxels * 100.0)
        # Mean per-cube percentages
        mean_normal  = np.mean(pct_normal_list)  if pct_normal_list  else 0.0
        mean_inverse = np.mean(pct_inverse_list) if pct_inverse_list else 0.0
        mean_no_fault = np.mean(pct_no_fault_list) if pct_no_fault_list else 0.0
        mean_pct['normal']  = mean_normal
        mean_pct['inverse'] = mean_inverse
        mean_pct['no_fault'] = mean_no_fault
        if pct_overlap_list:
            mean_pct['overlap'] = np.mean(pct_overlap_list)
    return overall_pct, mean_pct
