# stats_plot.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from constants import mask_mode, normal_colour, inverse_colour
from utilities import count_pixels, plot_histogram


def plot_dataset_stats(all_stats_data, base_out, image_dir, title_suffix="Dataset Statistics"):
    """
    Plot and save dataset statistics for splits: 'full', 'train', 'validation'.

    Parameters
    ----------
    all_stats_data : dict
        Dict with keys 'train', 'validation', 'full', each containing:
          - 'cube_level_params': list[dict] with 'num_faults', 'noise_sigma', 'faults'
          - 'all_fault_params': list[dict] with per-fault params (strike, dip, etc.)
    base_out : str
        Root folder containing split subdirs with 'fault/*.npy' masks.
    image_dir : str
        Folder to save the generated PNGs.
    title_suffix : str
        Text appended to each figure title (default: "Dataset Statistics").
    """
    os.makedirs(image_dir, exist_ok=True)

    for split_name in ['full', 'train', 'validation']:
        stats = all_stats_data.get(split_name)
        print(f"\n--- Dataset statistics for **{split_name.capitalize()}** split ---")
        if not stats or 'cube_level_params' not in stats or 'all_fault_params' not in stats:
            print(f"No statistics available for '{split_name}' split.")
            continue

        cube_params  = stats['cube_level_params']
        fault_params = stats['all_fault_params']

        # --- summary table
        df_params = pd.DataFrame(cube_params)
        cols = [c for c in ['num_faults', 'noise_sigma'] if c in df_params.columns]
        if cols and not df_params.empty:
            num_df = df_params[cols].apply(pd.to_numeric, errors='coerce')
            if num_df.select_dtypes(include=[np.number]).shape[1] > 0:
                display(num_df.describe())
            else:
                print("No numeric cube-level parameters to summarize.")
        else:
            print("No cube-level parameters recorded.")

        # --- gather masks & compute pixel percentages (verify via recount)
        mask_cubes = []
        splits_to_scan = ['train', 'validation'] if split_name == 'full' else [split_name]
        for split in splits_to_scan:
            mask_dir = os.path.join(base_out, split, "fault")
            if not os.path.isdir(mask_dir):
                continue
            for fname in sorted(os.listdir(mask_dir)):
                if fname.endswith(".npy"):
                    mask_cubes.append(np.load(os.path.join(mask_dir, fname)))
        overall_pct, mean_pct = count_pixels(mask_cubes, mask_mode)

        print("\nPixel-class distribution – OVERALL (% of all voxels):")
        for k, v in overall_pct.items():
            print(f"  {k:>10}: {v:6.2f} %")
        print("Pixel-class distribution – MEAN PER CUBE:")
        for k, v in mean_pct.items():
            print(f"  {k:>10}: {v:6.2f} %")

        # --- skip plots if no faults present
        if not fault_params:
            print("\n(No faults in this split to plot.)")
            continue

        strikes = [f['strike'] for f in fault_params]
        dips = [f['dip'] for f in fault_params]
        displacements = [f['applied_disp_signed'] for f in fault_params]

        # --- figure layout: 3 rows × 2 cols (last row full-width)
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(f"{split_name.capitalize()} {title_suffix}", fontsize=24, y=0.94)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])

        # 1) Fault Type Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if mask_mode == 0:
            total_faults = len(fault_params)
            ax1.bar(['Total Faults'], [total_faults], color='skyblue')
        else:
            num_normal  = sum(1 for f in fault_params if f['fault_type'] == 'Normal')
            num_inverse = sum(1 for f in fault_params if f['fault_type'] == 'Reverse')
            ax1.bar(['Normal', 'Reverse'], [num_normal, num_inverse],
                    color=[normal_colour, inverse_colour])
        ax1.set_title('Fault Type Distribution')
        ax1.set_ylabel('Number of Faults')

        # 2) Fault Strike Angles — Rose Diagram
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        if strikes:
            if mask_mode == 1:
                strikes_normal  = np.deg2rad([f['strike'] for f in fault_params if f['fault_type'] == 'Normal'])
                strikes_reverse = np.deg2rad([f['strike'] for f in fault_params if f['fault_type'] == 'Reverse'])
                num_bins = 18
                bins = np.linspace(0, 2*np.pi, num_bins + 1)
                cnt_norm, _ = np.histogram(strikes_normal, bins=bins)
                cnt_rev,  _ = np.histogram(strikes_reverse, bins=bins)
                centers = bins[:-1] + np.diff(bins)/2
                width = np.diff(bins)[0]
                ax2.bar(centers, cnt_norm, width=width, bottom=0.0, color=normal_colour,
                        edgecolor='black', alpha=0.8, label='Normal')
                ax2.bar(centers, cnt_rev, width=width, bottom=cnt_norm, color=inverse_colour,
                        edgecolor='black', alpha=0.8, label='Reverse')
                ax2.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0))
            else:
                angles = np.deg2rad(np.mod(strikes, 360.0))
                num_bins = 18
                bins = np.linspace(0, 2*np.pi, num_bins + 1)
                counts, _ = np.histogram(angles, bins=bins)
                centers = bins[:-1] + np.diff(bins)/2
                width = np.diff(bins)[0]
                ax2.bar(centers, counts, width=width, bottom=0.0, color='cornflowerblue',
                        edgecolor='black', alpha=0.9)
            ax2.set_title('Fault Strike Angles', va='bottom')
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_thetagrids([0, 90, 180, 270], labels=['N','E','S','W'])


        # 3) Fault Dip Angles
        ax3 = fig.add_subplot(gs[1, 0])
        if dips:
            if mask_mode == 1:
                dips_normal  = [f['dip'] for f in fault_params if f['fault_type'] == 'Normal']
                dips_reverse = [f['dip'] for f in fault_params if f['fault_type'] == 'Reverse']
                ax3.hist([dips_normal, dips_reverse], bins='auto', stacked=True,
                         color=[normal_colour, inverse_colour], edgecolor='black',
                         label=['Normal', 'Reverse'])
                ax3.legend()
            else:
                plot_histogram(dips, ax=ax3, title='Fault Dip Angles', xlabel='Dip (deg)')
            ax3.set_title('Fault Dip Angles'); ax3.set_xlabel('Dip (degrees)')

        # 4) Fault Displacement Histogram (signed)
        ax4 = fig.add_subplot(gs[1, 1])
        if displacements:
            low = np.floor(min(displacements))
            high = np.ceil(max(displacements))
            if mask_mode == 1:
                disp_norm  = [d for d in displacements if d >= 0]
                disp_rev   = [d for d in displacements if d < 0]
                ax4.hist([disp_norm, disp_rev], bins=40, range=(low, high), stacked=True,
                         color=[normal_colour, inverse_colour], edgecolor='black',
                         label=['Normal', 'Reverse'])
                ax4.legend()
            else:
                plot_histogram(displacements, ax=ax4, title='Fault Displacements',
                               xlabel='Slip (voxels)', bins=40, hist_range=(low, high), use_percentile=False)
            ax4.set_title('Fault Slip Magnitudes'); ax4.set_xlabel('Slip (voxels)')

        # 5) Pixel-Class Distribution Bar
        ax5 = fig.add_subplot(gs[2, :])
        categories = list(overall_pct.keys())
        if mask_mode == 0:
            categories = ['no_fault', 'fault']
            colors = ['dimgray', 'skyblue']
        else:
            if 'overlap' in categories and overall_pct.get('overlap', 0) == 0:
                categories.remove('overlap')
            ordered = ['no_fault', 'normal', 'inverse']
            categories = [c for c in ordered if c in categories]
            cmap = {'no_fault':'dimgray', 'normal':normal_colour, 'inverse':inverse_colour}
            colors = [cmap[c] for c in categories]
        ax5.bar(categories, [overall_pct[k] for k in categories], color=colors)
        ax5.set_ylim(0, 100); ax5.set_title('Pixel-Class Distribution (Overall % of Voxels)')
        ax5.set_ylabel('Percentage of Volume')
        for idx, key in enumerate(categories):
            ax5.text(idx, overall_pct[key] + 1, f"{overall_pct[key]:.1f}%", ha='center')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        img_file = os.path.join(image_dir, f"{split_name}_dataset_stats.png")
        plt.savefig(img_file, dpi=300)
        plt.show()
        print(f"Saved statistics figure to {img_file}")
