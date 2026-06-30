"""
plot swanson figs to show regional specificity of age effects on r_noise and r_signal

"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions
from scripts.utils.plot_utils import figure_style
from scripts.utils.io import read_table
import config as C
ANALYSIS_TAG = "_VISp_VISpm_separate"

br = BrainRegions()

def region_table(beryl_names):
    ROI_df = pd.DataFrame({'beryl_name': beryl_names, 'ROI': beryl_names})

    ROI_mapping = {
        'VISa': 'PPC', 'VISam': 'PPC',
        'APN': 'MBm', 'MRN': 'MBm',
        'ACAv': 'ACA', 'ACAd': 'ACA',
        'PL': 'mPFC', 'ILA': 'mPFC',
        'ORBm': 'ORB', 'ORBl': 'ORB', 'ORBvl': 'ORB',
        'TTd': 'OLF', 'DP': 'OLF', 'AON': 'OLF',
        'LSr': 'LS', 'LSc': 'LS', 'LSv': 'LS'
    }

    ROI_df['ROI'] = ROI_df['beryl_name'].map(ROI_mapping).fillna(ROI_df['beryl_name'])
    ROI_df['cosmos_name'] = ROI_df['beryl_name'].apply(lambda r: br.acronym2acronym(r, mapping='Cosmos')[0])
    ROI_df['swanson_name'] = ROI_df['beryl_name'].apply(lambda r: br.acronym2acronym(r, mapping='Swanson')[0])

    return ROI_df


def load_stats_results(y_col, n_permut=C.N_PERMUT_NEURAL_REGIONAL, window="pre"):
    """
    Load current-format regional permutation results.

    Current input format:
    proj2_noise_pairlevel_regional_permutation_{analysis_name}_{n_permut}perms_500ms{ANALYSIS_TAG}.csv

    analysis_name:
    - pre/post for r_noise
    - quench for delta_r_noise
    """

    if y_col == "delta_r_noise":
        analysis_name = "quench"
    elif y_col == "r_noise":
        if window not in ["pre", "post"]:
            raise ValueError("For r_noise, window must be 'pre' or 'post'.")
        analysis_name = window
    else:
        raise ValueError(
            f"Unsupported y_col: {y_col}. "
            "This loader currently supports r_noise and delta_r_noise only."
        )

    fname = (
        f"proj2_noise_pairlevel_regional_permutation_"
        f"{analysis_name}_{n_permut}perms_500ms{ANALYSIS_TAG}.csv"
    )

    path = C.RESULTSPATH / fname

    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    stats_df = read_table(path)

    print(f"[Loaded stats] {path}")
    print(stats_df.shape)
    print(stats_df[["analysis", "cluster_region", "y_col", "observed_val", "p_perm"]].head())

    return stats_df


def get_vmin_vmax(metric):
    ranges = {
        'r_noise': (-0.05, 0.05), 'r_signal': (-0.1, 0.1),
        'delta_r_noise': (-0.03, 0.03)
    }
    return ranges.get(metric, (-1, 1))


def plot_swanson(metric, ROI_df, stats_df, window="pre"):
    if metric == 'delta_r_noise':
        cmap = LinearSegmentedColormap.from_list(
            "swanson_cmap", C.COLORS_SWANSON_INVERT,
            N=256
        )
    else:
        cmap = LinearSegmentedColormap.from_list(
            "swanson_cmap", C.COLORS_SWANSON,
            N=256
        )

    merged_df = ROI_df.merge(
        stats_df[['cluster_region', 'observed_val', 'p_perm']],
        left_on='ROI', right_on='cluster_region', how='left'
    )

    vmin, vmax = get_vmin_vmax(metric)
    figure_style()

    fig, ax = plt.subplots(figsize=(2.5, 2))

    plot_swanson_vector(
        merged_df['beryl_name'], merged_df['observed_val'],
        cmap=cmap, vmin=vmin, vmax=vmax, br=br,
        empty_color='white', show_cbar=True, annotate=False,  
        annotate_list=merged_df['swanson_name'],  ax=ax,

    )

    for coll in ax.collections:
        try:
            coll.set_linewidth(0.2)
            coll.set_edgecolor('0.6')

        except Exception:
            pass

    for patch in ax.patches:
        try:
            patch.set_linewidth(0.2)
            patch.set_edgecolor('0.6')
        except Exception:
            pass

    for line in ax.lines:
        try:
            line.set_linewidth(0.2)
            line.set_color('0.6')
        except Exception:
            pass

    # brain region text
    for txt in ax.texts:
        txt.set_fontname("Arial")
        txt.set_fontsize(4)
        txt.set_fontweight("normal")

    # 2. fewer colorbar ticks
    if len(fig.axes) > 1:
        cax = fig.axes[-1]
        # adjust colorbar height
        pos = cax.get_position()
        new_height = pos.height * 0.55   # smaller than current height
        new_y0 = pos.y0 + (pos.height - new_height) / 2

        cax.set_position([
            pos.x0,
            new_y0,
            pos.width,
            new_height,
        ])

        if metric == 'r_noise':
            ticks = [-0.05, 0.0, 0.05]
        elif metric == 'r_signal':
            ticks = [-0.1, 0.0, 0.1]
        elif metric == 'delta_r_noise':
            ticks = [-0.03, 0.0, 0.03]
        else:
            ticks = [vmin, 0, vmax]

        cax.set_yticks(ticks)
        cax.set_yticklabels([f"{t:.2f}" if abs(t) < 1 else f"{t:g}" for t in ticks])

        cax.tick_params(axis='y', labelsize=4, width=0.15, length=1.2, pad=1)

        for label in cax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontsize(4)
            label.set_fontweight("normal")

        for spine in cax.spines.values():
            spine.set_linewidth(0.15)


    ax.set_axis_off()
    if metric == 'delta_r_noise':
        fname = f"Swanson_{metric}_500ms{ANALYSIS_TAG}.pdf"
    else:
        fname = f"Swanson_{metric}_{window}_500ms{ANALYSIS_TAG}.pdf"
    fig.savefig(os.path.join(C.FIGPATH, fname), dpi=300)
    print(f"[Saved figure] {os.path.join(C.FIGPATH, fname)}")


def main():

    # selected_metrics = ['r_noise']
    selected_metrics = ['delta_r_noise']
    ROI_df = region_table(C.BERYL_NAMES)

    for metric in selected_metrics:
        if metric in ['r_signal', 'r_noise']:
            for time_window in ['pre', 'post']:
                print(f"Plotting Swanson for {metric} ({time_window}-window)...")
                stats_df = load_stats_results(metric, window=time_window) 
                plot_swanson(metric, ROI_df, stats_df, window=time_window)
        else:
            print(f"Plotting Swanson for {metric}...")
            stats_df = load_stats_results(metric) 
            plot_swanson(metric, ROI_df, stats_df)

if __name__ == "__main__":
    main()

