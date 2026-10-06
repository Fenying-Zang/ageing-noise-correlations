"""
Figure 3c: Swanson maps of regional LMM slopes for r_noise and delta_r_noise.
The slopes are for the effect of age on r_noise or delta_r_noise

"""
#%%
import config as C
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions
from scripts.utils.plot_utils import figure_style
from scripts.utils.io import read_table

WINDOW_LEN = 500
REGIONAL_STATS_FILE = C.RESULTSPATH / f"noise_regional_slopes_BH_FDR_{WINDOW_LEN}ms_{C.RANDOM_FACTOR}.csv"

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


def load_stats_results(y_col, window="pre"):
    """
    Load regional LMM age slopes.

    r_noise:
        analysis = pre / post
    delta_r_noise:
        analysis = quench

    beta_age is the reported slope for age_years.
    """
    if y_col == "delta_r_noise":
        analysis_name = "quench"
    elif y_col == "r_noise":
        if window not in ["pre", "post"]:
            raise ValueError(
                "For r_noise, window must be 'pre' or 'post'."
            )
        analysis_name = window
    else:
        raise ValueError(
            f"Unsupported y_col: {y_col}. "
            "Supported metrics: r_noise and delta_r_noise."
        )

    if not REGIONAL_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Stats file not found: {REGIONAL_STATS_FILE}"
        )

    stats_df = read_table(REGIONAL_STATS_FILE)

    required = [
        "analysis",
        "random_factor",
        "cluster_region",
        "beta_age",
        "p_BH_FDR",
    ]
    missing = [col for col in required if col not in stats_df.columns]
    if missing:
        raise ValueError(f"Missing statistics columns: {missing}")

    stats_df = stats_df.loc[
        (stats_df["analysis"] == analysis_name)
        & (stats_df["random_factor"] == C.RANDOM_FACTOR),
        required,
    ].copy()

    if stats_df.empty:
        raise ValueError(
            f"No regional statistics for analysis='{analysis_name}', "
            f"random_factor='{C.RANDOM_FACTOR}'."
        )

    # Each region must have exactly one slope for this analysis/model.
    duplicates = stats_df.loc[
        stats_df["cluster_region"].duplicated(keep=False),
        "cluster_region",
    ].unique()

    if len(duplicates):
        raise ValueError(
            f"Duplicate regional statistics: {duplicates.tolist()}"
        )

    print(f"Loaded regional LMM slopes: {REGIONAL_STATS_FILE}")
    print(stats_df.shape)
    print(stats_df.head())

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
        stats_df[["cluster_region", "beta_age", "p_BH_FDR"]],
        left_on="ROI",
        right_on="cluster_region",
        how="left",
        validate="many_to_one",
    )

    vmin, vmax = get_vmin_vmax(metric)
    figure_style()
    fig, ax = plt.subplots(figsize=(2.5, 2))

    plot_swanson_vector(
        merged_df["beryl_name"],
        merged_df["beta_age"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        br=br,
        empty_color="white",
        show_cbar=True,
        annotate=False,
        annotate_list=merged_df["swanson_name"],
        ax=ax,
    )

    # 1. make outlines thinner + lighter
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
    analysis_name = "quench" if metric == "delta_r_noise" else window

    fname = (
        f"Swanson_{metric}_{analysis_name}_"
        f"{WINDOW_LEN}ms_LMM_{C.RANDOM_FACTOR}.pdf"
    )

    os.makedirs(C.FIGPATH, exist_ok=True)
    save_path = os.path.join(C.FIGPATH, fname)

    fig.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    return fig, ax


def main():
    selected_metrics = ["r_noise", "delta_r_noise"]

    ROI_df = region_table(C.BERYL_NAMES)

    for metric in selected_metrics:
        if metric == "r_noise":
            for time_window in ["pre", "post"]:
                print(
                    f"Plotting Swanson for {metric} "
                    f"({time_window}-window)..."
                )
                stats_df = load_stats_results(
                    metric,
                    window=time_window,
                )
                plot_swanson(
                    metric,
                    ROI_df,
                    stats_df,
                    window=time_window,
                )

        elif metric == "delta_r_noise":
            print(f"Plotting Swanson for {metric} (quench)...")
            stats_df = load_stats_results(metric)
            plot_swanson(metric, ROI_df, stats_df)

        else:
            raise ValueError(f"Unsupported metric: {metric}")


if __name__ == "__main__":
    main()