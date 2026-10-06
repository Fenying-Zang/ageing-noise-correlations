"""
Regional supplementary plots for the three noise-correlation validations.

Each figure uses the 17-region slice-organization layout. Within
each region, pair-level observations are binned exactly as in the pooled main
figure and shown as the bin mean +/- 95% CI. The annotation is read from the
regional LMM simple-slope results and contains only beta and BH-FDR p.
"""

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.utils import MM_TO_INCH
import figrid as fg
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style, create_slice_org_axes_17panels, format_p_value

# %% SETTINGS
FONT = "Arial"
AX_LABEL_SIZE = 6
TICK_SIZE = 4
PANEL_TEXT_SIZE = 5
WINDOW = "post"
WIN_LEN = 500
NBINS = 10
DATA_FILE = (
    C.DATAPATH
    / f"noise_pair_corr_merged_prepost_{WIN_LEN}ms.parquet"
)

REGIONAL_STATS_FILE = (
    C.RESULTSPATH
    / "noise_validation_regional_slopes_BH_FDR_"
      f"{WINDOW}_{WIN_LEN}ms_{C.RANDOM_FACTOR}.csv"
)

PANELS = [
    {
        "outcome": "r_noise",
        "predictor": "cluster_geo_mean_fr",
        "x_label": "Pair geometric mean firing rate (sp/s)",
        "beta_subscript": r"\mathrm{FR}",
        "beta_decimals": 4,
        "max_x": None,
        "bottom_ticks": [0, 50, 100, 150],
    },
    {
        "outcome": "r_noise",
        "predictor": "pair_distance",
        "x_label": "Pair distance (µm)",
        "beta_subscript": r"\mathrm{distance}",
        "beta_decimals": 5,
        "max_x": 1000,
        "bottom_ticks": [0, 500, 1000],
    },
    {
        "outcome": "r_noise",
        "predictor": "r_signal",
        "x_label": "Signal correlations",
        "beta_subscript": r"r_{\mathrm{signal}}",
        "beta_decimals": 3,
        "max_x": None,
        "bottom_ticks": [-0.5, 0, 0.5],
    },
]

BOTTOM_REGIONS = ["ACB", "OLF", "MBm", "PO"]
DARK_GRAY = "0.15"


# %% STYLE AND LOADING
def apply_figure_style():
    plt.rcParams.update({
        "font.family": FONT,
        "font.weight": "normal",
        "axes.labelsize": AX_LABEL_SIZE,
        "axes.labelweight": "normal",
        "axes.titlesize": PANEL_TEXT_SIZE,
        "axes.titleweight": "normal",
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "lines.linewidth": 0.7,
        "lines.markersize": 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


def style_axis(ax):
    ax.tick_params(
        axis="both", which="major",
        labelsize=TICK_SIZE, width=0.4, length=2, pad=1,
    )
    ax.tick_params(axis="both", which="minor", width=0.3, length=1.2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT)
        label.set_fontsize(TICK_SIZE)
        label.set_fontweight("normal")

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.4)


def standardize_pair_corr_columns(df):
    df = df.copy()
    column_map = {
        "mouse_age_noise": "mouse_age",
        "mouse_name_noise": "mouse_name",
        "session_eid_noise": "session_eid",
        "session_pid_noise": "session_pid",
        "cluster_region_noise": "cluster_region",
        "cluster_geo_mean_fr_noise": "cluster_geo_mean_fr",
        "pair_distance_noise": "pair_distance",
        "align_event_noise": "align_event",
        "n_trials_noise": "n_trials",
    }

    for old_col, new_col in column_map.items():
        if old_col not in df.columns:
            continue
        if new_col in df.columns:
            df[new_col] = df[new_col].combine_first(df[old_col])
        else:
            df[new_col] = df[old_col]

    return df


def load_post_data():
    df = standardize_pair_corr_columns(read_table(DATA_FILE))
    df["window"] = df["window"].astype(str)
    df = df[df["window"] == WINDOW].copy()

    print(f"[Loaded data] {DATA_FILE}")
    print(df.shape)
    return df


def load_regional_stats(outcome, predictor):
    stats = read_table(REGIONAL_STATS_FILE)
    stats = stats[
        (stats["window"].astype(str) == WINDOW)
        & (stats["outcome"] == outcome)
        & (stats["predictor"] == predictor)
        & (stats["random_factor"] == C.RANDOM_FACTOR)
    ].copy()

    print(
        f"[Loaded stats] {outcome} ~ {predictor}: "
        f"{len(stats)} regions from {REGIONAL_STATS_FILE}"
    )
    return stats


# %% PLOTTING HELPERS
def bin_mean_and_ci(x, y, nbins=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(x.min(), x.max(), nbins + 1)
    bin_index = np.digitize(x, edges) - 1

    x_mid, y_mean, y_ci = [], [], []
    for index in range(nbins):
        in_bin = bin_index == index
        if in_bin.sum() < 2:
            continue

        x_mid.append(0.5 * (edges[index] + edges[index + 1]))
        y_mean.append(y[in_bin].mean())
        y_ci.append(1.96 * y[in_bin].std(ddof=1) / np.sqrt(in_bin.sum()))

    return np.asarray(x_mid), np.asarray(y_mean), np.asarray(y_ci)


def plot_one_validation(df, panel):
    outcome = panel["outcome"]
    predictor = panel["predictor"]
    stats = load_regional_stats(outcome, predictor)

    plot_df = df.dropna(subset=[outcome, predictor, "cluster_region"]).copy()
    if panel["max_x"] is not None:
        plot_df = plot_df[plot_df[predictor] <= panel["max_x"]].copy()

    fig, axs = create_slice_org_axes_17panels(fg, MM_TO_INCH)
    figure_style()
    apply_figure_style()

    for region in C.ROIS_vis_seperate:
        ax = axs[region]
        region_data = plot_df[plot_df["cluster_region"] == region]
        region_stats = stats[stats["cluster_region"] == region]

        if region_data.empty or region_stats.empty:
            ax.axis("off")
            continue

        x_mid, y_mean, y_ci = bin_mean_and_ci(
            region_data[predictor],
            region_data[outcome],
            nbins=NBINS,
        )

        ax.plot(x_mid, y_mean, color=DARK_GRAY, linewidth=1.0, zorder=3)
        ax.fill_between(
            x_mid,
            y_mean - y_ci,
            y_mean + y_ci,
            color=DARK_GRAY,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

        beta = float(region_stats.iloc[0]["beta_predictor"])
        p_fdr = float(region_stats.iloc[0]["p_BH_FDR"])
        beta_text = f"{beta:.{panel['beta_decimals']}f}"
        p_text = format_p_value(p_fdr)

        annotation = (
            rf"$\beta_{{{panel['beta_subscript']}}} = {beta_text}, "
            rf"p_{{\mathrm{{fdr}}}}$ {p_text}"
        )
        ax.text(
            0.05,
            1.25,
            annotation,
            transform=ax.transAxes,
            fontsize=PANEL_TEXT_SIZE,
            va="top",
            linespacing=0.8,
        )

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        if region in BOTTOM_REGIONS:
            ax.set_xticks(panel["bottom_ticks"])
        else:
            ax.set_xticks([])

        style_axis(ax)
        sns.despine(offset=2, trim=False, ax=ax)

    fig.suptitle(
        f"Regional specificity: Noise correlations vs {panel['x_label'].lower()}",
        font=FONT,
        fontsize=8,
    )
    fig.supxlabel(panel["x_label"], font=FONT, fontsize=8).set_y(0.35)
    fig.supylabel("Noise correlations", font=FONT, fontsize=8).set_x(0.02)

    output_file = (
        C.FIGPATH
        / f"noise_validation_regional_{outcome}_vs_{predictor}_"
          f"{WINDOW}_{WIN_LEN}ms_sliceorg_LMM_FDR.pdf"
    )
    save_figure(fig, output_file, add_timestamp=True)
    print(f"[Saved] {output_file}")

    return fig, axs


# %% run the three supplementary figures
if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    post_data = load_post_data()

    for panel_settings in PANELS:
        plot_one_validation(post_data, panel_settings)
