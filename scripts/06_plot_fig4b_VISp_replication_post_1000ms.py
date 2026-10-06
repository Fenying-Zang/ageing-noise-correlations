"""
Figure 4b:VISp replication plot using the post-stimulus 1-s winsow.
Each coloured point is one insertion-level VISp estimate.
The grey line and annotation use the VISp slope from the mixed model.

"""
# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style, format_p_value
from scripts.utils.data_utils import add_age_group

# %% SETTINGS
FONT = "Arial"
WINDOW = "post"
WINDOW_LEN = 1000
REGION = "VISp"
FDR_ALPHA = 0.01
DATA_FILE = (C.DATAPATH / f"noise_pair_corr_merged_prepost_{WINDOW_LEN}ms.parquet")
REGIONAL_STATS_FILE = (C.RESULTSPATH / f"noise_regional_slopes_BH_FDR_{WINDOW_LEN}ms_{C.RANDOM_FACTOR}.csv")
OUTPUT_FILE = (C.FIGPATH / f"noise_VISp_replication_{WINDOW}_{WINDOW_LEN}ms_regional_slope_median.pdf")

# Liu et al. (2025) summary values used in the original figure.
LIU_AGE_MONTHS = np.array([8 * 7 / 30.44, 12.0])
LIU_R_NOISE = np.array([0.1941, 0.3110])

# %% STYLE AND PREPARATION
def apply_figure_style():
    plt.rcParams.update({
        "font.family": FONT,
        "font.weight": "normal",
        "axes.labelsize": 6,
        "axes.labelweight": "normal",
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.linewidth": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "lines.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


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


def load_visp_probe_data():
    pairs = standardize_pair_corr_columns(read_table(DATA_FILE))
    pairs["window"] = pairs["window"].astype(str)
    pairs = pairs[
        (pairs["window"] == WINDOW)
        & (pairs["cluster_region"] == REGION)
    ].dropna(subset=["r_noise", "mouse_age", "session_pid"])

    rows = []
    for pid, group in pairs.groupby("session_pid"):
        rows.append({
            "session_pid": pid,
            "mouse_age": group["mouse_age"].iloc[0],
            "mouse_age_months": group["mouse_age"].iloc[0] / 30.0,
            "n_pairs": len(group),
            "r_noise": group["r_noise"].median(),
        })

    probe = add_age_group(pd.DataFrame(rows))

    print(f"[Loaded data] {DATA_FILE}")
    print(
        f"{WINDOW} {REGION}: {len(pairs)} pairs, "
        f"{len(probe)} insertion-level points"
    )
    return probe


def load_visp_regional_slope():
    stats = read_table(REGIONAL_STATS_FILE)
    row = stats[
        (stats["analysis"].astype(str) == WINDOW)
        & (stats["random_factor"] == C.RANDOM_FACTOR)
        & (stats["cluster_region"] == REGION)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one {WINDOW} {REGION} row; found {len(row)} in "
            f"{REGIONAL_STATS_FILE}."
        )

    row = row.iloc[0]
    print(f"[Loaded stats] {REGIONAL_STATS_FILE}")
    print(
        f"beta_age={row['beta_age']:.6f}, "
        f"95% CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
        f"p_BH_FDR={row['p_BH_FDR']:.6g}"
    )
    return row


# %% PLOTTING
def reported_slope_line(probe, beta_age):
    """Draw the reported age slope through the centre of displayed points."""
    xline = np.linspace(
        probe["mouse_age_months"].min(),
        probe["mouse_age_months"].max(),
        200,
    )
    x_centre = probe["mouse_age_months"].mean()
    y_centre = probe["r_noise"].mean()
    age_difference_years = (xline - x_centre) * 30 / 365
    yline = y_centre + beta_age * age_difference_years
    return xline, yline


def plot_visp_replication(probe, stats_row, save=True):
    figure_style()
    apply_figure_style()

    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    sizes = np.sqrt(probe["n_pairs"]) * 7
    sns.scatterplot(
        data=probe,
        x="mouse_age_months",
        y="r_noise",
        hue="age_group",
        palette=C.PALETTE,
        s=sizes,
        alpha=0.8,
        marker=".",
        edgecolor="none",
        legend=False,
        ax=ax,
    )

    beta_age = float(stats_row["beta_age"])
    p_fdr = float(stats_row["p_BH_FDR"])

    if p_fdr < FDR_ALPHA:
        xline, yline = reported_slope_line(probe, beta_age)
        ax.plot(xline, yline, color="gray", linewidth=0.8)

    ax.plot(
        LIU_AGE_MONTHS,
        LIU_R_NOISE,
        linestyle="none",
        marker="D",
        color="black",
        markersize=2.5,
        zorder=10,
    )

    annotation = (
        rf"$\beta_{{\mathrm{{age}}}} = {beta_age:.3f}, "
        rf"p_{{\mathrm{{fdr}}}}$ {format_p_value(p_fdr)}"
    )
    ax.text(
        0.05,
        1.0,
        annotation,
        transform=ax.transAxes,
        fontsize=5,
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="D", linestyle="none", color="black",
            markersize=3, label="Liu et al., 2025",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", color="0.45",
            markersize=3, label="Zang et al., 2026",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=5,
        loc="center left",
        bbox_to_anchor=(1.02, 0.18),
        handletextpad=0.4,
    )

    ax.set_xlabel("Age (months)", fontsize=6)
    ax.set_ylabel("Noise correlations", fontsize=6)
    ax.set_xlim(1, 20.5)
    ax.set_xticks([2, 5, 10, 15, 20])
    ax.tick_params(
        axis="both", which="major", labelsize=4, width=0.4, length=2, pad=1
    )
    sns.despine(offset=2, trim=False, ax=ax)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.25, top=0.98)

    if save:
        save_figure(fig, OUTPUT_FILE, add_timestamp=False)
        print(f"[Saved] {OUTPUT_FILE}")

    return fig, ax


# %% RUN
if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    visp_probe_data = load_visp_probe_data()
    visp_stats = load_visp_regional_slope()
    plot_visp_replication(visp_probe_data, visp_stats, save=True)
