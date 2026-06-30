#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style
from scripts.utils.data_utils import add_age_group

FONT = "Arial"
AX_LABEL_SIZE = 6
TICK_SIZE = 4
PANEL_TEXT_SIZE = 5
ANALYSIS_TAG = "_VISp_VISpm_separate" #""

def apply_today_figure_style():
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
        "legend.fontsize": PANEL_TEXT_SIZE,
        "legend.title_fontsize": PANEL_TEXT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


def style_axis_text(ax):
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_SIZE, width=0.4, length=2, pad=1)
    ax.tick_params(axis="both", which="minor",
                   width=0.3, length=1.2)

    ax.xaxis.label.set_fontname(FONT)
    ax.yaxis.label.set_fontname(FONT)
    ax.xaxis.label.set_fontsize(AX_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AX_LABEL_SIZE)
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT)
        label.set_fontsize(TICK_SIZE)
        label.set_fontweight("normal")

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.4)


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def inv_fisher_z(z):
    return np.tanh(z)

def zmean(values, weights=None):
    z = fisher_z(np.asarray(values, dtype=float))
    if weights is None:
        return inv_fisher_z(np.nanmean(z))
    w = np.asarray(weights, dtype=float)
    return inv_fisher_z(np.nansum(w * z) / np.nansum(w))


def format_corr_label(kind):
    if kind == "noise":
        return r"$r_{\mathrm{noise}}$"
    elif kind == "signal":
        return r"$r_{\mathrm{signal}}$"
    return kind


def ci95_fisher(values):
    """
    95% CI after Fisher-z transform, then back-transform to r.
    Returns mean_r, low_r, high_r.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    z = fisher_z(values)
    z_mean = np.nanmean(z)

    if len(z) < 2:
        r_mean = inv_fisher_z(z_mean)
        return r_mean, np.nan, np.nan

    z_ci = 1.96 * np.nanstd(z, ddof=1) / np.sqrt(len(z))

    r_mean = inv_fisher_z(z_mean)
    r_low = inv_fisher_z(z_mean - z_ci)
    r_high = inv_fisher_z(z_mean + z_ci)

    return r_mean, r_low, r_high


def plot_pre_post_corr_pooled_pairs(
    kind="noise",
    save=True,
    fname=None,
):
    figure_style()
    apply_today_figure_style()

    value_col = f"r_{kind}"
    df = build_pre_post_pair_table(kind)
    df = df.dropna(subset=[value_col, "window", "age_group"])

    fig, ax = plt.subplots(1, 1, figsize=(1.375, 1.375))

    x_map = {"pre": 0, "post": 1}
    group_offsets = {"young": -0.08, "old": 0.08}

    for age_group, g in df.groupby("age_group"):
        color = C.PALETTE.get(age_group, "0.3")

        xs, means, lows, highs = [], [], [], []

        for window in ["pre", "post"]:
            vals = g.loc[g["window"].astype(str) == window, value_col].values
            mean_r, low_r, high_r = ci95_fisher(vals)

            xs.append(x_map[window] + group_offsets.get(age_group, 0))
            means.append(mean_r)
            lows.append(mean_r - low_r)
            highs.append(high_r - mean_r)

        ax.errorbar(
            xs, means,
            yerr=[lows, highs],
            color=color,
            marker="o",
            markersize=2.5,
            lw=0.7,
            elinewidth=0.6,
            capsize=1.5,
            label=age_group,
            zorder=3
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre", "post"])
    ax.set_xlabel("Time window")
    ax.set_ylabel(format_corr_label(kind))

    style_axis_text(ax)
    sns.despine(offset=2, trim=False, ax=ax)

    ax.legend(
        frameon=False,
        fontsize=PANEL_TEXT_SIZE,
        handlelength=1.0,
        loc="best"
    )

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.25, top=0.92)

    if fname is None:
        fname = C.FIGPATH / f"pre_post_{value_col}_pooled_pairs_by_age_group{ANALYSIS_TAG}.pdf"

    if save:
        save_figure(fig, fname, add_timestamp=True)

    return fig, ax, df


def standardize_pair_corr_columns(df):
    df = df.copy()

    rename_map = {
        "mouse_age_noise": "mouse_age",
        "mouse_name_noise": "mouse_name",
        "session_eid_noise": "session_eid",
        "cluster_geo_mean_fr_noise": "cluster_geo_mean_fr",
        "pair_distance_noise": "pair_distance",
        "align_event_noise": "align_event",
    }

    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    return df


def load_pair_corr_prepost(kind="noise"):
    path = C.DATAPATH / f"proj2_noise_pair_corr_merged_prepost_500ms{ANALYSIS_TAG}.parquet"
    df = read_table(path)
    df = standardize_pair_corr_columns(df)

    value_col = f"r_{kind}"
    df = df.dropna(subset=[value_col, "window", "mouse_age"]).copy()

    df["window"] = pd.Categorical(
        df["window"],
        categories=["pre", "post"],
        ordered=True,
    )

    df = add_age_group(df)

    print(f"[Loaded] {path}")
    print(df.shape)
    print(df["window"].value_counts(dropna=False))

    return df


def build_pre_post_pair_table(kind="noise"):
    value_col = f"r_{kind}"

    long_df = load_pair_corr_prepost(kind=kind)

    keep_cols = [
        "session_pid",
        "cluster_region",
        "cluster_id1",
        "cluster_id2",
        "mouse_age",
        "age_group",
        value_col,
        "window",
    ]

    long_df = long_df[keep_cols].copy()

    return long_df


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    plot_pre_post_corr_pooled_pairs(
        kind="noise",
        save=True,
        fname=C.FIGPATH / f"proj2_noise_fig02a_prepost_rnoise_pooled_pairs{ANALYSIS_TAG}.pdf",
    )
