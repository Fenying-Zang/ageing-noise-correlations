"""
Figure 3b: Global probe-level relationship between noise corr and age
Scatter plot of probe-level noise correlations vs. mouse age, with LMM slope and p-value.

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style, format_p_value
from scripts.utils.data_utils import add_age_group

FONT = "Arial"
AX_LABEL_SIZE = 6
TICK_SIZE = 4
PANEL_TEXT_SIZE = 5
WINDOW_LEN = 500
MODEL_COMPARISON_FILE = (C.RESULTSPATH / f"noise_three_model_AIC_BIC_{WINDOW_LEN}ms_session_eid.csv")
WINDOWS_TO_PLOT = ["pre", "post", "quench"]
YLIMS = {"pre": None, "post": None, "quench": (-0.15, 0.15)}

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


def pairs_to_probe_table(
    pairs_df,
    value_col="r_noise",
    out_col=None,
    min_pairs=5,
):
    """
    Aggregate pair-level correlations to one value per probe.
    Each output row = one session_pid.

    This pools all valid neuron pairs within a probe across recorded regions.
    """
    if out_col is None:
        out_col = value_col

    req = ["session_pid", "mouse_age", value_col]
    for c in req:
        if c not in pairs_df.columns:
            raise ValueError(f"Missing column in pairs_df: {c}")

    rows = []

    for pid, g in pairs_df.groupby("session_pid"):
        g = g.dropna(subset=[value_col]).copy()

        if len(g) < min_pairs:
            continue

        r_agg = np.nanmean(g[value_col].values)

        row = {
            "session_pid": pid,
            "mouse_age": g["mouse_age"].iloc[0],
            "mouse_age_months": g["mouse_age"].iloc[0] / 30.0,
            "n_pairs": len(g),
            out_col: r_agg,
        }

        if "cluster_region" in g.columns:
            row["n_regions"] = g["cluster_region"].nunique()

        rows.append(row)

    probe = pd.DataFrame(rows)

    if probe.empty:
        return probe

    probe = add_age_group(probe)

    print("[Probe-level aggregate]")
    print("rows:", len(probe))
    print("unique session_pid:", probe["session_pid"].nunique())
    if "n_regions" in probe.columns:
        print(probe["n_regions"].describe())

    return probe


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


def load_pair_corr_prepost(kind="noise"):
    path = C.DATAPATH / f"noise_pair_corr_merged_prepost_{WINDOW_LEN}ms.parquet"
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


def load_pair_corr_for_window_or_quench(kind="noise", window="pre"):
    """
    Load pair-level data from the formal merged pre/post table.
    Returns pair-level rows for pre/post, or paired delta rows for quench.
    Scatter plotting will aggregate these rows to session_pid x cluster_region.
    """
    if kind != "noise":
        raise ValueError("This figure script currently supports kind='noise' only.")

    df = load_pair_corr_prepost(kind=kind)

    if window in ["pre", "post"]:
        out = df[df["window"].astype(str) == window].copy()
        return out, "r_noise", "r_noise"

    if window == "quench":
        key_cols = [
            "session_pid",
            "cluster_region",
            "cluster_id1",
            "cluster_id2",
        ]

        pre = df[df["window"].astype(str) == "pre"].copy()
        post = df[df["window"].astype(str) == "post"].copy()

        keep_cols = key_cols + [
            "r_noise",
            "mouse_age",
            "age_group",
            "n_trials",
            "pair_distance",
            "cluster_geo_mean_fr",
        ]

        pre = pre[keep_cols].copy()
        post = post[key_cols + ["r_noise"]].copy()

        out = pre.merge(
            post,
            on=key_cols,
            how="inner",
            suffixes=("_pre", "_post"),
            validate="one_to_one",
        )

        out["delta_r_noise"] = out["r_noise_post"] - out["r_noise_pre"]

        return out, "delta_r_noise", "delta_r_noise"

    raise ValueError("window must be 'pre', 'post', or 'quench'.")


def load_global_stats(analysis_name):
    stats = read_table(MODEL_COMPARISON_FILE)
    stats = stats[
        (stats["analysis"] == analysis_name)
        & (stats["random_factor"] == C.RANDOM_FACTOR)
        & (stats["comparison"] == "common_age_effect")
    ].copy()

    print(f"[Loaded global LMM comparison] {MODEL_COMPARISON_FILE}")
    return stats


def line_from_reported_slope(df, y_col, beta_age):
    """Draw the reported LMM slope through the centre of the displayed dots."""
    xline = np.linspace(
        df["mouse_age_months"].min(),
        df["mouse_age_months"].max(),
        200,
    )
    x_centre = df["mouse_age_months"].mean()
    y_centre = df[y_col].mean()

    age_difference_years = (xline - x_centre) * 30 / 365
    yline = y_centre + beta_age * age_difference_years
    return xline, yline


def plot_scatter_pooled(
    df_probe,
    stats_df,
    y_col="r_noise",
    analysis_name="pre",
    granularity="probe_level",
    save=True,
    ylim=None,
):
    """
    Pooled scatter plot across regions.
    Each point = one session_pid × cluster_region aggregate.
    Point size ∝ sqrt(n_pairs).
    """
    figure_style()
    apply_today_figure_style()

    need = ["session_pid", "mouse_age_months", y_col]
    for c in need:
        if c not in df_probe.columns:
            raise ValueError(f"Missing column in df_probe: {c}")

    df = df_probe.dropna(subset=["mouse_age_months", y_col]).copy()

    beta_age = stats_df["beta_age_common"].values[0]
    # p_lrt = stats_df["LRT_p"].values[0]
    p_wald = stats_df["beta_age_common_p_wald"].values[0]

    if "n_pairs" in df.columns:
        sizes = np.sqrt(df["n_pairs"]) #* 2
    else:
        sizes = 10

    fig, ax = plt.subplots(1, 1, figsize=(1.375, 1.375))

    sns.scatterplot(
        x="mouse_age_months",
        y=y_col,
        data=df,
        hue="age_group" if "age_group" in df.columns else None,
        palette=C.PALETTE,
        s=sizes,
        alpha=0.8,
        marker=".",
        edgecolor="none",
        ax=ax,
        legend=False,
    )

    if p_wald < 0.05:
        xline, yline = line_from_reported_slope(
            df=df,
            y_col=y_col,
            beta_age=beta_age,
        )
        ax.plot(
            xline,
            yline,
            color="grey",
            linewidth=0.8,
        )

    p_wald_text = format_p_value(p_wald)

    txt = (
        rf"$\beta_{{\mathrm{{age}}}} = {beta_age:.3f}, "
        rf"p_{{\mathrm{{wald}}}}$ {p_wald_text}"
    )

    ax.text(
        0.05,
        1.0,
        txt,
        transform=ax.transAxes,
        fontsize=5,
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

    ax.set_xlabel("Age (months)")
    ax.set_ylabel(format_y_label(y_col))
    ax.set_xticks([5, 10, 15, 20])
    if ylim is not None:
        ax.set_ylim(ylim)

    style_axis_text(ax)
    sns.despine(offset=2, trim=False, ax=ax)

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.25, top=0.92)

    if save:
        fname = (
            C.FIGPATH
            / f"noise_fig03_{analysis_name}_{y_col}_age_scatter_"
              f"{granularity}_LMM.pdf"
        )
        save_figure(fig, fname, add_timestamp=True)

    return fig, ax, df


def format_y_label(y_col):
    if y_col == "r_noise":
        return r"$r_{\mathrm{noise}}$"
    if y_col == "delta_r_noise":
        return r"$\Delta r_{\mathrm{noise}}$"
    return y_col


def main_visual_corr(
    kind="noise",
    window="pre",
    plot_level="pooled",
    min_pairs=5,
    ylim=None
):
    """
    Build probe-region aggregates and plot global age scatter.
    Each point = one session_pid.
    """
    if plot_level != "pooled":
        raise ValueError("This script currently only handles plot_level='pooled'.")

    pairs, value_col, out_col = load_pair_corr_for_window_or_quench(
        kind=kind,
        window=window,
    )

    probe = pairs_to_probe_table(
        pairs_df=pairs,
        value_col=value_col,
        out_col=out_col,
        min_pairs=min_pairs,
    )

    if probe.empty:
        print("No probe-level entries after aggregation.")
        return

    probe = add_age_group(probe)

    stats_df = load_global_stats(analysis_name=window)

    fig, ax, plot_df = plot_scatter_pooled(
        df_probe=probe,
        stats_df=stats_df,
        y_col=out_col,
        analysis_name=window,
        granularity="probe_level",
        save=True,
        ylim=ylim
    )

    print("[Plot data]")
    print(plot_df.shape)
    print(plot_df[["mouse_age_months", out_col, "n_pairs"]].describe())

    return fig, ax, plot_df


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    for window in WINDOWS_TO_PLOT:
        main_visual_corr(
            kind="noise",
            window=window,
            plot_level="pooled",
            min_pairs=0,
            ylim=YLIMS[window],
        )