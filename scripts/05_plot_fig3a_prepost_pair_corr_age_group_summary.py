"""
Figure 3a: Pre/post pairwise correlation summary by age group.
This script computes and plots the mean pre/post pairwise correlation for each age group

"""

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
N_BOOT = 2000
BOOTSTRAP_SEED = 20260917
WIN_LENGTH = 500

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


def build_pre_post_session_table(kind="noise", min_pairs=2):

    """ descriptive pre/post table with one value per session and window."""

    value_col = f"r_{kind}"
    pairs = load_pair_corr_prepost(kind=kind)

    required_cols = [
        "session_eid",
        "session_pid",
        "cluster_region",
        "window",
        "mouse_age",
        "age_group",
        value_col,
    ]
    missing_cols = [c for c in required_cols if c not in pairs.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    pairs = pairs.dropna(subset=required_cols).copy()
    pairs["window"] = pairs["window"].astype(str)

    insertion_region = (
        pairs.groupby(
            [
                "session_eid",
                "session_pid",
                "cluster_region",
                "window",
                "mouse_age",
                "age_group",
            ],
            observed=True,
        )
        .agg(
            **{
                value_col: (value_col, "mean"),
                "n_pairs": (value_col, "size"),
            }
        )
        .reset_index()
    )
    insertion_region = insertion_region[
        insertion_region["n_pairs"] >= min_pairs
    ].copy()

    session_region = (
        insertion_region.groupby(
            [
                "session_eid",
                "cluster_region",
                "window",
                "mouse_age",
                "age_group",
            ],
            observed=True,
        )
        .agg(
            **{
                value_col: (value_col, "mean"),
                "n_insertions": ("session_pid", "nunique"),
                "n_pairs": ("n_pairs", "sum"),
            }
        )
        .reset_index()
    )

    matched_session_regions = (
        session_region.groupby(
            ["session_eid", "cluster_region"], observed=True
        )["window"]
        .nunique()
        .loc[lambda x: x == 2]
        .reset_index()[["session_eid", "cluster_region"]]
    )
    session_region = session_region.merge(
        matched_session_regions,
        on=["session_eid", "cluster_region"],
        how="inner",
    )

    # Each session contributes one descriptive value per window. Regions are
    # averaged only within that session.
    session_df = (
        session_region.groupby(
            ["session_eid", "window", "mouse_age", "age_group"],
            observed=True,
        )
        .agg(
            **{
                value_col: (value_col, "mean"),
                "n_regions": ("cluster_region", "nunique"),
                "n_insertions": ("n_insertions", "sum"),
                "n_pairs": ("n_pairs", "sum"),
            }
        )
        .reset_index()
    )

    matched_sessions = (
        session_df.groupby("session_eid", observed=True)["window"]
        .nunique()
        .loc[lambda x: x == 2]
        .index
    )
    session_df = session_df[
        session_df["session_eid"].isin(matched_sessions)
    ].copy()
    session_df["window"] = pd.Categorical(
        session_df["window"],
        categories=["pre", "post"],
        ordered=True,
    )

    return session_df, session_region


def bootstrap_session_summary(
    session_df,
    value_col,
    n_boot=N_BOOT,
    seed=BOOTSTRAP_SEED,
):
    """Age-group means and 95% CIs from session-level cluster bootstrap."""
    rng = np.random.default_rng(seed)
    rows = []

    for age_group, group_df in session_df.groupby("age_group", observed=True):
        wide = group_df.pivot(
            index="session_eid",
            columns="window",
            values=value_col,
        ).dropna(subset=["pre", "post"])

        session_ids = wide.index.to_numpy()
        if len(session_ids) == 0:
            continue

        observed_means = wide[["pre", "post"]].mean(axis=0)
        boot_means = np.empty((n_boot, 2), dtype=float)

        for i in range(n_boot):
            sampled_ids = rng.choice(
                session_ids,
                size=len(session_ids),
                replace=True,
            )
            boot_means[i, :] = (
                wide.loc[sampled_ids, ["pre", "post"]]
                .mean(axis=0)
                .to_numpy()
            )

        ci_low = np.percentile(boot_means, 2.5, axis=0)
        ci_high = np.percentile(boot_means, 97.5, axis=0)

        for j, window in enumerate(["pre", "post"]):
            rows.append({
                "age_group": age_group,
                "window": window,
                "mean": observed_means[window],
                "ci_low": ci_low[j],
                "ci_high": ci_high[j],
                "n_sessions": len(session_ids),
            })

    return pd.DataFrame(rows)


def plot_pre_post_corr_session_summary(
    kind="noise",
    min_pairs=2,
    n_boot=N_BOOT,
    save=True,
    fname=None,
):
    """Descriptive young/old means with session-clustered bootstrap CIs."""
    figure_style()
    apply_today_figure_style()

    value_col = f"r_{kind}"
    session_df, session_region_df = build_pre_post_session_table(
        kind=kind,
        min_pairs=min_pairs,
    )
    summary = bootstrap_session_summary(
        session_df,
        value_col=value_col,
        n_boot=n_boot,
    )

    print("[Session-level descriptive summary]")
    print("session_df:", session_df.shape)
    print("session_region_df:", session_region_df.shape)
    print(summary)

    fig, ax = plt.subplots(1, 1, figsize=(1.375, 1.375))

    x_map = {"pre": 0, "post": 1}
    group_offsets = {"young": -0.08, "old": 0.08}

    for age_group, group_summary in summary.groupby(
        "age_group", observed=True
    ):
        group_summary = (
            group_summary.set_index("window")
            .loc[["pre", "post"]]
            .reset_index()
        )
        color = C.PALETTE.get(age_group, "0.3")

        xs = [
            x_map[window] + group_offsets.get(age_group, 0)
            for window in group_summary["window"]
        ]
        means = group_summary["mean"].to_numpy()
        lows = means - group_summary["ci_low"].to_numpy()
        highs = group_summary["ci_high"].to_numpy() - means

        ax.errorbar(
            xs,
            means,
            yerr=[lows, highs],
            color=color,
            marker="o",
            markersize=2.5,
            lw=0.7,
            elinewidth=0.6,
            capsize=1.5,
            label=age_group,
            zorder=3,
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
        loc="best",
    )

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.25, top=0.92)

    if fname is None:
        fname = (
            C.FIGPATH
            / f"pre_post_{value_col}_{WIN_LENGTH}ms_session_summary_by_age_group.pdf"
        )

    if save:
        save_figure(fig, fname, add_timestamp=True)

    return fig, ax, summary, session_df, session_region_df


def format_corr_label(kind):
    if kind == "noise":
        return r"$r_{\mathrm{noise}}$"
    elif kind == "signal":
        return r"$r_{\mathrm{signal}}$"
    return kind


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
    path = C.DATAPATH / f"noise_pair_corr_merged_prepost_{WIN_LENGTH}ms.parquet"
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


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    plot_pre_post_corr_session_summary(
        kind="noise",
        min_pairs=2,
        n_boot=N_BOOT,
        save=True,
        fname=C.FIGPATH / f"noise_fig3a_prepost_rnoise_{WIN_LENGTH}ms_session_summary.pdf",
    )
