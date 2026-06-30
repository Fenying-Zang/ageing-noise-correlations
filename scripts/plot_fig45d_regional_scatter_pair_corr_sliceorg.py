#%%
import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ibl_style.utils import MM_TO_INCH
import figrid as fg
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import (
    figure_style,
    format_bf_annotation,
    create_slice_org_axes_17panels,
)
from scripts.utils.data_utils import add_age_group
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian

FONT = "Arial"
AX_LABEL_SIZE = 6
TICK_SIZE = 4
PANEL_TEXT_SIZE = 5
ANALYSIS_TAG = "_VISp_VISpm_separate"

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


def load_pair_corr_for_window_or_quench(kind="noise", window="pre"):
    """
    Load pair-level data from the formal merged pre/post table.
    Returns pair-level rows for pre/post, or paired delta rows for quench.
    Scatter plotting will aggregate these rows to session_pid × cluster_region.
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
        )

        out["delta_r_noise"] = out["r_noise_post"] - out["r_noise_pre"]

        return out, "delta_r_noise", "delta_r_noise"

    raise ValueError("window must be 'pre', 'post', or 'quench'.")


def format_y_label(y_col):
    if y_col == "r_noise":
        return r"$r_{\mathrm{noise}}$"
    if y_col == "delta_r_noise":
        return r"$\Delta r_{\mathrm{noise}}$"
    return y_col


def pairs_to_probe_region_table(
    pairs_df,
    value_col="r_noise",
    out_col=None,
    min_pairs=5,
    weight_by_trials=False,
):
    """
    Aggregate pair-level correlations to one value per probe-region entry.
    Each output row = one session_pid × cluster_region.
    """
    if out_col is None:
        out_col = value_col

    req = ["session_pid", "cluster_region", "mouse_age", value_col]
    for c in req:
        if c not in pairs_df.columns:
            raise ValueError(f"Missing column in pairs_df: {c}")

    rows = []

    for (pid, region), g in pairs_df.groupby(["session_pid", "cluster_region"]):
        g = g.dropna(subset=[value_col]).copy()

        if len(g) < min_pairs:
            continue

        if weight_by_trials and "n_trials" in g.columns:
            r_agg = zmean(g[value_col].values, weights=g["n_trials"].values)
        else:
            r_agg = zmean(g[value_col].values)

        rows.append({
            "session_pid": pid,
            "cluster_region": region,
            "mouse_age": g["mouse_age"].iloc[0],
            "mouse_age_months": g["mouse_age"].iloc[0] / 30.0,
            "n_pairs": len(g),
            out_col: r_agg,
        })

    probe_region = pd.DataFrame(rows)

    if probe_region.empty:
        return probe_region

    probe_region = add_age_group(probe_region)

    print("[Probe-region aggregate]")
    print("rows:", len(probe_region))
    print("unique session_pid:", probe_region["session_pid"].nunique())
    print("unique regions:", probe_region["cluster_region"].nunique())

    return probe_region


def load_regional_stats(analysis_name="pre"):
    bf_path = C.RESULTSPATH / f"proj2_noise_pairlevel_regional_BFs_pre_post_quench_500ms{ANALYSIS_TAG}.csv"

    permut_files = sorted(
        glob(str(
            C.RESULTSPATH
            / f"proj2_noise_pairlevel_regional_permutation_{analysis_name}_*perms_500ms{ANALYSIS_TAG}.csv"
        )),
        key=os.path.getmtime,
    )

    if not permut_files:
        raise FileNotFoundError(
            f"No regional permutation file found for analysis: {analysis_name}"
        )

    permut_path = Path(permut_files[-1])

    bf_df = read_table(bf_path)
    permut_df = read_table(permut_path)

    bf_df = bf_df[bf_df["analysis"] == analysis_name].copy()

    print(f"[Loaded regional BF] {bf_path}")
    print(f"[Loaded regional permutation] {permut_path}")

    return bf_df, permut_df


def fit_pairlevel_line_for_plot(df_pairs_region, y_col):
    """
    Fit the display line from pair-level rows, not from probe-region aggregates.

    Model:
        y_col ~ age_years + n_trials + pair_distance + cluster_geo_mean_fr

    The displayed line varies age while holding covariates at their regional medians.
    """
    required_cols = [
        "mouse_age",
        y_col,
        "n_trials",
        "pair_distance",
        "cluster_geo_mean_fr",
    ]

    missing_cols = [c for c in required_cols if c not in df_pairs_region.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for pair-level fitted line: {missing_cols}")

    fit_df = df_pairs_region[required_cols].copy()

    fit_df["age_years"] = fit_df["mouse_age"] / 365.0

    numeric_cols = [
        y_col,
        "age_years",
        "n_trials",
        "pair_distance",
        "cluster_geo_mean_fr",
    ]

    for col in numeric_cols:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")

    fit_df = fit_df.dropna(subset=numeric_cols).copy()
    fit_df = fit_df[np.isfinite(fit_df[numeric_cols]).all(axis=1)].copy()

    if fit_df.empty:
        return None, None

    formula = (
        f"{y_col} ~ age_years + n_trials + "
        "pair_distance + cluster_geo_mean_fr"
    )

    model = glm(
        formula=formula,
        data=fit_df,
        family=Gaussian(),
        eval_env=0,
    ).fit()

    xgrid_years = np.linspace(
        fit_df["age_years"].min(),
        fit_df["age_years"].max(),
        200,
    )

    new = pd.DataFrame({
        "age_years": xgrid_years,
        "n_trials": np.nanmedian(fit_df["n_trials"]),
        "pair_distance": np.nanmedian(fit_df["pair_distance"]),
        "cluster_geo_mean_fr": np.nanmedian(fit_df["cluster_geo_mean_fr"]),
    })

    yhat = model.predict(new)

    return xgrid_years * 12.0, yhat


def plot_scatter_by_region_simple(
    df_probe_region,
    df_pairs,
    permut_df,
    bf_df,
    y_col="r_noise",
    analysis_name="pre",
    granularity="probe_region_agg",
    save=True,
    title_prefix="Regional specificity",
):
    """
    Slice-organized scatter: one panel per region.
    Each point = one probe-region aggregate.
    df_probe_region must contain:
    ['session_pid', 'cluster_region', 'mouse_age_months', y_col]
    """
    fig, axs = create_slice_org_axes_17panels(fg, MM_TO_INCH)
    figure_style()
    apply_today_figure_style()

    need = ["session_pid", "cluster_region", "mouse_age_months", y_col]
    for c in need:
        if c not in df_probe_region.columns:
            raise ValueError(f"Missing column in df_probe_region: {c}")

    for region in C.ROIS_vis_seperate:
        ax = axs[region]
        sub = df_probe_region[df_probe_region["cluster_region"] == region].copy()
        sub_pairs = df_pairs[df_pairs["cluster_region"] == region].copy()

        if sub.empty:
            continue

        sub = sub.dropna(subset=["mouse_age_months", y_col])
        sub = sub.drop_duplicates(subset=["session_pid", "cluster_region"])

        sub_perm = permut_df[permut_df["cluster_region"] == region]
        sub_bf = bf_df[bf_df["cluster_region"] == region]

        if sub_perm.empty or sub_bf.empty:
            continue

        slope_age = sub_perm["observed_val"].values[0]
        p_perm = sub_perm["p_perm"].values[0]
        bf_conclusion = sub_bf["BF10_age_category"].values[0]
        bf10 = sub_bf["BF10_age"].values[0]

        if "n_pairs" in sub.columns:
            sizes = np.sqrt(sub["n_pairs"])
        else:
            sizes = 10

        sns.scatterplot(
            x="mouse_age_months",
            y=y_col,
            data=sub,
            hue="age_group" if "age_group" in sub.columns else None,
            marker=".",
            legend=False,
            s=sizes,
            palette=C.PALETTE,
            ax=ax,
            edgecolor="none",
            alpha=0.8,
        )

        if bf_conclusion in ("strong H1", "moderate H1"):
            xline, yline = fit_pairlevel_line_for_plot(
                df_pairs_region=sub_pairs,
                y_col=y_col,
            )

            if xline is not None:
                ax.plot(
                    xline,
                    yline,
                    color="grey",
                    linewidth=0.8,
                )

        txt = format_bf_annotation(
            slope_age,
            p_perm,
            bf10,
            bf_conclusion,
            beta_label="age",
            big_bf=100,
        )

        ax.text(
            0.05,
            1.25,
            txt,
            transform=ax.transAxes,
            fontsize=5,
            va="top",
            linespacing=0.8,
        )

        # ax.set_title(f"{region}", fontsize=8)
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")

        if region in ["ACB", "OLF", "MBm", "PO"]:
            ax.set_xticks([5, 10, 15, 20])
        else:
            ax.set_xticks([])

        style_axis_text(ax)
        sns.despine(offset=2, trim=False, ax=ax)

    fig.suptitle(f"{title_prefix}: {format_y_label(y_col)}", font="Arial", fontsize=8)
    fig.supxlabel("Age (months)", font="Arial", fontsize=8).set_y(0.35)

    if save:
        fname = (
            C.FIGPATH
            / f"proj2_noise_fig04_{analysis_name}_{y_col}_{granularity}_sliceorg{ANALYSIS_TAG}.pdf"
        )
        save_figure(fig, fname, add_timestamp=True)

    return fig, axs


def main_visual_corr_sliceorg(
    kind="noise",
    window="pre",
    min_pairs=5,
    weight_by_trials=False,
):
    """
    Build probe-region aggregates and plot slice-organized regional scatter.
    """
    pairs, value_col, out_col = load_pair_corr_for_window_or_quench(
        kind=kind,
        window=window,
    )

    probe_region = pairs_to_probe_region_table(
        pairs_df=pairs,
        value_col=value_col,
        out_col=out_col,
        min_pairs=min_pairs,
        weight_by_trials=weight_by_trials,
    )

    if probe_region.empty:
        print("No probe-region entries after aggregation.")
        return

    bf_df, permut_df = load_regional_stats(analysis_name=window)

    fig, axs = plot_scatter_by_region_simple(
        df_probe_region=probe_region,
        df_pairs=pairs,
        permut_df=permut_df,
        bf_df=bf_df,
        y_col=out_col,
        analysis_name=window,
        granularity="probe_region_agg",
        save=True,
        title_prefix=f"Regional specificity of {window} r_noise",
    )

    print("[Plot data]")
    print(probe_region.shape)
    print(probe_region.groupby("cluster_region").size())

    return fig, axs, probe_region


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()

    main_visual_corr_sliceorg(
        kind="noise",
        window="post",
        min_pairs=0,
        weight_by_trials=False,
    )
