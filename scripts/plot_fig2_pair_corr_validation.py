"""
script for plots for pairwise noise correlation vs distance, firing rate, etc.

- relationship between pairwise noise correlation and distance
- relationship between pairwise noise correlation and firing rate
- relationship between pairwise noise correlation and signal correlation

"""
#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style, create_slice_org_axes
from scripts.utils.data_utils import add_age_group
from ibl_style.utils import MM_TO_INCH
import figrid as fg
import config as C
import pandas as pd
from scripts.utils.plot_utils import format_bf_annotation_negative
from scripts.utils.noise_corr_utils import (
    compute_noise_corr_by_condition_average,
    compute_noise_corr_pooled_withincond_zscore,
    compute_signal_corr_by_condition_means,
    combine_corr_results
)

CORR_COLORS = {
    "r_noise": "0.15",
    "r_signal": "0.15",
}
ANALYSIS_TAG = "_VISp_VISpm_separate"


def load_pairlevel_omnibus_stats(y_col, x_col, window="post"):
    """
    Load pair-level omnibus stats table saved by the updated permutation script.
    """
    fname = (
        C.RESULTSPATH
        / f"pairlevel_Omnibus_{y_col}_vs_{x_col}_{window}_1000permutation_"
          f"{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_500{ANALYSIS_TAG}.csv"
    )

    print("Looking for:", fname)
    print("Exists?", fname.exists())

    if not fname.exists():
        return None

    df = pd.read_csv(fname)
    if df.empty:
        print(f"[Warn] empty stats file: {fname}")
        return None

    return df.iloc[0].to_dict()


def add_unified_stats_annotation(
    ax,
    y_col,
    x_col,
    window="post",
    beta_label=None,
    big_bf=100,
):
    """
    Add unified annotation using precomputed permutation + BF results.
    """
    row = load_pairlevel_omnibus_stats(
        y_col=y_col,
        x_col=x_col,
        window=window,
    )

    if row is None:
        return

    beta = row.get("observed_val", np.nan)
    p_perm = row.get("p_perm", np.nan)
    BF10 = row.get("BF10", np.nan)
    BF_conclusion = row.get("BF_conclusion", "")

    if not np.isfinite(beta) or not np.isfinite(p_perm):
        return

    if np.isinf(BF10):
        BF10 = big_bf + 10

    if beta_label is None:
        beta_label = x_col

    txt = format_bf_annotation_negative(
        beta,
        p_perm,
        BF10,
        BF_conclusion,
        beta_label=beta_label,
        big_bf=big_bf,
    )

    ax.text(
        0.05,
        1.0,
        txt,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
    )


def standardize_pair_corr_columns(df):
    """
    Standardize the new merged pre/post pair-correlation table
    so the plotting functions can keep using the old column names.
    """
    df = df.copy()

    col_map = {
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

    for old_col, new_col in col_map.items():
        if old_col not in df.columns:
            continue

        if new_col in df.columns:
            df[new_col] = df[new_col].combine_first(df[old_col])
        else:
            df[new_col] = df[old_col]

    return df


def load_pair_corr_for_validation_plot(window="post"):
    """
    Load the new merged pre/post pair-level correlation table.

    Returns:
      df_noise: rows with r_noise
      df_signal: rows with r_signal
      df_merged: rows with both r_noise and r_signal
    """
    path = C.DATAPATH / f"proj2_noise_pair_corr_merged_prepost_500ms{ANALYSIS_TAG}.parquet"

    df = read_table(path)
    df = standardize_pair_corr_columns(df)

    if "window" not in df.columns:
        raise ValueError("Expected column 'window' in the merged pre/post pair-correlation table.")

    df["window"] = df["window"].astype(str)

    if window is not None:
        df = df[df["window"] == window].copy()

    if df.empty:
        raise ValueError(f"No rows left after filtering window={window}.")

    if "mouse_age" not in df.columns:
        raise ValueError("Missing 'mouse_age' after column standardization.")

    df = add_age_group(df)
    df["mouse_age_months"] = df["mouse_age"] / 30.0
    df["age_years"] = df["mouse_age"] / 365.0

    if "r_noise" in df.columns:
        df_noise = df.dropna(subset=["r_noise"]).copy()
    else:
        df_noise = pd.DataFrame()

    if "r_signal" in df.columns:
        df_signal = df.dropna(subset=["r_signal"]).copy()
        df_merged = df.dropna(subset=["r_noise", "r_signal"]).copy()
    else:
        print("[Warn] r_signal not found. Signal-correlation plots will be skipped.")
        df_signal = pd.DataFrame()
        df_merged = pd.DataFrame()

    print(f"[Loaded] {path}")
    print(f"window={window}")
    print("df_noise:", df_noise.shape)
    print("df_signal:", df_signal.shape)
    print("df_merged:", df_merged.shape)

    return df_noise, df_signal, df_merged

DARK_GRAY = "0.15"
def plot_corr_pooled(
    df,
    x_col="pair_distance",     # e.g. 'pair_distance' or 'cluster_geo_mean_fr'
    y_col="r_noise",           # e.g. 'r_noise' or 'r_signal'
    hue=None,                  # e.g. 'cluster_region', 'mouse_age', 'project', or None
    mode="binned",             # 'binned' or 'scatter'
    nbins=20,
    qcut=False,
    max_x=None,
    sample_n=5000,             
    annotate_corr=True,        
    window="post",
    save=True,
    fname=None,
    line_color=DARK_GRAY,
):
    """
    通用 pooled 可视化：在一个大图中画 y_col 对 x_col 的关系。
    - mode='binned'：对 x 分箱画均值曲线（稳、快，适合大数据）
    - mode='scatter'：散点 + LOWESS（适合中小数据）
    - hue：可选分层（仅在 scatter 模式下建议使用；binned 模式也支持，但会对每个 hue 独立分箱）
    """
    figure_style()

    # ---- 基础清理 ----
    if x_col not in df or y_col not in df:
        print(f"[Skip] columns {x_col} or {y_col} missing.")
        return
    df = df.dropna(subset=[x_col, y_col])
    if max_x is not None:
        df = df[df[x_col] <= max_x]
    if df.empty:
        print("[Skip] empty df.")
        return

    if x_col == "cluster_geo_mean_fr" and "cluster_geo_mean_fr" not in df.columns:
        if {"cluster_id1_fr", "cluster_id2_fr"}.issubset(df.columns):
            df["cluster_geo_mean_fr"] = np.sqrt(df["cluster_id1_fr"] * df["cluster_id2_fr"])
        else:
            print("[Skip] missing FR columns to build 'cluster_geo_mean_fr'.")
            return

    fig, ax = plt.subplots(figsize=(3.6, 2.8))

    if mode == "scatter":
        if hue is None:
            sns.regplot(
                data=df, x=x_col, y=y_col,
                scatter_kws=dict(s=6, alpha=0.25, linewidths=0),
                line_kws=dict(lw=1.2), color='grey',
                lowess=True, ax=ax
            )
        else:
            palette = None
            if hasattr(C, "PALETTE") and isinstance(C.PALETTE, dict):
                palette = C.PALETTE
            for key, sub in df.groupby(hue):
                sub_plot = sub if len(sub) <= sample_n else sub.sample(sample_n, random_state=0)
                sns.regplot(
                    data=sub_plot, x=x_col, y=y_col,
                    scatter_kws=dict(s=6, alpha=0.18, linewidths=0),
                    line_kws=dict(lw=1.0),#color='grey',
                    lowess=True, ax=ax, label=str(key), color=None if palette is None else palette.get(key, None)
                )
            ax.legend(frameon=False, fontsize=8, title=hue)

        if annotate_corr:
            # unified stats annotation from saved permutation+BF results
            beta_label = {
                "pair_distance": "distance",
                "cluster_geo_mean_fr": "FR",
                "r_signal": "r_signal",
            }.get(x_col, x_col)
            # add_unified_stats_annotation(ax, y_col=y_col, x_col=x_col, beta_label=beta_label, big_bf=100)
            add_unified_stats_annotation(
                ax,
                y_col=y_col,
                x_col=x_col,
                window=window,
                beta_label=beta_label,
                big_bf=100,
            )


    else:
        def _bin_xy(x, y, nb, q):
            x = np.asarray(x); y = np.asarray(y)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size == 0:
                return np.array([]), np.array([]), np.array([])
            bins = np.unique(np.quantile(x, np.linspace(0, 1, nb+1))) if q else np.linspace(x.min(), x.max(), nb+1)
            idx = np.digitize(x, bins) - 1
            xm, ym, se, ci = [], [], [], []
            for b in range(len(bins)-1):
                m = idx == b
                if m.sum() < 2:
                    continue
                xm.append(0.5*(bins[b]+bins[b+1]))
                ym.append(y[m].mean())
                se.append(y[m].std(ddof=1)/np.sqrt(m.sum()))
                # se.append(y[m].std(ddof=1)/np.sqrt(m.sum()))
                ci.append(1.96 * y[m].std(ddof=1) / np.sqrt(m.sum()))
            return np.array(xm), np.array(ym), np.array(se), np.array(ci)

        if hue is None:
            xc, ym, se, ci = _bin_xy(df[x_col], df[y_col], nbins, qcut)
            if xc.size:
                # ax.plot(xc, ym, lw=1.6)
                ax.plot(xc, ym, lw=1.3, color=line_color, zorder=3)
                # ax.fill_between(xc, ym-se, ym+se, alpha=0.18, color=line_color,
                #                 zorder=2,linewidth=0)
                ax.fill_between(xc, ym-ci, ym+ci, alpha=0.18, color=line_color,
                                zorder=2,linewidth=0)
                
        else:
            palette = None
            if hasattr(C, "PALETTE") and isinstance(C.PALETTE, dict):
                palette = C.PALETTE
            for key, sub in df.groupby(hue):
                xc, ym, se, ci = _bin_xy(sub[x_col], sub[y_col], nbins, qcut)
                if not xc.size:
                    continue
                color = line_color if palette is None else palette.get(key, line_color)

                # ax.plot(xc, ym, lw=1.3, label=str(key), color=None if palette is None else palette.get(key, None)) #color='brown'
                ax.plot(xc, ym, lw=1.3, label=str(key), color=color)

                # ax.fill_between(xc, ym-se, ym+se, alpha=0.14, linewidth=0, color=color)
                ax.fill_between(xc, ym-ci, ym+ci, alpha=0.14, linewidth=0, color=color)
            ax.legend(frameon=False, fontsize=8, title=hue)

        if annotate_corr:
            # unified stats annotation from saved permutation+BF results
            beta_label = {
                "pair_distance": "distance",
                "cluster_geo_mean_fr": "FR",
                "r_signal": "r_signal",
            }.get(x_col, x_col)
            # add_unified_stats_annotation(ax, y_col=y_col, x_col=x_col, beta_label=beta_label, big_bf=100)
            add_unified_stats_annotation(
                ax,
                y_col=y_col,
                x_col=x_col,
                window=window,
                beta_label=beta_label,
                big_bf=100,
            )

    xmax = df[x_col].max()
    if max_x is not None:
        ax.set_xlim(right=max_x if max_x is not None else None)
    if x_col == "pair_distance": 
        ax.set_xlabel(f"{x_col} (µm)")
    elif x_col == "cluster_geo_mean_fr":
        ax.set_xlabel("Pair geometric mean firing rate (sp/s)")
    else:
        ax.set_xlabel(x_col)
    ax.set_ylabel(y_col.replace("_", " "))
    # ---- Axis font sizes (match usual pooled-plot style) ----
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    sns.despine()
    plt.tight_layout()

    if save:
        if fname is None:
            fname = C.FIGPATH / f"pooled_{y_col}_vs_{x_col}{'_by_' + hue if hue else ''}.pdf"
        save_figure(fig, fname, add_timestamp=False)

    return fig


def plot_corr_bivariate_distribution(
    df,
    x_col="pair_distance",
    y_col="r_noise",
    window="post",
    bins=40,
    max_x=None,
    overlay_binned=True,
    nbins_line=12,
    annotate_stats=True,
    save=True,
    fname=None,
):
    """
    Bivariate distribution plot for pair-level validation.

    Shows the joint density of pair-level observations using sns.histplot(x, y),
    with optional binned mean trend overlay.
    """
    figure_style()

    needed_cols = [x_col, y_col]
    df = df.dropna(subset=needed_cols).copy()

    if max_x is not None:
        df = df[df[x_col] <= max_x].copy()

    if df.empty:
        print(f"[Skip] empty df for {y_col} vs {x_col}")
        return None

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    sns.histplot(
        data=df,
        x=x_col,
        y=y_col,
        bins=bins,
        cbar=True,
        pthresh=0.01,
        ax=ax,
    )

    if overlay_binned:
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        bins_line = np.linspace(x.min(), x.max(), nbins_line + 1)
        idx = np.digitize(x, bins_line) - 1

        xm, ym = [], []
        for b in range(len(bins_line) - 1):
            m = idx == b
            if m.sum() < 10:
                continue
            xm.append(0.5 * (bins_line[b] + bins_line[b + 1]))
            ym.append(np.nanmean(y[m]))

        ax.plot(xm, ym, color="black", lw=0.8)

    if annotate_stats:
        beta_label = {
            "pair_distance": "distance",
            "cluster_geo_mean_fr": "FR",
            "r_signal": "r_signal",
        }.get(x_col, x_col)

        add_unified_stats_annotation(
            ax,
            y_col=y_col,
            x_col=x_col,
            window=window,
            beta_label=beta_label,
            big_bf=100,
        )

    if x_col == "pair_distance":
        ax.set_xlabel("Pair distance (µm)")
    elif x_col == "cluster_geo_mean_fr":
        ax.set_xlabel("Pair geometric mean firing rate (sp/s)")
    elif x_col == "r_signal":
        ax.set_xlabel(r"$r_{\mathrm{signal}}$")
    else:
        ax.set_xlabel(x_col.replace("_", " "))

    if y_col == "r_noise":
        ax.set_ylabel(r"$r_{\mathrm{noise}}$")
    elif y_col == "r_signal":
        ax.set_ylabel(r"$r_{\mathrm{signal}}$")
    else:
        ax.set_ylabel(y_col.replace("_", " "))

    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        if fname is None:
            fname = (
                C.FIGPATH
                / f"bivariate_{y_col}_vs_{x_col}_{window}{ANALYSIS_TAG}.pdf"
            )
        save_figure(fig, fname, add_timestamp=False)

    return fig


def plot_corr_slice_org(
    df,
    x_col="pair_distance",     
    y_col="r_noise",           
    hue=None,                   
    regions=None,
    mode="binned",              
    nbins=20,
    qcut=False,
    max_x=None,
    sample_n=2000,
    save=True,
    fname=None,
):

    figure_style()

    # ---- clean data ----
    if x_col not in df or y_col not in df:
        print(f"[Skip] columns {x_col} or {y_col} missing.")
        return
    df = df.dropna(subset=[x_col, y_col, "cluster_region"])
    if max_x is not None:
        df = df[df[x_col] <= max_x]
    if df.empty:
        print("[Skip] empty df.")
        return

    # add age_group
    if hue == "age_group" and "age_group" not in df.columns and "mouse_age" in df.columns:
        df["age_group"] = np.where(df["mouse_age"] > C.AGE_GROUP_THRESHOLD, "old", "young")

    # palette
    palette = getattr(C, "PALETTE", None)
    hue_order = ["young", "old"] if hue == "age_group" else None

    # ---- prepare region list ----
    if regions is None:
        regions_all = getattr(C, "ROIS_RS", None) or getattr(C, "ROIS", None)
        regions = [r for r in regions_all if r in df["cluster_region"].unique()]
    else:
        regions = [r for r in regions if r in df["cluster_region"].unique()]

    # ---- slice-org ----
    fig, axs = create_slice_org_axes(fg, MM_TO_INCH)
    for region in regions:
        ax = axs[region]
        sub = df[df["cluster_region"] == region]
        if sub.empty:
            ax.axis("off")
            continue

        # ---- plot ----
        if mode == "scatter":
            if hue is None:
                sns.regplot(data=sub, x=x_col, y=y_col,
                            scatter_kws=dict(s=6, alpha=0.25, linewidths=0),
                            line_kws=dict(lw=1.0), lowess=True, ax=ax)
            else:
                for key, sub2 in sub.groupby(hue):
                    sub2_plot = sub2 if len(sub2) <= sample_n else sub2.sample(sample_n, random_state=0)
                    sns.regplot(
                        data=sub2_plot, x=x_col, y=y_col,
                        scatter_kws=dict(s=5, alpha=0.18, linewidths=0),
                        line_kws=dict(lw=1.0),
                        lowess=True, ax=ax,
                        label=str(key),
                        color=None if palette is None else palette.get(key, None),
                    )
        else:
            # bin 
            if hue is None:
                _subgroups = [("all", sub)]
            else:
                _subgroups = sub.groupby(hue)

            for key, sub2 in _subgroups:
                x, y = sub2[x_col].to_numpy(), sub2[y_col].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                if x.size == 0:
                    continue
                bins = np.unique(np.quantile(x, np.linspace(0, 1, nbins+1))) if qcut else np.linspace(x.min(), x.max(), nbins+1)
                idx = np.digitize(x, bins) - 1
                xm, ym, se, ci = [], [], [], []
                for b in range(len(bins)-1):
                    m = idx == b
                    if m.sum() < 2:
                        continue
                    xm.append(0.5*(bins[b]+bins[b+1]))
                    ym.append(y[m].mean())
                    se.append(y[m].std(ddof=1)/np.sqrt(m.sum()))
                    ci.append(1.96 * y[m].std(ddof=1) / np.sqrt(m.sum()))
                color = None if palette is None else palette.get(key, None)
                ax.plot(xm, ym, lw=1.3, label=None if hue is None else str(key), color=color)
                ax.fill_between(xm, np.array(ym)-np.array(ci), np.array(ym)+np.array(ci),
                                alpha=0.2, linewidth=0, color=color)

        ax.set_title(region, fontsize=8)
        ax.set_xlabel(x_col.replace("_", " "), fontsize=7)
        ax.set_ylabel(y_col.replace("_", " "), fontsize=7)
        sns.despine(ax=ax)

    for key, ax in axs.items():
        if key not in regions:
            ax.axis("off")

    if save:
        if fname is None:
            fname = C.FIGPATH / f"sliceorg_{y_col}_vs_{x_col}{'_by_' + hue if hue else ''}.pdf"
        save_figure(fig, fname, add_timestamp=True)
    return fig



if __name__ == "__main__":
    
    window = "post"
    df_noise, df_signal, df_merged = load_pair_corr_for_validation_plot(window=window)

    print(df_noise.columns)
    print(df_noise.head())
    print(df_noise.cluster_region.unique())

    # Noise corr vs distance
    plot_corr_pooled(
        df_noise,
        x_col="pair_distance",
        y_col="r_noise",
        mode="binned",
        nbins=10,
        max_x=1000,
        window=window,
        save=True,
        fname=C.FIGPATH / f"pooled_r_noise_vs_pair_distance_{window}{ANALYSIS_TAG}.pdf",
    )

    # Noise corr vs firing rate
    plot_corr_pooled(
        df_noise,
        x_col="cluster_geo_mean_fr",
        y_col="r_noise",
        mode="binned",
        nbins=10,
        window=window,
        save=True,
        fname=C.FIGPATH / f"pooled_r_noise_vs_cluster_geo_mean_fr_{window}{ANALYSIS_TAG}.pdf",
    )

    # Signal corr vs distance
    if not df_signal.empty:
        plot_corr_pooled(
            df_signal,
            x_col="pair_distance",
            y_col="r_signal",
            mode="binned",
            max_x=1000,
            nbins=10,
            window=window,
            save=True,
            fname=C.FIGPATH / f"pooled_r_signal_vs_pair_distance_{window}{ANALYSIS_TAG}.pdf",
        )

    # Signal corr vs firing rate
    if not df_signal.empty:
        plot_corr_pooled(
            df_signal,
            x_col="cluster_geo_mean_fr",
            y_col="r_signal",
            mode="binned",
            nbins=10,
            window=window,
            save=True,
            fname=C.FIGPATH / f"pooled_r_signal_vs_cluster_geo_mean_fr_{window}{ANALYSIS_TAG}.pdf",
        )

    # Noise corr vs signal corr
    if not df_merged.empty:
        plot_corr_pooled(
            df_merged,
            x_col="r_signal",
            y_col="r_noise",
            mode="binned",
            nbins=10,
            window=window,
            save=True,
            fname=C.FIGPATH / f"pooled_r_noise_vs_r_signal_{window}{ANALYSIS_TAG}.pdf",
        )

