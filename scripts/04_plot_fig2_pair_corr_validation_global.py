"""
Main figure 2efg: validation plots using common-effect LMM estimates.
The three panels show pair-level relationships between noise correlation and
- firing rate
- pair distance
- signal correlation
Plotting uses binned means +/- 95% CI.
Annotations come from the M_predictor model and contain only the common slope and its two-sided Wald p-value.

"""
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as C
from scripts.utils.io import read_table, save_figure
from scripts.utils.plot_utils import figure_style, format_p_value
# %% SETTINGS
WINDOW = "post"
WIN_LEN = 500
NBINS = 10
DARK_GRAY = "0.15"
DATA_FILE = (C.DATAPATH / f"noise_pair_corr_merged_prepost_{WIN_LEN}ms.parquet")
GLOBAL_STATS_FILE = (
    C.RESULTSPATH
    / "noise_validation_three_model_AIC_BIC_"
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
    },
    {
        "outcome": "r_noise",
        "predictor": "pair_distance",
        "x_label": "Pair distance (µm)",
        "beta_subscript": r"\mathrm{distance}",
        "beta_decimals": 5,
        "max_x": 1000,
    },
    {
        "outcome": "r_noise",
        "predictor": "r_signal",
        "x_label": "Signal correlations",
        "beta_subscript": r"r_{\mathrm{signal}}",
        "beta_decimals": 3,
        "max_x": None,
    },
]


# %% LOADING
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


def load_validation_data():
    df = standardize_pair_corr_columns(read_table(DATA_FILE))
    df["window"] = df["window"].astype(str)
    df = df[df["window"] == WINDOW].copy()

    print(f"[Loaded data] {DATA_FILE}")
    print(f"window={WINDOW}, rows={len(df)}")
    return df


def load_common_effect(outcome, predictor):
    stats = read_table(GLOBAL_STATS_FILE)
    row = stats[
        (stats["window"].astype(str) == WINDOW)
        & (stats["outcome"] == outcome)
        & (stats["predictor"] == predictor)
        & (stats["random_factor"] == C.RANDOM_FACTOR)
        & (stats["comparison"] == "common_predictor_effect")
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one common-effect row for {outcome} ~ {predictor}; "
            f"found {len(row)} in {GLOBAL_STATS_FILE}."
        )

    return row.iloc[0]


# %% PLOTTING HELPERS
def bin_mean_and_ci(x, y, nbins=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    edges = np.linspace(x.min(), x.max(), nbins + 1)
    bin_index = np.digitize(x, edges) - 1
    bin_index = np.clip(bin_index, 0, nbins - 1)

    x_mid, y_mean, y_ci = [], [], []
    for index in range(nbins):
        in_bin = bin_index == index
        if in_bin.sum() < 2:
            continue
        x_mid.append(0.5 * (edges[index] + edges[index + 1]))
        y_mean.append(y[in_bin].mean())
        y_ci.append(1.96 * y[in_bin].std(ddof=1) / np.sqrt(in_bin.sum()))

    return np.asarray(x_mid), np.asarray(y_mean), np.asarray(y_ci)


def add_common_effect_annotation(ax, panel):
    row = load_common_effect(panel["outcome"], panel["predictor"])
    beta = float(row["beta_predictor_common"])
    p_wald = float(row["beta_predictor_common_p_wald"])

    beta_text = f"{beta:.{panel['beta_decimals']}f}"
    annotation = (
        rf"$\beta_{{{panel['beta_subscript']}}} = {beta_text}, "
        rf"p_{{\mathrm{{wald}}}}$ {format_p_value(p_wald)}"
    )

    ax.text(
        0.05,
        1.0,
        annotation,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
    )


def plot_validation_panel(df, panel):
    outcome = panel["outcome"]
    predictor = panel["predictor"]

    plot_df = df.dropna(subset=[outcome, predictor]).copy()
    if panel["max_x"] is not None:
        plot_df = plot_df[plot_df[predictor] <= panel["max_x"]].copy()

    x_mid, y_mean, y_ci = bin_mean_and_ci(
        plot_df[predictor], plot_df[outcome], nbins=NBINS
    )

    figure_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(x_mid, y_mean, color=DARK_GRAY, linewidth=1.3)
    ax.fill_between(
        x_mid,
        y_mean - y_ci,
        y_mean + y_ci,
        color=DARK_GRAY,
        alpha=0.18,
        linewidth=0,
    )

    add_common_effect_annotation(ax, panel)

    ax.set_xlabel(panel["x_label"], fontsize=9)
    ax.set_ylabel("Noise correlations", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()

    output_file = (
        C.FIGPATH
        / f"pooled_{outcome}_vs_{predictor}_{WINDOW}.pdf"
    )
    save_figure(fig, output_file, add_timestamp=False)
    print(f"[Saved] {output_file}")
    return fig, ax


# %% RUN THE THREE PANELS
if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    validation_data = load_validation_data()

    for panel_settings in PANELS:
        plot_validation_panel(validation_data, panel_settings)
