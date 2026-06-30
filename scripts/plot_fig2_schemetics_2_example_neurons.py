"""
plot panel bcd for paper figure1 
illustrating the pairwise correlation calculation

"""
#%%
import os
import one
import matplotlib.pyplot as plt
import config as C
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from scripts.utils.behavior_utils import clean_rts

PALETTE9_CONTRAST = {
    -100:  "#2166AC",
    -25:   "#4393C3",
    -12.5: "#92C5DE",
    -6.25: "#D1E5F0",
    0:     "#D9D9D9",
    6.25:  "#FDDBC7",
    12.5:  "#F4A582",
    25:    "#D6604D",
    100:   "#B2182B",
}

CORR_COLORS = {
    "r_noise": "#7B4F9D",
    "r_signal": "#E78B2C",
}


def clean_rt_table(trials_table, rt_variable):
    
    trials_table['rt_raw'] = trials_table[rt_variable].copy()
    trials_table['rt'] = clean_rts(trials_table[rt_variable], cutoff=C.RT_CUTOFF)    
    return trials_table


def count_spikes_by_trial(spike_times_one_neuron, trial_onsets, t0=0, t1=0.5):
    spike_times_one_neuron = np.asarray(spike_times_one_neuron)
    trial_onsets = np.asarray(trial_onsets)

    win_starts = trial_onsets + t0
    win_ends = trial_onsets + t1

    left = np.searchsorted(spike_times_one_neuron, win_starts, side="left")
    right = np.searchsorted(spike_times_one_neuron, win_ends, side="left")

    return right - left


def safe_pearsonr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return pearsonr(x, y)[0]


def get_signed_contrast_order(values):
    values = np.asarray(pd.Series(values).dropna().unique())

    # handle both percent scale and proportion scale
    if np.nanmax(np.abs(values)) <= 1.5:
        default_order = [-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]
    else:
        default_order = [-100, -25, -12.5, -6.25, 0, 6.25, 12.5, 25, 100]

    return [v for v in default_order if v in values]


def nearest_condition(values, target=100):
    values = np.asarray(pd.Series(values).dropna().unique())

    if np.nanmax(np.abs(values)) <= 1.5 and target == 100:
        target = 1

    return values[np.argmin(np.abs(values - target))]


def make_example_pair_trial_table(
    spikes,
    trials,
    event,
    cluster_id1,
    cluster_id2,
    t0=0,
    t1=0.5,
    cond_col="signed_contrast",
):
    spike_t1 = spikes["times"][spikes["clusters"] == cluster_id1]
    spike_t2 = spikes["times"][spikes["clusters"] == cluster_id2]

    trial_onsets = trials[event].to_numpy()

    n1 = count_spikes_by_trial(spike_t1, trial_onsets, t0=t0, t1=t1)
    n2 = count_spikes_by_trial(spike_t2, trial_onsets, t0=t0, t1=t1)

    out = pd.DataFrame({
        "trial_id": np.arange(len(trials)),
        cond_col: trials[cond_col].to_numpy(),
        "neuron1_spikes": n1,
        "neuron2_spikes": n2,
    })

    return out


def make_signed_contrast_color_map(values, palette=PALETTE9_CONTRAST):
    signed_levels = get_signed_contrast_order(values)

    # handle proportion scale: -1, -0.25, ..., 1
    if np.nanmax(np.abs(signed_levels)) <= 1.5:
        palette = {k / 100: v for k, v in palette.items()}

    missing = [lev for lev in signed_levels if lev not in palette]
    if len(missing) > 0:
        raise ValueError(f"Missing colors for contrast levels: {missing}")

    return {lev: palette[lev] for lev in signed_levels}


def plot_example_noise_signal_pair(
    counts_df,
    pair_info=None,
    cond_col="signed_contrast",
    example_cond=100,
    save=False,
    fname=None,
):
    plt.rcParams["font.family"] = "Arial"

    cond_order = get_signed_contrast_order(counts_df[cond_col])
    cond_for_noise = nearest_condition(counts_df[cond_col], target=example_cond)

    tune = (
        counts_df
        .groupby(cond_col)[["neuron1_spikes", "neuron2_spikes"]]
        .mean()
        .reindex(cond_order)
        .reset_index()
    )
    print(tune)

    # signed contrast color map
    signed_color_map = make_signed_contrast_color_map(counts_df[cond_col])
    tune["point_color"] = tune[cond_col].map(signed_color_map)

    noise_df = counts_df[counts_df[cond_col] == cond_for_noise].copy()
    r_noise_example = safe_pearsonr(
        noise_df["neuron1_spikes"],
        noise_df["neuron2_spikes"],
    )

    r_signal_example = safe_pearsonr(
        tune["neuron1_spikes"],
        tune["neuron2_spikes"],
    )

    fig, axs = plt.subplots(1, 3, figsize=(5, 1.7))

    # b: tuning-like curves
    ax = axs[0]
    x = np.arange(len(tune))

    # lines indicate neuron identity
    ax.plot(
        x, tune["neuron1_spikes"],
        lw=0.8, color="0.4", label="Neuron 1", zorder=1
    )
    ax.plot(
        x, tune["neuron2_spikes"],
        lw=0.8, color="0.7", label="Neuron 2", zorder=1
    )

    # points indicate abs contrast level
    ax.scatter(
        x, tune["neuron1_spikes"],
        s=15, c=tune["point_color"], edgecolors="none", zorder=2
    )
    ax.scatter(
        x, tune["neuron2_spikes"],
        s=15, c=tune["point_color"], edgecolors="none", zorder=2
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c).replace(".0", "") for c in tune[cond_col]], rotation=45, fontsize=4)
    ax.set_xlabel("Signed contrast (%)", fontsize=6)
    ax.set_ylabel("Mean spike count", fontsize=6)
    ax.legend(frameon=False, fontsize=5, handlelength=1.2)

    # c: noise correlation within highest condition
    ax = axs[1]
    ax.scatter(
        noise_df["neuron1_spikes"],
        noise_df["neuron2_spikes"],
        s=8,  linewidths=0, #alpha=0.65,
        # color=PALETTE5[-1]
        color=signed_color_map[cond_for_noise]
    )
    ax.set_xlabel("Neuron 1 spike count", fontsize=6)
    ax.set_ylabel("Neuron 2 spike count", fontsize=6)
    ax.text(
        0.05, 0.95,
        f"$r_{{noise}}$ = {r_noise_example:.2f}\ncontrast = {cond_for_noise:g}",
        transform=ax.transAxes,
        va="top", fontsize=5
    )

    # d: signal correlation across condition means
    ax = axs[2]
    ax.scatter(
        tune["neuron1_spikes"],
        tune["neuron2_spikes"],
        s=15, linewidths=0, #alpha=0.9, 
        c=tune["point_color"]
    )

    ax.set_xlabel("Neuron 1 mean spike count", fontsize=6)
    ax.set_ylabel("Neuron 2 mean spike count", fontsize=6)
    ax.text(
        0.05, 0.95,
        f"$r_{{signal}}$ = {r_signal_example:.2f}",
        transform=ax.transAxes,
        va="top", fontsize=5
    )

    for ax in axs:
        ax.tick_params(axis="both", labelsize=4, length=2)
        sns.despine(ax=ax)

    if pair_info is not None:
        title = (
            f"{pair_info.get('cluster_region', '')}, "
            f"{pair_info.get('pair_distance', np.nan):.0f} µm"
        )
        fig.suptitle(title, fontsize=6, y=1.05)

    fig.tight_layout(w_pad=1.2)

    if save and fname is not None:
        fig.savefig(fname)

    return fig


example_pid = '810416d1-ce3c-4b2e-a827-6cdc65c8cd5c'
example_eid = 'e6043c7d-8f6e-4b66-8309-2ec0abac0f79'
cluster_id1  = 555
cluster_id2 = 575
one = ONE()

eid, pname = one.pid2eid(example_pid)
trials_path = os.path.join(C.DATAPATH, 'ibl_included_eids_trials_table2025_full.csv')
trials_table = pd.read_csv(trials_path)

trials_table_example_session = trials_table[trials_table['eid'] == example_eid]
print(trials_table_example_session.shape[0])
trials_table_example_session['exclude_nan_event_mask'] = np.where(trials_table_example_session[C.EVENT_LIST].notna().all(axis=1), 1, 0)
trials_table_example_session = trials_table_example_session[trials_table_example_session['exclude_nan_event_mask'] == 1]
trials_table_example_session = trials_table_example_session.head(400).reset_index(drop=True)
trials_table_example_session = clean_rt_table(trials_table_example_session, C.RT_VARIABLE_NAME)
trials_table_example_session = trials_table_example_session[~trials_table_example_session['rt'].isna()]
    
#load spikes and clusters for the example session
sl = SpikeSortingLoader(one=one, pid=example_pid)
spikes, clusters, channels = sl.load_spike_sorting(revision='2024-05-06', good_units=True)
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels, compute_metrics=False)

counts_df = make_example_pair_trial_table(
    spikes=spikes,
    trials=trials_table_example_session,
    event="stimOn_times",
    cluster_id1=cluster_id1,
    cluster_id2=cluster_id2,
    t0=0,
    t1=0.5,
    cond_col="signed_contrast",
)

fig = plot_example_noise_signal_pair(
    counts_df,
    # pair_info=example.to_dict(),
    example_cond=100,
    save=True,
    fname=C.FIGPATH / "example_pair_noise_signal_schematic_9colors.pdf",
)

