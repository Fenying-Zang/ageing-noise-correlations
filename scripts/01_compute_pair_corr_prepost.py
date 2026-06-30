
""""
Script to calculate pairwise noise and signal correlations for neurons.

"""
#%%
import config as C
import logging
import traceback
import os
import pandas as pd
import numpy as np
from pathlib import Path
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from scripts.utils.neuron_utils import cal_presence_ratio, combine_regions, smoothing_sliding
from scripts.utils.behavior_utils import clean_rts
from glob import glob
import logging
from scripts.utils.io import read_table
import pingouin as pg
from scripts.utils.noise_corr_utils import (
    compute_noise_corr_by_condition_average,
    compute_noise_corr_pooled_withincond_zscore,
    compute_signal_corr_by_condition_means,
)
import gc
log = logging.getLogger(__name__)

def clean_rt_table(trials_table, rt_variable):
    
    trials_table['rt_raw'] = trials_table[rt_variable].copy()
    trials_table['rt'] = clean_rts(trials_table[rt_variable], cutoff=C.RT_CUTOFF)    
    return trials_table


def extract_mouse_info (trials_table, eid):
    #extract trials and subject info from trials table
    trials = trials_table.loc[trials_table['eid']==eid] 
    subject = trials_table[trials_table['eid'] == eid]['mouse_name'].iloc[0]
    sex = trials_table[trials_table['eid'] == eid]['mouse_sex'].iloc[0]
    age_at_recording=  trials_table[trials_table['eid'] == eid]['mouse_age'].iloc[0]
    sess_date = trials_table[trials_table['eid'] == eid]['date'].iloc[0]

    return trials, subject, sex, age_at_recording, sess_date


def map_event():
    if C.ALIGN_EVENT == 'stim': #'stim','move'
        event = 'stimOn_times' #movement, feedback
    elif C.ALIGN_EVENT == 'move':
        event = 'firstMovement_times' #movement, feedback
    elif C.ALIGN_EVENT == 'feedback':
        event = 'feedback_times' #movement, feedback
    return event


def load_and_prepare_trials(trial_type, event_list, clean_rt, rt_variable_name):
    """
    Load and filter trials table.
    Returns: cleaned trials_table
    """
    trials_path = os.path.join(C.DATAPATH, 'ibl_included_eids_trials_table2025_full.csv')
    try:
        trials_table = pd.read_csv(trials_path)
    except Exception as err:
        print(f'Error loading trials table: {err}')
        return None

    if event_list:
        trials_table['exclude_nan_event_mask'] = np.where(trials_table[event_list].notna().all(axis=1), 1, 0)
        trials_table = trials_table[trials_table['exclude_nan_event_mask'] == 1]

    if trial_type == 'first400':
        trials_table = trials_table.groupby('eid').head(400).reset_index(drop=True)

    if clean_rt:
        trials_table = clean_rt_table(trials_table, rt_variable_name)
        trials_table = trials_table[~trials_table['rt'].isna()]
    
    return trials_table


def filter_spikes_by_cluster(spikes, clusters_ids, pid, pid_no_spikes):
    """
    Filters spikes based on selected cluster IDs.

    Returns:
        spike_idx: boolean mask
        has_valid_spikes: bool
        updated pid_no_spikes (optional)
    """
    spike_idx = np.isin(spikes['clusters'], clusters_ids)
    if np.sum(spike_idx) == 0:
        print(f"{pid} — No spikes in selected C.ROIS.")
        pid_no_spikes.append(pid)
        return spike_idx, False, pid_no_spikes
    return spike_idx, True, pid_no_spikes


def load_sorting_and_clusters(row, one, no_iblsortor):
    """
    Load spikes, clusters, and channels for a given row.
    Updates and returns no_iblsortor list if fallback is used.
    """
    pid = row['pid']
    if row['project'] == 'brainwidemap':
        sl = SpikeSortingLoader(one=one, pid=pid)
        spikes, clusters, channels = sl.load_spike_sorting(revision='2024-05-06', good_units=True)
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels, compute_metrics=False)

    elif row['project'] == 'learninglifespan':
        sl = SpikeSortingLoader(pid=pid, one=one, spike_sorter='iblsort')
        try:
            spikes, clusters, channels = sl.load_spike_sorting(enforce_version=True)
        except AssertionError:
            spikes, clusters, channels = sl.load_spike_sorting(enforce_version=False)
            no_iblsortor.append(pid)
        clusters = sl.merge_clusters(spikes, clusters, channels)

    return spikes, clusters, channels, no_iblsortor


def compute_cluster_metrics(spikes, clusters, trials, br, hist_win=10):
    """
    Compute presence ratio and firing rate for clusters within trial-aligned window.
    Also maps brain regions to Beryl and merged region.

    Returns:
        clusters: updated DataFrame with metrics and region labels
        spike_times_btw: filtered spike times (within window)
        spike_clusters: corresponding spike cluster ids
    """
    event = map_event()
    start_point = trials[event].min() - 0.4
    end_point = trials[event].max() + 1

    mask = (spikes['times'] >= start_point) & (spikes['times'] <= end_point)
    spike_times_btw = spikes['times'][mask]
    spike_clusters = spikes['clusters'][mask]
    cluster_ids = clusters['cluster_id']

    pr_poi, fr_poi = cal_presence_ratio(start_point, end_point, spike_times_btw, spike_clusters, cluster_ids, hist_win=hist_win)
    clusters['presence_ratio_poi'] = pr_poi
    clusters['firing_rate_poi'] = fr_poi

    clusters['Beryl'] = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
    clusters['Beryl_merge'] = combine_regions(clusters['Beryl'])

    return clusters, spike_times_btw, spike_clusters


def compute_neural_yield(clusters, channels, ROIs, firing_rate_threshold, presence_ratio_threshold, pid, eid, subject, age_at_recording, br):
    """
    Given clusters and channels with metrics and region info,
    compute the number of good units and channels per brain region (yield).
    
    Returns:
        yield_table: DataFrame with n_channel, n_cluster, and metadata
        clusters_ids: selected good cluster IDs
        cluster_idx: boolean mask of selected clusters
    """
    # Handle channels
    channels['Beryl'] = br.id2acronym(channels['atlas_id'], mapping='Beryl')
    channels['Beryl_merge'] = combine_regions(channels['Beryl'])

    try:
        channels_df = pd.DataFrame.from_dict(channels)
    except Exception as err:
        print(f"Error creating channels_df, falling back: {err}")
        for key, value in channels.items():
            print(f"{key}: {len(value)}")

        selected_columns = ['x', 'y', 'z', 'acronym', 'atlas_id', 'Beryl', 'Beryl_merge', 'rawInd']
        filtered_data = {key: channels[key] for key in selected_columns if key in channels}
        channels_df = pd.DataFrame(filtered_data)

    channels_good = (
        channels_df[channels_df["Beryl_merge"].isin(ROIs)]
        .groupby("Beryl_merge")[["rawInd"]]
        .nunique()
        .reset_index()
        .rename(columns={"rawInd": "n_channel"})
    )

    # Select good clusters
    cluster_idx = (
        np.isin(clusters['Beryl_merge'], ROIs) &
        (clusters['label'] == 1) &
        (clusters['firing_rate_poi'] > firing_rate_threshold) &
        (clusters['presence_ratio_poi'] > presence_ratio_threshold)
    )
    clusters_ids = clusters['cluster_id'][cluster_idx]
    clusters_df = pd.DataFrame.from_dict(clusters)

    cluster_good = (
        clusters_df.loc[cluster_idx]
        .groupby("Beryl_merge")["cluster_id"]
        .nunique()
        .reset_index()
        .rename(columns={"cluster_id": "n_cluster"})
    )

    # Merge channel and cluster counts
    yield_table = pd.merge(channels_good, cluster_good, on="Beryl_merge", how="outer").reset_index(drop=True)
    yield_table['pid'] = pid
    yield_table['eid'] = eid
    yield_table['subject'] = subject
    yield_table['age_at_recording'] = age_at_recording

    return yield_table, clusters_ids, cluster_good, cluster_idx


def compute_noise_corr_trialwise_direct(
    spikes_times, spikes_clusters, cluster_ids_in_region,
    trial_onsets, t0, t1,
    subtract_condition_mean=True,
    min_trials=25,
):
    """
    Classic noise correlation (trial-to-trial).

    Count spikes for each trial x neuron within the window [onset + t0, onset + t1],
    forming a matrix `counts` with shape (n_trials, n_neurons). Compute pairwise
    correlations across the trial dimension.

    Parameters
    ----------
    spikes_times : 1D np.ndarray
        Spike timestamps in seconds.
    spikes_clusters : 1D np.ndarray
        Cluster ID for each spike; same length as spikes_times.
    cluster_ids_in_region : Iterable
        Iterable of cluster IDs (the good units within this region).
    trial_onsets : 1D np.ndarray
        Alignment event timestamps (one per trial).
    t0, t1 : float
        Window edges relative to the alignment event in seconds, e.g., (0, 0.25).
    subtract_condition_mean : bool
        If True, subtract the across-trial mean from each neuron before computing correlations.
    min_trials : int
        Minimum number of trials required to compute correlations.

    Returns
    -------
    corr_df : pandas.DataFrame
        DataFrame with columns: ['cluster_id1', 'cluster_id2', 'r', 'n_trials'].
    """
    cluster_ids = np.asarray(list(cluster_ids_in_region))
    n_trials = len(trial_onsets)
    if n_trials < min_trials or cluster_ids.size < 2:
        return pd.DataFrame()

    # trial window to be used
    win_starts = np.asarray(trial_onsets) + t0
    win_ends   = np.asarray(trial_onsets) + t1

    # extract spike times for each neuron
    counts = np.zeros((n_trials, cluster_ids.size), dtype=float)
    for j, cid in enumerate(cluster_ids):
        ts = spikes_times[spikes_clusters == cid]
        # searchsorted 
        left_idx  = np.searchsorted(ts, win_starts, side='left')
        right_idx = np.searchsorted(ts, win_ends,   side='left')
        counts[:, j] = right_idx - left_idx

    if subtract_condition_mean:
        counts = counts - counts.mean(axis=0, keepdims=True)

    # trials × neurons → corr over trials
    R = np.corrcoef(counts, rowvar=False)  # (n_neurons, n_neurons)
    iu, ju = np.triu_indices(R.shape[0], k=1)
    r_vals = R[iu, ju]

    corr_df = pd.DataFrame({
        "cluster_id1": cluster_ids[iu],
        "cluster_id2": cluster_ids[ju],
        "r": r_vals,
        "n_trials": n_trials,
        "t0": t0,
        "t1": t1,
    })
    return corr_df


if __name__ == "__main__":
    
    outdirs = {}
    for window_name, win_cfg in C.NC_WINDOWS.items():
        outdir = C.RESULTSPATH / win_cfg["subfoldername"]
        outdir.mkdir(parents=True, exist_ok=True)
        outdirs[window_name] = outdir

    trials_table = load_and_prepare_trials(
        C.TRIAL_TYPE, C.EVENT_LIST, C.CLEAN_RT, C.RT_VARIABLE_NAME
    )

    if trials_table is None:
        print("Failed to load trials.")
        raise SystemExit(1)

    print(len(set(trials_table.eid)))

    recordings_filtered = read_table(C.DATAPATH / "BWM_LL_release_afterQC_df.csv")

    ba = AllenAtlas()
    br = BrainRegions()
    one = ONE()

    pid_no_spikes = []
    no_iblsortor = []

    print(f"Total recordings to process: {len(recordings_filtered)}")

    for index, row in recordings_filtered.iterrows():
    # for index, row in recordings_filtered.head(2).iterrows():
        pid = row["pid"]

        if pid in C.PIDS_WITHOUT_ILBLSORTOR:
            continue

        done_files = {
            window_name: outdirs[window_name] / f"{pid}.noise_signal.done"
            for window_name in C.NC_WINDOWS
        }

        if all(done_file.exists() for done_file in done_files.values()):
            print(f"[Skip all windows] {index}||{pid} already processed.")
            continue

        print(f"PID {index + 1}/{len(recordings_filtered)}: {pid}")

        spikes = None
        clusters = None
        channels = None
        trials = None
        clusters_included = None
        spike_corr_mask = None
        spike_times_corr = None
        spike_clusters_corr = None

        try:
            eid, pname = one.pid2eid(pid)
            eid = str(eid)

            trials, subject, sex, age_at_recording, sess_date = extract_mouse_info(
                trials_table, eid
            )

            spikes, clusters, channels, no_iblsortor = load_sorting_and_clusters(
                row, one, no_iblsortor
            )

            event = map_event()

            clusters, spike_times_btw, spike_clusters = compute_cluster_metrics(
                spikes, clusters, trials, br
            )

            yield_table, clusters_ids, cluster_good, cluster_idx = compute_neural_yield(
                clusters,
                channels,
                C.ROIS,#ROI_TEMP, #
                C.FIRING_RATE_THRESHOLD,
                C.PRESENCE_RATIO_THRESHOLD,
                pid,
                eid,
                subject,
                age_at_recording,
                br,
            )

            print(len(clusters_ids))
            print(cluster_good)

            clusters = pd.DataFrame.from_dict(clusters)
            clusters_included = clusters[clusters["cluster_id"].isin(clusters_ids)]

            if len(clusters_included) < 2:
                print(
                    f"PID {pid} has less than 2 good units; "
                    "skipping noise correlation computation."
                )
                continue

            # For this pid, keep only spikes that can possibly enter
            # any pre/post window.
            all_t0 = []
            all_t1 = []

            for window_cfg in C.NC_WINDOWS.values():
                all_t0.append(window_cfg["nc_window"][0])
                all_t1.append(window_cfg["nc_window"][1])

                if C.SC_ENABLE:
                    all_t0.append(window_cfg["sc_window"][0])
                    all_t1.append(window_cfg["sc_window"][1])

            min_t0 = min(all_t0)
            max_t1 = max(all_t1)

            start_point = trials[event].min() + min_t0
            end_point = trials[event].max() + max_t1

            spike_corr_mask = (
                (spikes["times"] >= start_point)
                & (spikes["times"] <= end_point)
            )

            spike_times_corr = spikes["times"][spike_corr_mask]
            spike_clusters_corr = spikes["clusters"][spike_corr_mask]

            print(
                f"Using {len(spike_times_corr)} spikes for corr "
                f"out of {len(spikes['times'])}"
            )

            cond_labels = trials["signed_contrast"].values

            for window_name, window_cfg in C.NC_WINDOWS.items():

                outdir = outdirs[window_name]
                done_file = done_files[window_name]

                if done_file.exists():
                    print(f"[Skip {window_name}] {index}||{pid} already processed.")
                    continue

                print(f"[Window {window_name}] {pid}")

                noise_corr_result_pid = []
                signal_corr_result_pid = []

                nc_t0, nc_t1 = window_cfg["nc_window"]

                if C.SC_ENABLE:
                    sc_t0, sc_t1 = window_cfg["sc_window"]

                for region, region_cluster_table in clusters_included.groupby("Beryl_merge"):

                    print(region)

                    region_cluster_ids = region_cluster_table["cluster_id"].values
                    region_spike_idx = np.isin(spike_clusters_corr, region_cluster_ids)

                    if np.sum(region_spike_idx) == 0:
                        print(f"{pid} — No spikes in selected region {region}.")
                        pid_no_spikes.append(pid)
                        continue

                    region_spike_t = spike_times_corr[region_spike_idx]
                    region_spike_clu = spike_clusters_corr[region_spike_idx]

                    n_units = region_cluster_table["cluster_id"].nunique()
                    n_pairs = n_units * (n_units - 1) // 2
                    print(f"{region}: {n_units} units, {n_pairs} pairs")

                    if C.NC_MODE == "by_condition_avg":

                        per_cond, per_pair = compute_noise_corr_by_condition_average(
                            spike_t=region_spike_t,
                            spike_clu=region_spike_clu,
                            cluster_ids=list(region_cluster_table["cluster_id"]),
                            depth_map=region_cluster_table.set_index("cluster_id")["depths"],
                            trial_onsets=trials[event].values,
                            cond_labels=cond_labels,
                            t0=nc_t0,
                            t1=nc_t1,
                            min_trials_per_cond=C.NC_MIN_TRIALS_PER_COND,
                            outlier_sd=C.NC_OUTLIER_SD,
                        )

                        corr_df = per_pair.copy()
                        corr_df["agg_note"] = "per_condition_mean_r"

                    elif C.NC_MODE == "pooled_zscore":

                        corr_df = compute_noise_corr_pooled_withincond_zscore(
                            spike_t=region_spike_t,
                            spike_clu=region_spike_clu,
                            cluster_ids=list(region_cluster_table["cluster_id"]),
                            depth_map=region_cluster_table.set_index("cluster_id")["depths"],
                            trial_onsets=trials[event].values,
                            cond_labels=cond_labels,
                            t0=nc_t0,
                            t1=nc_t1,
                            min_trials_per_cond=C.NC_MIN_TRIALS_PER_COND,
                            min_trials_total=C.NC_MIN_TRIALS_TOTAL,
                            outlier_sd=C.NC_OUTLIER_SD,
                        )

                        corr_df["agg_note"] = "pooled_zscore_single_r"

                    else:
                        raise ValueError(f"Unknown NC_MODE: {C.NC_MODE}")

                    if not corr_df.empty:
                        corr_df["session_pid"] = pid
                        corr_df["session_eid"] = eid
                        corr_df["mouse_name"] = subject
                        corr_df["mouse_age"] = age_at_recording
                        corr_df["cluster_region"] = region
                        corr_df["align_event"] = event

                        corr_df["window"] = window_name
                        corr_df["nc_t0"] = nc_t0
                        corr_df["nc_t1"] = nc_t1

                        noise_corr_result_pid.append(corr_df)

                    if C.SC_ENABLE:

                        sc_df = compute_signal_corr_by_condition_means(
                            spike_t=region_spike_t,
                            spike_clu=region_spike_clu,
                            cluster_ids=list(region_cluster_table["cluster_id"]),
                            depth_map=region_cluster_table.set_index("cluster_id")["depths"],
                            trial_onsets=trials[event].values,
                            cond_labels=cond_labels,
                            t0=sc_t0,
                            t1=sc_t1,
                            min_trials_per_cond=C.NC_MIN_TRIALS_PER_COND,
                            min_conds=C.SC_MIN_CONDS, 

                        )

                        if not sc_df.empty:
                            sc_df["session_pid"] = pid
                            sc_df["session_eid"] = eid
                            sc_df["mouse_name"] = subject
                            sc_df["mouse_age"] = age_at_recording
                            sc_df["cluster_region"] = region
                            sc_df["align_event"] = event

                            sc_df["window"] = window_name
                            sc_df["sc_t0"] = sc_t0
                            sc_df["sc_t1"] = sc_t1

                            signal_corr_result_pid.append(sc_df)

                    # release region-level temporary objects
                    region_spike_t = None
                    region_spike_clu = None
                    region_spike_idx = None
                    corr_df = None

                    if C.SC_ENABLE:
                        sc_df = None

                if noise_corr_result_pid:
                    out_noise_pid = pd.concat(noise_corr_result_pid, ignore_index=True)
                    tmp = outdir / f"pair_noise_{pid}.parquet.part"
                    final = outdir / f"pair_noise_{pid}.parquet"
                    out_noise_pid.to_parquet(tmp, index=False)
                    os.replace(tmp, final)

                    out_noise_pid = None

                if signal_corr_result_pid:
                    out_signal_pid = pd.concat(signal_corr_result_pid, ignore_index=True)
                    tmp = outdir / f"pair_signal_{pid}.parquet.part"
                    final = outdir / f"pair_signal_{pid}.parquet"
                    out_signal_pid.to_parquet(tmp, index=False)
                    os.replace(tmp, final)

                    out_signal_pid = None

                if not noise_corr_result_pid and not signal_corr_result_pid:
                    print(f"[No result {window_name}] {pid}")

                noise_corr_result_pid = None
                signal_corr_result_pid = None

                done_file.touch()
                print(f"[Done {window_name}] {index} || {pid}")

                gc.collect()

        except Exception as err:
            print(f"Error on PID {pid} (index {index}): {err}")
            traceback.print_exc()
            logging.error(f"PID {pid} (index {index}) — {err}")
            logging.error(traceback.format_exc())
            continue

        finally:
            spikes = None
            clusters = None
            channels = None
            trials = None
            clusters_included = None

            spike_corr_mask = None
            spike_times_corr = None
            spike_clusters_corr = None

            spike_times_btw = None
            spike_clusters = None

            yield_table = None
            clusters_ids = None
            cluster_good = None
            cluster_idx = None

            gc.collect()


# %%
