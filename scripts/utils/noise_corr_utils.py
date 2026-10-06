# scripts/utils/noise_corr_utils.py

import numpy as np
import pandas as pd

# ---------- core helpers ----------
def corrcoef_upper_triangle(X):
    """
    X: (n_trials, n_neurons). Corr across the trial dimension.
    Returns upper-triangular pairwise r plus index arrays (i < j).
    """
    R = np.corrcoef(X, rowvar=False)
    iu, ju = np.triu_indices(R.shape[0], k=1)
    return R[iu, ju], iu, ju


def counts_matrix_direct(spike_t, spike_clu, cluster_ids, win_starts, win_ends):
    """
    Build trial-by-neuron spike-count matrix using non-overlapping windows.

    spike_t: 1D np.array of spike times (s)
    spike_clu: 1D np.array of cluster ids (same length as spike_t)
    cluster_ids: iterable of cluster ids to include (columns)
    win_starts, win_ends: arrays of per-trial window edges

    Returns
    -------
    counts : (n_trials, n_neurons) float
    cluster_ids : np.array of ids in the same order as columns
    """
    cluster_ids = np.asarray(list(cluster_ids))
    counts = np.zeros((win_starts.size, cluster_ids.size), dtype=float)
    for j, cid in enumerate(cluster_ids):
        ts = spike_t[spike_clu == cid]
        li = np.searchsorted(ts, win_starts, side='left')
        ri = np.searchsorted(ts, win_ends,   side='left')
        counts[:, j] = ri - li
    return counts, cluster_ids


def _remove_outlier_trials(x, y, sd=3.0):
    """
    Drop trials where either variable exceeds mean ± sd*std (Ruff & Cohen, 2014).
    Returns a boolean mask of trials to keep.
    """
    xi = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    yi = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    return (np.abs(xi) <= sd) & (np.abs(yi) <= sd)


# ---------- Path A: per-condition rSC, then simple average across conditions ----------
def compute_noise_corr_by_condition_average(
    spike_t, spike_clu, cluster_ids, depth_map,
    trial_onsets, cond_labels, t0, t1,
    min_trials_per_cond=20,
    outlier_sd=None
):
    """
    Literature-aligned main analysis (Ecker, 2014; Ruff & Cohen, 2014):
      1) For each condition c: build counts in [onset+t0, onset+t1], compute pairwise Pearson r across trials.
      2) For each pair, take the simple arithmetic mean of per-condition r values.

    Parameters
    ----------
    cond_labels : 1D array-like of length n_trials
        Condition label per trial (e.g., signed contrast).
    outlier_sd : float or None
        If set (e.g., 3.0), drop trials > sd from mean for either unit (per pair, per condition).

    Returns
    -------
    per_condition : DataFrame
        Columns: ['cluster_id1','cluster_id2','r','n_trials','condition','t0','t1'].
    per_pair_mean : DataFrame
        Pair-level simple mean across conditions.
        Columns: ['cluster_id1','cluster_id2','r_mean','n_conds'].
    """
    cond_labels = np.asarray(cond_labels)
    event_onsets = np.asarray(trial_onsets)

    per_condition_results = []
    unique_conds = np.unique(cond_labels)

    for c in unique_conds:
        idx = np.where(cond_labels == c)[0]
        if idx.size < min_trials_per_cond:
            continue

        ws = event_onsets[idx] + t0
        we = event_onsets[idx] + t1
        Xc, clu_ids = counts_matrix_direct(spike_t, spike_clu, cluster_ids, ws, we)
        if Xc.shape[0] < 2 or Xc.shape[1] < 2:
            continue

        firing_rates = (Xc.mean(axis=0, keepdims=True) / (t1 - t0)).ravel()

        # Compute pairwise r within this condition
        iu, ju = np.triu_indices(Xc.shape[1], k=1)
        rvals = np.empty(iu.size, dtype=float)
        for k, (a, b) in enumerate(zip(iu, ju)):
            if outlier_sd is not None:
                keep = _remove_outlier_trials(Xc[:, a], Xc[:, b], sd=outlier_sd)
                if keep.sum() < min_trials_per_cond:
                    rvals[k] = np.nan
                else:
                    rvals[k] = np.corrcoef(Xc[keep, a], Xc[keep, b])[0, 1]
            else:
                # Corr over trials
                rvals[k] = np.corrcoef(Xc[:, a], Xc[:, b])[0, 1]

        depths = depth_map.reindex(clu_ids).to_numpy()


        df_c = pd.DataFrame({
            "cluster_id1": clu_ids[iu],
            "cluster_id1_fr": firing_rates[iu],
            "cluster_id2": clu_ids[ju],
            "cluster_id2_fr": firing_rates[ju],
            "cluster_geo_mean_fr": np.sqrt(firing_rates[iu] * firing_rates[ju]),
            "pair_distance": np.abs(depths[iu] - depths[ju]),
            "r_noise": rvals,
            "n_trials": idx.size,
            "condition": c,
            "t0": t0, "t1": t1
        })
        per_condition_results.append(df_c)

    if not per_condition_results:
        return pd.DataFrame(), pd.DataFrame()

    per_condition = pd.concat(per_condition_results, ignore_index=True)

    # Simple arithmetic mean across conditions (no Fisher-z)
    per_pair_mean = (
        per_condition
        .groupby(["cluster_id1", "cluster_id2"], as_index=False)
        .agg(r_noise=("r_noise", "mean"), n_conds=("condition", "nunique"))
    )

    return per_condition, per_pair_mean


# ---------- Path B: within-condition z-score, pool all trials, single r ----------
def compute_noise_corr_pooled_withincond_zscore(
    spike_t, spike_clu, cluster_ids, depth_map,
    trial_onsets, cond_labels, t0, t1,
    min_trials_per_cond=10,
    min_trials_total=50,
    outlier_sd=None,
    eps=1e-8
):
    """
    Robustness path (Ruff & Cohen alt.): z-score within each condition (per neuron),
    then concatenate trials across all conditions and compute a single rSC.

    Returns
    -------
    df : DataFrame
        Columns: ['cluster_id1','cluster_id2','r','n_trials','t0','t1','pool_mode']
        with pool_mode='zscore'.
    """
    cond_labels = np.asarray(cond_labels)
    event_onsets = np.asarray(trial_onsets)
    unique_conds = np.unique(cond_labels)

    X_list = []
    kept_conds = []
    sum_counts = None
    n_trials_total = 0
    clu_ids = None
    for c in unique_conds:
        idx = np.where(cond_labels == c)[0]
        if idx.size < min_trials_per_cond:
            continue
        ws = event_onsets[idx] + t0
        we = event_onsets[idx] + t1
        Xc, clu_ids = counts_matrix_direct(spike_t, spike_clu, cluster_ids, ws, we)
        if sum_counts is None:
            sum_counts = Xc.sum(axis=0, keepdims=False)
        else:
            sum_counts += Xc.sum(axis=0, keepdims=False)
        n_trials_total += Xc.shape[0]

        # within-condition z-score per neuron
        mu = Xc.mean(axis=0, keepdims=True)
        sd = Xc.std(axis=0, ddof=1, keepdims=True)
        X_list.append((Xc - mu) / (sd + eps))
        kept_conds.append(c)

    if not X_list:
        return pd.DataFrame()

    X = np.vstack(X_list)
    if X.shape[0] < min_trials_total or X.shape[1] < 2:
        return pd.DataFrame()

    firing_rates = (sum_counts / n_trials_total) / (t1 - t0)   # shape: (n_clusters,)

    iu, ju = np.triu_indices(X.shape[1], k=1)
    rvals = np.empty(iu.size, dtype=float)
    for k, (a, b) in enumerate(zip(iu, ju)):
        if outlier_sd is not None:
            keep = _remove_outlier_trials(X[:, a], X[:, b], sd=outlier_sd)
            if keep.sum() < min_trials_total:
                rvals[k] = np.nan
            else:
                rvals[k] = np.corrcoef(X[keep, a], X[keep, b])[0, 1]
        else:
            rvals[k] = np.corrcoef(X[:, a], X[:, b])[0, 1]

    # depths = cluster_depths.to_numpy()          # ← 不 reset_index
    depths = depth_map.reindex(clu_ids).to_numpy()

    df = pd.DataFrame({
        "cluster_id1": clu_ids[iu],
        "cluster_id1_fr": firing_rates[iu],
        "cluster_id2": clu_ids[ju],
        "cluster_id2_fr": firing_rates[ju],
        "cluster_geo_mean_fr": np.sqrt(firing_rates[iu] * firing_rates[ju]),
        #TODO: check if the indexing is correct
        "pair_distance": np.abs(depths[iu] - depths[ju]),
        "r_noise": rvals,
        "n_trials": X.shape[0],
        "n_conds": len(kept_conds),
        "conditions": ",".join(map(str, kept_conds)),
        "t0": t0, "t1": t1,
        "pool_mode": "zscore"
    })
    return df


# ---------- Signal correlation  ----------
def compute_signal_corr_by_condition_means(
    spike_t, spike_clu, cluster_ids, depth_map,
    trial_onsets, cond_labels, t0, t1,
    min_trials_per_cond=10, min_conds=4, eps=1e-8
):
    """
    Signal correlation between tuning vectors across conditions:
      - For each condition, compute mean spike count in [onset+t0, onset+t1].
      - Build tuning vectors (mean per condition) per neuron.
      - Pearson correlation across conditions for each neuron pair.

    Returns
    -------
    df : DataFrame
        Columns include: ['cluster_id1','cluster_id2','r_signal','n_conds','t0','t1'].
    """
    cond_labels = np.asarray(cond_labels)
    event_onsets = np.asarray(trial_onsets)
    unique_conds = np.unique(cond_labels)

    # Collect per-condition means for each neuron
    tuning_list = []
    kept_conds = []
    sum_counts = None
    n_trials_total = 0
    clu_ids = None

    for c in unique_conds:
        idx = np.where(cond_labels == c)[0]
        if idx.size < min_trials_per_cond:
            continue
        ws = event_onsets[idx] + t0
        we = event_onsets[idx] + t1
        Xc, clu_ids = counts_matrix_direct(spike_t, spike_clu, cluster_ids, ws, we)
        tuning_list.append(Xc.mean(axis=0))  # mean over trials (per neuron)
        kept_conds.append(c)

        sc = Xc.sum(axis=0, keepdims=False)
        if sum_counts is None:
            sum_counts = sc.astype(float, copy=False)
        else:
            sum_counts += sc
        n_trials_total += Xc.shape[0]

    if not tuning_list or len(kept_conds) < min_conds: # previous:2
        return pd.DataFrame()

    # tuning: (n_conds_kept, n_neurons)
    tuning = np.vstack(tuning_list)

    # corr across conditions for each pair of neurons
    # compute correlation across rows (conditions), so set rowvar=True then index columns appropriately
    # R = np.corrcoef(tuning, rowvar=True)  # (n_conds, n_conds) if used wrongly; we need per-pair across conditions
    # Instead, for clarity, compute pairwise correlations explicitly:
    n_neurons = tuning.shape[1]
    iu, ju = np.triu_indices(n_neurons, k=1)
    r_sig = np.empty(iu.size, dtype=float)
    for k, (a, b) in enumerate(zip(iu, ju)):
        r_sig[k] = np.corrcoef(tuning[:, a], tuning[:, b])[0, 1]

    dur = (t1 - t0)
    firing_rates = (sum_counts / max(n_trials_total, 1)) / max(dur, eps)

    # depths = cluster_depths.to_numpy()
    depths = depth_map.reindex(clu_ids).to_numpy()

    df = pd.DataFrame({
        "cluster_id1": clu_ids[iu],
        "cluster_id2": clu_ids[ju],
        "cluster_id1_fr": firing_rates[iu],
        "cluster_id2_fr": firing_rates[ju],
        "cluster_geo_mean_fr": np.sqrt(firing_rates[iu] * firing_rates[ju]),
        "pair_distance": np.abs(depths[iu] - depths[ju]),
        "r_signal": r_sig,
        "n_conds": len(kept_conds),
        "t0": t0, "t1": t1
    })
    return df


def combine_corr_results(df_noise, df_signal):
    common_cols = [
        "session_pid",
        "cluster_region",
        "cluster_id1",
        "cluster_id2",
        "window",
    ]

    common_cols = [c for c in common_cols if c in df_noise.columns and c in df_signal.columns]

    merged = pd.merge(
        df_noise,
        df_signal,
        on=common_cols,
        how="inner",
        suffixes=("_noise", "_signal"),
    )

    return merged
