"""
Permutation tests at the pair level for relationships between r_noise
and geometric / tuning covariates (pair_distance, cluster_geo_mean_fr, r_signal).

Omnibus + region-specific, compatible with existing age-effect permutation code.
"""

# %% imports
import pandas as pd
import numpy as np
from statsmodels.genmod.families import Gaussian
import logging
import config as C
from scripts.utils.io import read_table
from scripts.utils.stats_utils import run_permutation_test
from scripts.utils.plot_utils import plot_permut_test
from scripts.utils.data_utils import (
    add_age_group,
    bf_gaussian_via_pearson,
    interpret_bayes_factor,
)
log = logging.getLogger(__name__)

FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = "labels1_global"  
ANALYSIS_TAG = "_VISp_VISpm_separate"


PREDICTORS_BY_METRIC = {
    "r_noise": ["pair_distance", "cluster_geo_mean_fr", "r_signal"],
    "r_signal": ["pair_distance", "cluster_geo_mean_fr"],
}
VALIDATION_WINDOWS = ["post"]


def drop_term_from_formula(formula, term="C(cluster_region)"):
    """Remove a single RHS term from a Patsy-style formula safely."""
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    terms = [t.strip() for t in rhs.split("+")]
    terms = [t for t in terms if t != term and t != ""]
    rhs_new = " + ".join(terms) if terms else "1"
    return f"{lhs} ~ {rhs_new}"

def standardize_pair_corr_columns(df):
    """
    Standardize the new merged pre/post pair-correlation table
    so that the rest of the validation script can keep using the old column names.
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


def load_pair_corr_prepost_for_validation(window="post"):
    """
    Load the new merged pre/post pair-level correlation table.

    The returned df should contain:
      r_noise
      r_signal, if available
      pair_distance
      cluster_geo_mean_fr
      session_pid
      cluster_region
      mouse_age
      age_years
      window
    """
    path = C.DATAPATH / f"proj2_noise_pair_corr_merged_prepost_500ms{ANALYSIS_TAG}.parquet"

    log.info(f"Loading merged pre/post pair correlations from {path}")
    df = read_table(path)
    df = standardize_pair_corr_columns(df)

    if "window" not in df.columns:
        raise ValueError("Expected a 'window' column in the new pre/post pair-correlation table.")

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

    log.info(f"[Loaded validation df] window={window}, shape={df.shape}")
    log.info(f"Columns available: {list(df.columns)}")

    return df


def load_pair_level_tables(window="post"):
    """
    Keep the old script interface, but load from the new merged pre/post table.

    Previously:
      df_noise came from all_pair_neural_metrics_r_noise_500.parquet
      df_signal came from all_pair_neural_metrics_r_signal_500.parquet
      df_merged came from combine_corr_results(df_noise, df_signal)

    Now:
      one merged pre/post file is loaded directly.
    """
    df_all = load_pair_corr_prepost_for_validation(window=window)

    if "r_noise" in df_all.columns:
        df_noise = df_all.dropna(subset=["r_noise"]).copy()
    else:
        df_noise = pd.DataFrame()

    if "r_signal" in df_all.columns:
        df_signal = df_all.dropna(subset=["r_signal"]).copy()
        df_merged = df_all.dropna(subset=["r_noise", "r_signal"]).copy()
    else:
        log.warning("Column 'r_signal' not found. r_signal validation and r_noise ~ r_signal will be skipped.")
        df_signal = pd.DataFrame()
        df_merged = pd.DataFrame()

    return df_noise, df_signal, df_merged


def main(plot_permt_result=False):
    """
      - r_noise ~ pair_distance
      - r_noise ~ cluster_geo_mean_fr
      - r_noise ~ r_signal

    output:
      - Omnibus: pairlevel_Omnibus_{metric}_vs_{x_col}_...csv
      - Regional: pairlevel_Regional_{metric}_vs_{x_col}_...csv
    """
    for window in VALIDATION_WINDOWS:
    
        df_noise, df_signal, df_merged = load_pair_level_tables(window=window)

    selected_metrics = ["r_noise", "r_signal"]

    for metric in selected_metrics:
        predictors = PREDICTORS_BY_METRIC.get(metric, [])
        if not predictors:
            continue

        if metric == "r_noise":
            base_df = df_noise
        elif metric == "r_signal":
            base_df = df_signal
        else:
            continue

        for x_col in predictors:
            log.info(f"\n[Permutation] Testing {metric} ~ {x_col}, window={window}")

            if x_col == "r_signal" and metric == "r_noise":
                df_use = df_merged.copy()
            else:
                df_use = base_df.copy()

            needed_cols = [metric, x_col, "session_pid"]

            missing = [c for c in needed_cols if c not in df_use.columns]
            if missing:
                log.warning(f"Skip {metric} ~ {x_col}: missing columns {missing}")
                continue

            df_use = df_use.dropna(subset=needed_cols).reset_index(drop=True)

            if df_use.empty:
                log.warning(f"Skip {metric} ~ {x_col}: df_use is empty after dropping NaNs.")
                continue
            df_use["age_years"] = df_use[x_col].values   
            formula_full = f"{metric} ~ age_years"       
            labels1 = df_use["age_years"].values       
            
            # ---- Bayes factor (Pearson r) ----
            BF_dict = bf_gaussian_via_pearson(df_use, y_col=metric, x_col=x_col)
            
            shuffling = SHUFFLING
            group_labels = df_use["session_pid"].values  

            observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
                data=df_use,
                age_labels=labels1,
                group_labels=group_labels,
                formula=formula_full,
                family_func=FAMILY_FUNC,
                shuffling=shuffling,
                n_permut=C.N_PERMUT_NEURAL_OMNIBUS,
                n_jobs=N_JOBS,
                random_state=C.RANDOM_STATE,
                plot=False,
            )

            log.info(
                f"Omnibus {metric} ~ {x_col}: beta={observed_val:.4f}, p_perm={p_perm:.4g}"
            )

            if plot_permt_result:
                plot_permut_test(
                    null_dist=valid_null,
                    observed_val=observed_val,
                    p=p_perm,
                    mark_p=None,
                    metric=f"{metric}_vs_{x_col}",
                    save_path=C.FIGPATH,
                    show=True,
                    region="Omnibus",
                )

            out_omni = pd.DataFrame(
                [{
                    "y_col": metric,
                    "x_col": x_col,
                    "n_perm": C.N_PERMUT_NEURAL_OMNIBUS,
                    "formula": formula_full,
                    "shuffling": shuffling,
                    "observed_val": observed_val,
                    "observed_val_p": observed_val_p,
                    "p_perm": p_perm,
                    "pearson_r": BF_dict["r"],
                    "BF10": BF_dict["BF10"],
                    "BF_conclusion": interpret_bayes_factor(BF_dict["BF10"]),
                    "ave_null_dist": valid_null.mean() if len(valid_null) else np.nan,
                    "n_pairs": len(df_use),
                }]
            )

            omni_fname = (
                C.RESULTSPATH
                / f"pairlevel_Omnibus_{metric}_vs_{x_col}_{window}_1000permutation_"
                  f"{C.ALIGN_EVENT}_{C.TRIAL_TYPE}_500{ANALYSIS_TAG}.csv"
            
            )
            out_omni.to_csv(omni_fname, index=False)
            log.info(f"[Saved omnibus] {omni_fname.resolve()}")


if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    main(plot_permt_result=False)
