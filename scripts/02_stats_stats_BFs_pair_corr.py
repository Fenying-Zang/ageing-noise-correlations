"""
Computing BayesFactors for
- pre noise correlations
- post noise correlations
- delta noise correlations

"""
#%%
import pandas as pd
import numpy as np
import os
import platform
from scripts.utils.data_utils import interpret_bayes_factor, add_age_group
from scripts.utils.io import read_table, get_suffix
import config as C
from rpy2.robjects import Formula
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter    
from rpy2.robjects import default_converter, pandas2ri  
import logging

ANALYSIS_TAG = "_visp"  

log = logging.getLogger(__name__)
# ---------- R env ----------
os.environ['R_HOME'] = r"C:/Program Files/R/R-4.5.1"#Note: updated R version here
os.environ['R_USER'] = os.path.expanduser("~")
os.environ["R_DISABLE_CONSOLE_OUTPUT"] = "TRUE"
os.environ['RPY2_CFFI_MODE'] = 'ABI'  

# Load BayesFactor 
bayesfactor = importr('BayesFactor')

# Disable JIT (avoids rpy2 issues on Windows)
ro.r('compiler::enableJIT(0)')

try:  
    if platform.system() == "Windows":
        ro.r('Sys.setlocale("LC_CTYPE", "English_United States")')
    else:
        ro.r('Sys.setlocale("LC_ALL", "English_United_States.UTF-8")')
except Exception:
    pass  


def def_BF_formula(metric, log_transform=False):
    if metric in ['r_noise','delta_r_noise']:
        formula_full_str = f'{metric} ~ age_years + n_trials + cluster_region + pair_distance + cluster_geo_mean_fr'
        formula_reduced_age_str = f'{metric} ~ n_trials + cluster_region + pair_distance + cluster_geo_mean_fr'
        formula_reduced_region_str = f'{metric} ~ age_years + n_trials + pair_distance + cluster_geo_mean_fr'
    else:
        formula_full_str = f'{metric} ~ age_years + cluster_region + pair_distance + cluster_geo_mean_fr'
        formula_reduced_age_str = f'{metric} ~  cluster_region + pair_distance + cluster_geo_mean_fr'
        formula_reduced_region_str = f'{metric} ~ age_years + pair_distance + cluster_geo_mean_fr'

    return formula_full_str, formula_reduced_age_str, formula_reduced_region_str


def def_BF_formula_region(metric, log_transform=False):
    if metric in ['r_noise','delta_r_noise']:
        formula_full_str = f'{metric} ~ age_years + n_trials + pair_distance + cluster_geo_mean_fr'
        formula_reduced_age_str = f'{metric} ~ n_trials + pair_distance + cluster_geo_mean_fr'
    else:
        formula_full_str = f'{metric} ~ age_years + pair_distance + cluster_geo_mean_fr'
        formula_reduced_age_str = f'{metric} ~ pair_distance + cluster_geo_mean_fr'
    return formula_full_str, formula_reduced_age_str  


def compute_bayes_factor(
    df,
    metric,
    formula_full_str=None,
    formula_reduced_age_str=None,
    formula_reduced_region_str=None,
    posterior_iterations=10000,
):
    # pandas -> R
    with localconverter(default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)

    ro.globalenv["df_r"] = r_df
    ro.globalenv["formula_full"] = Formula(formula_full_str)
    ro.globalenv["formula_reduced_age"] = Formula(formula_reduced_age_str)

    if formula_reduced_region_str is not None:
        ro.globalenv["formula_reduced_region"] = Formula(formula_reduced_region_str)

    # Compute Bayes factors
    ro.r("""
        library(BayesFactor)

        bf_full <- lmBF(formula_full, data = df_r)
        bf_no_age <- lmBF(formula_reduced_age, data = df_r)
        bf_age <- bf_full / bf_no_age

        assign("bf_age", bf_age, envir = .GlobalEnv)
        assign("bf_full", bf_full, envir = .GlobalEnv)
    """)

    if formula_reduced_region_str is not None:
        ro.r("""
            bf_no_region <- lmBF(formula_reduced_region, data = df_r)
            bf_region <- bf_full / bf_no_region
            assign("bf_region", bf_region, envir = .GlobalEnv)
        """)

    BF10_age = float(ro.r("extractBF(bf_age)$bf[1]")[0])

    BF10_region = np.nan
    if formula_reduced_region_str is not None:
        BF10_region = float(ro.r("extractBF(bf_region)$bf[1]")[0])

    # Optional posterior summary
    if posterior_iterations is not None and posterior_iterations > 0:
        ro.globalenv["posterior_iterations"] = posterior_iterations

        ro.r("""
            chains <- posterior(bf_full, iterations = posterior_iterations)

            summary_stats <- as.data.frame(summary(chains)$statistics)
            summary_quants <- as.data.frame(summary(chains)$quantiles)

            assign("summary_stats", summary_stats, envir = .GlobalEnv)
            assign("summary_quants", summary_quants, envir = .GlobalEnv)
        """)

        with localconverter(default_converter + pandas2ri.converter):
            stats_df = ro.conversion.rpy2py(ro.r["summary_stats"])
            quants_df = ro.conversion.rpy2py(ro.r["summary_quants"])

        summary_df = pd.concat([stats_df, quants_df], axis=1)

    else:
        summary_df = pd.DataFrame()

    return BF10_age, BF10_region, summary_df


def safe_lookup(df, row, col, default=np.nan):
    try:
        if row in df.index:
            return df.loc[row, col]

        matches = [idx for idx in df.index if str(idx).startswith(f"{row}-")]
        if len(matches) == 1:
            return df.loc[matches[0], col]

        return default

    except Exception:
        return default


def clean_dependent_variable(df, y_col):
    return df[np.isfinite(df[y_col]) & ~df[y_col].isna()].copy()


def load_pair_corr_prepost():

    metrics_path = C.DATAPATH / f"proj2_noise_pair_corr_merged_prepost_500ms{ANALYSIS_TAG}.parquet"
    df = read_table(metrics_path)
    print(f"[Loaded] {metrics_path}")
    print(df.shape)
    print(df["window"].value_counts(dropna=False))

    return df


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


def get_window_table(df, window):
    out = df[df["window"] == window].copy()
    out = standardize_pair_corr_columns(out)
    out = clean_dependent_variable(out, "r_noise")
    out = add_age_group(out)

    print(f"[{window}] rows after cleaning: {len(out)}")
    return out


def get_delta_table(df):
    key_cols = [
        "session_pid",
        "cluster_region",
        "cluster_id1",
        "cluster_id2",
    ]

    pre = standardize_pair_corr_columns(df[df["window"] == "pre"].copy())
    post = standardize_pair_corr_columns(df[df["window"] == "post"].copy())

    keep_cols = key_cols + [
        "r_noise",
        "n_trials",
        "mouse_age",
        "mouse_name",
        "session_eid",
        "cluster_geo_mean_fr",
        "pair_distance",
    ]

    pre = pre[keep_cols].copy()
    post = post[key_cols + ["r_noise"]].copy()

    delta = pd.merge(
        pre,
        post,
        on=key_cols,
        how="inner",
        suffixes=("_pre", "_post"),
    )

    delta["delta_r_noise"] = delta["r_noise_post"] - delta["r_noise_pre"]

    delta = clean_dependent_variable(delta, "delta_r_noise")
    delta = add_age_group(delta)

    print(f"[delta_r_noise] rows after cleaning: {len(delta)}")
    return delta


def main(log_transform=False):

    df_prepost = load_pair_corr_prepost()

    analysis_tables = {
        "pre": {
            "metric": "r_noise",
            "df": get_window_table(df_prepost, "pre"),
        },
        "post": {
            "metric": "r_noise",
            "df": get_window_table(df_prepost, "post"),
        },
        "quench": {
            "metric": "delta_r_noise",
            "df": get_delta_table(df_prepost),
        },
    }

    result_list = []

    for analysis_name, analysis_info in analysis_tables.items():
        metric = analysis_info["metric"]
        neural_metrics2use = analysis_info["df"]

        print("\n==============================")
        print(f"Running global BF: {analysis_name} | {metric}")
        print(f"Rows: {len(neural_metrics2use)}")
        print("==============================\n")

        formula_full_str, formula_reduced_age_str, formula_reduced_region_str = def_BF_formula(
            metric, log_transform=log_transform
        )

        BF10_age, BF10_region, chain_table = compute_bayes_factor(
            neural_metrics2use,
            metric=metric,
            formula_full_str=formula_full_str,
            formula_reduced_age_str=formula_reduced_age_str,
            formula_reduced_region_str=formula_reduced_region_str,
        )

        chain_file = (
            C.RESULTSPATH
            / f"proj2_noise_pairlevel_global_BF_chain_{analysis_name}_{metric}_500ms{ANALYSIS_TAG}.csv"
        )
        chain_table.to_csv(chain_file)

        BF10_age_category = interpret_bayes_factor(BF10_age)
        BF10_region_category = interpret_bayes_factor(BF10_region)

        result_df = pd.DataFrame({
            "analysis": [analysis_name],
            "level": ["global"],
            "metric": [metric],
            "formula_full": [formula_full_str],
            "formula_reduced_age": [formula_reduced_age_str],
            "formula_reduced_region": [formula_reduced_region_str],
            "BF10_age": [BF10_age],
            "BF10_age_category": [BF10_age_category],
            "BF10_region": [BF10_region],
            "BF10_region_category": [BF10_region_category],
            "mean_age": [safe_lookup(chain_table, "age_years", "Mean")],
            "low_ci_age": [safe_lookup(chain_table, "age_years", "2.5%")],
            "high_ci_age": [safe_lookup(chain_table, "age_years", "97.5%")],
            "n_rows": [len(neural_metrics2use)],
            "chain_file": [chain_file.name],
        })

        result_list.append(result_df)

    final_df = pd.concat(result_list, ignore_index=True)
    final_df["age_ci_conclusion"] = ~(
        (final_df["low_ci_age"] < 0) & (final_df["high_ci_age"] > 0)
    )

    filename = C.RESULTSPATH / f"proj2_noise_pairlevel_global_BFs_pre_post_quench_500ms{ANALYSIS_TAG}.csv"
    final_df.to_csv(filename, index=False)

    print(f"[Saved global BF table] {filename}")
    print(final_df)


def run_region_bfs(log_transform=False, min_rows=100):
    
    df_prepost = load_pair_corr_prepost()

    analysis_tables = {
        "pre": {
            "metric": "r_noise",
            "df": get_window_table(df_prepost, "pre"),
        },
        "post": {
            "metric": "r_noise",
            "df": get_window_table(df_prepost, "post"),
        },
        "quench": {
            "metric": "delta_r_noise",
            "df": get_delta_table(df_prepost),
        },
    }

    result_region_list = []

    for analysis_name, analysis_info in analysis_tables.items():
        metric = analysis_info["metric"]
        neural_metrics2use = analysis_info["df"]

        formula_full_r, formula_reduced_age_r = def_BF_formula_region(
            metric, log_transform=log_transform
        )
        regions_to_run = sorted(neural_metrics2use["cluster_region"].dropna().unique())

        for region in regions_to_run:
            region_df = neural_metrics2use[
                neural_metrics2use["cluster_region"] == region
            ].copy()

            print("\n==============================")
            print(f"Running regional BF: {analysis_name} | {metric} | {region}")
            print(f"Rows: {len(region_df)}")
            print("==============================\n")

            if len(region_df) < min_rows:
                print(f"[Skip] Too few rows: {region}")
                continue

            try:
                BF10_age_region, _, chain_table_region = compute_bayes_factor(
                    region_df,
                    metric=metric,
                    formula_full_str=formula_full_r,
                    formula_reduced_age_str=formula_reduced_age_r,
                    formula_reduced_region_str=None,
                    posterior_iterations=0,

                )

                chain_file = (
                    C.RESULTSPATH
                    / f"proj2_noise_pairlevel_regional_BF_chain_{analysis_name}_{region}_{metric}_500ms{ANALYSIS_TAG}.csv"
                )
                # chain_table_region.to_csv(chain_file)

                result_df_region = pd.DataFrame({
                    "analysis": [analysis_name],
                    "level": ["region"],
                    "cluster_region": [region],
                    "metric": [metric],
                    "formula_full": [formula_full_r],
                    "formula_reduced_age": [formula_reduced_age_r],
                    "BF10_age": [BF10_age_region],
                    "BF10_age_category": [interpret_bayes_factor(BF10_age_region)],
                    # "mean_age": [safe_lookup(chain_table_region, "age_years", "Mean")],
                    # "low_ci_age": [safe_lookup(chain_table_region, "age_years", "2.5%")],
                    # "high_ci_age": [safe_lookup(chain_table_region, "age_years", "97.5%")],
                    "mean_age": [np.nan],
                    "low_ci_age": [np.nan],
                    "high_ci_age": [np.nan],
                    "n_rows": [len(region_df)],
                    # "chain_file": [chain_file.name],
                })

                result_df_region["age_ci_conclusion"] = np.nan

                result_region_list.append(result_df_region)

            except Exception as err:
                print(f"[Error] {analysis_name} | {region}: {err}")
                continue

    if result_region_list:
        final_df_region = pd.concat(result_region_list, ignore_index=True)

        filename = (
            C.RESULTSPATH
            / f"proj2_noise_pairlevel_regional_BFs_pre_post_quench_500ms{ANALYSIS_TAG}.csv"
        )
        final_df_region.to_csv(filename, index=False)

        print(f"[Saved regional BF table] {filename}")
        print(final_df_region[[
            "analysis",
            "cluster_region",
            "metric",
            "BF10_age",
            "BF10_age_category",
            "mean_age",
            "low_ci_age",
            "high_ci_age",
            "age_ci_conclusion",
            "n_rows",
        ]])

    return final_df_region


if __name__ == "__main__":

    # run global
    # main(log_transform=False)

    # only run regional 
    run_region_bfs(log_transform=False, min_rows=100)


# %%
