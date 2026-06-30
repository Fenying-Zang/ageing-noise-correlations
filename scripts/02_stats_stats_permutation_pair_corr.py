"""
replicate the permutation test at the pair level for 
- pre noise correlations
- post noise correlations
- delta noise correlations

"""
#%%#
import pandas as pd
import numpy as np
from statsmodels.genmod.families import Gaussian
import config as C
from scripts.utils.plot_utils import plot_permut_test
from scripts.utils.io import read_table, get_suffix
from scripts.utils.stats_utils import run_permutation_test  
from scripts.utils.io import read_table
import logging

ANALYSIS_TAG = "_visp"

log = logging.getLogger(__name__)
FAMILY_FUNC = Gaussian()
N_JOBS = 6
SHUFFLING = 'labels1_based_on_2'  


def def_glm_formula(metric):
    """Return GLM formula string for a metric, switching covariates by metric /log_transform."""
    if metric in ['r_noise', 'delta_r_noise']:
        formula2use = f"{metric} ~ age_years + C(cluster_region) + n_trials + cluster_geo_mean_fr + pair_distance"
    else:
        formula2use = f"{metric} ~ age_years + C(cluster_region) + cluster_geo_mean_fr + pair_distance"
    return formula2use


def drop_term_from_formula(formula, term="C(cluster_region)"):
    """Remove a single RHS term from a Patsy-style formula safely."""
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    terms = [t.strip() for t in rhs.split("+")]
    terms = [t for t in terms if t != term and t != ""]
    rhs_new = " + ".join(terms) if terms else "1"
    return f"{lhs} ~ {rhs_new}"


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


def add_age_columns(df):
    df = df.copy()
    df["age_group"] = df["mouse_age"].map(
        lambda x: "old" if x > C.AGE_GROUP_THRESHOLD else "young"
    )
    df["mouse_age_months"] = df["mouse_age"] / 30
    df["age_years"] = df["mouse_age"] / 365

    return df


def get_window_table(df, window):
    out = df[df["window"] == window].copy()
    out = standardize_pair_corr_columns(out)
    out = clean_dependent_variable(out, "r_noise")
    out = add_age_columns(out)

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
    delta = add_age_columns(delta)

    print(f"[delta_r_noise] rows after cleaning: {len(delta)}")
    return delta


def main(plot_permt_result=False, run_regions=False):
    
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

    global_results = []

    for analysis_name, analysis_info in analysis_tables.items():
        metric = analysis_info["metric"]
        neural_metrics2use = analysis_info["df"]

        this_age = neural_metrics2use["age_years"].values
        this_eid = neural_metrics2use["session_eid"].values
        formula_full = def_glm_formula(metric)

        print("\n==============================")
        print(f"Running global permutation: {analysis_name} | {metric}")
        print(f"Rows: {len(neural_metrics2use)}")
        print(f"Formula: {formula_full}")
        print("==============================\n")

        observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
            data=neural_metrics2use,
            age_labels=this_age,
            group_labels=this_eid,
            formula=formula_full,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=C.N_PERMUT_NEURAL_OMNIBUS,
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE,
            plot=False,
        )

        global_results.append({
            "analysis": analysis_name,
            "level": "global",
            "cluster_region": "Omnibus",
            "y_col": metric,
            "n_perm": C.N_PERMUT_NEURAL_OMNIBUS,
            "formula": formula_full,
            "observed_val": observed_val,
            "observed_val_p": observed_val_p,
            "p_perm": p_perm,
            "ave_null_dist": valid_null.mean(),
            "std_null_dist": valid_null.std(),
            "n_rows": len(neural_metrics2use),
            "n_sessions": neural_metrics2use["session_eid"].nunique(),
        })

    global_out = (
        C.RESULTSPATH
        / f"proj2_noise_pairlevel_global_permutation_pre_post_quench_{C.N_PERMUT_NEURAL_OMNIBUS}perms_500ms{ANALYSIS_TAG}.csv"
    )
    global_df = pd.DataFrame(global_results)
    global_df.to_csv(global_out, index=False)
    print(f"[Saved global permutation table] {global_out}")

    if not run_regions:
        return global_df

    regional_tables = []

    for analysis_name in analysis_tables.keys():
        regional_df = run_region_permutations_for_analysis(
            analysis_name=analysis_name,
            plot_permt_result=plot_permt_result,
            min_rows=100,
        )
        regional_tables.append(regional_df)

    regional_all = pd.concat(regional_tables, ignore_index=True)

    regional_all_out = (
        C.RESULTSPATH
        / f"proj2_noise_pairlevel_regional_permutation_pre_post_quench_{C.N_PERMUT_NEURAL_REGIONAL}perms_500ms{ANALYSIS_TAG}.csv"
    )
    regional_all.to_csv(regional_all_out, index=False)
    print(f"[Saved combined regional permutation table] {regional_all_out}")

    return global_df, regional_all


def run_region_permutations_for_analysis(
    analysis_name="pre",
    plot_permt_result=False,
    min_rows=100,
):

    df_prepost = load_pair_corr_prepost()

    if analysis_name == "pre":
        metric = "r_noise"
        neural_metrics2use = get_window_table(df_prepost, "pre")
    elif analysis_name == "post":
        metric = "r_noise"
        neural_metrics2use = get_window_table(df_prepost, "post")
    elif analysis_name == "quench":
        metric = "delta_r_noise"
        neural_metrics2use = get_delta_table(df_prepost)
    else:
        raise ValueError("analysis_name must be 'pre', 'post', or 'quench'")

    formula_full = def_glm_formula(metric)
    formula_region = drop_term_from_formula(formula_full, "C(cluster_region)")

    region_results = []

    regions_list = sorted(neural_metrics2use["cluster_region"].dropna().unique())
    print(f"Regions to run for {analysis_name}: {regions_list}")

    for region in regions_list:
        region_data = neural_metrics2use[
            neural_metrics2use["cluster_region"] == region
        ].copy()

        print("\n==============================")
        print(f"Running regional permutation: {analysis_name} | {metric} | {region}")
        print(f"Rows: {len(region_data)}")
        print(f"Unique sessions: {region_data['session_eid'].nunique()}")
        print(f"Formula: {formula_region}")
        print("==============================\n")

        if len(region_data) < min_rows:
            print(f"[Skip] Too few rows: {region}")
            continue

        this_age = region_data["age_years"].values
        this_eid = region_data["session_eid"].values

        observed_val, observed_val_p, p_perm, valid_null = run_permutation_test(
            data=region_data,
            age_labels=this_age,
            group_labels=this_eid,
            formula=formula_region,
            family_func=FAMILY_FUNC,
            shuffling=SHUFFLING,
            n_permut=C.N_PERMUT_NEURAL_REGIONAL,
            n_jobs=N_JOBS,
            random_state=C.RANDOM_STATE,
            plot=False,
        )

        print(
            f"Region permutation results for {analysis_name} | {region}: "
            f"beta = {observed_val:.6f}, p_perm = {p_perm:.6f}"
        )

        region_results.append({
            "analysis": analysis_name,
            "level": "region",
            "cluster_region": region,
            "y_col": metric,
            "n_perm": C.N_PERMUT_NEURAL_REGIONAL,
            "formula": formula_region,
            "observed_val": observed_val,
            "observed_val_p": observed_val_p,
            "p_perm": p_perm,
            "ave_null_dist": valid_null.mean(),
            "std_null_dist": valid_null.std(),
            "n_rows": len(region_data),
            "n_sessions": region_data["session_eid"].nunique(),
        })

        if plot_permt_result:
            plot_permut_test(
                null_dist=valid_null,
                observed_val=observed_val,
                p=p_perm,
                mark_p=None,
                metric=metric,
                save_path=C.FIGPATH,
                show=True,
                region=region,
            )

    out = (
        C.RESULTSPATH
        / f"proj2_noise_pairlevel_regional_permutation_{analysis_name}_{C.N_PERMUT_NEURAL_REGIONAL}perms_500ms{ANALYSIS_TAG}.csv"
    )

    pd.DataFrame(region_results).to_csv(out, index=False)

    print(f"[Saved regional permutation table] {out}")
    print(pd.DataFrame(region_results))

    return pd.DataFrame(region_results)


if __name__ == "__main__":
    from scripts.utils.io import setup_logging
    setup_logging()
    #For both global and regional permutations, we can run them together or separately. 
    main(plot_permt_result=False,  run_regions=False)
 
    # separate run for regional permutations (if we want to iterate faster on the regional part without re-running the global permutations each time)
    # for analysis in ["pre", "post", "quench"]:
    #     run_region_permutations_for_analysis(
    #         analysis_name=analysis,
    #         plot_permt_result=False,
    #         min_rows=100,
    #     )

