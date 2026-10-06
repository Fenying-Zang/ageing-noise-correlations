"""
Model comparison for pair-level noise correlations.
Models fitted for pre, post, and delta (post - pre) noise correlations.
    M0            y ~ region + covariates + (1 | session)
    M_age         y ~ age + region + covariates + (1 | session)
    M_interaction y ~ age * region + covariates + (1 | session)

Comparisons:
    M_age vs M0             common age effect
    M_interaction vs M_age  age-by-region interaction
    M_interaction vs M0     any age-related structure

Here we report AIC, BIC, and regional age slopes from the joint interaction model.
"""
# %% 1. SETTINGS
import os
import platform
import numpy as np
import pandas as pd
import config as C
from scripts.utils.io import read_table

NC_WINDOWS = C.NC_WINDOWS_500
RUN_TAG = NC_WINDOWS["pre"]["subfoldername"].removeprefix("pair_corr_pre_")
DATA_FILE = (C.DATAPATH / f"noise_pair_corr_merged_prepost_{RUN_TAG}.parquet")
RANDOM_FACTORS = ["session_eid"]
ANALYSES = ["pre", "post", "quench"]
COVARIATES = ["n_trials", "cluster_geo_mean_fr", "pair_distance"]
SCALE_COVARIATES = True
CI_LEVEL = 0.95
FDR_ALPHA = 0.01
R_HOME = r"C:/Program Files/R/R-4.5.1"

# %% 2. LOAD AND PREPARE PRE, POST, AND DELTA TABLES
df = read_table(DATA_FILE)
print(f"[Loaded] {DATA_FILE}")
print(df.shape)
print(df["window"].value_counts(dropna=False))
# Some merged files retain the suffix from the noise-correlation table.
column_map = {
    "mouse_age_noise": "mouse_age",
    "mouse_name_noise": "mouse_name",
    "session_eid_noise": "session_eid",
    "session_pid_noise": "session_pid",
    "pid_noise": "pid",
    "cluster_region_noise": "cluster_region",
    "n_trials_noise": "n_trials",
    "cluster_geo_mean_fr_noise": "cluster_geo_mean_fr",
    "pair_distance_noise": "pair_distance",
    "align_event_noise": "align_event",
}

for old_column, new_column in column_map.items():
    if old_column in df.columns:
        if new_column in df.columns:
            df[new_column] = df[new_column].combine_first(df[old_column])
        else:
            df[new_column] = df[old_column]

pre = df.loc[df["window"] == "pre"].copy()
post = df.loc[df["window"] == "post"].copy()

pre["age_years"] = pre["mouse_age"] / 365
post["age_years"] = post["mouse_age"] / 365

pre = pre.loc[np.isfinite(pre["r_noise"])].copy()
post = post.loc[np.isfinite(post["r_noise"])].copy()

# Build quench = post - pre for the same neuron pair.
PAIR_KEYS = ["session_pid", "cluster_region", "cluster_id1", "cluster_id2"]
PRE_COLUMNS = list(
    dict.fromkeys(
        PAIR_KEYS
        + [
            "r_noise",
            "n_trials",
            "mouse_age",
            "mouse_name",
            "session_eid",
            "cluster_geo_mean_fr",
            "pair_distance",
        ]
        + RANDOM_FACTORS
    )
)

quench = pre[PRE_COLUMNS].merge(
    post[PAIR_KEYS + ["r_noise"]],
    on=PAIR_KEYS,
    how="inner",
    suffixes=("_pre", "_post"),
    validate="one_to_one",
)
quench["delta_r_noise"] = quench["r_noise_post"] - quench["r_noise_pre"]
quench["age_years"] = quench["mouse_age"] / 365
quench = quench.loc[np.isfinite(quench["delta_r_noise"])].copy()

analysis_tables = {
    "pre": (pre, "r_noise"),
    "post": (post, "r_noise"),
    "quench": (quench, "delta_r_noise"),
}

for name, (table, metric) in analysis_tables.items():
    print(f"[{name}] {metric}: {len(table)} rows")


# %% 3. START R AND DEFINE THE MODEL FIT
if platform.system() == "Windows":
    os.environ.setdefault("R_HOME", R_HOME)

os.environ.setdefault("R_USER", os.path.expanduser("~"))
os.environ.setdefault("R_DISABLE_CONSOLE_OUTPUT", "TRUE")
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

importr("lme4")
importr("emmeans")

ro.r(
    r"""
    fit_three_models <- function(
        df_r, formula_M0, formula_M_age, formula_M_interaction,
        random_factor, ci_level, fdr_alpha
    ) {
        library(lme4)
        library(emmeans)

        df_r$cluster_region <- factor(df_r$cluster_region)
        df_r[[random_factor]] <- factor(df_r[[random_factor]])

        control <- lmerControl(
            optimizer = "bobyqa",
            optCtrl = list(maxfun = 200000)
        )

        M0 <- lmer(
            as.formula(formula_M0), data = df_r,
            REML = FALSE, control = control, na.action = na.fail
        )
        M_age <- lmer(
            as.formula(formula_M_age), data = df_r,
            REML = FALSE, control = control, na.action = na.fail
        )
        M_interaction <- lmer(
            as.formula(formula_M_interaction), data = df_r,
            REML = FALSE, control = control, na.action = na.fail
        )

        one_comparison <- function(simple, complex, label, simple_name,
                                   complex_name) {
            logLik_simple <- logLik(simple)
            logLik_complex <- logLik(complex)
            LRT_chisq <- 2 * (
                as.numeric(logLik_complex) - as.numeric(logLik_simple)
            )
            LRT_df <- attr(logLik_complex, "df") - attr(logLik_simple, "df")

            data.frame(
                comparison = label,
                simple_model = simple_name,
                complex_model = complex_name,
                LRT_chisq = LRT_chisq,
                LRT_df = LRT_df,
                LRT_p = pchisq(LRT_chisq, df = LRT_df,
                               lower.tail = FALSE),
                AIC_simple = AIC(simple),
                AIC_complex = AIC(complex),
                delta_AIC_simple_minus_complex = AIC(simple) - AIC(complex),
                BIC_simple = BIC(simple),
                BIC_complex = BIC(complex),
                delta_BIC_simple_minus_complex = BIC(simple) - BIC(complex),
                singular_simple = isSingular(simple, tol = 1e-4),
                singular_complex = isSingular(complex, tol = 1e-4)
            )
        }

        comparisons <- rbind(
            one_comparison(
                M0, M_age, "common_age_effect", "M0", "M_age"
            ),
            one_comparison(
                M_age, M_interaction, "age_by_region_interaction",
                "M_age", "M_interaction"
            ),
            one_comparison(
                M0, M_interaction, "any_age_related_structure",
                "M0", "M_interaction"
            )
        )

        # Coefficient from M_age: the common age slope across regions.
        age_coef <- coef(summary(M_age))["age_years", ]
        comparisons$beta_age_common <- NA_real_
        comparisons$beta_age_common_se <- NA_real_
        comparisons$beta_age_common_ci_low <- NA_real_
        comparisons$beta_age_common_ci_high <- NA_real_
        comparisons$beta_age_common_p_wald <- NA_real_

        common_row <- comparisons$comparison == "common_age_effect"
        comparisons$beta_age_common[common_row] <- age_coef["Estimate"]
        comparisons$beta_age_common_se[common_row] <- age_coef["Std. Error"]
        comparisons$beta_age_common_ci_low[common_row] <-
            age_coef["Estimate"] - qnorm(1 - (1 - ci_level) / 2) *
            age_coef["Std. Error"]
        comparisons$beta_age_common_ci_high[common_row] <-
            age_coef["Estimate"] + qnorm(1 - (1 - ci_level) / 2) *
            age_coef["Std. Error"]
        comparisons$beta_age_common_p_wald[common_row] <-
            2 * pnorm(abs(age_coef["Estimate"] / age_coef["Std. Error"]),
                      lower.tail = FALSE)

        # One age slope per region from the joint interaction model.
        trends <- emtrends(
            M_interaction,
            specs = ~ cluster_region,
            var = "age_years",
            lmer.df = "asymptotic"
        )
        slopes <- as.data.frame(summary(
            trends, infer = c(TRUE, TRUE), level = ci_level, adjust = "none"
        ))

        names(slopes)[grepl("\\.trend$", names(slopes))] <- "beta_age"
        names(slopes)[names(slopes) == "asymp.LCL"] <- "ci_low"
        names(slopes)[names(slopes) == "asymp.UCL"] <- "ci_high"
        names(slopes)[names(slopes) == "z.ratio"] <- "z_value"
        names(slopes)[names(slopes) == "p.value"] <- "p_raw"

        slopes$p_BH_FDR <- p.adjust(slopes$p_raw, method = "BH")
        slopes$significant_BH_FDR <- slopes$p_BH_FDR < fdr_alpha

        slopes <- slopes[, c(
            "cluster_region", "beta_age", "SE", "df", "ci_low", "ci_high",
            "z_value", "p_raw", "p_BH_FDR", "significant_BH_FDR"
        )]

        list(comparisons = comparisons, slopes = slopes)
    }
    """
)


def to_r(dataframe):
    """Convert a pandas DataFrame to an R data.frame."""
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(dataframe)


def to_pandas(r_dataframe):
    """Convert an R data.frame to pandas."""
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_dataframe).reset_index(drop=True)


# %% 4. FIT MODELS AND SAVE RESULTS
comparison_results = []
slope_results = []

random_tag = "_and_".join(RANDOM_FACTORS)
comparison_file = (
    C.RESULTSPATH
    / f"noise_three_model_AIC_BIC_{RUN_TAG}_{random_tag}.csv"
)
slope_file = (
    C.RESULTSPATH
    / f"noise_regional_slopes_BH_FDR_{RUN_TAG}_{random_tag}.csv"
)

for random_factor in RANDOM_FACTORS:
    for analysis in ANALYSES:
        table, metric = analysis_tables[analysis]

        model_columns = [
            metric,
            "age_years",
            "cluster_region",
            *COVARIATES,
            random_factor,
        ]
        model_data = table[model_columns].replace(
            [np.inf, -np.inf], np.nan
        ).dropna().copy()

        if SCALE_COVARIATES:
            for covariate in COVARIATES:
                model_data[covariate] = (
                    model_data[covariate] - model_data[covariate].mean()
                ) / model_data[covariate].std(ddof=0)

        fixed_covariates = " + ".join(COVARIATES)
        formula_M0 = (
            f"{metric} ~ cluster_region + {fixed_covariates} "
            f"+ (1 | {random_factor})"
        )
        formula_M_age = (
            f"{metric} ~ age_years + cluster_region + {fixed_covariates} "
            f"+ (1 | {random_factor})"
        )
        formula_M_interaction = (
            f"{metric} ~ age_years * cluster_region + {fixed_covariates} "
            f"+ (1 | {random_factor})"
        )

        print("\n========================================")
        print(f"{analysis} | {metric} | random = {random_factor}")
        print(
            f"rows = {len(model_data)} | "
            f"groups = {model_data[random_factor].nunique()} | "
            f"regions = {model_data['cluster_region'].nunique()}"
        )
        print(f"M0:            {formula_M0}")
        print(f"M_age:         {formula_M_age}")
        print(f"M_interaction: {formula_M_interaction}")
        print("========================================")

        ro.globalenv["model_df"] = to_r(model_data)
        ro.globalenv["formula_M0"] = formula_M0
        ro.globalenv["formula_M_age"] = formula_M_age
        ro.globalenv["formula_M_interaction"] = formula_M_interaction
        ro.globalenv["random_factor"] = random_factor
        ro.globalenv["ci_level"] = CI_LEVEL
        ro.globalenv["fdr_alpha"] = FDR_ALPHA

        fitted = ro.r(
            "fit_three_models(model_df, formula_M0, formula_M_age, "
            "formula_M_interaction, random_factor, ci_level, fdr_alpha)"
        )

        comparisons = to_pandas(fitted.rx2("comparisons"))
        comparisons.insert(0, "analysis", analysis)
        comparisons.insert(1, "metric", metric)
        comparisons.insert(2, "random_factor", random_factor)
        comparisons["n_rows"] = len(model_data)
        comparisons["n_random_groups"] = model_data[random_factor].nunique()
        comparisons["n_regions"] = model_data["cluster_region"].nunique()
        comparisons["formula_M0"] = formula_M0
        comparisons["formula_M_age"] = formula_M_age
        comparisons["formula_M_interaction"] = formula_M_interaction
        comparison_results.append(comparisons)

        slopes = to_pandas(fitted.rx2("slopes"))
        slopes.insert(0, "analysis", analysis)
        slopes.insert(1, "metric", metric)
        slopes.insert(2, "random_factor", random_factor)
        slopes["fdr_alpha"] = FDR_ALPHA
        slopes["n_rows_model"] = len(model_data)
        slopes["n_random_groups_model"] = model_data[
            random_factor
        ].nunique()
        slopes["formula_M_interaction"] = formula_M_interaction
        slope_results.append(slopes)

        # Save incrementally so completed analyses remain available if a later
        # model is interrupted.
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(comparison_results, ignore_index=True).to_csv(
            comparison_file, index=False
        )
        pd.concat(slope_results, ignore_index=True).to_csv(
            slope_file, index=False
        )
        print(f"[Saved] {comparison_file}")
        print(f"[Saved] {slope_file}")


# %% 5. SHOW THE FINAL TABLES
comparison_results = pd.concat(comparison_results, ignore_index=True)
slope_results = pd.concat(slope_results, ignore_index=True)

print("\nMODEL COMPARISONS")
print(
    comparison_results[
        [
            "analysis",
            "random_factor",
            "comparison",
            "LRT_chisq",
            "LRT_df",
            "LRT_p",
            "delta_AIC_simple_minus_complex",
            "delta_BIC_simple_minus_complex",
            "singular_complex",
        ]
    ].to_string(index=False)
)

print("\nREGIONAL AGE SLOPES")
print(
    slope_results[
        [
            "analysis",
            "random_factor",
            "cluster_region",
            "beta_age",
            "ci_low",
            "ci_high",
            "p_raw",
            "p_BH_FDR",
            "significant_BH_FDR",
        ]
    ].to_string(index=False)
)

print("\nInterpretation of the deltas:")
print("  positive delta AIC/BIC -> favors the complex model")
print("  negative delta AIC/BIC -> favors the simple model")
