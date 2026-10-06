"""
Validate canonical pair-level correlation relationships with mixed models.
For every outcome/predictor pair, fit:
    M0            outcome ~ region + (1 | session)
    M_predictor   outcome ~ predictor + region + (1 | session)
    M_interaction outcome ~ predictor * region + (1 | session)

Here we report AIC, BIC, and regional predictor slopes with BH-FDR.
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
DATA_FILE = (C.DATAPATH / f"proj2_noise_pair_corr_merged_prepost_{RUN_TAG}.parquet")
VALIDATION_WINDOWS = ["post"]
RANDOM_FACTORS = ["session_eid"]
PREDICTORS_BY_METRIC = { "r_noise": ["pair_distance", "cluster_geo_mean_fr", "r_signal"],
    # "r_signal": ["pair_distance", "cluster_geo_mean_fr"],
}
SCALE_PREDICTOR = False
CI_LEVEL = 0.95
FDR_ALPHA = 0.01
R_HOME = r"C:/Program Files/R/R-4.5.1"

# %% 2. LOAD THE PAIR-LEVEL TABLE
df = read_table(DATA_FILE)

# Fill the standard column names from suffixed merged-table columns when needed.
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

for old_column, new_column in column_map.items():
    if old_column in df.columns:
        if new_column in df.columns:
            df[new_column] = df[new_column].combine_first(df[old_column])
        else:
            df[new_column] = df[old_column]

df["window"] = df["window"].astype(str)

print(f"[Loaded] {DATA_FILE}")
print(df.shape)
print(df["window"].value_counts(dropna=False))


# %% 3. START R AND DEFINE THE THREE-MODEL ANALYSIS
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
    fit_validation_models <- function(
        df_r, outcome, predictor,
        formula_M0, formula_M_predictor, formula_M_interaction,
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
        M_predictor <- lmer(
            as.formula(formula_M_predictor), data = df_r,
            REML = FALSE, control = control, na.action = na.fail
        )
        M_interaction <- lmer(
            as.formula(formula_M_interaction), data = df_r,
            REML = FALSE, control = control, na.action = na.fail
        )

        one_comparison <- function(simple, complex, label,
                                   simple_name, complex_name) {
            ll_simple <- logLik(simple)
            ll_complex <- logLik(complex)
            LRT_chisq <- 2 * (
                as.numeric(ll_complex) - as.numeric(ll_simple)
            )
            LRT_df <- attr(ll_complex, "df") - attr(ll_simple, "df")

            data.frame(
                comparison = label,
                simple_model = simple_name,
                complex_model = complex_name,
                LRT_chisq = LRT_chisq,
                LRT_df = LRT_df,
                LRT_p = pchisq(
                    LRT_chisq, df = LRT_df, lower.tail = FALSE
                ),
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
                M0, M_predictor,
                "common_predictor_effect", "M0", "M_predictor"
            ),
            one_comparison(
                M_predictor, M_interaction,
                "predictor_by_region_interaction",
                "M_predictor", "M_interaction"
            ),
            one_comparison(
                M0, M_interaction,
                "any_predictor_related_structure", "M0", "M_interaction"
            )
        )

        # Common predictor coefficient from M_predictor.
        predictor_coef <- coef(summary(M_predictor))[predictor, ]
        comparisons$beta_predictor_common <- NA_real_
        comparisons$beta_predictor_common_se <- NA_real_
        comparisons$beta_predictor_common_ci_low <- NA_real_
        comparisons$beta_predictor_common_ci_high <- NA_real_
        comparisons$beta_predictor_common_p_wald <- NA_real_

        common_row <- comparisons$comparison == "common_predictor_effect"
        critical_z <- qnorm(1 - (1 - ci_level) / 2)
        comparisons$beta_predictor_common[common_row] <-
            predictor_coef["Estimate"]
        comparisons$beta_predictor_common_se[common_row] <-
            predictor_coef["Std. Error"]
        comparisons$beta_predictor_common_ci_low[common_row] <-
            predictor_coef["Estimate"] -
            critical_z * predictor_coef["Std. Error"]
        comparisons$beta_predictor_common_ci_high[common_row] <-
            predictor_coef["Estimate"] +
            critical_z * predictor_coef["Std. Error"]
        comparisons$beta_predictor_common_p_wald[common_row] <-
            2 * pnorm(
                abs(predictor_coef["Estimate"] /
                    predictor_coef["Std. Error"]),
                lower.tail = FALSE
            )

        # One predictor slope per region from the joint interaction model.
        trends <- emtrends(
            M_interaction,
            specs = ~ cluster_region,
            var = predictor,
            lmer.df = "asymptotic"
        )
        slopes <- as.data.frame(summary(
            trends, infer = c(TRUE, TRUE), level = ci_level, adjust = "none"
        ))

        names(slopes)[grepl("\\.trend$", names(slopes))] <- "beta_predictor"
        names(slopes)[names(slopes) == "asymp.LCL"] <- "ci_low"
        names(slopes)[names(slopes) == "asymp.UCL"] <- "ci_high"
        names(slopes)[names(slopes) == "z.ratio"] <- "z_value"
        names(slopes)[names(slopes) == "p.value"] <- "p_raw"

        slopes$p_BH_FDR <- p.adjust(slopes$p_raw, method = "BH")
        slopes$significant_BH_FDR <- slopes$p_BH_FDR < fdr_alpha
        slopes <- slopes[, c(
            "cluster_region", "beta_predictor", "SE", "df",
            "ci_low", "ci_high", "z_value", "p_raw",
            "p_BH_FDR", "significant_BH_FDR"
        )]

        list(comparisons = comparisons, slopes = slopes)
    }
    """
)


def to_r(dataframe):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(dataframe)


def to_pandas(r_dataframe):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_dataframe).reset_index(drop=True)


# %% 4. FIT EACH VALIDATION RELATIONSHIP AND SAVE INCREMENTALLY
comparison_results = []
slope_results = []

random_tag = "_and_".join(RANDOM_FACTORS)
window_tag = "_".join(VALIDATION_WINDOWS)

comparison_file = (
    C.RESULTSPATH
    / "proj2_noise_validation_three_model_AIC_BIC_"
      f"{window_tag}_{RUN_TAG}_{random_tag}.csv"
)
slope_file = (
    C.RESULTSPATH
    / "proj2_noise_validation_regional_slopes_BH_FDR_"
      f"{window_tag}_{RUN_TAG}_{random_tag}.csv"
)

for window in VALIDATION_WINDOWS:
    window_df = df.loc[df["window"] == window].copy()

    for outcome, predictors in PREDICTORS_BY_METRIC.items():
        for predictor in predictors:
            if outcome not in window_df.columns or predictor not in window_df.columns:
                print(f"[Skipped] {outcome} ~ {predictor}: column unavailable")
                continue

            for random_factor in RANDOM_FACTORS:
                model_columns = [
                    outcome, predictor, "cluster_region", random_factor
                ]
                model_data = window_df[model_columns].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna().copy()

                if SCALE_PREDICTOR:
                    model_data[predictor] = (
                        model_data[predictor] - model_data[predictor].mean()
                    ) / model_data[predictor].std(ddof=0)

                formula_M0 = (
                    f"{outcome} ~ cluster_region + (1 | {random_factor})"
                )
                formula_M_predictor = (
                    f"{outcome} ~ {predictor} + cluster_region "
                    f"+ (1 | {random_factor})"
                )
                formula_M_interaction = (
                    f"{outcome} ~ {predictor} * cluster_region "
                    f"+ (1 | {random_factor})"
                )

                print("\n========================================")
                print(
                    f"{window} | {outcome} ~ {predictor} | "
                    f"random = {random_factor}"
                )
                print(
                    f"rows = {len(model_data)} | "
                    f"groups = {model_data[random_factor].nunique()} | "
                    f"regions = {model_data['cluster_region'].nunique()}"
                )
                print(f"M0:            {formula_M0}")
                print(f"M_predictor:   {formula_M_predictor}")
                print(f"M_interaction: {formula_M_interaction}")
                print("========================================")

                ro.globalenv["model_df"] = to_r(model_data)
                ro.globalenv["outcome"] = outcome
                ro.globalenv["predictor"] = predictor
                ro.globalenv["formula_M0"] = formula_M0
                ro.globalenv["formula_M_predictor"] = formula_M_predictor
                ro.globalenv["formula_M_interaction"] = formula_M_interaction
                ro.globalenv["random_factor"] = random_factor
                ro.globalenv["ci_level"] = CI_LEVEL
                ro.globalenv["fdr_alpha"] = FDR_ALPHA

                fitted = ro.r(
                    "fit_validation_models("
                    "model_df, outcome, predictor, formula_M0, "
                    "formula_M_predictor, formula_M_interaction, "
                    "random_factor, ci_level, fdr_alpha)"
                )

                comparisons = to_pandas(fitted.rx2("comparisons"))
                comparisons.insert(0, "window", window)
                comparisons.insert(1, "outcome", outcome)
                comparisons.insert(2, "predictor", predictor)
                comparisons.insert(3, "random_factor", random_factor)
                comparisons["n_rows"] = len(model_data)
                comparisons["n_random_groups"] = model_data[
                    random_factor
                ].nunique()
                comparisons["n_regions"] = model_data[
                    "cluster_region"
                ].nunique()
                comparisons["predictor_scaled"] = SCALE_PREDICTOR
                comparisons["formula_M0"] = formula_M0
                comparisons["formula_M_predictor"] = formula_M_predictor
                comparisons["formula_M_interaction"] = formula_M_interaction
                comparison_results.append(comparisons)

                slopes = to_pandas(fitted.rx2("slopes"))
                slopes.insert(0, "window", window)
                slopes.insert(1, "outcome", outcome)
                slopes.insert(2, "predictor", predictor)
                slopes.insert(3, "random_factor", random_factor)
                slopes["fdr_alpha"] = FDR_ALPHA
                slopes["predictor_scaled"] = SCALE_PREDICTOR
                slopes["n_rows_model"] = len(model_data)
                slopes["n_random_groups_model"] = model_data[
                    random_factor
                ].nunique()
                slopes["formula_M_interaction"] = formula_M_interaction
                slope_results.append(slopes)

                comparison_file.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(comparison_results, ignore_index=True).to_csv(
                    comparison_file, index=False
                )
                pd.concat(slope_results, ignore_index=True).to_csv(
                    slope_file, index=False
                )
                print(f"[Saved] {comparison_file}")
                print(f"[Saved] {slope_file}")


# %% 5. SHOW FINAL RESULTS
comparison_results = pd.concat(comparison_results, ignore_index=True)
slope_results = pd.concat(slope_results, ignore_index=True)

print("\nMODEL COMPARISONS")
print(
    comparison_results[
        [
            "window",
            "outcome",
            "predictor",
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

print("\nREGIONAL SLOPES")
print(
    slope_results[
        [
            "window",
            "outcome",
            "predictor",
            "cluster_region",
            "beta_predictor",
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