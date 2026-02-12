"""
Parent-order-level analyses for internalization market impact study.

Analyses:
  A. Descriptive statistics
  B. OLS regressions (ITT + dose-response)
  C. Propensity score matching / IPW
  D. Non-parametric dose-response curve
"""
import numpy as np
import pandas as pd

from config import (
    CONTINUOUS_CONTROLS,
    CATEGORICAL_FE,
    ENTITY_FE,
    TREATMENT_COLS_ITT,
    TREATMENT_COLS_DOSE,
    OUTCOME_VARS,
    PSM_COVARIATES,
    CRB_BUCKET_LABELS,
    REGRESSION_SAMPLE_SIZE,
)
from utils import (
    run_ols,
    extract_treatment_coefficients,
    estimate_propensity_scores,
    nearest_neighbor_match,
    compute_ipw_weights,
    compute_smd,
    compute_weighted_smd,
    compute_adjusted_means,
    bootstrap_mean_ci,
)


# ===================================================================
# A.  Descriptive analysis
# ===================================================================

def descriptive_summary(df):
    """Produce summary statistics comparing isInt groups and CRBPct groups.

    Returns
    -------
    dict with keys:
      - "isInt_comparison": DataFrame comparing isInt=True vs False
      - "hasCRB_comparison": DataFrame comparing hasCRB=True vs False (among isInt=True)
      - "crbpct_distribution": DataFrame of CRBPct distribution stats among isInt=True
      - "outcome_by_bucket": DataFrame of mean outcomes per CRBPctBucket
    """
    results = {}

    # --- isInt comparison ------------------------------------------------
    stats_rows = []
    compare_cols = CONTINUOUS_CONTROLS + OUTCOME_VARS + ["CRBPct", "ATSPINPct"]
    for col in compare_cols:
        if col not in df.columns:
            continue
        for flag_val, label in [(True, "isInt=True"), (False, "isInt=False")]:
            sub = df.loc[df["isInt"] == flag_val, col]
            stats_rows.append({
                "group": label,
                "variable": col,
                "mean": sub.mean(),
                "median": sub.median(),
                "std": sub.std(),
                "n": sub.notna().sum(),
            })
    results["isInt_comparison"] = pd.DataFrame(stats_rows)

    # --- hasCRB comparison (within isInt=True) ---------------------------
    enabled = df[df["isInt"] == True]
    stats_rows = []
    for col in compare_cols:
        if col not in enabled.columns:
            continue
        for flag_val, label in [(True, "hasCRB"), (False, "noCRB")]:
            sub = enabled.loc[enabled["hasCRB"] == flag_val, col]
            stats_rows.append({
                "group": label,
                "variable": col,
                "mean": sub.mean(),
                "median": sub.median(),
                "std": sub.std(),
                "n": sub.notna().sum(),
            })
    results["hasCRB_comparison"] = pd.DataFrame(stats_rows)

    # --- CRBPct distribution among enabled orders ------------------------
    crb_nz = enabled.loc[enabled["CRBPct"] > 0, "CRBPct"]
    results["crbpct_distribution"] = pd.DataFrame([{
        "mean": crb_nz.mean(),
        "median": crb_nz.median(),
        "std": crb_nz.std(),
        "p25": crb_nz.quantile(0.25),
        "p75": crb_nz.quantile(0.75),
        "p90": crb_nz.quantile(0.90),
        "n_nonzero": len(crb_nz),
        "n_enabled": len(enabled),
        "pct_with_crb": len(crb_nz) / len(enabled) * 100,
    }])

    # --- Mean outcomes by CRBPctBucket -----------------------------------
    bucket_stats = (
        df.groupby("CRBPctBucket", observed=True)[OUTCOME_VARS]
        .agg(["mean", "std", "count"])
    )
    results["outcome_by_bucket"] = bucket_stats

    return results


# ===================================================================
# B.  OLS regressions
# ===================================================================

def run_all_regressions(df, cluster_col="RIC"):
    """Run ITT and dose-response regressions for all outcome variables.

    Returns
    -------
    dict with keys:
      - "itt_results": list of statsmodels results (one per outcome)
      - "dose_results": list of statsmodels results
      - "itt_coefficients": DataFrame of treatment coefficients across outcomes
      - "dose_coefficients": DataFrame
      - "itt_enabled_results": list (ITT regression within isInt=True using hasCRB)
      - "itt_enabled_coefficients": DataFrame
    """
    all_controls = CONTINUOUS_CONTROLS
    all_fe = CATEGORICAL_FE + ENTITY_FE
    sample_n = REGRESSION_SAMPLE_SIZE

    results = {}

    # --- ITT: isInt on full population -----------------------------------
    itt_res = []
    itt_coefs = []
    for outcome in OUTCOME_VARS:
        res = run_ols(df, outcome, ["isInt"], all_controls, all_fe,
                      cluster_col=cluster_col, sample_n=sample_n)
        itt_res.append(res)
        coef_df = extract_treatment_coefficients(res, ["isInt"])
        coef_df["outcome"] = outcome
        itt_coefs.append(coef_df)

    results["itt_results"] = itt_res
    results["itt_coefficients"] = pd.concat(itt_coefs, ignore_index=True)

    # --- Dose-response: CRBPct + ATSPINPct on full population ------------
    dose_res = []
    dose_coefs = []
    for outcome in OUTCOME_VARS:
        res = run_ols(df, outcome, TREATMENT_COLS_DOSE, all_controls, all_fe,
                      cluster_col=cluster_col, sample_n=sample_n)
        dose_res.append(res)
        coef_df = extract_treatment_coefficients(res, TREATMENT_COLS_DOSE)
        coef_df["outcome"] = outcome
        dose_coefs.append(coef_df)

    results["dose_results"] = dose_res
    results["dose_coefficients"] = pd.concat(dose_coefs, ignore_index=True)

    # --- Within-enabled: hasCRB among isInt=True -------------------------
    enabled = df[df["isInt"] == True].copy()
    en_res = []
    en_coefs = []
    for outcome in OUTCOME_VARS:
        res = run_ols(enabled, outcome, ["CRBPct"], all_controls, all_fe,
                      cluster_col=cluster_col, sample_n=sample_n)
        en_res.append(res)
        coef_df = extract_treatment_coefficients(res, ["CRBPct"])
        coef_df["outcome"] = outcome
        en_coefs.append(coef_df)

    results["itt_enabled_results"] = en_res
    results["itt_enabled_coefficients"] = pd.concat(en_coefs, ignore_index=True)

    return results


# ===================================================================
# C.  Propensity Score Matching / IPW
# ===================================================================

def run_psm_analysis(df, treatment_col="hasCRB", max_sample=2_000_000,
                     caliper_mult=0.2):
    """Run propensity-score analysis: matching + IPW.

    Uses a subsample for nearest-neighbor matching (expensive) and full
    sample for IPW.

    Parameters
    ----------
    df : DataFrame – should already have derived columns
    treatment_col : str
    max_sample : int – cap for NN matching
    caliper_mult : float – caliper as multiple of PS std

    Returns
    -------
    dict with keys:
      - "smd_before": DataFrame
      - "smd_after_ipw": DataFrame
      - "smd_after_nn": DataFrame
      - "ipw_outcomes": DataFrame – weighted mean outcomes by group
      - "nn_outcomes": DataFrame – matched mean outcomes
      - "propensity_scores": Series
    """
    work = df.dropna(subset=PSM_COVARIATES + [treatment_col]).copy()
    results = {}

    # --- Propensity scores -----------------------------------------------
    ps = estimate_propensity_scores(work, treatment_col, PSM_COVARIATES)
    work["ps"] = ps
    results["propensity_scores"] = work["ps"]

    # --- SMD before ------------------------------------------------------
    results["smd_before"] = compute_smd(work, treatment_col, PSM_COVARIATES)

    # --- IPW (full data) ------------------------------------------------
    weights = compute_ipw_weights(ps, work[treatment_col].astype(int))
    results["smd_after_ipw"] = compute_weighted_smd(
        work, treatment_col, PSM_COVARIATES, weights
    )

    ipw_rows = []
    for outcome in OUTCOME_VARS:
        if outcome not in work.columns:
            continue
        for tval, label in [(True, "treated"), (False, "control")]:
            mask = work[treatment_col].astype(bool) == tval
            vals = work.loc[mask, outcome].values
            w = weights[mask]
            valid = np.isfinite(vals) & (w > 0)
            if valid.sum() == 0:
                continue
            wmean = np.average(vals[valid], weights=w[valid])
            ipw_rows.append({
                "outcome": outcome,
                "group": label,
                "weighted_mean": wmean,
                "n": int(valid.sum()),
            })
    results["ipw_outcomes"] = pd.DataFrame(ipw_rows)

    # --- Nearest-neighbor matching (subsample) ---------------------------
    if len(work) > max_sample:
        work_nn = work.sample(n=max_sample, random_state=42)
    else:
        work_nn = work

    treated = work_nn[work_nn[treatment_col].astype(bool)]
    control = work_nn[~work_nn[treatment_col].astype(bool)]

    if len(treated) > 0 and len(control) > 0:
        ps_t = treated["ps"].values
        ps_c = control["ps"].values

        caliper = caliper_mult * work_nn["ps"].std()
        indices, distances = nearest_neighbor_match(ps_t, ps_c, caliper=caliper)

        # build matched dataset
        valid_mask = indices[:, 0] >= 0
        matched_t = treated.iloc[valid_mask].reset_index(drop=True)
        matched_c = control.iloc[indices[valid_mask, 0]].reset_index(drop=True)

        # SMD after matching
        matched_all = pd.concat([
            matched_t.assign(**{treatment_col: True}),
            matched_c.assign(**{treatment_col: False}),
        ], ignore_index=True)
        results["smd_after_nn"] = compute_smd(
            matched_all, treatment_col, PSM_COVARIATES
        )

        # Outcome comparison
        nn_rows = []
        for outcome in OUTCOME_VARS:
            if outcome not in matched_t.columns:
                continue
            mt_vals = matched_t[outcome].dropna()
            mc_vals = matched_c[outcome].dropna()
            diff_mean, diff_lo, diff_hi = bootstrap_mean_ci(
                mt_vals.values - mc_vals.iloc[:len(mt_vals)].values, n_boot=1000
            )
            nn_rows.append({
                "outcome": outcome,
                "treated_mean": mt_vals.mean(),
                "control_mean": mc_vals.mean(),
                "diff": diff_mean,
                "diff_ci_lower": diff_lo,
                "diff_ci_upper": diff_hi,
                "n_pairs": len(mt_vals),
            })
        results["nn_outcomes"] = pd.DataFrame(nn_rows)
    else:
        results["smd_after_nn"] = pd.DataFrame()
        results["nn_outcomes"] = pd.DataFrame()

    return results


# ===================================================================
# D.  Non-parametric dose-response curve
# ===================================================================

def compute_dose_response(df, n_boot=500):
    """Compute confounder-adjusted dose-response curves.

    For each outcome variable, computes adjusted means per CRBPctBucket.

    Parameters
    ----------
    df : DataFrame
    n_boot : int – bootstrap iterations for CIs

    Returns
    -------
    dict[str, DataFrame] – keyed by outcome variable
    """
    results = {}
    for outcome in OUTCOME_VARS:
        adj = compute_adjusted_means(
            df, outcome, "CRBPctBucket",
            CONTINUOUS_CONTROLS, CATEGORICAL_FE,
            n_boot=n_boot,
        )
        results[outcome] = adj
    return results


# ===================================================================
# Convenience: run everything
# ===================================================================

def run_full_parent_analysis(df, cluster_col="RIC"):
    """Execute all parent-level analyses and return combined results dict."""
    print("  [1/4] Descriptive analysis ...")
    desc = descriptive_summary(df)

    print("  [2/4] OLS regressions ...")
    reg = run_all_regressions(df, cluster_col=cluster_col)

    print("  [3/4] Propensity score analysis ...")
    psm = run_psm_analysis(df)

    print("  [4/4] Dose-response curves ...")
    dr = compute_dose_response(df)

    return {
        "descriptive": desc,
        "regression": reg,
        "psm": psm,
        "dose_response": dr,
    }
