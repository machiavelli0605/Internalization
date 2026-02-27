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
from sklearn.neighbors import NearestNeighbors

from config import (
    CATEGORICAL_FE,
    CONTINUOUS_CONTROLS,
    CRB_BUCKET_LABELS,
    ENTITY_FE,
    EXACT_MATCH_COLS,
    N_NEIGHBORS,
    OUTCOME_VARS,
    PSM_COVARIATES,
    REGRESSION_SAMPLE_SIZE,
    TREATMENT_COLS_DOSE,
    TREATMENT_COLS_ITT,
)
from utils import (
    bootstrap_mean_ci,
    compute_adjusted_means,
    compute_ipw_weights,
    compute_smd,
    compute_weighted_smd,
    estimate_propensity_scores,
    extract_treatment_coefficients,
    nearest_neighbor_match,
    run_ols,
)

# ===================================================================
# A.  Descriptive analysis
# ===================================================================


def stratified_sample(
    df: pd.DataFrame,
    strata_cols,
    n_total: int,
    random_state: int = 42,
    treatmetn_col: str | None = None,
    preserve_treatment_share: bool = True,
    min_per_stratum: int = 0,
):
    """Proportional stratified sampling, optionally preserving treatment share.

    Parameters
    ----------
    df : DataFrame to sample from.
    strata_cols : column(s) defining strata.
    n_total : target total sample size (may be smaller if df is smaller).
    random_state : RNG seed.
    treatmetn_col : if given *and* preserve_treatment_share is True, the
        treatment/control ratio within each stratum is preserved.
    min_per_stratum : guarantee at least this many rows per stratum (if the
        stratum has enough rows).
    """
    if n_total >= len(df):
        return df

    rng = np.random.RandomState(random_state)

    if not strata_cols:
        return df.sample(n=n_total, random_state=rng)

    groups = df.groupby(strata_cols, observed=True)
    stratum_sizes = groups.size()
    n_strata = len(stratum_sizes)

    # --- allocate budget proportionally, respecting min_per_stratum ----------
    raw_alloc = (stratum_sizes / stratum_sizes.sum() * n_total).values
    alloc = np.maximum(np.round(raw_alloc).astype(int), min_per_stratum)
    # clamp to actual stratum size
    alloc = np.minimum(alloc, stratum_sizes.values)
    # redistribute any excess (from clamping or rounding) iteratively
    for _ in range(n_strata):
        gap = n_total - alloc.sum()
        if gap <= 0:
            break
        room = stratum_sizes.values - alloc
        expandable = room > 0
        if not expandable.any():
            break
        # distribute gap proportionally among strata with room
        extra = np.zeros_like(alloc)
        weights = room[expandable] / room[expandable].sum()
        extra[expandable] = np.round(weights * gap).astype(int)
        extra = np.minimum(extra, room)
        alloc += extra
    # if rounding pushed us over, trim from largest allocations
    while alloc.sum() > n_total:
        over = alloc.sum() - n_total
        idx = np.argsort(-alloc)
        for i in idx[:over]:
            if alloc[i] > min_per_stratum:
                alloc[i] -= 1

    alloc_map = dict(zip(stratum_sizes.index, alloc))

    # --- sample within each stratum ----------------------------------------
    parts = []
    for key, grp in groups:
        n_grp = alloc_map[key]
        if n_grp <= 0:
            continue
        if n_grp >= len(grp):
            parts.append(grp)
            continue

        if treatmetn_col and preserve_treatment_share and treatmetn_col in grp.columns:
            treated = grp[grp[treatmetn_col].astype(bool)]
            control = grp[~grp[treatmetn_col].astype(bool)]
            n_t = int(round(len(treated) / len(grp) * n_grp))
            n_t = max(0, min(n_t, len(treated)))
            n_c = min(n_grp - n_t, len(control))
            if n_t > 0:
                parts.append(treated.sample(n=n_t, random_state=rng))
            if n_c > 0:
                parts.append(control.sample(n=n_c, random_state=rng))
        else:
            parts.append(grp.sample(n=n_grp, random_state=rng))

    return pd.concat(parts, axis=0)


def common_support_summary(ps, treat_mask):
    ps_t = ps[treat_mask]
    ps_c = ps[~treat_mask]
    lo_c, hi_c = np.min(ps_c), np.max(ps_c)
    outside = (ps_t < lo_c) | (ps_t > hi_c)
    return {
        "control_min": float(lo_c),
        "control_max": float(hi_c),
        "treated_outside_range_n": int(outside.sum()),
        "treated_ouside_range_pct": float(100 * outside.mean()) if len(ps_t) else np.nan,
    }


def quantiles(x):
    q = np.nanpercentile(x, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    keys = ["min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return {k: float(v) for k, v in zip(keys, q)}


def nearest_distance_summary(score, treat_mask):
    s_t = score[treat_mask].reshape(-1, 1)
    s_c = score[~treat_mask].reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1).fit(s_c)
    dist, _ = nn.kneighbors(s_t)
    dist = dist.ravel()
    return {**quantiles(dist), "mean": float(np.mean(dist)), "n": int(len(dist))}


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
            stats_rows.append(
                {
                    "group": label,
                    "variable": col,
                    "mean": sub.mean(),
                    "median": sub.median(),
                    "std": sub.std(),
                    "n": sub.notna().sum(),
                }
            )
    results["isInt_comparison"] = pd.DataFrame(stats_rows)

    # --- hasCRB comparison (within isInt=True) ---------------------------
    enabled = df[df["isInt"] == True]
    stats_rows = []
    for col in compare_cols:
        if col not in enabled.columns:
            continue
        for flag_val, label in [(True, "hasCRB"), (False, "noCRB")]:
            sub = enabled.loc[enabled["hasCRB"] == flag_val, col]
            stats_rows.append(
                {
                    "group": label,
                    "variable": col,
                    "mean": sub.mean(),
                    "median": sub.median(),
                    "std": sub.std(),
                    "n": sub.notna().sum(),
                }
            )
    results["hasCRB_comparison"] = pd.DataFrame(stats_rows)

    # --- CRBPct distribution among enabled orders ------------------------
    crb_nz = enabled.loc[enabled["CRBPct"] > 0, "CRBPct"]
    results["crbpct_distribution"] = pd.DataFrame(
        [
            {
                "mean": crb_nz.mean(),
                "median": crb_nz.median(),
                "std": crb_nz.std(),
                "p25": crb_nz.quantile(0.25),
                "p75": crb_nz.quantile(0.75),
                "p90": crb_nz.quantile(0.90),
                "n_nonzero": len(crb_nz),
                "n_enabled": len(enabled),
                "pct_with_crb": len(crb_nz) / len(enabled) * 100 if len(enabled) > 0 else 0,
            }
        ]
    )

    # --- Mean outcomes by CRBPctBucket -----------------------------------
    available_outcomes = [o for o in OUTCOME_VARS if o in df.columns]
    if available_outcomes:
        bucket_stats = df.groupby("CRBPctBucket", observed=True)[available_outcomes].agg(["mean", "std", "count"])
    else:
        bucket_stats = pd.DataFrame()
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

    def _run_batch(data, treatments, label):
        """Run OLS for all outcomes, collecting results and coefficients."""
        res_list = []
        coef_list = []
        for outcome in OUTCOME_VARS:
            res = run_ols(
                data,
                outcome,
                treatments,
                all_controls,
                all_fe,
                cluster_col=cluster_col,
                sample_n=sample_n,
            )
            res_list.append(res)
            coef_df = extract_treatment_coefficients(res, treatments)
            if not coef_df.empty:
                coef_df["outcome"] = outcome
                coef_list.append(coef_df)
        coefs = pd.concat(coef_list, ignore_index=True) if coef_list else pd.DataFrame()
        return res_list, coefs

    # --- ITT: isInt on full population -----------------------------------
    itt_res, itt_coefs = _run_batch(df, ["isInt"], "ITT")
    results["itt_results"] = itt_res
    results["itt_coefficients"] = itt_coefs

    # --- Dose-response: CRBPct + ATSPINPct on full population ------------
    dose_res, dose_coefs = _run_batch(df, TREATMENT_COLS_DOSE, "dose")
    results["dose_results"] = dose_res
    results["dose_coefficients"] = dose_coefs

    # --- Within-enabled: CRBPct among isInt=True -------------------------
    enabled = df[df["isInt"].astype(bool)].copy()
    if len(enabled) > 0:
        en_res, en_coefs = _run_batch(enabled, ["CRBPct"], "enabled")
    else:
        en_res, en_coefs = [], pd.DataFrame()
    results["itt_enabled_results"] = en_res
    results["itt_enabled_coefficients"] = en_coefs

    return results


# ===================================================================
# C.  Propensity Score Matching / IPW
# ===================================================================


def run_psm_analysis(
    df,
    treatment_col="hasCRB",
    max_sample=500_000_000,
    caliper_mult=0.2,
):
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
    available_covs = [c for c in PSM_COVARIATES if c in df.columns]
    exact_cols = [c for c in EXACT_MATCH_COLS if c in df.columns]
    ps_covs = [c for c in available_covs if c not in exact_cols]
    EPS = 1e-8

    def dist_to_weight(d, eps=EPS, scheme="inverse"):
        d = np.asarray(d, dtype=float)
        if scheme == "exp":
            return np.exp(-d)
        return 1.0 / (d + eps)

    needed = ps_covs + exact_cols + [treatment_col]
    work = df.dropna(subset=needed).copy()

    results = {
        "smd_before": pd.DataFrame(),
        "smd_after_ipw": pd.DataFrame(),
        "smd_after_nn": pd.DataFrame(),
        "ipw_outcomes": pd.DataFrame(),
        "nn_outcomes": pd.DataFrame(),
        "propensity_scores": pd.Series(dtype=float),
        # diagnostics
        "strata_counts_before": pd.DataFrame(),
        "strata_counts_after": pd.DataFrame(),
        "overlap_summary_before": {},
        "overlap_summary_after": {},
        "match_retention_by_stratum": pd.DataFrame(),
        "common_support_ps": {},
        "common_support_ps_logit": {},
        "ps_quantiles_before": {},
        "ps_logit_quantiles_before": {},
        "nn_distance_ps_before": {},
        "nn_distance_ps_logit_before": {},
    }

    if len(work) < 20 or work[treatment_col].astype(bool).nunique() < 2:
        return results

    # --- Propensity scores -----------------------------------------------
    work["ps"] = np.nan
    if exact_cols:
        for _, g in work.groupby(exact_cols, dropna=False, observed=True):
            if g[treatment_col].astype(bool).nunique() < 2 or len(g) < 10:
                continue
            ps_g = estimate_propensity_scores(g, treatment_col, ps_covs)
            work.loc[g.index, "ps"] = ps_g
    else:
        work["ps"] = estimate_propensity_scores(work, treatment_col, ps_covs)

    # Drop rows where PS is missing (e.g. too-small strata)
    work = work.dropna(subset=["ps"]).copy()
    results["propensity_scores"] = work["ps"]
    p = np.clip(work["ps"].astype(float).values, 1e-6, 1 - 1e-6)
    work["ps_logit"] = np.log(p / (1 - p))

    # Propensity score diagnostics
    tmask = work[treatment_col].astype(bool).values
    results["common_support_ps"] = common_support_summary(work["ps"].values, tmask)
    results["common_support_ps_logit"] = common_support_summary(work["ps_logit"].values, tmask)
    ps_diag = work["ps"].values
    ps_logit_diag = work["ps_logit"].values
    tr_diag = work[treatment_col].astype(bool).values
    results["ps_quantiles_before"] = {
        "treated": {**quantiles(ps_diag[tr_diag]), "n": int(np.sum(tr_diag))},
        "control": {**quantiles(ps_diag[~tr_diag]), "n": int(np.sum(~tr_diag))},
    }

    results["ps_logit_quantiles_before"] = {
        "treated": {**quantiles(ps_logit_diag[tr_diag]), "n": int(np.sum(tr_diag))},
        "control": {**quantiles(ps_logit_diag[~tr_diag]), "n": int(np.sum(~tr_diag))},
    }
    results["nn_distance_ps_before"] = nearest_distance_summary(work["ps"].values, tmask)
    results["nn_distance_ps_logit_before"] = nearest_distance_summary(work["ps_logit"].values, tmask)

    if exact_cols:
        g_before = work.groupby(exact_cols, dropna=False, observed=True)[treatment_col]
        strata_counts_before = g_before.agg(
            size="size",
            n_treated=lambda s: int(s.astype(bool).sum())
        ).reset_index()
        strata_counts_before["n_control"] = strata_counts_before["size"] - strata_counts_before["n_treated"]
        strata_counts_before["mean"] = strata_counts_before["n_treated"] / strata_counts_before["size"]
        results["strata_counts_before"] = strata_counts_before
        overlap_mask = (strata_counts_before["n_treated"] > 0) & (strata_counts_before["n_control"] > 0)
        n_strata = int(len(strata_counts_before))
        n_overlap = int(overlap_mask.sum())

        results["overlap_summary_before"] = {
            "n_strata": n_strata,
            "n_overlap_strata": n_overlap,
            "pct_overlap_strata": float(100*n_overlap/n_strata) if n_strata else np.nan,
        }

    results["smd_before"] = compute_smd(work, treatment_col, available_covs)

    # --- IPW (full data) ------------------------------------------------
    weights = compute_ipw_weights(np.clip(work["ps"].values, 0.01, 0.99), work[treatment_col].astype(int).values)
    results["smd_after_ipw"] = compute_weighted_smd(work, treatment_col, ps_covs, weights)

    ipw_rows = []
    for outcome in OUTCOME_VARS:
        if outcome not in work.columns:
            continue
        for tval, label in [(True, "treated"), (False, "control")]:
            mask = work[treatment_col].astype(bool) == tval
            vals = work.loc[mask, outcome].values
            w = weights[mask.values]
            valid = np.isfinite(vals) & (w > 0)
            if valid.sum() == 0:
                continue
            wmean = np.average(vals[valid], weights=w[valid])
            ipw_rows.append(
                {
                    "outcome": outcome,
                    "group": label,
                    "weighted_mean": np.average(vals[valid], weights=w[valid]),
                    "n": int(valid.sum()),
                }
            )
    results["ipw_outcomes"] = pd.DataFrame(ipw_rows)

    # --- Nearest-neighbor matching (subsample) ---------------------------
    if len(work) > max_sample:
        treated_all = work[work[treatment_col].astype(bool)]
        budget = max_sample - len(treated_all)
        if budget <= 0:
            work_nn = stratified_sample(treated_all, strata_cols=exact_cols, n_total=max_sample, random_state=42, treatmetn_col=None)
        else:
            controls = work[~work[treatment_col].astype(bool)]
            controls_s = stratified_sample(controls, strata_cols=exact_cols, n_total=budget, random_state=42, treatmetn_col=None, preserve_treatment_share=False)
            work_nn = pd.concat([treated_all, controls_s], axis=0)
    else:
        work_nn = work

    matched_t_list = []
    matched_c_long_list = []
    pair_id_counter = 0
    
    group_iter = work_nn.groupby(exact_cols, dropna=False, observed=True) if exact_cols else [(None, work_nn)]
    k = N_NEIGHBORS
    
    for _, g in group_iter:
        treated_g = g[g[treatment_col].astype(bool)].copy().reset_index(drop=True)
        control_g = g[~g[treatment_col].astype(bool)].copy().reset_index(drop=True)

        if len(treated_g) == 0 or len(control_g) == 0:
            continue

        ps_t = treated_g["ps_logit"].values
        ps_c = control_g["ps_logit"].values

        ps_std = g["ps_logit"].std()
        caliper = caliper_mult * ps_std if ps_std > 0 else 0.1
        
        indices, distances = nearest_neighbor_match(ps_t, ps_c, n_neighbors=k, caliper=caliper)

        valid_any = (indices >= 0).any(axis=1)
        if valid_any.sum() == 0:
            continue

        mt = treated_g.loc[valid_any].copy().reset_index(drop=True)
        nbr_idx = indices[valid_any]
        nbr_dist = distances[valid_any]

        n_mt = len(mt)
        pair_ids = np.arange(pair_id_counter, pair_id_counter + n_mt)
        pair_id_counter += n_mt
        mt["pair_id"] = pair_ids

        rows = []
        for i, pid in enumerate(pair_ids):
            for j in range(k):
                c_idx = int(nbr_idx[i, j])
                if c_idx < 0:
                    continue
                d = float(nbr_dist[i, j])
                rows.append((pid, c_idx, d, j))

        if not rows:
            continue

        c_long = pd.DataFrame(rows, columns=["pair_id", "c_idx", "distance", "nn_rank"])
        c_long["w_raw"] = dist_to_weight(c_long["distance"].values, scheme="inverse")
        wsum = c_long.groupby("pair_id")["w_raw"].transform("sum")
        c_long["match_weight"] = c_long["w_raw"] / wsum

        c_attached = control_g.iloc[c_long["c_idx"].values].copy().reset_index(drop=True)
        c_attached["pair_id"] = c_long["pair_id"].values
        c_attached["distance"] = c_long["distance"].values
        c_attached["nn_rank"] = c_long["nn_rank"].values
        c_attached["match_weight"] = c_long["match_weight"].values
        matched_t_list.append(mt)
        matched_c_long_list.append(c_attached)

    if matched_t_list:
        matched_t = pd.concat(matched_t_list, ignore_index=True)
        matched_c_long = pd.concat(matched_c_long_list, ignore_index=True)
        matched_all_long = pd.concat([matched_t.assign(**{treatment_col: True, "match_weight": 1.0}),
                                      matched_c_long.assign(**{treatment_col: False})], ignore_index=True)

        if exact_cols:
            g_after = matched_all_long.groupby(exact_cols, dropna=False, observed=True)[treatment_col]
            strata_counts_after = g_after.agg(size="size", n_treated=lambda s: int(s.astype(bool).sum())).reset_index()
            strata_counts_after["n_control"] = strata_counts_after["size"] - strata_counts_after["n_treated"]
            strata_counts_after["mean"] = strata_counts_after["n_treated"] / strata_counts_after["size"]
            results["strata_counts_after"] = strata_counts_after

            overlap_mask_a = (strata_counts_after["n_treated"] > 0) & (strata_counts_after["n_control"] > 0)
            n_strata_a = int(len(strata_counts_after))
            n_overlap_a = int(overlap_mask_a.sum())
            
            results["overlap_summary_after"] = {
                "n_strata": n_strata_a,
                "n_overlap_strata": n_overlap_a,
                "pct_overlap_strata": float(100.0*n_overlap_a/n_strata_a) if n_strata_a else np.nan,
            }

            if not results["strata_counts_before"].empty and not strata_counts_after.empty:
                before_sizes = results["strata_counts_before"][exact_cols+["size"]].rename(columns={"size": "size_before"})
                treated_after = matched_t.groupby(exact_cols, dropna=False, observed=True).size().reset_index(name="treated_after")
                retention = before_sizes.merge(treated_after, on=exact_cols, how="left")
                retention["treated_after"] = retention["treated_after"].fillna(0).astype(int)
                retention["matched_retention_pct"] = np.where(retention["size_before"] > 0, 100.0*retention["treated_after"]/retention["size_before"], np.nan)
                results["match_retention_by_stratum"] = retention
            else:
                results["match_retention_by_stratum"] = pd.DataFrame()
        results["smd_after_nn"] = compute_weighted_smd(matched_all_long, treatment_col, ps_covs, matched_all_long["match_weight"].values)

        # Outcome comparison — keep pairs aligned (same positional index)
        nn_rows = []
        for outcome in OUTCOME_VARS:
            if outcome not in matched_t.columns or outcome not in matched_c_long.columns:
                continue
            t_df = matched_t[["pair_id", outcome]].copy()
            c_df = matched_c_long[["pair_id", outcome, "match_weight"]].copy()
            t_df = t_df[np.isfinite(t_df[outcome].values)]
            c_df = c_df[np.isfinite(c_df[outcome].values) & (c_df["match_weight"].values > 0)]

            if t_df.empty or c_df.empty:
                continue

            c_df["w_y"] = c_df[outcome] * c_df["match_weight"]
            ctrl_mean = c_df.groupby("pair_id").agg(control_sum=("w_y", "sum"), wsum=("match_weight", "sum"),
                                                    k_used=(outcome, "size")).reset_index()
            ctrl_mean["control_mean"] = ctrl_mean["control_sum"]/ctrl_mean["wsum"]
            merged = t_df.merge(ctrl_mean[["pair_id", "control_mean", "k_used"]], on="pair_id", how="inner")
            if merged.empty:
                continue

            diffs = merged[outcome].values - merged["control_mean"].values
            diff_mean, diff_lo, diff_hi = bootstrap_mean_ci(diffs, n_boot=1000)

            nn_rows.append({"outcome": outcome, "treated_mean": float(np.nanmean(merged[outcome].values)),
                            "control_mean": float(np.nanmean(merged["control_mean"].values)), "diff": diff_mean,
                            "diff_ci_lower": diff_lo, "diff_ci_upper": diff_hi, "n_pairs": int(len(merged)),
                            "avg_k_used": float(np.nanmean(merged["k_used"].values)),})
        results["nn_outcomes"] = pd.DataFrame(nn_rows)
    else:
        results["smd_after_nn"] = pd.DataFrame()
        results["nn_outcomes"] = pd.DataFrame()
        if exact_cols:
            results["strata_counts_after"] = pd.DataFrame()
            results["overlap_summary_after"] = {
                "n_strata": 0,
                "n_overlap_strata": 0,
                "pct_overlap_strata": np.nan
            }
            results["match_retention_by_stratum"] = pd.DataFrame()

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
            df,
            outcome,
            "CRBPctBucket",
            CONTINUOUS_CONTROLS,
            CATEGORICAL_FE,
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
