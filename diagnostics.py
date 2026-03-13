"""
PSM diagnostic functions for internalization market impact study.

Diagnostics:
  1. Stratum-level ATT decomposition
  2. Leave-one-out stratum analysis
  3. PS model AUROC
  4. Variance ratio
  5. Covariate SMD by stratum
  6. Prognostic scores
  7. PS specification sensitivity
  8. Rosenbaum bounds
  9. E-value
  10. Match quality by stratum
  11. Common support summary
  12. Pre-matching PS diagnostics
  13. Strata counts & overlap
  14. Match retention by stratum
  15. Run all PSM diagnostics
  16. ATT heterogeneity scan
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils import bootstrap_mean_ci

# ===================================================================
# 1. Stratum-level ATT decomposition
# ===================================================================


def _compute_pair_diffs(matched_t, matched_c_long, outcome):
    """Compute per-pair outcome difference (treated - weighted control mean).

    Returns DataFrame with columns: pair_id, diff, plus any columns from
    matched_t that are useful for grouping.
    """
    t_df = matched_t[["pair_id", outcome]].copy()
    c_df = matched_c_long[["pair_id", outcome, "match_weight"]].copy()

    t_df = t_df[np.isfinite(t_df[outcome].values)]
    c_df = c_df[np.isfinite(c_df[outcome].values) & (c_df["match_weight"].values > 0)]

    if t_df.empty or c_df.empty:
        return pd.DataFrame()

    c_df["w_y"] = c_df[outcome] * c_df["match_weight"]
    ctrl_agg = (
        c_df.groupby("pair_id")
        .agg(
            control_sum=("w_y", "sum"),
            wsum=("match_weight", "sum"),
        )
        .reset_index()
    )
    ctrl_agg["control_mean"] = ctrl_agg["control_sum"] / ctrl_agg["wsum"]

    merged = t_df.merge(
        ctrl_agg[["pair_id", "control_mean"]], on="pair_id", how="inner"
    )
    if merged.empty:
        return pd.DataFrame()

    merged["diff"] = merged[outcome] - merged["control_mean"]
    return merged


def stratum_att_decomposition(matched_t, matched_c_long, exact_cols, outcome_vars):
    """Per-stratum ATT with bootstrap CIs and contribution weights.

    Returns DataFrame with columns:
        stratum, outcome, att, ci_lower, ci_upper, n_treated, n_control,
        contribution_weight
    """
    if matched_t.empty or matched_c_long.empty or not exact_cols:
        return pd.DataFrame()

    n_total = len(matched_t)
    rows = []

    for outcome in outcome_vars:
        if outcome not in matched_t.columns or outcome not in matched_c_long.columns:
            continue

        pair_diffs = _compute_pair_diffs(matched_t, matched_c_long, outcome)
        if pair_diffs.empty:
            continue

        # Attach stratum info from matched_t
        stratum_info = matched_t[["pair_id"] + exact_cols].drop_duplicates("pair_id")
        pair_diffs = pair_diffs.merge(stratum_info, on="pair_id", how="left")

        group_col = exact_cols[0] if len(exact_cols) == 1 else exact_cols
        for stratum_key, grp in pair_diffs.groupby(group_col, observed=True):
            stratum_label = (
                stratum_key if isinstance(stratum_key, str) else str(stratum_key)
            )
            diffs = grp["diff"].values
            if len(diffs) == 0:
                continue

            att, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=1000)

            # Count controls for this stratum
            stratum_pairs = set(grp["pair_id"])
            n_ctrl = matched_c_long[
                matched_c_long["pair_id"].isin(stratum_pairs)
            ].shape[0]

            rows.append(
                {
                    "stratum": stratum_label,
                    "outcome": outcome,
                    "att": att,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n_treated": len(diffs),
                    "n_control": n_ctrl,
                    "contribution_weight": len(diffs) / n_total if n_total > 0 else 0.0,
                }
            )

    return pd.DataFrame(rows)


# ===================================================================
# 2. Leave-one-out stratum analysis
# ===================================================================


def leave_one_out_att(matched_t, matched_c_long, exact_cols, outcome_vars):
    """Recompute overall ATT excluding each stratum one at a time.

    Returns DataFrame with columns:
        excluded_stratum, outcome, att_without, att_full, delta
    """
    if matched_t.empty or matched_c_long.empty or not exact_cols:
        return pd.DataFrame()

    rows = []

    for outcome in outcome_vars:
        if outcome not in matched_t.columns or outcome not in matched_c_long.columns:
            continue

        full_diffs = _compute_pair_diffs(matched_t, matched_c_long, outcome)
        if full_diffs.empty:
            continue

        stratum_info = matched_t[["pair_id"] + exact_cols].drop_duplicates("pair_id")
        full_diffs = full_diffs.merge(stratum_info, on="pair_id", how="left")

        att_full = float(np.nanmean(full_diffs["diff"].values))

        group_col = exact_cols[0] if len(exact_cols) == 1 else exact_cols
        strata = full_diffs.groupby(group_col, observed=True)
        for stratum_key, _ in strata:
            stratum_label = (
                stratum_key if isinstance(stratum_key, str) else str(stratum_key)
            )

            # Exclude this stratum
            if len(exact_cols) == 1:
                mask = full_diffs[exact_cols[0]] != stratum_key
            else:
                mask = pd.Series(True, index=full_diffs.index)
                for col, val in zip(exact_cols, stratum_key):
                    mask &= full_diffs[col] != val

            remaining = full_diffs.loc[mask, "diff"].values
            if len(remaining) == 0:
                continue

            att_without = float(np.nanmean(remaining))

            rows.append(
                {
                    "excluded_stratum": stratum_label,
                    "outcome": outcome,
                    "att_without": att_without,
                    "att_full": att_full,
                    "delta": att_without - att_full,
                }
            )

    return pd.DataFrame(rows)


# ===================================================================
# 3. PS model AUROC
# ===================================================================


def ps_model_auroc(ps, treatment):
    """Compute AUROC of propensity scores vs actual treatment assignment.

    Returns dict with auroc, n_treated, n_control.
    """
    ps = np.asarray(ps, dtype=float)
    treatment = np.asarray(treatment, dtype=int)

    valid = np.isfinite(ps)
    ps = ps[valid]
    treatment = treatment[valid]

    n_t = int((treatment == 1).sum())
    n_c = int((treatment == 0).sum())

    if n_t == 0 or n_c == 0 or len(ps) == 0:
        return {"auroc": np.nan, "n_treated": n_t, "n_control": n_c}

    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(treatment, ps))
    return {"auroc": auroc, "n_treated": n_t, "n_control": n_c}


# ===================================================================
# 4. Variance ratio
# ===================================================================


def variance_ratio(df, treatment_col, covariates, weights=None):
    """Compute Var(treated)/Var(control) per covariate.

    Target range: 0.5 to 2.0.  Outside this range indicates distributional
    imbalance not captured by SMD.

    Returns DataFrame with columns: covariate, vr
    """
    t_mask = df[treatment_col].astype(bool).values
    c_mask = ~t_mask

    rows = []
    for cov in covariates:
        if cov not in df.columns:
            continue
        x = df[cov].values.astype(float)

        if weights is not None:
            w = np.asarray(weights, dtype=float)

            def _wvar(vals, wts):
                wts = wts[np.isfinite(vals)]
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2 or wts.sum() == 0:
                    return np.nan
                wm = np.average(vals, weights=wts)
                return np.average((vals - wm) ** 2, weights=wts)

            var_t = _wvar(x[t_mask], w[t_mask])
            var_c = _wvar(x[c_mask], w[c_mask])
        else:
            xt = x[t_mask]
            xc = x[c_mask]
            var_t = np.nanvar(xt, ddof=1) if np.isfinite(xt).sum() > 1 else np.nan
            var_c = np.nanvar(xc, ddof=1) if np.isfinite(xc).sum() > 1 else np.nan

        vr = var_t / var_c if var_c and var_c > 0 else np.nan

        rows.append({"covariate": cov, "vr": vr})

    return pd.DataFrame(rows)


# ===================================================================
# 5. Covariate SMD by stratum
# ===================================================================


def covariate_smd_by_stratum(df, treatment_col, covariates, exact_cols):
    """Compute SMD within each stratum separately.

    Returns DataFrame with columns: stratum, covariate, smd
    """
    if df.empty or not exact_cols:
        return pd.DataFrame()

    group_col = exact_cols[0] if len(exact_cols) == 1 else exact_cols
    rows = []
    for stratum_key, grp in df.groupby(group_col, observed=True):
        stratum_label = (
            stratum_key if isinstance(stratum_key, str) else str(stratum_key)
        )
        treated = grp[grp[treatment_col].astype(bool)]
        control = grp[~grp[treatment_col].astype(bool)]

        if len(treated) < 2 or len(control) < 2:
            continue

        for cov in covariates:
            if cov not in grp.columns:
                continue
            mt = treated[cov].mean()
            mc = control[cov].mean()
            vt = treated[cov].var()
            vc = control[cov].var()
            pooled_sd = np.sqrt((vt + vc) / 2)
            smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append({"stratum": stratum_label, "covariate": cov, "smd": smd})

    return pd.DataFrame(rows)


# ===================================================================
# 6. Prognostic scores
# ===================================================================


def prognostic_scores(df, treatment_col, covariates, outcome):
    """Fit OLS of outcome ~ covariates on the control group only.

    Identifies which covariates most strongly predict the outcome,
    guiding confounder selection.

    Returns DataFrame with columns: covariate, coef, se, pvalue, r_squared
    """
    controls = df[~df[treatment_col].astype(bool)].copy()
    available = [c for c in covariates if c in controls.columns]

    if len(controls) < len(available) + 2 or outcome not in controls.columns:
        return pd.DataFrame()

    work = controls[available + [outcome]].dropna()
    if len(work) < len(available) + 2:
        return pd.DataFrame()

    import statsmodels.api as sm

    X = sm.add_constant(work[available].astype(float))
    y = work[outcome].astype(float)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return pd.DataFrame()

    rows = []
    for cov in available:
        if cov not in model.params.index:
            continue
        rows.append(
            {
                "covariate": cov,
                "coef": float(model.params[cov]),
                "se": float(model.bse[cov]),
                "pvalue": float(model.pvalues[cov]),
                "r_squared": float(model.rsquared),
            }
        )

    return pd.DataFrame(rows)


# ===================================================================
# 10. Match quality by stratum
# ===================================================================


def match_quality_by_stratum(matched_t, matched_c_long, exact_cols):
    """Per-stratum match quality: distance statistics and effective k.

    Returns DataFrame with columns:
        stratum, mean_dist, median_dist, max_dist, mean_k, min_k
    """
    if matched_t.empty or matched_c_long.empty or not exact_cols:
        return pd.DataFrame()

    if "distance" not in matched_c_long.columns:
        return pd.DataFrame()

    # Count neighbors per pair
    k_per_pair = matched_c_long.groupby("pair_id").size().reset_index(name="k_used")

    # Attach stratum from matched_t
    stratum_info = matched_t[["pair_id"] + exact_cols].drop_duplicates("pair_id")
    k_per_pair = k_per_pair.merge(stratum_info, on="pair_id", how="left")

    # Per-pair mean distance
    dist_per_pair = (
        matched_c_long.groupby("pair_id")["distance"]
        .mean()
        .reset_index(name="mean_pair_dist")
    )
    k_per_pair = k_per_pair.merge(dist_per_pair, on="pair_id", how="left")

    group_col = exact_cols[0] if len(exact_cols) == 1 else exact_cols
    rows = []
    for stratum_key, grp in k_per_pair.groupby(group_col, observed=True):
        stratum_label = (
            stratum_key if isinstance(stratum_key, str) else str(stratum_key)
        )

        # Distance stats from the raw control-level distances
        stratum_pairs = set(grp["pair_id"])
        stratum_dists = matched_c_long.loc[
            matched_c_long["pair_id"].isin(stratum_pairs), "distance"
        ].values
        stratum_dists = stratum_dists[np.isfinite(stratum_dists)]

        rows.append(
            {
                "stratum": stratum_label,
                "mean_dist": float(np.mean(stratum_dists))
                if len(stratum_dists) > 0
                else np.nan,
                "median_dist": float(np.median(stratum_dists))
                if len(stratum_dists) > 0
                else np.nan,
                "max_dist": float(np.max(stratum_dists))
                if len(stratum_dists) > 0
                else np.nan,
                "mean_k": float(grp["k_used"].mean()),
                "min_k": int(grp["k_used"].min()),
            }
        )

    return pd.DataFrame(rows)


# ===================================================================
# 8. Rosenbaum bounds
# ===================================================================


def rosenbaum_bounds(diffs, gamma_range=None):
    """Sensitivity analysis for matched pair differences.

    For each Gamma (odds of differential treatment assignment due to
    unmeasured confounding), computes the upper-bound p-value of a
    Wilcoxon signed-rank test.

    Parameters
    ----------
    diffs : array-like - matched pair outcome differences
    gamma_range : list of floats - Gamma values to test (default 1.0 to 3.0)

    Returns DataFrame with columns: gamma, p_upper
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return pd.DataFrame()

    if gamma_range is None:
        gamma_range = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    from scipy.stats import wilcoxon

    # Absolute ranks
    abs_diffs = np.abs(diffs)
    # Remove zeros for ranking
    nonzero = abs_diffs > 0
    if nonzero.sum() < 2:
        return pd.DataFrame()

    diffs_nz = diffs[nonzero]
    abs_nz = abs_diffs[nonzero]
    n = len(diffs_nz)

    # Rank the absolute differences
    ranks = np.argsort(np.argsort(abs_nz)) + 1.0  # simple ranking

    rows = []
    for gamma in gamma_range:
        if gamma == 1.0:
            # Standard Wilcoxon signed-rank test
            try:
                _, p = wilcoxon(diffs_nz, alternative="two-sided")
            except ValueError:
                p = np.nan
            rows.append({"gamma": gamma, "p_upper": float(p)})
        else:
            # Under Gamma, the probability that a positive diff is
            # actually positive ranges from 1/(1+Gamma) to Gamma/(1+Gamma).
            # T+ = sum of ranks where diff > 0
            positive = diffs_nz > 0
            T_plus = float(ranks[positive].sum())
            T_total = float(ranks.sum())

            # Under Gamma, expected T+ under null:
            p_gamma = gamma / (1 + gamma)
            E_T = T_total * p_gamma
            V_T = (ranks**2).sum() * gamma / (1 + gamma) ** 2
            sd_T = np.sqrt(V_T) if V_T > 0 else 1e-10

            # Normal approximation
            from scipy.stats import norm

            z = (T_plus - E_T) / sd_T
            p_upper = float(2 * norm.sf(abs(z)))  # two-sided
            rows.append({"gamma": gamma, "p_upper": p_upper})

    return pd.DataFrame(rows)


# ===================================================================
# 9. E-value
# ===================================================================


def e_value(att, se):
    """Compute E-value: minimum confounding strength to explain away the result.

    Uses the VanderWeele & Ding (2017) approximation for continuous outcomes.

    Parameters
    ----------
    att : float - average treatment effect
    se : float - standard error of the ATT

    Returns dict with e_value_point, e_value_ci
    """

    def _e_from_rr(rr):
        rr = abs(rr)
        if rr <= 1.0:
            return 1.0
        return rr + np.sqrt(rr * (rr - 1))

    # Approximate RR using exp(0.91 * |att/se|)
    z_point = abs(att) / se if se > 0 else 0.0
    rr_point = np.exp(0.91 * z_point) if z_point > 0 else 1.0

    # CI bound: use the CI-closest-to-null
    z_ci = max(0, z_point - 1.96)
    rr_ci = np.exp(0.91 * z_ci) if z_ci > 0 else 1.0

    return {
        "e_value_point": _e_from_rr(rr_point),
        "e_value_ci": _e_from_rr(rr_ci),
    }


# ===================================================================
# 7. PS specification sensitivity
# ===================================================================


def _run_quick_psm_att(
    df, treatment_col, ps_covs, exact_cols, outcome, caliper_mult, n_neighbors
):
    """Run a simplified PSM pipeline for a single outcome.

    Returns (att, ci_lo, ci_hi, auroc, mean_smd_after) or None.
    """
    from utils import (
        compute_weighted_smd,
        estimate_propensity_scores,
        nearest_neighbor_match,
    )

    needed = ps_covs + exact_cols + [treatment_col]
    work = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    if len(work) < 20 or work[treatment_col].astype(bool).nunique() < 2:
        return None

    # PS estimation
    work["ps"] = np.nan
    if exact_cols:
        for _, g in work.groupby(exact_cols, dropna=False, observed=True):
            if g[treatment_col].astype(bool).nunique() < 2 or len(g) < 10:
                continue
            ps_g = estimate_propensity_scores(g, treatment_col, ps_covs)
            work.loc[g.index, "ps"] = ps_g
    else:
        work["ps"] = estimate_propensity_scores(work, treatment_col, ps_covs)

    work = work.dropna(subset=["ps"]).copy()
    if len(work) < 20 or work[treatment_col].astype(bool).nunique() < 2:
        return None

    p = np.clip(work["ps"].astype(float).values, 1e-6, 1 - 1e-6)
    work["ps_logit"] = np.log(p / (1 - p))

    # AUROC
    auroc_res = ps_model_auroc(
        work["ps"].values, work[treatment_col].astype(int).values
    )
    auroc = auroc_res["auroc"]

    # NN matching
    EPS = 1e-8
    matched_t_list = []
    matched_c_long_list = []
    pair_id_counter = 0
    group_iter = (
        work.groupby(exact_cols, dropna=False, observed=True)
        if exact_cols
        else [(None, work)]
    )

    for _, g in group_iter:
        treated_g = g[g[treatment_col].astype(bool)].copy().reset_index(drop=True)
        control_g = g[~g[treatment_col].astype(bool)].copy().reset_index(drop=True)
        if len(treated_g) == 0 or len(control_g) == 0:
            continue

        ps_std = g["ps_logit"].std()
        caliper = caliper_mult * ps_std if ps_std > 0 else 0.1
        indices, distances = nearest_neighbor_match(
            treated_g["ps_logit"].values,
            control_g["ps_logit"].values,
            n_neighbors=n_neighbors,
            caliper=caliper,
        )
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

        c_rows = []
        for i, pid in enumerate(pair_ids):
            for j in range(nbr_idx.shape[1]):
                c_idx = int(nbr_idx[i, j])
                if c_idx < 0:
                    continue
                d = float(nbr_dist[i, j])
                c_rows.append((pid, c_idx, d))

        if not c_rows:
            continue

        c_long = pd.DataFrame(c_rows, columns=["pair_id", "c_idx", "distance"])
        c_long["w_raw"] = 1.0 / (c_long["distance"].values + EPS)
        wsum = c_long.groupby("pair_id")["w_raw"].transform("sum")
        c_long["match_weight"] = c_long["w_raw"] / wsum

        c_attached = (
            control_g.iloc[c_long["c_idx"].values].copy().reset_index(drop=True)
        )
        c_attached["pair_id"] = c_long["pair_id"].values
        c_attached["match_weight"] = c_long["match_weight"].values
        matched_t_list.append(mt)
        matched_c_long_list.append(c_attached)

    if not matched_t_list:
        return None

    matched_t = pd.concat(matched_t_list, ignore_index=True)
    matched_c_long = pd.concat(matched_c_long_list, ignore_index=True)

    # Compute ATT
    pair_diffs = _compute_pair_diffs(matched_t, matched_c_long, outcome)
    if pair_diffs.empty:
        return None

    diffs = pair_diffs["diff"].values
    att, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=500)

    # Mean SMD after matching
    all_covs = [c for c in ps_covs if c in matched_t.columns]
    matched_all = pd.concat(
        [
            matched_t.assign(**{treatment_col: True, "match_weight": 1.0}),
            matched_c_long.assign(**{treatment_col: False}),
        ],
        ignore_index=True,
    )
    smd_after = compute_weighted_smd(
        matched_all, treatment_col, all_covs, matched_all["match_weight"].values
    )
    mean_smd = float(smd_after["smd"].abs().mean()) if not smd_after.empty else np.nan

    return att, ci_lo, ci_hi, auroc, mean_smd


def ps_specification_sensitivity(
    df,
    treatment_col,
    base_covariates,
    exact_cols,
    outcome,
    caliper_mult=0.2,
    n_neighbors=10,
):
    """Test ATT sensitivity to different PS model specifications.

    Specifications tested:
      1. Base model (all covariates)
      2-N. Leave-one-out (drop each covariate)
      N+1. Base + quadratic terms
      N+2. Base + pairwise interactions of top-2 prognostic covariates

    Returns DataFrame with columns:
        spec_name, covariates_used, att, ci_lower, ci_upper, auroc, mean_smd_after
    """
    available = [c for c in base_covariates if c in df.columns]
    if len(available) < 2 or len(df) < 40:
        return pd.DataFrame()

    specs = []

    # 1. Base
    specs.append(("base", available))

    # 2-N. Leave-one-out
    for cov in available:
        remaining = [c for c in available if c != cov]
        if len(remaining) < 1:
            continue
        specs.append((f"drop_{cov}", remaining))

    # N+2. Interactions of top-2 prognostic covariates
    prog = prognostic_scores(df, treatment_col, available, outcome)
    if not prog.empty and len(prog) >= 2:
        top2 = prog.reindex(prog["coef"].abs().sort_values(ascending=False).index)[:2]
        c1, c2 = top2["covariate"].values
        inter_col = f"{c1}_x_{c2}"
        if inter_col not in df.columns:
            df[inter_col] = df[c1].astype(float) * df[c2].astype(float)
        inter_covs = available + [inter_col]
        specs.append(("interactions", inter_covs))

    rows = []
    for spec_name, covs in specs:
        print(f"        {spec_name}")
        result = _run_quick_psm_att(
            df,
            treatment_col,
            covs,
            exact_cols,
            outcome,
            caliper_mult,
            n_neighbors,
        )
        if result is None:
            continue
        att_val, ci_lo, ci_hi, auroc, mean_smd = result
        rows.append(
            {
                "spec_name": spec_name,
                "covariates_used": ", ".join(covs),
                "att": att_val,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "auroc": auroc,
                "mean_smd_after": mean_smd,
            }
        )

    return pd.DataFrame(rows)


# ===================================================================
# 11. Common support & PS distribution helpers
# ===================================================================


def quantiles(x):
    """Compute standard quantile summary for an array."""
    q = np.nanpercentile(x, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    keys = ["min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return {k: float(v) for k, v in zip(keys, q)}


def common_support_summary(ps, treat_mask):
    """Summarize common support: control PS range and treated orders outside it."""
    ps_t = ps[treat_mask]
    ps_c = ps[~treat_mask]
    lo_c, hi_c = np.min(ps_c), np.max(ps_c)
    outside = (ps_t < lo_c) | (ps_t > hi_c)
    return {
        "control_min": float(lo_c),
        "control_max": float(hi_c),
        "treated_outside_range_n": int(outside.sum()),
        "treated_outside_range_pct": float(100 * outside.mean())
        if len(ps_t)
        else np.nan,
    }


def nearest_distance_summary(score, treat_mask):
    """Quantile summary of nearest-neighbor distances (treated -> control)."""
    s_t = score[treat_mask].reshape(-1, 1)
    s_c = score[~treat_mask].reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1).fit(s_c)
    dist, _ = nn.kneighbors(s_t)
    dist = dist.ravel()
    return {**quantiles(dist), "mean": float(np.mean(dist)), "n": int(len(dist))}


# ===================================================================
# 12. Pre-matching PS diagnostics
# ===================================================================


def pre_matching_ps_diagnostics(work, treatment_col):
    """Compute pre-matching propensity score diagnostics.

    Parameters
    ----------
    work : DataFrame -- must have 'ps' and 'ps_logit' columns
    treatment_col : str

    Returns
    -------
    dict with keys: common_support_ps, common_support_ps_logit,
        ps_quantiles_before, ps_logit_quantiles_before,
        nn_distance_ps_before, nn_distance_ps_logit_before
    """
    tmask = work[treatment_col].astype(bool).values
    ps_vals = work["ps"].values
    ps_logit_vals = work["ps_logit"].values
    n_t = int(np.sum(tmask))
    n_c = int(np.sum(~tmask))

    return {
        "common_support_ps": common_support_summary(ps_vals, tmask),
        "common_support_ps_logit": common_support_summary(ps_logit_vals, tmask),
        "ps_quantiles_before": {
            "treated": {**quantiles(ps_vals[tmask]), "n": n_t},
            "control": {**quantiles(ps_vals[~tmask]), "n": n_c},
        },
        "ps_logit_quantiles_before": {
            "treated": {**quantiles(ps_logit_vals[tmask]), "n": n_t},
            "control": {**quantiles(ps_logit_vals[~tmask]), "n": n_c},
        },
        "nn_distance_ps_before": nearest_distance_summary(ps_vals, tmask),
        "nn_distance_ps_logit_before": nearest_distance_summary(ps_logit_vals, tmask),
    }


# ===================================================================
# 13. Strata counts & overlap
# ===================================================================


def compute_strata_counts(work, exact_cols, treatment_col):
    """Compute strata counts and overlap summary.

    Returns
    -------
    (strata_counts_df, overlap_summary_dict)
    """
    g = work.groupby(exact_cols, dropna=False, observed=True)[treatment_col]
    sc = g.agg(size="size", n_treated=lambda s: int(s.astype(bool).sum())).reset_index()
    sc["n_control"] = sc["size"] - sc["n_treated"]
    sc["mean"] = sc["n_treated"] / sc["size"]

    overlap_mask = (sc["n_treated"] > 0) & (sc["n_control"] > 0)
    n_strata = int(len(sc))
    n_overlap = int(overlap_mask.sum())
    overlap_summary = {
        "n_strata": n_strata,
        "n_overlap_strata": n_overlap,
        "pct_overlap_strata": float(100 * n_overlap / n_strata) if n_strata else np.nan,
    }

    return sc, overlap_summary


# ===================================================================
# 14. Match retention by stratum
# ===================================================================


def compute_match_retention(strata_counts_before, matched_t, exact_cols):
    """Compute match retention by stratum.

    Returns DataFrame with before/after treated counts and retention percentage.
    """
    if strata_counts_before.empty:
        return pd.DataFrame()

    before_sizes = strata_counts_before[exact_cols + ["size", "n_treated"]].rename(
        columns={"size": "size_before", "n_treated": "n_treated_before"}
    )
    treated_after = (
        matched_t.groupby(exact_cols, dropna=False, observed=True)
        .size()
        .reset_index(name="treated_after")
    )
    retention = before_sizes.merge(treated_after, on=exact_cols, how="left")
    retention["treated_after"] = retention["treated_after"].fillna(0).astype(int)
    retention["matched_retention_pct"] = np.where(
        retention["n_treated_before"] > 0,
        100.0 * retention["treated_after"] / retention["n_treated_before"],
        np.nan,
    )
    return retention


# ===================================================================
# 15. Dose-response strata diagnostics
# ===================================================================


def dose_strata_diagnostics(work, exact_cols, treatment_col):
    """Compute strata diagnostics for a dose-response bucket.

    Returns dict with 'strata_counts' and 'overlap_summary'.
    """
    g_ct = work.groupby(exact_cols, dropna=False, observed=True)[treatment_col]
    sc = g_ct.agg(size="size", n_treated=lambda s: int(s.sum())).reset_index()
    sc["n_control"] = sc["size"] - sc["n_treated"]
    overlap = (sc["n_treated"] > 0) & (sc["n_control"] > 0)
    return {
        "strata_counts": sc,
        "overlap_summary": {
            "n_strata": int(len(sc)),
            "n_overlap": int(overlap.sum()),
            "pct_overlap": float(100 * overlap.sum() / len(sc)) if len(sc) else np.nan,
        },
    }


# ===================================================================
# 16. Run all PSM diagnostics
# ===================================================================


def run_psm_diagnostics(df, psm_results, treatment_col="hasCRB"):
    """Run all PSM diagnostics given a completed PSM analysis.

    Parameters
    ----------
    df : DataFrame - the full dataset (for specification sensitivity)
    psm_results : dict - output of run_psm_analysis()
    treatment_col : str

    Returns
    -------
    dict with diagnostic results
    """
    from config import EXACT_MATCH_COLS, N_NEIGHBORS, OUTCOME_VARS, PSM_COVARIATES

    matched_t = psm_results.get("matched_t", pd.DataFrame())
    matched_c_long = psm_results.get("matched_c_long", pd.DataFrame())
    exact_cols = [c for c in EXACT_MATCH_COLS if c in df.columns]
    available_covs = [c for c in PSM_COVARIATES if c in df.columns]

    diag = {}

    # 1. Stratum ATT decomposition
    print("    Stratum ATT decomposition ...")
    diag["stratum_att"] = stratum_att_decomposition(
        matched_t,
        matched_c_long,
        exact_cols,
        OUTCOME_VARS,
    )

    # 2. Leave-one-out
    print("    Leave-one-out ...")
    diag["leave_one_out"] = leave_one_out_att(
        matched_t,
        matched_c_long,
        exact_cols,
        OUTCOME_VARS,
    )

    # 3. AUROC
    print("    AUROC ...")
    ps = psm_results.get("propensity_scores", pd.Series(dtype=float))
    if not ps.empty and treatment_col in df.columns:
        treat_vals = (
            df.loc[ps.index, treatment_col].astype(int).values
            if ps.index.isin(df.index).all()
            else np.array([])
        )
        if len(treat_vals) == len(ps):
            diag["auroc"] = ps_model_auroc(ps.values, treat_vals)
        else:
            diag["auroc"] = {"auroc": np.nan, "n_treated": 0, "n_control": 0}
    else:
        diag["auroc"] = {"auroc": np.nan, "n_treated": 0, "n_control": 0}

    # 4. Variance ratio (before and after)
    print("    Variance ratio ...")
    needed = available_covs + [treatment_col]
    work = df.dropna(subset=[c for c in needed if c in df.columns])
    diag["variance_ratio_before"] = (
        variance_ratio(work, treatment_col, available_covs)
        if not work.empty
        else pd.DataFrame()
    )

    if not matched_t.empty and not matched_c_long.empty:
        matched_all = pd.concat(
            [
                matched_t.assign(**{treatment_col: True, "match_weight": 1.0}),
                matched_c_long.assign(**{treatment_col: False}),
            ],
            ignore_index=True,
        )
        diag["variance_ratio_after"] = variance_ratio(
            matched_all,
            treatment_col,
            available_covs,
            weights=matched_all["match_weight"].values,
        )
    else:
        diag["variance_ratio_after"] = pd.DataFrame()

    # 5. SMD by stratum
    print("    SMD by stratum ...")
    diag["smd_by_stratum"] = (
        covariate_smd_by_stratum(
            work,
            treatment_col,
            available_covs,
            exact_cols,
        )
        if not work.empty
        else pd.DataFrame()
    )

    # 6. Prognostic scores
    print("    Prognostic scores ...")
    diag["prognostic"] = (
        prognostic_scores(
            work,
            treatment_col,
            available_covs,
            "tempImpactBps",
        )
        if not work.empty and "tempImpactBps" in work.columns
        else pd.DataFrame()
    )

    # 10. Match quality by stratum
    print("    Match Quality by stratum ...")
    diag["match_quality"] = match_quality_by_stratum(
        matched_t,
        matched_c_long,
        exact_cols,
    )

    # 8. Rosenbaum bounds (on tempImpactBps pair diffs)
    print("    Rosenbaum bounds ...")
    if (
        not matched_t.empty
        and not matched_c_long.empty
        and "tempImpactBps" in matched_t.columns
    ):
        pair_diffs = _compute_pair_diffs(matched_t, matched_c_long, "tempImpactBps")
        if not pair_diffs.empty:
            diag["rosenbaum_bounds"] = rosenbaum_bounds(pair_diffs["diff"].values)
        else:
            diag["rosenbaum_bounds"] = pd.DataFrame()
    else:
        diag["rosenbaum_bounds"] = pd.DataFrame()

    # 9. E-values (for each outcome)
    print("    E-values ...")
    nn_outcomes = psm_results.get("nn_outcomes", pd.DataFrame())
    e_vals = {}
    if not nn_outcomes.empty:
        for _, row in nn_outcomes.iterrows():
            att_val = row["diff"]
            se_approx = (row["diff_ci_upper"] - row["diff_ci_lower"]) / (2 * 1.96)
            if se_approx > 0:
                e_vals[row["outcome"]] = e_value(att_val, se_approx)
    diag["e_values"] = e_vals

    # 7. PS specification sensitivity
    # print("    PS Specification sensitivity ...")
    # if not df.empty and treatment_col in df.columns and "tempImpactBps" in df.columns:
    #     diag["spec_sensitivity"] = ps_specification_sensitivity(
    #         df,
    #         treatment_col,
    #         available_covs,
    #         exact_cols,
    #         "tempImpactBps",
    #         caliper_mult=0.2,
    #         n_neighbors=N_NEIGHBORS,
    #     )
    # else:
    #     diag["spec_sensitivity"] = pd.DataFrame()

    return diag


# ===================================================================
# 17. ATT heterogeneity scan
# ===================================================================


def att_heterogeneity_scan(
    matched_t,
    matched_c_long,
    outcome,
    skip_cols=None,
    max_categorical_unique=50,
    n_quantile_bins=4,
    min_group_size=30,
    corr_threshold=0.05,
):
    """Scan all columns in matched_t for ATT heterogeneity on a given outcome.

    For each column not in skip_cols:
      - Categorical (<=max_categorical_unique unique values): group pair diffs
        by value, report per-group ATT and count.
      - Continuous (>max_categorical_unique unique values): report correlation
        with pair diffs, then bin into quantiles and report per-bin ATT.

    Parameters
    ----------
    matched_t : DataFrame - matched treated observations (must have pair_id)
    matched_c_long : DataFrame - matched controls (must have pair_id, match_weight)
    outcome : str - outcome variable to compute diffs on
    skip_cols : list[str] | None - columns to exclude from the scan
    max_categorical_unique : int - threshold to distinguish categorical vs continuous
    n_quantile_bins : int - number of quantile bins for continuous columns
    min_group_size : int - minimum pairs in a group to report
    corr_threshold : float - minimum |correlation| to include continuous columns

    Returns
    -------
    dict with keys:
      - "categorical": DataFrame (column, value, att, count, std)
      - "continuous": DataFrame (column, correlation, bin, att, count, std)
    """
    if matched_t.empty or matched_c_long.empty:
        return {"categorical": pd.DataFrame(), "continuous": pd.DataFrame()}

    pair_diffs = _compute_pair_diffs(matched_t, matched_c_long, outcome)
    if pair_diffs.empty:
        return {"categorical": pd.DataFrame(), "continuous": pd.DataFrame()}

    # Attach all matched_t columns for slicing
    extra_cols = [c for c in matched_t.columns if c not in pair_diffs.columns]
    pair_diffs = pair_diffs.merge(
        matched_t[["pair_id"] + extra_cols], on="pair_id", how="left"
    )

    default_skip = {
        "pair_id",
        "diff",
        "control_mean",
        outcome,
        "ps",
        "ps_logit",
        "match_weight",
    }
    if skip_cols:
        default_skip |= set(skip_cols)

    pair_diffs["EffectiveStartTime"] = pd.to_datetime(
        pair_diffs["EffectiveStartTime"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    pair_diffs["EffectiveEndTime"] = pd.to_datetime(
        pair_diffs["EffectiveEndTime"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    pair_diffs["EffectiveStartTime"] = (
        pair_diffs["EffectiveStartTime"].dt.hour * 60
    ) + (pair_diffs["EffectiveStartTime"].dt.minute)
    pair_diffs["EffectiveEndTime"] = (pair_diffs["EffectiveEndTime"].dt.hour * 60) + (
        pair_diffs["EffectiveEndTime"].dt.minute
    )

    candidates = [c for c in pair_diffs.columns if c not in default_skip]
    diffs = pair_diffs["diff"].values

    cat_rows = []
    cont_rows = []

    for col in candidates:
        if col not in [
            "Notional",
            "EffectiveStartTime",
            "EffectiveEndTime",
            "qtyOverADV",
            "PcpRate",
            "RiskAversion",
            "ATSPINQty",
            "LitQty",
            "ELPQty",
            "adv",
            "tickrule",
            "dailyvol",
            "duration_mins",
        ]:
            continue
        print(col)
        vals = pair_diffs[col]
        n_unique = vals.nunique()
        if n_unique < 2:
            continue

        if n_unique <= max_categorical_unique:
            # Categorical scan
            for value, grp in pair_diffs.groupby(col, observed=True):
                if len(grp) < min_group_size:
                    continue
                g_diffs = grp["diff"].values
                cat_rows.append(
                    {
                        "column": col,
                        "value": str(value),
                        "att": float(np.nanmean(g_diffs)),
                        "count": len(g_diffs),
                        "std": float(np.nanstd(g_diffs, ddof=1))
                        if len(g_diffs) > 1
                        else np.nan,
                    }
                )
        else:
            # Continuous scan
            col_vals = vals.values.astype(float)
            valid = np.isfinite(col_vals) & np.isfinite(diffs)
            if valid.sum() < min_group_size:
                continue

            corr = float(np.corrcoef(col_vals[valid], diffs[valid])[0, 1])

            # Bin into quantiles
            pair_diffs["_bin"] = pd.qcut(vals, q=n_quantile_bins, duplicates="drop")

            for bin_label, grp in pair_diffs.groupby("_bin", observed=True):
                if len(grp) < min_group_size:
                    continue
                g_diffs = grp["diff"].values
                cont_rows.append(
                    {
                        "column": col,
                        "correlation": corr,
                        "bin": str(bin_label),
                        "att": float(np.nanmean(g_diffs)),
                        "count": len(g_diffs),
                        "std": float(np.nanstd(g_diffs, ddof=1))
                        if len(g_diffs) > 1
                        else np.nan,
                    }
                )

            pair_diffs.drop(columns=["_bin"], inplace=True)

    return {
        "categorical": pd.DataFrame(cat_rows),
        "continuous": pd.DataFrame(cont_rows),
    }


if __name__ == "__main__":
    matched_t = pd.read_csv("output/results/psm_matched_t.csv")
    matched_c_long = pd.read_csv("output/results/psm_matched_c_long.csv")
    matched_t = matched_t.loc[matched_t.Strategy == "VWAP"]
    matched_c_long = matched_c_long.loc[matched_c_long.Strategy == "VWAP"]
    results = att_heterogeneity_scan(matched_t, matched_c_long, outcome="tempImpactBps")
    results["categorical"].to_csv("output/results/categorical.csv")
    results["continuous"].to_csv("output/results/continuous.csv")
