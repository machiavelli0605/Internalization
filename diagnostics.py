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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    ctrl_agg = c_df.groupby("pair_id").agg(
        control_sum=("w_y", "sum"),
        wsum=("match_weight", "sum"),
    ).reset_index()
    ctrl_agg["control_mean"] = ctrl_agg["control_sum"] / ctrl_agg["wsum"]

    merged = t_df.merge(ctrl_agg[["pair_id", "control_mean"]], on="pair_id", how="inner")
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
            stratum_label = stratum_key if isinstance(stratum_key, str) else str(stratum_key)
            diffs = grp["diff"].values
            if len(diffs) == 0:
                continue

            att, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=1000)

            # Count controls for this stratum
            stratum_pairs = set(grp["pair_id"])
            n_ctrl = matched_c_long[matched_c_long["pair_id"].isin(stratum_pairs)].shape[0]

            rows.append({
                "stratum": stratum_label,
                "outcome": outcome,
                "att": att,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_treated": len(diffs),
                "n_control": n_ctrl,
                "contribution_weight": len(diffs) / n_total if n_total > 0 else 0.0,
            })

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
            stratum_label = stratum_key if isinstance(stratum_key, str) else str(stratum_key)

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

            rows.append({
                "excluded_stratum": stratum_label,
                "outcome": outcome,
                "att_without": att_without,
                "att_full": att_full,
                "delta": att_without - att_full,
            })

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
        stratum_label = stratum_key if isinstance(stratum_key, str) else str(stratum_key)
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
        rows.append({
            "covariate": cov,
            "coef": float(model.params[cov]),
            "se": float(model.bse[cov]),
            "pvalue": float(model.pvalues[cov]),
            "r_squared": float(model.rsquared),
        })

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
    dist_per_pair = matched_c_long.groupby("pair_id")["distance"].mean().reset_index(name="mean_pair_dist")
    k_per_pair = k_per_pair.merge(dist_per_pair, on="pair_id", how="left")

    group_col = exact_cols[0] if len(exact_cols) == 1 else exact_cols
    rows = []
    for stratum_key, grp in k_per_pair.groupby(group_col, observed=True):
        stratum_label = stratum_key if isinstance(stratum_key, str) else str(stratum_key)

        # Distance stats from the raw control-level distances
        stratum_pairs = set(grp["pair_id"])
        stratum_dists = matched_c_long.loc[
            matched_c_long["pair_id"].isin(stratum_pairs), "distance"
        ].values
        stratum_dists = stratum_dists[np.isfinite(stratum_dists)]

        rows.append({
            "stratum": stratum_label,
            "mean_dist": float(np.mean(stratum_dists)) if len(stratum_dists) > 0 else np.nan,
            "median_dist": float(np.median(stratum_dists)) if len(stratum_dists) > 0 else np.nan,
            "max_dist": float(np.max(stratum_dists)) if len(stratum_dists) > 0 else np.nan,
            "mean_k": float(grp["k_used"].mean()),
            "min_k": int(grp["k_used"].min()),
        })

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
            V_T = (ranks ** 2).sum() * gamma / (1 + gamma) ** 2
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


def _run_quick_psm_att(df, treatment_col, ps_covs, exact_cols, outcome,
                       caliper_mult, n_neighbors):
    """Run a simplified PSM pipeline for a single outcome.

    Returns (att, ci_lo, ci_hi, auroc, mean_smd_after) or None.
    """
    from utils import estimate_propensity_scores, nearest_neighbor_match, compute_weighted_smd

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
    auroc_res = ps_model_auroc(work["ps"].values, work[treatment_col].astype(int).values)
    auroc = auroc_res["auroc"]

    # NN matching
    EPS = 1e-8
    matched_t_list = []
    matched_c_long_list = []
    pair_id_counter = 0
    group_iter = (work.groupby(exact_cols, dropna=False, observed=True)
                  if exact_cols else [(None, work)])

    for _, g in group_iter:
        treated_g = g[g[treatment_col].astype(bool)].copy().reset_index(drop=True)
        control_g = g[~g[treatment_col].astype(bool)].copy().reset_index(drop=True)
        if len(treated_g) == 0 or len(control_g) == 0:
            continue

        ps_std = g["ps_logit"].std()
        caliper = caliper_mult * ps_std if ps_std > 0 else 0.1
        indices, distances = nearest_neighbor_match(
            treated_g["ps_logit"].values, control_g["ps_logit"].values,
            n_neighbors=n_neighbors, caliper=caliper,
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

        c_attached = control_g.iloc[c_long["c_idx"].values].copy().reset_index(drop=True)
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
    matched_all = pd.concat([
        matched_t.assign(**{treatment_col: True, "match_weight": 1.0}),
        matched_c_long.assign(**{treatment_col: False}),
    ], ignore_index=True)
    smd_after = compute_weighted_smd(matched_all, treatment_col, all_covs,
                                     matched_all["match_weight"].values)
    mean_smd = float(smd_after["smd"].abs().mean()) if not smd_after.empty else np.nan

    return att, ci_lo, ci_hi, auroc, mean_smd


def ps_specification_sensitivity(df, treatment_col, base_covariates, exact_cols,
                                  outcome, caliper_mult=0.2, n_neighbors=10):
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

    # N+1. Quadratic terms
    df = df.copy()
    quad_covs = available.copy()
    for cov in available:
        sq_col = f"{cov}_sq"
        if sq_col not in df.columns:
            df[sq_col] = df[cov].astype(float) ** 2
        quad_covs.append(sq_col)
    specs.append(("quadratic", quad_covs))

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
        result = _run_quick_psm_att(
            df, treatment_col, covs, exact_cols, outcome,
            caliper_mult, n_neighbors,
        )
        if result is None:
            continue
        att_val, ci_lo, ci_hi, auroc, mean_smd = result
        rows.append({
            "spec_name": spec_name,
            "covariates_used": ", ".join(covs),
            "att": att_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "auroc": auroc,
            "mean_smd_after": mean_smd,
        })

    return pd.DataFrame(rows)
