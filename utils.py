"""
Shared utility functions for internalization analysis.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols as smf_ols
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ===================================================================
# Regression helpers
# ===================================================================

def build_ols_formula(outcome, treatments, continuous_controls, categorical_fe):
    """Build a statsmodels OLS formula string.

    Parameters
    ----------
    outcome : str
    treatments : list[str]
    continuous_controls : list[str]
    categorical_fe : list[str]  – wrapped with C()

    Returns
    -------
    str
    """
    rhs_parts = list(treatments) + list(continuous_controls)
    rhs_parts += [f"C({c})" for c in categorical_fe]
    rhs = " + ".join(rhs_parts)
    return f"{outcome} ~ {rhs}"


def run_ols(df, outcome, treatments, continuous_controls, categorical_fe,
            cluster_col=None, sample_n=None):
    """Run OLS regression and return results.

    Parameters
    ----------
    df : DataFrame
    outcome : str
    treatments : list[str]
    continuous_controls : list[str]
    categorical_fe : list[str]
    cluster_col : str or None – column for clustered SEs
    sample_n : int or None – if set, sample this many rows

    Returns
    -------
    statsmodels RegressionResults, or None if fitting fails
    """
    all_cols = [outcome] + treatments + continuous_controls
    present = [c for c in all_cols if c in df.columns]
    work = df.dropna(subset=present)
    if sample_n and len(work) > sample_n:
        work = work.sample(n=sample_n, random_state=42)

    # need at least more rows than parameters
    n_params = len(treatments) + len(continuous_controls) + 2  # rough lower bound
    if len(work) < n_params:
        return None

    # filter categorical FE to those with >1 level in the working data
    active_fe = [c for c in categorical_fe if c in work.columns and work[c].nunique() > 1]

    formula = build_ols_formula(outcome, treatments, continuous_controls, active_fe)
    try:
        model = smf_ols(formula, data=work)
        if cluster_col and cluster_col in work.columns:
            groups = work[cluster_col]
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            result = model.fit(cov_type="HC1")
        return result
    except Exception:
        return None


def extract_treatment_coefficients(result, treatments):
    """Pull coefficients, SEs, CIs, and p-values for treatment variables.

    Returns
    -------
    DataFrame with columns: coef, se, ci_lower, ci_upper, pvalue, nobs
    """
    if result is None:
        return pd.DataFrame()
    rows = []
    conf = result.conf_int()
    for t in treatments:
        # handle bool treatment mapped to True label
        label = f"{t}[T.True]" if t in ("isInt", "hasCRB") else t
        if label not in result.params.index:
            label = t  # fallback
        if label not in result.params.index:
            continue
        rows.append({
            "treatment": t,
            "coef": result.params[label],
            "se": result.bse[label],
            "ci_lower": conf.loc[label, 0],
            "ci_upper": conf.loc[label, 1],
            "pvalue": result.pvalues[label],
            "nobs": int(result.nobs),
            "r2": result.rsquared,
        })
    return pd.DataFrame(rows)


# ===================================================================
# Propensity Score Matching / IPW
# ===================================================================

def estimate_propensity_scores(df, treatment_col, covariates, max_iter=1000):
    """Estimate propensity scores via logistic regression.

    Returns
    -------
    np.ndarray of propensity scores (same length as df)
    """
    X = df[covariates].copy()
    y = df[treatment_col].astype(int)

    # drop rows with NaN in covariates
    mask = X.notna().all(axis=1)
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    ps = np.full(len(df), np.nan)

    # need both treatment classes present and enough observations
    if len(X_clean) < 10 or y_clean.nunique() < 2:
        # degenerate case: return naive proportion as PS
        if len(y_clean) > 0:
            ps[mask.values] = y_clean.mean()
        return ps

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    lr = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=1.0)
    try:
        lr.fit(X_scaled, y_clean)
        ps[mask.values] = lr.predict_proba(X_scaled)[:, 1]
    except Exception:
        # fallback: return naive proportion
        ps[mask.values] = y_clean.mean()
    return ps


def nearest_neighbor_match(ps_treated, ps_control, n_neighbors=1, caliper=None):
    """Match treated to control units on propensity score.

    Parameters
    ----------
    ps_treated : array-like – propensity scores for treated group
    ps_control : array-like – propensity scores for control group
    n_neighbors : int
    caliper : float or None – max allowed distance

    Returns
    -------
    matched_indices : ndarray of shape (n_treated, n_neighbors) – indices into
                      the control array
    distances : ndarray – matching distances
    """
    ps_t = np.asarray(ps_treated).reshape(-1, 1)
    ps_c = np.asarray(ps_control).reshape(-1, 1)

    if len(ps_t) == 0 or len(ps_c) == 0:
        return (np.array([], dtype=int).reshape(0, n_neighbors),
                np.array([], dtype=float).reshape(0, n_neighbors))

    # can't request more neighbors than control units
    effective_k = min(n_neighbors, len(ps_c))
    nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn.fit(ps_c)
    distances, indices = nn.kneighbors(ps_t)

    if caliper is not None:
        # flag matches outside caliper as -1
        indices[distances > caliper] = -1
        distances[distances > caliper] = np.nan

    return indices, distances


def compute_ipw_weights(ps, treatment, trim_pct=0.05):
    """Compute inverse-propensity weights with trimming.

    Parameters
    ----------
    ps : array-like – propensity scores
    treatment : array-like – binary treatment indicator
    trim_pct : float – trim tails of PS distribution

    Returns
    -------
    weights : ndarray
    """
    ps = np.asarray(ps, dtype=float)
    treatment = np.asarray(treatment, dtype=int)

    # trim extreme propensity scores
    lo, hi = np.nanpercentile(ps, [trim_pct * 100, (1 - trim_pct) * 100])
    trimmed = (ps >= lo) & (ps <= hi)

    weights = np.zeros_like(ps)
    # treated: w = 1/ps; control: w = 1/(1-ps)
    weights[treatment == 1] = 1.0 / np.clip(ps[treatment == 1], 1e-6, None)
    weights[treatment == 0] = 1.0 / np.clip(1 - ps[treatment == 0], 1e-6, None)
    weights[~trimmed] = 0.0

    return weights


# ===================================================================
# Covariate balance (Standardized Mean Difference)
# ===================================================================

def compute_smd(df, treatment_col, covariates):
    """Compute standardized mean differences for each covariate.

    SMD = (mean_treated - mean_control) / sqrt((var_t + var_c) / 2)

    Returns
    -------
    DataFrame with columns: covariate, smd, mean_treated, mean_control
    """
    treated = df[df[treatment_col].astype(bool)]
    control = df[~df[treatment_col].astype(bool)]

    rows = []
    for cov in covariates:
        mt = treated[cov].mean()
        mc = control[cov].mean()
        vt = treated[cov].var()
        vc = control[cov].var()
        pooled_sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({
            "covariate": cov,
            "smd": smd,
            "mean_treated": mt,
            "mean_control": mc,
        })
    return pd.DataFrame(rows)


def compute_weighted_smd(df, treatment_col, covariates, weights):
    """Compute weighted SMD after IPW/matching reweighting."""
    t_mask = df[treatment_col].astype(bool).values
    c_mask = ~t_mask
    w = np.asarray(weights)

    rows = []
    for cov in covariates:
        x = df[cov].values.astype(float)
        # weighted means
        mt = np.average(x[t_mask], weights=w[t_mask]) if w[t_mask].sum() > 0 else np.nan
        mc = np.average(x[c_mask], weights=w[c_mask]) if w[c_mask].sum() > 0 else np.nan
        # unweighted pooled SD for denominator (standard practice)
        vt = np.var(x[t_mask])
        vc = np.var(x[c_mask])
        pooled_sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"covariate": cov, "smd": smd, "mean_treated": mt, "mean_control": mc})
    return pd.DataFrame(rows)


# ===================================================================
# Bootstrap confidence intervals
# ===================================================================

def bootstrap_ci(data, statistic_func, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : array-like
    statistic_func : callable – takes array, returns scalar
    n_boot : int
    ci : float

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)

    point = statistic_func(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic_func(sample)

    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (point, lo, hi)


def bootstrap_mean_ci(data, n_boot=1000, ci=0.95, seed=42):
    """Convenience wrapper: bootstrap CI for the mean."""
    return bootstrap_ci(data, np.nanmean, n_boot=n_boot, ci=ci, seed=seed)


# ===================================================================
# Winsorization
# ===================================================================

def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize a Series at the given percentiles."""
    lo_val = series.quantile(lower)
    hi_val = series.quantile(upper)
    return series.clip(lower=lo_val, upper=hi_val)


# ===================================================================
# Fixed-effects demeaning (within estimator)
# ===================================================================

def demean_by_groups(df, value_cols, group_cols):
    """Demean columns within groups (for absorbing high-dim FE).

    Parameters
    ----------
    df : DataFrame
    value_cols : list[str] – numeric columns to demean
    group_cols : list[str] – categorical columns defining groups

    Returns
    -------
    DataFrame with demeaned value columns (same index as df)
    """
    demeaned = df[value_cols].copy()
    for gc in group_cols:
        group_means = df.groupby(gc)[value_cols].transform("mean")
        demeaned = demeaned - group_means
    # add back the grand mean so coefficients are interpretable
    demeaned = demeaned + df[value_cols].mean()
    return demeaned


# ===================================================================
# Adjusted means by bucket (for dose-response)
# ===================================================================

def compute_adjusted_means(df, outcome, bucket_col, continuous_controls,
                           categorical_fe, n_boot=500):
    """Compute confounder-adjusted means per bucket.

    Approach: residualize outcome on controls, then compute mean residual
    per bucket and add back the grand mean.

    Returns
    -------
    DataFrame with columns: bucket, adj_mean, ci_lower, ci_upper, n
    """
    available_controls = [c for c in continuous_controls if c in df.columns]
    work = df.dropna(subset=[outcome] + available_controls).copy()

    if len(work) == 0:
        return pd.DataFrame(columns=["bucket", "adj_mean", "ci_lower", "ci_upper", "n"])

    # residualize outcome on controls
    active_fe = [c for c in categorical_fe if c in work.columns and work[c].nunique() > 1]
    formula = build_ols_formula(outcome, [], available_controls, active_fe)
    try:
        res = smf_ols(formula, data=work).fit()
        work["_residual"] = res.resid + work[outcome].mean()
    except Exception:
        # fallback: use raw outcome
        work["_residual"] = work[outcome]

    rows = []
    for bucket, grp in work.groupby(bucket_col, observed=True):
        vals = grp["_residual"].dropna().values
        if len(vals) == 0:
            continue
        mean_val, ci_lo, ci_hi = bootstrap_mean_ci(vals, n_boot=n_boot)
        rows.append({
            "bucket": bucket,
            "adj_mean": mean_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n": len(vals),
        })
    return pd.DataFrame(rows)
