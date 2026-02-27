# Dose-Response PSM Design

**Date**: 2026-02-27
**Status**: Approved
**Replaces**: Section D (non-parametric dose-response via OLS residualization)

## Problem

The current dose-response analysis (section D) residualizes outcomes on controls via OLS, then computes adjusted means per CRBPct bucket. This relies on the linear model being correctly specified. If confounders affect both CRBPct and outcomes non-linearly, the residualization doesn't fully remove selection bias.

## Solution

Replace section D with **per-bucket binary propensity score matching**, aligned with the methodology in section C (binary PSM). For each non-zero CRBPct dose bucket, match treated orders against CRBPct=0% controls using the same k:1 stratified nearest-neighbor matching infrastructure.

## Approach: Separate Binary PSM per Dose Bucket

For each of the 5 non-zero dose buckets (0-5%, 5-15%, 15-30%, 30-50%, 50%+):

1. **Define comparison**: Treated = orders in this bucket. Control = orders with CRBPct = 0%.
2. **Estimate PS**: Binary logistic regression on `PSM_COVARIATES`, stratified by `EXACT_MATCH_COLS` (same as section C).
3. **Transform**: Logit scale, clip to [1e-6, 1-1e-6] (same as section C).
4. **Match**: k:1 NN (`N_NEIGHBORS=10`) on logit(PS) with caliper = `caliper_mult * std(ps_logit)` (same as section C).
5. **Weight**: Inverse-distance weighting of k neighbors (same as section C).
6. **ATT**: Weighted mean difference (treated - matched control) with bootstrap CIs (same as section C).
7. **Balance**: SMD before/after matching per bucket.

The 0% control pool is shared across all bucket comparisons.

## Alternatives Considered

- **Generalized Propensity Score (GPS)**: Multinomial PS model across all dose levels. Theoretically clean but harder to implement, validate, and diagnose.
- **Continuous treatment IPW**: Model CRBPct density via beta regression. Most data-efficient but fragile, requires density modeling of a zero-inflated bounded variable.

Approach B (separate binary PSM) was chosen for simplicity, reuse of existing infrastructure, and 1:1 alignment with section C.

## File Changes

### `parent_analysis.py`
- Remove `compute_dose_response()`
- Add `compute_dose_response_psm(df, treatment_col="CRBPct", caliper_mult=0.2)`
- Update `run_full_parent_analysis()` to call the new function

### `run_analysis.py`
- Update `_print_dose_response_summary()` to print ATT per bucket (format: `bucket vs 0%: ATT=X CI=[lo, hi] n_treated=N n_ctrl=M`)

### `plots.py`
- Update P4 plot: x=bucket, y=ATT with CI error bars, horizontal line at 0, one panel per outcome

### `test_pipeline.py`
- Add test for `compute_dose_response_psm()` with synthetic data covering multiple CRBPct buckets

### `README.md`
- Update section D methodology: "Non-parametric dose-response" -> "Dose-Response PSM"
- Describe per-bucket binary matching aligned with section C

### `config.py`
- No changes needed. Reuses existing: `CRB_BUCKET_EDGES`, `CRB_BUCKET_LABELS`, `PSM_COVARIATES`, `EXACT_MATCH_COLS`, `N_NEIGHBORS`.

## Return Structure

```python
{
    "att_by_bucket": DataFrame(
        bucket, outcome, att, ci_lower, ci_upper,
        n_treated, n_matched_controls, avg_k_used
    ),
    "balance_by_bucket": {
        "(0-5%]": {"smd_before": DataFrame, "smd_after": DataFrame},
        ...
    },
    "diagnostics_by_bucket": {
        "(0-5%]": {"strata_counts": DataFrame, "overlap_summary": dict, "ps_quantiles": dict},
        ...
    },
}
```
