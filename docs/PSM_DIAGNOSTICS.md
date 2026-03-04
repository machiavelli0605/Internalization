# PSM Diagnostics: Investigating Negative Internalization ATT

## The Problem

The PSM analysis may report a **negative tempImpactBps ATT** — meaning internalized (CRB) orders appear to have *worse* temporary market impact than non-CRB orders. This is counterintuitive: if internalization avoids sending flow to lit venues, it should *reduce* adverse price movement.

A negative ATT does not necessarily mean internalization harms execution. It can arise from:

1. **Stratum composition** — one Strategy stratum dominates the sample and happens to have a negative ATT, masking positive effects in others.
2. **Unobserved confounding** — the propensity model fails to capture a variable that drives both CRB usage and worse outcomes (e.g., urgency, volatility regime).
3. **PS model misspecification** — the logistic model for propensity scores is too rigid, producing poor overlap or balance.
4. **Matched sample artifacts** — poor match quality in certain strata, or too few neighbors within caliper.

The diagnostics in `diagnostics.py` are designed to pinpoint which of these is the root cause.

## Running the Diagnostics

Diagnostics run automatically as part of the standard pipeline:

```bash
python run_analysis.py --parent-only --parent-data data/parent_orders.parquet
```

Output includes:
- **Console** — the `PSM DIAGNOSTICS` section printed after PSM results
- **CSV** — `output/results/psm_diagnostics_*.csv` files for each diagnostic
- **Plots** — `output/plots/P8_*` through `P13_*`

To run diagnostics programmatically:

```python
from parent_analysis import run_psm_analysis, run_psm_diagnostics

psm = run_psm_analysis(df, treatment_col="hasCRB")
diag = run_psm_diagnostics(df, psm)
```

## Step-by-Step Investigation

### Step 1: Identify the Offending Stratum

**Diagnostic:** Stratum ATT Decomposition (`diag["stratum_att"]`)
**Plot:** P9 — Per-Stratum ATT Waterfall

This breaks the overall ATT into per-stratum components, weighted by each stratum's share of treated units.

```
  Per-Stratum ATT Decomposition:
    tempImpactBps:
      VWAP          ATT=  -3.421  CI=[-5.1, -1.8]  n=1,200  weight=65%
      TWAP          ATT=  +1.832  CI=[+0.2, +3.4]  n=  650  weight=35%
```

**What to look for:**
- Which stratum has a negative ATT? (e.g., VWAP above)
- Is that stratum's contribution weight large enough to drag the overall ATT negative?
- Does the other stratum show the expected positive effect?

If VWAP has 65% weight and ATT = -3.4 while TWAP has 35% weight and ATT = +1.8, the blended ATT is `0.65 * (-3.4) + 0.35 * 1.8 = -1.58`. The negative overall ATT is driven entirely by VWAP.

### Step 2: Confirm with Leave-One-Out

**Diagnostic:** Leave-One-Out ATT (`diag["leave_one_out"]`)
**Plot:** P10 — Leave-One-Out Stratum Sensitivity

Recomputes the overall ATT after excluding each stratum.

```
  Leave-One-Out Stratum Sensitivity:
    tempImpactBps:
      Exclude VWAP          ATT=  +1.832  (full=-1.580, delta=+3.412)
      Exclude TWAP          ATT=  -3.421  (full=-1.580, delta=-1.841)
```

**What to look for:**
- If excluding a stratum flips the ATT sign from negative to positive, that stratum is the driver.
- A large positive `delta` for "Exclude X" means stratum X is pulling the ATT down.

### Step 3: Check Match Quality in the Problem Stratum

**Diagnostic:** Match Quality by Stratum (`diag["match_quality"]`)

```
  stratum  mean_dist  median_dist  max_dist  mean_k  min_k
  VWAP     0.085      0.062        0.412     8.2     3
  TWAP     0.031      0.024        0.198     9.7     7
```

**What to look for:**
- Higher `mean_dist` or `max_dist` in the problem stratum means treated units are being matched to distant controls — the "matches" may not be comparable.
- Low `min_k` means some treated units had very few neighbors within caliper, reducing the effective sample.
- If VWAP has 2-3x the match distances of TWAP, the VWAP ATT may be driven by bad matches rather than a real effect.

### Step 4: Check Covariate Balance Within the Stratum

**Diagnostic:** Covariate SMD by Stratum (`diag["smd_by_stratum"]`)

```
  stratum  covariate        smd
  VWAP     log_qtyOverADV   0.42
  VWAP     PcpRate         -0.38
  TWAP     log_qtyOverADV   0.08
  TWAP     PcpRate          0.05
```

**What to look for:**
- SMD > 0.25 within a stratum means the propensity model failed to achieve balance there.
- Systematic imbalance in key confounders (especially `log_qtyOverADV` and `PcpRate`) can bias the stratum ATT.
- Compare problem stratum SMDs vs healthy stratum — are specific covariates much more imbalanced?

### Step 5: Check PS Model Quality

**Diagnostic:** PS Model AUROC (`diag["auroc"]`)

```
  PS Model AUROC: 0.723  (n_treated=1,850, n_control=3,200)
```

**Interpretation:**
- AUROC 0.5–0.7: Weak separation (good for matching — treated and controls are similar)
- AUROC 0.7–0.8: Moderate separation (acceptable, check overlap)
- AUROC > 0.8: Strong separation (concerning — treated and controls may be structurally different, meaning few good matches exist)

**Plot:** P8 — PS Overlap Density

Look at the per-stratum panels. If the problem stratum shows poor overlap (treated and control PS distributions barely overlap), the matching is forced to use distant pairs.

### Step 6: Check Variance Ratios

**Diagnostic:** Variance Ratio (`diag["variance_ratio_before"]`, `diag["variance_ratio_after"]`)

```
  Variance Ratios (before matching):
    log_qtyOverADV            VR=3.412 ***
    PcpRate                   VR=1.021
    log_adv                   VR=0.834
```

**What to look for:**
- VR outside [0.5, 2.0] (flagged with `***`) means distributional imbalance not captured by SMD.
- A VR of 3.4 for `log_qtyOverADV` means the treated group has 3.4x the variance — even if means match, the distributions are very different.
- Check `variance_ratio_after` to see if matching improved or worsened the ratio.

### Step 7: Check Which Covariates Predict the Outcome

**Diagnostic:** Prognostic Scores (`diag["prognostic"]`)
**Plot:** P11 — Prognostic Covariate Importance

This fits `tempImpactBps ~ covariates` on the **control group only**, identifying which variables most strongly predict the outcome.

```
  Prognostic Covariate Importance:
    log_qtyOverADV            coef=  -4.231  SE=0.512  p=0.0000***
    PcpRate                   coef=  +2.140  SE=0.890  p=0.0162*
    duration_mins             coef=  -0.032  SE=0.015  p=0.0340*
    log_adv                   coef=  +0.180  SE=0.320  p=0.5740
    start_mins                coef=  +0.001  SE=0.003  p=0.8120
```

**What to look for:**
- Covariates with large, significant coefficients are strong outcome predictors.
- If a strong predictor (e.g., `log_qtyOverADV`) also has poor balance (from Step 4), that's likely driving the biased ATT.
- Covariates with near-zero coefficients and high p-values are not prognostic and could potentially be dropped from the PS model without loss.
- The R-squared tells you how much of the outcome variance is explained by observed covariates — low R-squared suggests unobserved confounders matter.

### Step 8: Test PS Model Robustness

**Diagnostic:** PS Specification Sensitivity (`diag["spec_sensitivity"]`)
**Plot:** P13 — PS Specification Sensitivity

Runs the full PSM pipeline under different PS model specifications:

```
  PS Specification Sensitivity (tempImpactBps):
    base                 ATT=  -1.580  CI=[-3.1, -0.1]  AUROC=0.723
    drop_log_qtyOverADV  ATT=  -2.910  CI=[-4.5, -1.3]  AUROC=0.681
    drop_PcpRate         ATT=  -0.320  CI=[-1.8, +1.2]  AUROC=0.718
    drop_log_adv         ATT=  -1.620  CI=[-3.2, -0.1]  AUROC=0.720
    drop_duration_mins   ATT=  -1.540  CI=[-3.1, +0.0]  AUROC=0.719
    drop_start_mins      ATT=  -1.590  CI=[-3.1, -0.1]  AUROC=0.722
    quadratic            ATT=  -1.210  CI=[-2.7, +0.3]  AUROC=0.735
    interactions          ATT=  -0.980  CI=[-2.5, +0.5]  AUROC=0.731
```

**What to look for:**
- **Sensitivity to specific covariates**: If dropping `PcpRate` flips the ATT from -1.6 to -0.3 (nearly null), PcpRate is doing heavy lifting in the PS model and may be introducing bias rather than removing it.
- **Functional form**: If "quadratic" or "interactions" moves the ATT toward zero, the linear PS model is misspecified.
- **Stability**: If the ATT is similar across all specs, the result is robust to PS model choice. If it swings wildly, the negative ATT is fragile.
- **AUROC changes**: Large AUROC changes when dropping a variable suggest that variable strongly separates treated/control — which may indicate structural differences the matching cannot overcome.

### Step 9: Assess Sensitivity to Unmeasured Confounding

**Diagnostic:** Rosenbaum Bounds (`diag["rosenbaum_bounds"]`)
**Plot:** P12 — Rosenbaum Bounds

Tests how much unmeasured confounding (Gamma) would be needed to explain away the result.

```
  Rosenbaum Bounds: result insensitive up to Gamma=1.3
```

**Interpretation:**
- Gamma = 1.0 is no confounding (standard Wilcoxon test).
- The result "breaks" (p > 0.05) at Gamma = 1.3, meaning an unmeasured confounder that changes the odds of treatment by just 30% could explain the negative ATT.
- Gamma < 1.5: Result is **fragile** — easily explained by a moderate unmeasured confounder.
- Gamma > 2.0: Result is **robust** — would require a very strong unmeasured confounder.

**Diagnostic:** E-value (`diag["e_values"]`)

```
  E-values (unmeasured confounding sensitivity):
    tempImpactBps             E-value=2.45  (CI bound: 1.31)
```

The E-value is the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need with both treatment and outcome to fully explain the observed ATT. An E-value of 2.45 means the confounder would need a 2.45x association with both — but the CI bound of 1.31 means it could explain the confidence interval with just a 1.31x association.

## Decision Framework

After running through the steps above, you should be in one of these situations:

### Scenario A: Single Stratum Drives the Negative ATT

**Symptoms:** Step 1 shows one stratum with a large negative ATT and high weight. Step 2 confirms excluding it flips the sign. Other strata show positive ATTs.

**Actions:**
- Investigate that stratum's match quality (Step 3) and covariate balance (Step 4).
- If match quality is poor, consider increasing `N_NEIGHBORS` or relaxing the caliper for that stratum.
- If the negative effect is real (good balance, good match quality), it may reflect a genuine stratum-specific phenomenon (e.g., VWAP orders internalize differently than TWAP).
- Report results by stratum rather than pooling.

### Scenario B: PS Model Misspecification

**Symptoms:** Step 8 shows the ATT is sensitive to functional form (quadratic/interactions move it toward zero). Step 5 shows high AUROC or poor overlap in P8.

**Actions:**
- Add quadratic terms or interactions to `PSM_COVARIATES` in `config.py`.
- Consider trimming extreme propensity scores more aggressively.
- If overlap is fundamentally poor, consider restricting analysis to the region of common support.

### Scenario C: Missing Confounder

**Symptoms:** Step 9 shows low Gamma breakpoint (< 1.5) and low E-value CI bound. Step 7 shows low R-squared. Step 6 shows high variance ratios.

**Actions:**
- Look for additional covariates not currently in the model (volatility regime, market conditions, order complexity).
- The prognostic analysis (Step 7) identifies which types of variables would be most useful — look for similar but unmeasured variables.
- Consider a different identification strategy (e.g., regression discontinuity, instrumental variables).

### Scenario D: Result is Robust

**Symptoms:** ATT is consistently negative across all strata, all PS specifications, with high Gamma breakpoint (> 2.0) and good covariate balance.

**Conclusion:** The negative effect appears genuine. Internalization may not improve temporary impact for this dataset/time period, possibly because internalized flow is adversely selected or because the internalization price is not favorable enough.

## Diagnostics Reference

| Key | Type | Description |
|---|---|---|
| `stratum_att` | DataFrame | Per-stratum ATT with bootstrap CIs and contribution weights |
| `leave_one_out` | DataFrame | ATT recomputed excluding each stratum one at a time |
| `auroc` | dict | AUROC of propensity score model |
| `variance_ratio_before` | DataFrame | Var(treated)/Var(control) per covariate before matching |
| `variance_ratio_after` | DataFrame | Same, after matching |
| `smd_by_stratum` | DataFrame | Standardized mean differences within each stratum |
| `prognostic` | DataFrame | OLS coefficients predicting outcome from covariates (control group) |
| `match_quality` | DataFrame | Per-stratum distance statistics and effective k |
| `rosenbaum_bounds` | DataFrame | Gamma vs upper-bound p-value for sensitivity analysis |
| `e_values` | dict | E-value for each outcome |
| `spec_sensitivity` | DataFrame | ATT under different PS model specifications |

## Plots Reference

| Plot | File | Shows |
|---|---|---|
| P8 | `P8_ps_overlap_density.png` | KDE of propensity scores by treatment group, overall and per stratum |
| P9 | `P9_stratum_att_waterfall.png` | Horizontal bars: per-stratum ATT with CIs and contribution weights |
| P10 | `P10_leave_one_out.png` | Forest plot: overall ATT vs ATT excluding each stratum |
| P11 | `P11_prognostic_importance.png` | Bar chart: which covariates predict tempImpactBps in controls |
| P12 | `P12_rosenbaum_bounds.png` | Line plot: Gamma vs p-value, with 0.05 threshold marked |
| P13 | `P13_spec_sensitivity.png` | Forest plot: ATT across PS model specifications |
