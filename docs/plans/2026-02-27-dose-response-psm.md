# Dose-Response PSM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the OLS residualization-based dose-response (section D) with per-bucket binary propensity score matching against 0% CRB controls.

**Architecture:** For each of the 5 non-zero CRBPct buckets, construct a binary comparison (bucket vs 0%), estimate stratified propensity scores, run k:1 NN matching on logit(PS), and compute ATT with bootstrap CIs. Reuses all existing matching infrastructure from section C.

**Tech Stack:** pandas, numpy, scikit-learn (LogisticRegression, NearestNeighbors), existing utils.py functions.

**Design doc:** `docs/plans/2026-02-27-dose-response-psm-design.md`

---

### Task 1: Implement `compute_dose_response_psm()` in parent_analysis.py

**Files:**
- Modify: `parent_analysis.py:618-648` (replace `compute_dose_response`)

**Step 1: Replace the function**

Delete `compute_dose_response` (lines 618-648) and replace with:

```python
# ===================================================================
# D.  Dose-response PSM
# ===================================================================


def compute_dose_response_psm(df, caliper_mult=0.2):
    """Dose-response via per-bucket binary PSM against 0% CRB controls.

    For each non-zero CRBPct bucket, matches treated orders to CRBPct=0%
    controls using stratified k:1 nearest-neighbor matching on logit(PS),
    aligned with run_psm_analysis() methodology (section C).

    Parameters
    ----------
    df : DataFrame – must have CRBPctBucket, PSM_COVARIATES, EXACT_MATCH_COLS
    caliper_mult : float – caliper as multiple of PS logit std

    Returns
    -------
    dict with keys:
      - "att_by_bucket": DataFrame (bucket, outcome, att, ci_lower, ci_upper,
         n_treated, n_matched_controls, avg_k_used)
      - "balance_by_bucket": dict of {bucket: {"smd_before": DF, "smd_after": DF}}
      - "diagnostics_by_bucket": dict of {bucket: {"strata_counts": DF, "overlap_summary": dict}}
    """
    available_covs = [c for c in PSM_COVARIATES if c in df.columns]
    exact_cols = [c for c in EXACT_MATCH_COLS if c in df.columns]
    ps_covs = [c for c in available_covs if c not in exact_cols]
    k = N_NEIGHBORS
    EPS = 1e-8

    def dist_to_weight(d, eps=EPS):
        d = np.asarray(d, dtype=float)
        return 1.0 / (d + eps)

    # Control pool: orders with CRBPct == 0
    controls = df[df["CRBPct"] == 0].copy()
    # Non-zero buckets (skip the "0%" label)
    dose_buckets = [b for b in CRB_BUCKET_LABELS if b != "0%"]

    att_rows = []
    balance_by_bucket = {}
    diagnostics_by_bucket = {}

    for bucket in dose_buckets:
        treated = df[df["CRBPctBucket"] == bucket].copy()
        if len(treated) < 10 or len(controls) < 10:
            continue

        # Create binary treatment column for this comparison
        work = pd.concat([
            treated.assign(_dose_treated=True),
            controls.assign(_dose_treated=False),
        ], axis=0, ignore_index=True)

        needed = ps_covs + exact_cols + ["_dose_treated"]
        work = work.dropna(subset=needed).copy()

        if len(work) < 20 or work["_dose_treated"].nunique() < 2:
            continue

        # --- Stratified PS estimation (same as section C) ---
        work["ps"] = np.nan
        if exact_cols:
            for _, g in work.groupby(exact_cols, dropna=False, observed=True):
                if g["_dose_treated"].nunique() < 2 or len(g) < 10:
                    continue
                ps_g = estimate_propensity_scores(g, "_dose_treated", ps_covs)
                work.loc[g.index, "ps"] = ps_g
        else:
            work["ps"] = estimate_propensity_scores(work, "_dose_treated", ps_covs)

        work = work.dropna(subset=["ps"]).copy()
        if len(work) < 20 or work["_dose_treated"].nunique() < 2:
            continue

        p = np.clip(work["ps"].astype(float).values, 1e-6, 1 - 1e-6)
        work["ps_logit"] = np.log(p / (1 - p))

        # --- Balance before ---
        smd_before = compute_smd(work, "_dose_treated", available_covs)

        # --- Strata diagnostics ---
        bucket_diag = {}
        if exact_cols:
            g_ct = work.groupby(exact_cols, dropna=False, observed=True)["_dose_treated"]
            sc = g_ct.agg(size="size", n_treated=lambda s: int(s.sum())).reset_index()
            sc["n_control"] = sc["size"] - sc["n_treated"]
            bucket_diag["strata_counts"] = sc
            overlap = (sc["n_treated"] > 0) & (sc["n_control"] > 0)
            bucket_diag["overlap_summary"] = {
                "n_strata": int(len(sc)),
                "n_overlap": int(overlap.sum()),
                "pct_overlap": float(100 * overlap.sum() / len(sc)) if len(sc) else np.nan,
            }

        # --- k:1 NN matching within strata (same as section C) ---
        matched_t_list = []
        matched_c_long_list = []
        pair_id_counter = 0

        group_iter = (
            work.groupby(exact_cols, dropna=False, observed=True)
            if exact_cols else [(None, work)]
        )

        for _, g in group_iter:
            treated_g = g[g["_dose_treated"]].copy().reset_index(drop=True)
            control_g = g[~g["_dose_treated"]].copy().reset_index(drop=True)

            if len(treated_g) == 0 or len(control_g) == 0:
                continue

            ps_std = g["ps_logit"].std()
            caliper = caliper_mult * ps_std if ps_std > 0 else 0.1

            indices, distances = nearest_neighbor_match(
                treated_g["ps_logit"].values,
                control_g["ps_logit"].values,
                n_neighbors=k,
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
            c_long["w_raw"] = dist_to_weight(c_long["distance"].values)
            wsum = c_long.groupby("pair_id")["w_raw"].transform("sum")
            c_long["match_weight"] = c_long["w_raw"] / wsum

            c_attached = control_g.iloc[c_long["c_idx"].values].copy().reset_index(drop=True)
            c_attached["pair_id"] = c_long["pair_id"].values
            c_attached["match_weight"] = c_long["match_weight"].values
            matched_t_list.append(mt)
            matched_c_long_list.append(c_attached)

        if not matched_t_list:
            balance_by_bucket[bucket] = {"smd_before": smd_before, "smd_after": pd.DataFrame()}
            diagnostics_by_bucket[bucket] = bucket_diag
            continue

        matched_t = pd.concat(matched_t_list, ignore_index=True)
        matched_c_long = pd.concat(matched_c_long_list, ignore_index=True)

        # --- Balance after ---
        matched_all = pd.concat([
            matched_t.assign(_dose_treated=True, match_weight=1.0),
            matched_c_long.assign(_dose_treated=False),
        ], ignore_index=True)
        smd_after = compute_weighted_smd(
            matched_all, "_dose_treated", ps_covs,
            matched_all["match_weight"].values,
        )

        balance_by_bucket[bucket] = {"smd_before": smd_before, "smd_after": smd_after}
        diagnostics_by_bucket[bucket] = bucket_diag

        # --- ATT per outcome ---
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
            ctrl_mean = c_df.groupby("pair_id").agg(
                control_sum=("w_y", "sum"),
                wsum=("match_weight", "sum"),
                k_used=(outcome, "size"),
            ).reset_index()
            ctrl_mean["control_mean"] = ctrl_mean["control_sum"] / ctrl_mean["wsum"]

            merged = t_df.merge(
                ctrl_mean[["pair_id", "control_mean", "k_used"]],
                on="pair_id", how="inner",
            )
            if merged.empty:
                continue

            diffs = merged[outcome].values - merged["control_mean"].values
            att, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=1000)

            att_rows.append({
                "bucket": bucket,
                "outcome": outcome,
                "att": att,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_treated": int(len(merged)),
                "n_matched_controls": int(len(matched_c_long)),
                "avg_k_used": float(np.nanmean(merged["k_used"].values)),
            })

    return {
        "att_by_bucket": pd.DataFrame(att_rows),
        "balance_by_bucket": balance_by_bucket,
        "diagnostics_by_bucket": diagnostics_by_bucket,
    }
```

Note: this function also needs `CRB_BUCKET_LABELS` imported from config. It is NOT currently imported — add it. Check the existing imports at the top of parent_analysis.py (lines 15-27) and add `CRB_BUCKET_LABELS` to the config import.

**Step 2: Verify no syntax errors**

Run: `python -c "from parent_analysis import compute_dose_response_psm; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add parent_analysis.py
git commit -m "feat: implement dose-response PSM replacing OLS residualization"
```

---

### Task 2: Wire up in `run_full_parent_analysis()`

**Files:**
- Modify: `parent_analysis.py:656-675`

**Step 1: Update the orchestrator**

Replace lines 667-668:
```python
    print("  [4/4] Dose-response curves ...")
    dr = compute_dose_response(df)
```
With:
```python
    print("  [4/4] Dose-response PSM ...")
    dr = compute_dose_response_psm(df)
```

Also remove the now-unused import of `compute_adjusted_means` from utils (line 30) if it's only used by the old `compute_dose_response`. Check first — `compute_adjusted_means` is imported at line 30 of parent_analysis.py. Search for other uses: if `compute_adjusted_means` is not used anywhere else in parent_analysis.py, remove it from the import.

**Step 2: Verify import works**

Run: `python -c "from parent_analysis import run_full_parent_analysis; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add parent_analysis.py
git commit -m "feat: wire dose-response PSM into analysis pipeline"
```

---

### Task 3: Update `_print_dose_response_summary()` in run_analysis.py

**Files:**
- Modify: `run_analysis.py:90-99`

**Step 1: Replace the print function**

Replace the function body (lines 90-99) with:

```python
def _print_dose_response_summary(dr_results):
    """Print dose-response PSM ATT table."""
    att = dr_results.get("att_by_bucket")
    if att is None or att.empty:
        print("\n  No dose-response PSM results available.")
        return
    for outcome in att["outcome"].unique():
        print(f"\n  {outcome}:")
        sub = att[att["outcome"] == outcome]
        for _, row in sub.iterrows():
            print(
                f"    {row['bucket']:12s} vs 0%:  ATT={row['att']:+8.3f}  "
                f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                f"n_treated={row['n_treated']:,.0f}"
            )
```

**Step 2: Verify no syntax errors**

Run: `python -c "from run_analysis import _print_dose_response_summary; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add run_analysis.py
git commit -m "feat: update dose-response summary printer for PSM ATT output"
```

---

### Task 4: Update P4 plot in plots.py

**Files:**
- Modify: `plots.py:191-234` (the `plot_dose_response` function)
- Modify: `plots.py:638-641` (the call site in `generate_parent_plots`)

**Step 1: Replace the P4 plot function**

Replace lines 191-234 with:

```python
# =====================================================================
# P4.  Dose-response PSM
# =====================================================================

def plot_dose_response(dose_response_results):
    """Plot ATT by CRBPct bucket from dose-response PSM."""
    att = dose_response_results.get("att_by_bucket")
    if att is None or att.empty:
        return

    outcomes = [o for o in OUTCOME_VARS if o in att["outcome"].values]
    n = len(outcomes)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        dr = att[att["outcome"] == outcome].copy()
        if dr.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(dr))
        ax.errorbar(
            x, dr["att"].values,
            yerr=[dr["att"].values - dr["ci_lower"].values,
                  dr["ci_upper"].values - dr["att"].values],
            fmt="o-", capsize=4, color=COLOR_CRB, linewidth=2, markersize=7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(dr["bucket"].values, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("ATT vs 0% CRB (bps)")
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)

        # annotate sample sizes
        for i, (_, row) in enumerate(dr.iterrows()):
            ax.annotate(
                f'n={row["n_treated"]:,.0f}',
                (i, row["ci_lower"]),
                textcoords="offset points", xytext=(0, -12),
                fontsize=7, ha="center", color="grey",
            )

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("P4: Dose-Response PSM — ATT vs 0% CRB by Dose Level",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P4_dose_response")
```

**Step 2: Update the call site** in `generate_parent_plots` (line 638-641)

The call `plot_dose_response(dr)` already passes the dose_response dict. The new function reads `dr.get("att_by_bucket")` which matches the new return structure. No change needed to the call site.

**Step 3: Also update the module docstring** at line 8:

Change `P4. Dose-response curve` to `P4. Dose-response PSM`.

**Step 4: Verify no syntax errors**

Run: `python -c "from plots import plot_dose_response; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add plots.py
git commit -m "feat: update P4 plot for dose-response PSM ATT visualization"
```

---

### Task 5: Fix test_pipeline.py synthetic data and add dose-response PSM validation

**Files:**
- Modify: `test_pipeline.py:95-134` (add missing venue columns to synthetic data)
- Modify: `test_pipeline.py:266-309` (add dose-response PSM result validation)

**Step 1: Fix synthetic data**

In `make_synthetic_parents()`, the synthetic DataFrame (line 95-134) is missing `ELPQty` and `FeeFeeQty` columns now required by `VENUE_QTY_COLS` in config. Add them as zero columns alongside the other venue columns (after line 120):

```python
        "ELPQty": np.zeros(n),
        "FeeFeeQty": np.zeros(n),
```

**Step 2: Add dose-response PSM validation**

After the results summary section (around line 269), add validation that the dose-response PSM produced results:

```python
    # Validate dose-response PSM results
    dr = parent_results.get("dose_response", {})
    att = dr.get("att_by_bucket")
    if att is not None and not att.empty:
        print(f"\n  Dose-response PSM: {len(att)} bucket-outcome ATT estimates")
        assert "att" in att.columns, "att_by_bucket missing 'att' column"
        assert "bucket" in att.columns, "att_by_bucket missing 'bucket' column"
        assert "outcome" in att.columns, "att_by_bucket missing 'outcome' column"
        assert len(att["bucket"].unique()) > 0, "No dose buckets matched"
        print("   Dose-response PSM validation: OK")
    else:
        print("\n  WARNING: Dose-response PSM produced no results (may need larger synthetic data)")
```

**Step 3: Run the test**

Run: `python test_pipeline.py`
Expected: Pipeline completes with `Pipeline test PASSED.` and dose-response PSM validation OK.

**Step 4: Commit**

```bash
git add test_pipeline.py
git commit -m "test: fix synthetic data for new venue cols, add dose-response PSM validation"
```

---

### Task 6: Update README.md

**Files:**
- Modify: `README.md` (Methodology section, around line 189-195)

**Step 1: Replace section D description**

Find the section:
```
**4. Non-Parametric Dose-Response**

Bins CRBPct into quantiles, residualizes outcomes on controls, and computes adjusted means per bin with bootstrap CIs. This captures non-linear relationships.
```

Replace with:
```
**4. Dose-Response PSM**

For each non-zero CRBPct dose bucket, constructs a binary comparison against CRBPct=0% controls. Estimates stratified propensity scores (within `EXACT_MATCH_COLS` strata), runs k:1 nearest-neighbor matching on logit(PS) with caliper and inverse-distance weighting — identical to the section C methodology. Computes the Average Treatment effect on the Treated (ATT) per dose level with bootstrap CIs. This provides a causal dose-response curve showing the benefit of each CRB dosage level relative to no internalization.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for dose-response PSM methodology"
```

---

### Task 7: Run full pipeline test and push

**Step 1: Run the full test**

Run: `python test_pipeline.py`
Expected: All passes, P4 plot generated, dose-response PSM validation OK.

**Step 2: Push**

```bash
git push origin master
```
