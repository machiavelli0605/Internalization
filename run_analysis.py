"""
Main orchestration script for internalization market impact analysis.

Usage:
    python run_analysis.py                          # full run
    python run_analysis.py --parent-only            # parent analysis only
    python run_analysis.py --exec-only              # execution analysis only
    python run_analysis.py --parent-data path.parquet --exec-data path.parquet
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from config import (
    CONTINUOUS_CONTROLS,
    EXECUTION_DATA_PATH,
    OUTCOME_VARS,
    PARENT_DATA_PATH,
    PLOT_DIR,
    PSM_COVARIATES,
    RESULT_DIR,
)
from data_prep import load_execution_data, load_parent_data, sample_parent_orders
from execution_analysis import run_full_execution_analysis
from parent_analysis import run_full_parent_analysis
from plots import generate_execution_plots, generate_parent_plots

warnings.filterwarnings("ignore", category=FutureWarning)


def _print_regression_summary(reg_results):
    """Print a concise table of regression coefficients."""
    for label, key in [
        ("ITT (isInt, Full Pop.)", "itt_coefficients"),
        ("Dose-Response (CRBPct, Full Pop.)", "dose_coefficients"),
        ("Dose-Response (CRBPct, Enabled Only)", "itt_enabled_coefficients"),
    ]:
        df = reg_results.get(key)
        if df is None or df.empty:
            continue
        print(f"\n  {label}:")
        # for dose, show only CRBPct
        if "CRBPct" in df["treatment"].values:
            df = df[df["treatment"] == "CRBPct"]
        for _, row in df.iterrows():
            sig = ""
            if row["pvalue"] < 0.001:
                sig = "***"
            elif row["pvalue"] < 0.01:
                sig = "**"
            elif row["pvalue"] < 0.05:
                sig = "*"
            print(
                f"    {row['outcome']:25s}  coef={row['coef']:+8.3f}  "
                f"SE={row['se']:.3f}  p={row['pvalue']:.4f}{sig}  "
                f"n={row['nobs']:,.0f}"
            )


def _print_psm_summary(psm_results):
    """Print PSM outcome comparison summary."""
    nn = psm_results.get("nn_outcomes")
    if nn is not None and not nn.empty:
        print("\n  Nearest-Neighbor Matched Outcome Differences (Treated − Control):")
        for _, row in nn.iterrows():
            print(
                f"    {row['outcome']:25s}  diff={row['diff']:+8.3f}  "
                f"CI=[{row['diff_ci_lower']:+.3f}, {row['diff_ci_upper']:+.3f}]  "
                f"n_pairs={row['n_pairs']:,.0f}"
            )

    ipw = psm_results.get("ipw_outcomes")
    if ipw is not None and not ipw.empty:
        print("\n  IPW-Weighted Outcome Means:")
        pivot = ipw.pivot(index="outcome", columns="group", values="weighted_mean")
        if "treated" in pivot.columns and "control" in pivot.columns:
            for outcome in pivot.index:
                t_val = pivot.loc[outcome, "treated"]
                c_val = pivot.loc[outcome, "control"]
                print(
                    f"    {outcome:25s}  treated={t_val:+8.3f}  "
                    f"control={c_val:+8.3f}  diff={t_val - c_val:+8.3f}"
                )


def _print_psm_diagnostics(psm_diag):
    """Print PSM diagnostic summary."""
    # AUROC
    auroc = psm_diag.get("auroc", {})
    if auroc.get("auroc") is not None and not np.isnan(auroc.get("auroc", np.nan)):
        print(
            f"\n  PS Model AUROC: {auroc['auroc']:.3f}  "
            f"(n_treated={auroc['n_treated']:,}, n_control={auroc['n_control']:,})"
        )

    # Stratum ATT
    stratum_att = psm_diag.get("stratum_att", pd.DataFrame())
    if not stratum_att.empty:
        print("\n  Per-Stratum ATT Decomposition:")
        for outcome in stratum_att["outcome"].unique():
            print(f"    {outcome}:")
            sub = stratum_att[stratum_att["outcome"] == outcome]
            for _, row in sub.iterrows():
                print(
                    f"      {row['stratum']:12s}  ATT={row['att']:+8.3f}  "
                    f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                    f"n={row['n_treated']:,}  weight={row['contribution_weight']:.1%}"
                )

    # Leave-one-out
    loo = psm_diag.get("leave_one_out", pd.DataFrame())
    if not loo.empty:
        print("\n  Leave-One-Out Stratum Sensitivity:")
        for outcome in loo["outcome"].unique():
            print(f"    {outcome}:")
            sub = loo[loo["outcome"] == outcome]
            for _, row in sub.iterrows():
                direction = "+" if row["delta"] > 0 else ""
                print(
                    f"      Exclude {row['excluded_stratum']:12s}  "
                    f"ATT={row['att_without']:+8.3f}  "
                    f"(full={row['att_full']:+.3f}, delta={direction}{row['delta']:.3f})"
                )

    # Variance ratio
    vr_before = psm_diag.get("variance_ratio_before", pd.DataFrame())
    if not vr_before.empty:
        print("\n  Variance Ratios (before matching):")
        for _, row in vr_before.iterrows():
            flag = " ***" if (row["vr"] < 0.5 or row["vr"] > 2.0) else ""
            print(f"    {row['covariate']:25s}  VR={row['vr']:.3f}{flag}")

    # Prognostic scores
    prog = psm_diag.get("prognostic", pd.DataFrame())
    if not prog.empty:
        print(
            "\n  Prognostic Covariate Importance (predicting tempImpactBps in controls):"
        )
        for _, row in prog.iterrows():
            sig = ""
            if row["pvalue"] < 0.001:
                sig = "***"
            elif row["pvalue"] < 0.01:
                sig = "**"
            elif row["pvalue"] < 0.05:
                sig = "*"
            print(
                f"    {row['covariate']:25s}  coef={row['coef']:+8.3f}  "
                f"SE={row['se']:.3f}  p={row['pvalue']:.4f}{sig}"
            )

    # E-values
    e_vals = psm_diag.get("e_values", {})
    if e_vals:
        print("\n  E-values (unmeasured confounding sensitivity):")
        for outcome, ev in e_vals.items():
            print(
                f"    {outcome:25s}  E-value={ev['e_value_point']:.2f}  "
                f"(CI bound: {ev['e_value_ci']:.2f})"
            )

    # Rosenbaum bounds
    rb = psm_diag.get("rosenbaum_bounds", pd.DataFrame())
    if not rb.empty:
        crossings = rb[rb["p_upper"] >= 0.05]
        if not crossings.empty:
            gamma_break = crossings["gamma"].iloc[0]
            print(
                f"\n  Rosenbaum Bounds: result insensitive up to Gamma={gamma_break:.1f}"
            )
        else:
            print(f"\n  Rosenbaum Bounds: result robust at all tested Gamma values")

    # Spec sensitivity
    spec = psm_diag.get("spec_sensitivity", pd.DataFrame())
    if not spec.empty:
        print("\n  PS Specification Sensitivity (tempImpactBps):")
        for _, row in spec.iterrows():
            print(
                f"    {row['spec_name']:20s}  ATT={row['att']:+8.3f}  "
                f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                f"AUROC={row.get('auroc', np.nan):.3f}"
            )


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


def _print_markout_summary(exec_results):
    """Print markout curve summary."""
    signed = exec_results.get("signed_markouts")
    if signed is not None and not signed.empty:
        print("\n  Signed Markout Curves:")
        for is_int in [True, False]:
            label = "CRB" if is_int else "Non-CRB"
            sub = signed[signed["group"] == is_int].sort_values("horizon_sec")
            vals = "  ".join(
                f"{int(r['horizon_sec'])}s={r['mean']:+.3f}" for _, r in sub.iterrows()
            )
            print(f"    {label:10s}  {vals}")

    within = exec_results.get("within_order", {})
    paired = within.get("paired_diff")
    if paired is not None and not paired.empty:
        signed_p = paired[~paired["is_absolute"]].sort_values("horizon_sec")
        if not signed_p.empty:
            print("\n  Within-Order Paired Difference (CRB − Non-CRB, signed):")
            for _, row in signed_p.iterrows():
                print(
                    f"    {int(row['horizon_sec'])}s:  diff={row['mean_diff']:+.3f}  "
                    f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                    f"n={row['n_orders']:,.0f}"
                )


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Internalization market-impact analysis"
    )
    parser.add_argument("--parent-data", default=str(PARENT_DATA_PATH))
    parser.add_argument("--exec-data", default=str(EXECUTION_DATA_PATH))
    parser.add_argument("--parent-only", action="store_true")
    parser.add_argument("--exec-only", action="store_true")
    parser.add_argument(
        "--cluster-col", default="RIC", help="Column for clustered standard errors"
    )
    parser.add_argument(
        "--exec-sample",
        type=int,
        default=5_000_000,
        help="Sample size for execution-level KDE plots",
    )
    parser.add_argument(
        "--exclude-auctions",
        action="store_true",
        help="Exclude auction fills from analysis",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.exclude_auctions:
        print("MODE: Excluding auction fills from analysis")

    # -----------------------------------------------------------------
    # Parent-level analysis
    # -----------------------------------------------------------------
    if not args.exec_only:
        print("=" * 60)
        print("PARENT-LEVEL ANALYSIS")
        print("=" * 60)

        print(f"\nLoading parent data from {args.parent_data} ...")
        filters = [("EnState", "==", "F")]
        if args.exclude_auctions:
            filters += [("IncludeOpen", "!=", True), ("IncludeClose", "!=", True)]
        parent_df = load_parent_data(
            args.parent_data, exclude_auctions=args.exclude_auctions
        )
        print(f"  Loaded {len(parent_df):,} parent orders.")
        print(
            f"  isInt=True: {parent_df['isInt'].sum():,}  "
            f"({parent_df['isInt'].mean() * 100:.1f}%)"
        )
        print(
            f"  hasCRB: {parent_df['hasCRB'].sum():,}  "
            f"({parent_df['hasCRB'].mean() * 100:.1f}%)"
        )

        print("\nRunning parent-level analyses ...")
        cap_quantile = 0.99
        for c in set(CONTINUOUS_CONTROLS + PSM_COVARIATES):
            if pd.api.types.is_numeric_dtype(
                parent_df[c]
            ) or pd.api.types.is_bool_dtype(parent_df[c]):
                parent_df[c] = pd.to_numeric(parent_df[c], errors="coerce")
                arr = parent_df[c].to_numpy(dtype=float, copy=False)
                nonfinite = ~np.isfinite(arr)
                finite_vals = parent_df.loc[~nonfinite, c].dropna()
                if len(finite_vals) > 0:
                    lo = finite_vals.quantile(1 - cap_quantile)
                    hi = finite_vals.quantile(cap_quantile)
                    parent_df.loc[
                        np.isposinf(parent_df[c]) | (parent_df[c] > hi), c
                    ] = hi
                    parent_df.loc[
                        np.isneginf(parent_df[c]) | (parent_df[c] < lo), c
                    ] = lo
                else:
                    parent_df.loc[nonfinite, c] = np.nan
        parent_results = run_full_parent_analysis(
            parent_df, cluster_col=args.cluster_col
        )

        # --- Print summaries ---
        # print("\n" + "-" * 60)
        # print("REGRESSION RESULTS")
        # print("-" * 60)
        # _print_regression_summary(parent_results["regression"])

        print("\n" + "-" * 60)
        print("PROPENSITY SCORE RESULTS")
        print("-" * 60)
        _print_psm_summary(parent_results["psm"])

        psm_diag = parent_results.get("psm_diagnostics", {})
        if psm_diag:
            print("\n" + "-" * 60)
            print("PSM DIAGNOSTICS")
            print("-" * 60)
            _print_psm_diagnostics(psm_diag)

        # print("\n" + "-" * 60)
        # print("DOSE-RESPONSE RESULTS")
        # print("-" * 60)
        # _print_dose_response_summary(parent_results["dose_response"])

        for tor, res in parent_results.items():
            for dn, dv in res.items():
                try:
                    if isinstance(dv, pd.DataFrame):
                        df = dv
                    elif isinstance(dv, pd.Series):
                        df = dv.to_frame()
                    elif isinstance(dv, dict) and all(
                        np.isscalar(x) for x in dv.values()
                    ):
                        df = pd.DataFrame([dv])
                    elif isinstance(dv, list) and dv and isinstance(dv[0], dict):
                        df = pd.DataFrame(dv)
                    else:
                        continue
                    df.to_csv(f"{RESULT_DIR}/{tor}_{dn}.csv", index=False)
                except (ValueError, TypeError):
                    continue
        # --- Generate plots ---
        generate_parent_plots(parent_df, parent_results)

    # -----------------------------------------------------------------
    # Execution-level analysis
    # -----------------------------------------------------------------
    if not args.parent_only:
        print("\n" + "=" * 60)
        print("EXECUTION-LEVEL ANALYSIS")
        print("=" * 60)

        print(f"\nRunning execution-level analyses on {args.exec_data} ...")

        # We need parent_df for within-order analysis
        if args.exec_only:
            print(f"  Loading parent data for within-order analysis ...")
            parent_df = load_parent_data(
                args.parent_data, exclude_auctions=args.exclude_auctions
            )

        exec_df = load_execution_data(
            path=args.exec_data, exclude_auctions=args.exclude_auctions
        )
        exec_results = run_full_execution_analysis(parent_df, exec_df=exec_df)

        # --- Print summaries ---
        print("\n" + "-" * 60)
        print("MARKOUT RESULTS")
        print("-" * 60)
        _print_markout_summary(exec_results)

        # --- Generate plots ---
        # For E6 (KDE), we need a sample loaded into memory
        print(
            f"\n  Loading execution sample (n={args.exec_sample:,}) for KDE plots ..."
        )
        exec_sample = exec_df.sample(
            n=min(args.exec_sample, len(exec_df)), random_state=42
        )
        generate_execution_plots(exec_results, exec_df_sample=exec_sample)

    # -----------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Elapsed: {elapsed / 60:.1f} minutes.")
    print(f"Plots saved to: {PLOT_DIR.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
