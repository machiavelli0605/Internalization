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

import pandas as pd

from config import PARENT_DATA_PATH, EXECUTION_DATA_PATH, OUTCOME_VARS, PLOT_DIR
from data_prep import load_parent_data, load_execution_data, sample_parent_orders
from parent_analysis import run_full_parent_analysis
from execution_analysis import run_full_execution_analysis
from plots import generate_parent_plots, generate_execution_plots

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
            print(f"    {row['outcome']:25s}  coef={row['coef']:+8.3f}  "
                  f"SE={row['se']:.3f}  p={row['pvalue']:.4f}{sig}  "
                  f"n={row['nobs']:,.0f}")


def _print_psm_summary(psm_results):
    """Print PSM outcome comparison summary."""
    nn = psm_results.get("nn_outcomes")
    if nn is not None and not nn.empty:
        print("\n  Nearest-Neighbor Matched Outcome Differences (Treated − Control):")
        for _, row in nn.iterrows():
            print(f"    {row['outcome']:25s}  diff={row['diff']:+8.3f}  "
                  f"CI=[{row['diff_ci_lower']:+.3f}, {row['diff_ci_upper']:+.3f}]  "
                  f"n_pairs={row['n_pairs']:,.0f}")

    ipw = psm_results.get("ipw_outcomes")
    if ipw is not None and not ipw.empty:
        print("\n  IPW-Weighted Outcome Means:")
        pivot = ipw.pivot(index="outcome", columns="group", values="weighted_mean")
        if "treated" in pivot.columns and "control" in pivot.columns:
            for outcome in pivot.index:
                t_val = pivot.loc[outcome, "treated"]
                c_val = pivot.loc[outcome, "control"]
                print(f"    {outcome:25s}  treated={t_val:+8.3f}  "
                      f"control={c_val:+8.3f}  diff={t_val - c_val:+8.3f}")


def _print_dose_response_summary(dr_results):
    """Print dose-response table."""
    for outcome, dr in dr_results.items():
        print(f"\n  {outcome}:")
        for _, row in dr.iterrows():
            print(f"    {row['bucket']:12s}  adj_mean={row['adj_mean']:+8.3f}  "
                  f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                  f"n={row['n']:,.0f}")


def _print_markout_summary(exec_results):
    """Print markout curve summary."""
    signed = exec_results.get("signed_markouts")
    if signed is not None and not signed.empty:
        print("\n  Signed Markout Curves:")
        for is_int in [True, False]:
            label = "CRB" if is_int else "Non-CRB"
            sub = signed[signed["group"] == is_int].sort_values("horizon_sec")
            vals = "  ".join(
                f"{int(r['horizon_sec'])}s={r['mean']:+.3f}"
                for _, r in sub.iterrows()
            )
            print(f"    {label:10s}  {vals}")

    within = exec_results.get("within_order", {})
    paired = within.get("paired_diff")
    if paired is not None and not paired.empty:
        signed_p = paired[~paired["is_absolute"]].sort_values("horizon_sec")
        if not signed_p.empty:
            print("\n  Within-Order Paired Difference (CRB − Non-CRB, signed):")
            for _, row in signed_p.iterrows():
                print(f"    {int(row['horizon_sec'])}s:  diff={row['mean_diff']:+.3f}  "
                      f"CI=[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
                      f"n={row['n_orders']:,.0f}")


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
    parser.add_argument("--cluster-col", default="RIC",
                        help="Column for clustered standard errors")
    parser.add_argument("--exec-sample", type=int, default=5_000_000,
                        help="Sample size for execution-level KDE plots")
    parser.add_argument("--exclude-auctions", action="store_true",
                        help="Exclude auction fills from analysis")
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
        parent_df = load_parent_data(args.parent_data,
                                     exclude_auctions=args.exclude_auctions)
        print(f"  Loaded {len(parent_df):,} parent orders.")
        print(f"  isInt=True: {parent_df['isInt'].sum():,}  "
              f"({parent_df['isInt'].mean()*100:.1f}%)")
        print(f"  hasCRB: {parent_df['hasCRB'].sum():,}  "
              f"({parent_df['hasCRB'].mean()*100:.1f}%)")

        print("\nRunning parent-level analyses ...")
        parent_results = run_full_parent_analysis(
            parent_df, cluster_col=args.cluster_col
        )

        # --- Print summaries ---
        print("\n" + "-" * 60)
        print("REGRESSION RESULTS")
        print("-" * 60)
        _print_regression_summary(parent_results["regression"])

        print("\n" + "-" * 60)
        print("PROPENSITY SCORE RESULTS")
        print("-" * 60)
        _print_psm_summary(parent_results["psm"])

        print("\n" + "-" * 60)
        print("DOSE-RESPONSE RESULTS")
        print("-" * 60)
        _print_dose_response_summary(parent_results["dose_response"])

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
            parent_df = load_parent_data(args.parent_data,
                                         exclude_auctions=args.exclude_auctions)

        exec_results = run_full_execution_analysis(
            parent_df, exec_path=args.exec_data,
            exclude_auctions=args.exclude_auctions
        )

        # --- Print summaries ---
        print("\n" + "-" * 60)
        print("MARKOUT RESULTS")
        print("-" * 60)
        _print_markout_summary(exec_results)

        # --- Generate plots ---
        # For E6 (KDE), we need a sample loaded into memory
        print(f"\n  Loading execution sample (n={args.exec_sample:,}) for KDE plots ...")
        try:
            exec_sample = load_execution_data(args.exec_data,
                                              exclude_auctions=args.exclude_auctions)
            if len(exec_sample) > args.exec_sample:
                exec_sample = exec_sample.sample(n=args.exec_sample, random_state=42)
        except (MemoryError, Exception) as e:
            print(f"    Full load failed ({e.__class__.__name__}); sampling via chunks ...")
            from data_prep import iter_execution_chunks
            chunks = []
            total = 0
            for chunk in iter_execution_chunks(args.exec_data,
                                                exclude_auctions=args.exclude_auctions):
                n_take = min(len(chunk), args.exec_sample - total)
                if n_take <= 0:
                    break
                chunks.append(chunk.sample(n=n_take, random_state=42)
                              if n_take < len(chunk) else chunk)
                total += n_take
                if total >= args.exec_sample:
                    break
            exec_sample = pd.concat(chunks, ignore_index=True) if chunks else None

        generate_execution_plots(exec_results, exec_df_sample=exec_sample)

    # -----------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Elapsed: {elapsed / 60:.1f} minutes.")
    print(f"Plots saved to: {PLOT_DIR.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
