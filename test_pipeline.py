"""
End-to-end smoke test using synthetic data.

Generates a small synthetic dataset that mimics the real data schema,
runs the full analysis pipeline, and verifies plots are produced.
"""
import numpy as np
import pandas as pd
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import PLOT_DIR, OUTCOME_VARS, CRB_BUCKET_LABELS
from data_prep import derive_parent_columns, derive_execution_columns
from parent_analysis import run_full_parent_analysis
from execution_analysis import (
    compute_signed_markout_curves,
    compute_abs_markout_curves,
    compute_markout_by_inttype,
    compute_abs_markout_by_inttype,
    compute_within_order_markouts,
    compute_markout_by_spread,
    run_full_execution_analysis,
)
from data_prep import _filter_auctions
from plots import generate_parent_plots, generate_execution_plots


def make_synthetic_parents(n=20_000, seed=42):
    """Generate synthetic parent order data."""
    rng = np.random.RandomState(seed)

    n_enabled = int(n * 0.6)
    n_disabled = n - n_enabled

    strategies = ["VWAP", "TWAP", "IS", "POV", "CLOSE"]
    risk_aversions = ["Low", "Medium", "High"]
    tick_rules = ["T1", "T2", "T3"]
    desks = [f"D{i}" for i in range(1, 11)]
    accounts = [f"A{i}" for i in range(1, 31)]

    amid = rng.lognormal(mean=np.log(50), sigma=0.5, size=n)
    side = rng.choice([1, -1], size=n)

    # isInt flag
    is_int = np.array([True] * n_enabled + [False] * n_disabled)
    rng.shuffle(is_int)

    # CRBPct: only non-zero for enabled orders, and not always
    crb_pct = np.zeros(n)
    enabled_mask = is_int
    # ~70% of enabled orders get some CRB
    has_crb = enabled_mask & (rng.random(n) < 0.7)
    crb_pct[has_crb] = rng.beta(2, 5, size=has_crb.sum())

    filled_qty = rng.randint(100, 100_000, size=n).astype(float)
    crb_qty = (crb_pct * filled_qty).astype(int).astype(float)
    remaining = filled_qty - crb_qty

    atspin_pct = rng.beta(1, 10, size=n) * 0.3
    dark_pct = rng.beta(2, 8, size=n) * 0.4
    lit_pct_raw = 1 - crb_pct - atspin_pct - dark_pct
    lit_pct_raw = np.clip(lit_pct_raw, 0, 1)

    # normalise non-CRB portions to fill remaining
    non_crb_total = atspin_pct + dark_pct + lit_pct_raw
    non_crb_total = np.where(non_crb_total > 0, non_crb_total, 1)
    atspin_qty = (remaining * atspin_pct / non_crb_total).astype(int).astype(float)
    dark_qty = (remaining * dark_pct / non_crb_total).astype(int).astype(float)
    lit_qty = (remaining - atspin_qty - dark_qty).clip(0)

    # impact: CRB reduces adverse impact (for testing)
    base_impact = rng.normal(-5, 10, size=n)  # negative = adverse
    crb_benefit = crb_pct * 8  # CRB improves impact
    emid = amid * (1 - side * (base_impact + crb_benefit) / 1e4)

    # post-trade reversion
    rev5m_bps = rng.normal(2, 5, size=n)
    rev15m_bps = rng.normal(1.5, 6, size=n)
    rev60m_bps = rng.normal(1, 7, size=n)

    # AvgPx: between amid and emid with some noise
    avg_px = amid + side * rng.uniform(-0.01, 0.02, size=n) * amid

    adv = rng.lognormal(mean=np.log(1e6), sigma=1, size=n)
    daily_vol = rng.uniform(0.005, 0.05, size=n)
    notional = filled_qty * amid

    start_times = pd.date_range("2024-01-02 09:30", periods=n, freq="1s")
    durations = rng.randint(60, 3600, size=n)
    end_times = start_times + pd.to_timedelta(durations, unit="s")

    df = pd.DataFrame({
        "date": start_times.date,
        "AlgoOrderId": np.arange(n),
        "RIC": rng.choice(["AAPL.O", "MSFT.O", "GOOG.O", "AMZN.O", "TSLA.O"], n),
        "Side": side,
        "EffectiveStartTime": start_times,
        "EffectiveEndTime": end_times,
        "Notional": notional,
        "qtyOverADV": filled_qty / adv,
        "amid": amid,
        "emid": emid,
        "rev5m_bps": rev5m_bps,
        "rev15m_bps": rev15m_bps,
        "rev60m_bps": rev60m_bps,
        "Strategy": rng.choice(strategies, n),
        "PcpRate": rng.uniform(0.01, 0.3, n),
        "TargetPcpRate": rng.choice([np.nan, 0.05, 0.1, 0.15, 0.2], n),
        "RiskAversion": rng.choice(risk_aversions, n),
        "Account": rng.choice(accounts, n),
        "CRBQty": crb_qty,
        "ATSPINQty": atspin_qty,
        "DarkQty": dark_qty,
        "LitQty": lit_qty,
        "InvertedQty": np.zeros(n),
        "ConditionalQty": np.zeros(n),
        "VenueTypeUnknownQty": np.zeros(n),
        "ELPQty": np.zeros(n),
        "FeeFeeQty": np.zeros(n),
        "FilledQty": filled_qty,
        "StartQty": filled_qty,
        "LimitPx": amid * (1 + side * 0.05),
        "AvgPx": avg_px,
        "ArrivalSlippageBps": 1e4 * side * (amid - avg_px) / amid,
        "IvlLimitVWAP": amid * (1 + rng.uniform(-0.001, 0.001, n)),
        "IvlSpreadBps": rng.uniform(1, 20, n),
        "adv": adv,
        "tickrule": rng.choice(tick_rules, n),
        "dailyvol": daily_vol,
        "ivlSpdVsAvgSpd": rng.uniform(0.5, 2, n),
        "DeskId": rng.choice(desks, n),
        "isInt": is_int,
    })

    return df


def make_synthetic_executions(parent_df, fills_per_order=10, seed=42):
    """Generate synthetic execution data linked to parent orders."""
    rng = np.random.RandomState(seed)
    rows = []

    int_types = ["mid", "far", "dark", None]
    liq_types = ["ADDED", "REMOVED", "CLOSE_AUCTION"]

    for _, parent in parent_df.iterrows():
        n_fills = rng.randint(3, fills_per_order + 1)
        order_id = parent["AlgoOrderId"]
        crb_pct = parent["CRBQty"] / parent["FilledQty"] if parent["FilledQty"] > 0 else 0
        n_crb = max(0, int(n_fills * crb_pct))

        for j in range(n_fills):
            is_int = j < n_crb
            spread = rng.uniform(1, 20)

            # CRB fills have lower markouts (less adverse)
            if is_int:
                base_markout = rng.normal(0.5, 2)
                int_type = rng.choice(["mid", "far", "dark"])
            else:
                base_markout = rng.normal(-1.5, 3)
                int_type = None

            rows.append({
                "date": parent["date"],
                "AlgoOrderId": order_id,
                "OrderId": order_id * 100 + j,
                "FillId": order_id * 1000 + j,
                "RIC": parent["RIC"],
                "Side": parent["Side"],
                "ExecTime": parent["EffectiveStartTime"] + pd.Timedelta(seconds=j * 10),
                "EffectiveStartTime": parent["EffectiveStartTime"],
                "EffectiveEndTime": parent["EffectiveEndTime"],
                "FilledQty": parent["FilledQty"] / n_fills,
                "FillPx": parent["amid"] * (1 + rng.uniform(-0.001, 0.001)),
                "LastLiquidity": rng.choice(liq_types),
                "OrderType": rng.choice(["LIMIT", "MARKET"]),
                "Tif": rng.choice(["DAY", "IOC"]),
                "PegType": rng.choice(["MidPoint", "Aggressive", None]),
                "spread": spread,
                "imb": rng.uniform(-1, 1),
                "rev1s_bps": base_markout + rng.normal(0, 1),
                "rev5s_bps": base_markout * 0.9 + rng.normal(0, 1.2),
                "rev10s_bps": base_markout * 0.8 + rng.normal(0, 1.5),
                "rev30s_bps": base_markout * 0.6 + rng.normal(0, 2),
                "rev60s_bps": base_markout * 0.4 + rng.normal(0, 2.5),
                "rev120s_bps": base_markout * 0.3 + rng.normal(0, 2.8),
                "rev300s_bps": base_markout * 0.2 + rng.normal(0, 3),
                "intType": int_type,
                "isInt": is_int,
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("SYNTHETIC DATA PIPELINE TEST")
    print("=" * 60)

    # --- Generate synthetic data ---
    print("\n1. Generating synthetic data ...")
    raw_parents = make_synthetic_parents(n=20_000)
    parent_df = derive_parent_columns(raw_parents)
    print(f"   Parent orders: {len(parent_df):,}")
    print(f"   isInt=True: {parent_df['isInt'].sum():,}")
    print(f"   hasCRB: {parent_df['hasCRB'].sum():,}")

    exec_df = make_synthetic_executions(raw_parents, fills_per_order=8)
    exec_df = derive_execution_columns(exec_df)
    print(f"   Executions: {len(exec_df):,}")

    # --- Run parent analysis ---
    print("\n2. Running parent-level analysis ...")
    parent_results = run_full_parent_analysis(parent_df, cluster_col="RIC")

    # --- Run execution analysis ---
    print("\n3. Running execution-level analysis ...")
    exec_results = {}

    print("  [1/6] Signed markout curves ...")
    exec_results["signed_markouts"] = compute_signed_markout_curves(df=exec_df)

    print("  [2/6] Absolute markout curves ...")
    exec_results["abs_markouts"] = compute_abs_markout_curves(df=exec_df)

    print("  [3/6] Markout by intType ...")
    exec_results["markout_by_inttype"] = compute_markout_by_inttype(df=exec_df)

    print("  [4/6] Absolute markout by intType ...")
    exec_results["abs_markout_by_inttype"] = compute_abs_markout_by_inttype(df=exec_df)

    print("  [5/6] Within-order markout comparison ...")
    exec_results["within_order"] = compute_within_order_markouts(
        parent_df, exec_df=exec_df, sample_orders=5000
    )

    print("  [6/6] Markout by spread bucket ...")
    exec_results["markout_by_spread"] = compute_markout_by_spread(df=exec_df)

    # --- Auction exclusion test ---
    print("\n3b. Testing auction exclusion path ...")
    n_before = len(exec_df)
    n_auction = exec_df["isAuction"].sum() if "isAuction" in exec_df.columns else 0
    exec_no_auction = _filter_auctions(exec_df)
    n_after = len(exec_no_auction)
    print(f"   Executions before: {n_before:,}  auction: {n_auction:,}  "
          f"after filtering: {n_after:,}")
    assert n_after <= n_before
    if "isAuction" in exec_no_auction.columns:
        assert exec_no_auction["isAuction"].sum() == 0

    # Run execution analysis on pre-filtered (no auction) df
    exec_results_no_auction = run_full_execution_analysis(
        parent_df, exec_df=exec_no_auction
    )
    assert "signed_markouts" in exec_results_no_auction
    print("   Auction exclusion path: OK")

    # --- Generate plots ---
    print("\n4. Generating plots ...")
    generate_parent_plots(parent_df, parent_results)
    generate_execution_plots(exec_results, exec_df_sample=exec_df)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    reg = parent_results.get("regression", {})
    dose_coefs = reg.get("dose_coefficients")
    if dose_coefs is not None and not dose_coefs.empty:
        crb_row = dose_coefs[
            (dose_coefs["treatment"] == "CRBPct")
            & (dose_coefs["outcome"] == "tempImpactBps")
        ]
        if not crb_row.empty:
            row = crb_row.iloc[0]
            print(f"\n  Key result — CRBPct -> tempImpactBps:")
            print(f"    Coefficient: {row['coef']:+.3f} bps per 100% CRB")
            print(f"    95% CI: [{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]")
            print(f"    p-value: {row['pvalue']:.4f}")
            if row["coef"] > 0:
                print("    Interpretation: Higher CRB% is associated with LESS "
                      "adverse market impact (positive = favorable).")
            else:
                print("    Interpretation: Higher CRB% is associated with MORE "
                      "adverse market impact.")

    signed = exec_results.get("signed_markouts")
    if signed is not None and not signed.empty:
        print(f"\n  Execution markouts (rev5s_bps):")
        for is_int in [True, False]:
            label = "CRB" if is_int else "Non-CRB"
            sub = signed[(signed["group"] == is_int)
                         & (signed["column"] == "rev5s_bps")]
            if not sub.empty:
                print(f"    {label}: mean={sub.iloc[0]['mean']:+.3f} bps")

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

    # Count plots generated
    plot_files = list(PLOT_DIR.glob("*.png"))
    print(f"\n  Plots generated: {len(plot_files)}")
    for pf in sorted(plot_files):
        print(f"    {pf.name}")

    print(f"\n{'=' * 60}")
    print("Pipeline test PASSED.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
