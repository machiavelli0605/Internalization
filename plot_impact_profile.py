"""
Plot market impact profile vs order completion percentage for
PSM-matched treatment and control groups.

Inputs:
  1. psm_matched_t.csv        – matched treated AlgoOrderIds (from run_analysis.py)
  2. psm_matched_c_long.csv   – matched control AlgoOrderIds with match_weight
  3. impact profile CSV        – per-AlgoOrder impact at each completion pct,
     with columns: date, AlgoOrderId,
       Impact_till_0%, Impact_till_10%, ..., Impact_till_100%,
       weight_0%, weight_10%, ..., weight_100%  (ATT estimation weights)

Usage:
    python plot_impact_profile.py --impact path/to/impact_profiles.csv
                                  [--matched-dir output/results]
                                  [--out output/plots/impact_profile.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPLETION_PCTS = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
IMPACT_COLS = [f"Impact_till_{p}%" for p in COMPLETION_PCTS]
WEIGHT_COLS = [f"weight_{p}%" for p in COMPLETION_PCTS]


def load_matched_ids(matched_dir: Path):
    """Return treatment AlgoOrderIds and control DataFrame with match_weight."""
    t = pd.read_csv(matched_dir / "psm_matched_t.csv")
    c = pd.read_csv(matched_dir / "psm_matched_c_long.csv")
    return t["AlgoOrderId"].unique(), c[["AlgoOrderId", "match_weight"]]


N_BOOT = 1000
CI_LEVEL = 0.95


def _weighted_mean(vals, weights):
    """Weighted mean, returns NaN if no valid data."""
    if len(vals) == 0:
        return np.nan
    return np.average(vals, weights=weights)


def weighted_avg_profile(
    impact_df: pd.DataFrame,
    match_weights: pd.Series | None = None,
    n_boot: int = N_BOOT,
    ci: float = CI_LEVEL,
    seed: int = 42,
):
    """Compute weighted-average impact with bootstrap CIs at each completion pct.

    For each completion pct p:
        numerator   = sum( Impact_till_p% * weight_p% * match_weight )
        denominator = sum( weight_p% * match_weight )

    weight_p% are ATT estimation weights from the impact data.
    match_weight (control group only) is the PSM nearest-neighbour weight.

    Rows with NaN impact at a given pct are excluded from that pct's average.

    Returns (means, ci_lo, ci_hi) arrays of length len(COMPLETION_PCTS).
    """
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    means = np.full(len(COMPLETION_PCTS), np.nan)
    ci_lo = np.full(len(COMPLETION_PCTS), np.nan)
    ci_hi = np.full(len(COMPLETION_PCTS), np.nan)

    for i, (icol, wcol) in enumerate(zip(IMPACT_COLS, WEIGHT_COLS)):
        mask = impact_df[icol].notna()
        n = mask.sum()
        if n == 0:
            continue
        w = impact_df.loc[mask, wcol].fillna(1.0).values
        if match_weights is not None:
            w = w * match_weights.loc[mask].values
        vals = impact_df.loc[mask, icol].values

        means[i] = _weighted_mean(vals, w)

        boot_stats = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_stats[b] = _weighted_mean(vals[idx], w[idx])
        ci_lo[i] = np.percentile(boot_stats, 100 * alpha)
        ci_hi[i] = np.percentile(boot_stats, 100 * (1 - alpha))

    return means, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser(description="Impact profile plot")
    parser.add_argument("--impact", required=True, help="Path to impact profiles CSV")
    parser.add_argument(
        "--matched-dir",
        default="output/results",
        help="Directory containing psm_matched_t/c_long CSVs",
    )
    parser.add_argument(
        "--out",
        default="output/plots/impact_profile.png",
        help="Output plot path",
    )
    args = parser.parse_args()

    matched_dir = Path(args.matched_dir)
    impact = pd.read_csv(args.impact)

    # Validate columns
    missing = [c for c in IMPACT_COLS + WEIGHT_COLS if c not in impact.columns]
    if missing:
        raise ValueError(f"Missing columns in impact CSV: {missing}")

    treat_ids, ctrl_df = load_matched_ids(matched_dir)

    # --- Treatment group ---
    treat_impact = impact[impact["AlgoOrderId"].isin(treat_ids)].copy()

    # --- Control group (merge to get match_weight, can have duplicates) ---
    ctrl_impact = ctrl_df.merge(impact, on="AlgoOrderId", how="inner")

    print(f"Treatment orders matched to impact data: {len(treat_impact)}")
    print(f"Control rows (with NN duplicates) matched: {len(ctrl_impact)}")

    treat_mean, treat_lo, treat_hi = weighted_avg_profile(treat_impact)
    ctrl_mean, ctrl_lo, ctrl_hi = weighted_avg_profile(
        ctrl_impact,
        match_weights=ctrl_impact["match_weight"].reset_index(drop=True),
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts = COMPLETION_PCTS

    ax.plot(pcts, treat_mean, marker="o", label="Treatment (CRB)", color="#2ca02c")
    ax.fill_between(pcts, treat_lo, treat_hi, color="#2ca02c", alpha=0.15)

    ax.plot(pcts, ctrl_mean, marker="s", label="Control (non-CRB)", color="#d62728")
    ax.fill_between(pcts, ctrl_lo, ctrl_hi, color="#d62728", alpha=0.15)

    ax.set_xlabel("Order Completion (%)")
    ax.set_ylabel("Market Impact (bps)")
    ax.set_title("Market Impact Profile vs Order Completion")
    ax.set_xticks(pcts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
