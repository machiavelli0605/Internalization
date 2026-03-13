"""
Plot market impact profile vs order completion percentage for
PSM-matched treatment and control groups.

Inputs:
  1. psm_matched_t.csv        – matched treated keys (date, AlgoOrderId)
  2. psm_matched_c_long.csv   – matched control keys (date, AlgoOrderId) with match_weight
  3. impact profile CSV        – per-(date, AlgoOrderId) impact at each completion pct,
     with columns: date, AlgoOrderId,
       Impact_till_0%, Impact_till_10%, ..., Impact_till_100%,
       weight_0%, weight_10%, ..., weight_100%  (ATT estimation weights)

Notes:
    - (date, AlgoOrderId) is treated as the unique key
    - For each completion pct p, we only include rows where BOTH:
        Impact_till_p% is non-null and weight_p% is non-null
    - Control combined weight per observation is:
        match_weight * weight_p%
      (match_weight from PSM matched control CSV, weight_p% from impact file)

Usage:
    python plot_impact_profile.py [--matched-dir output/results] [--out output/impact_profile.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPLETION_PCTS = list(range(0, 91, 10))  # 0, 10, 20, ..., 100
IMPACT_COLS = [f"{p}CumImpactFill" for p in COMPLETION_PCTS]
WEIGHT_COLS = [f"{p}CumNotional" for p in COMPLETION_PCTS]

N_BOOT = 1000
CI_LEVEL = 0.95

KEY_COLS = ["date", "AlgoOrderId"]


def read_impact(path: str) -> pd.DataFrame:
    """Read impact profiles from parquet or csv based on file extension."""
    p = Path(path)
    return pd.read_parquet(p, engine="pyarrow")


def load_matched_keys(matched_dir: Path):
    """
    Return:
        - treatment keys DataFrame with columns [date, AlgoOrderId]
        - control keys DataFrame with columns [date, AlgoOrderId, match_weight]
    """
    t = pd.read_csv(matched_dir / "psm_matched_t.csv")
    c = pd.read_csv(matched_dir / "psm_matched_c_long.csv")
    for df, name, req in [
        (t, "psm_matched_t.csv", KEY_COLS),
        (c, "psm_matched_c_long.csv", KEY_COLS + ["matched_weight"]),
    ]:
        missing = [col for col in req if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}. Expected at least: {req}"
            )
    t_keys = t[KEY_COLS].drop_duplicates()
    c_keys = c[KEY_COLS + ["match_weight"]].copy()

    return t_keys, c_keys


def _weighted_mean(vals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, returns NaN if no valid data."""
    if vals.size == 0:
        return np.nan
    wsum = np.sum(weights)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.nan
    return np.average(vals, weights=weights)


def weighted_avg_profile(
    df: pd.DataFrame,
    match_weight_col: pd.Series | None = None,
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

    Returns (means, ci_lo, ci_hi) arrays of length len(COMPLETION_PCTS).
    """
    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2
    means = np.full(len(COMPLETION_PCTS), np.nan, dtype=float)
    ci_lo = np.full(len(COMPLETION_PCTS), np.nan, dtype=float)
    ci_hi = np.full(len(COMPLETION_PCTS), np.nan, dtype=float)

    for i, (icol, wcol) in enumerate(zip(IMPACT_COLS, WEIGHT_COLS)):
        mask = df[icol].notna() & df[wcol].notna()

        if match_weights_col is not None:
            mask = mask & df[match_weights_col].notna()

        if mask.sum() == 0:
            continue

        vals = df.loc[mask, icol].to_numpy(dtype=float)
        base_w = df.loc[mask, icol].to_numpy(dtype=float)

        if match_weight_col is not None:
            mw = df.loc[mask, match_weight_col].to_numpy(dtype=float)
            w = base_w * mw
        else:
            w = base_w

        good = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        vals = vals[good]
        w = w[good]

        n = vals.size
        if n == 0:
            continue

        means[i] = _weighted_mean(vals, w)

        # Bootstrap: resample rows with replacement, keep their combined weights
        boot_stats = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_stats[b] = _weighted_mean(vals[idx], w[idx])

        ci_lo[i] = np.percentile(boot_stats, 100 * alpha)
        ci_hi[i] = np.percentile(boot_stats, 100 * (1 - alpha))

    return means, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser(description="Impact profile plot")
    parser.add_argument(
        "--impact-dir",
        default="data/mkt_impact_data",
        help="Path to impact profiles (.parquet or .csv)",
    )
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
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Bootstrap reps")
    parser.add_argument(
        "--ci", type=float, default=CI_LEVEL, help="CI level (e.g. 0.95)"
    )
    args = parser.parse_args()

    matched_dir = Path(args.matched_dir)
    impact = read_impact(args.impact_dir)

    # Validate columns
    missing_keys = [c for c in KEY_COLS if c not in impact.columns]
    if missing_keys:
        raise ValueError(
            f"Impact file missing key columns {missing_keys}. "
            f"Expected unique key (date, AlgoOrderId)."
        )

    treat_keys, ctrl_keys = load_matched_keys(matched_dir)

    # --- Treatment group ---
    treat_impact = impact.merge(treat_keys, on=KEY_COLS, how="inner")

    # --- Control group (merge to get match_weight, can have duplicates) ---
    ctrl_impact = ctrl_keys.merge(impact, on=KEY_COLS, how="inner")

    print(f"Treatment orders matched to impact data: {len(treat_impact)}")
    print(f"Control rows (with NN duplicates) matched: {len(ctrl_impact)}")

    treat_mean, treat_lo, treat_hi = weighted_avg_profile(
        treat_impact, match_weight_col=None, n_boot=args.n_boot, ci=args.ci, seed=42
    )
    ctrl_mean, ctrl_lo, ctrl_hi = weighted_avg_profile(
        ctrl_impact,
        match_weight_col="match_weight",
        n_boot=args.n_boot,
        ci=args.ci,
        seed=42,
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
