"""
Execution-level analyses for internalization market impact study.

All functions are designed to work with chunked processing for the 550M-row
execution dataset.  Each "compute_*" function can either:
  (a) accept a pre-loaded DataFrame, or
  (b) iterate over chunks via ``iter_execution_chunks`` and accumulate stats.

Analyses:
  A. Signed markout curves (CRB vs non-CRB)
  B. Absolute markout curves
  C. Markout by intType
  D. Within-order markout comparison
  E. Markout by spread bucket
"""
import numpy as np
import pandas as pd

from config import MARKOUT_HORIZONS_SEC, EXEC_CHUNK_SIZE
from data_prep import iter_execution_chunks, load_executions_for_orders
from utils import bootstrap_mean_ci


# ===================================================================
# Helpers
# ===================================================================

def _rev_cols(prefix="rev", suffix="s_bps", horizons=None):
    """Return list of reversion column names for the given horizons."""
    horizons = horizons or MARKOUT_HORIZONS_SEC
    return [f"{prefix}{h}{suffix}" for h in horizons]


def _abs_rev_cols(horizons=None):
    """Absolute markout column names."""
    horizons = horizons or MARKOUT_HORIZONS_SEC
    return [f"abs_rev{h}s_bps" for h in horizons]


def _available(df, cols):
    """Return the subset of *cols* that exist in *df*."""
    return [c for c in cols if c in df.columns]


# ===================================================================
# Core accumulator: group-level markout stats
# ===================================================================

class MarkoutAccumulator:
    """Online accumulator for computing group-level markout means & CIs.

    Accumulates (sum, sum_of_squares, count) per group so that the final
    mean & SE can be computed without loading all data at once.
    """

    def __init__(self, group_col, value_cols):
        self.group_col = group_col
        self.value_cols = value_cols
        self._sum = {}   # {group: {col: float}}
        self._ssq = {}   # sum of squares
        self._cnt = {}   # count

    def add_chunk(self, df):
        for grp, sub in df.groupby(self.group_col, observed=True):
            if grp not in self._sum:
                self._sum[grp] = {c: 0.0 for c in self.value_cols}
                self._ssq[grp] = {c: 0.0 for c in self.value_cols}
                self._cnt[grp] = {c: 0 for c in self.value_cols}
            for c in self.value_cols:
                if c not in sub.columns:
                    continue
                vals = sub[c].dropna().values
                self._sum[grp][c] += vals.sum()
                self._ssq[grp][c] += (vals ** 2).sum()
                self._cnt[grp][c] += len(vals)

    def result(self):
        """Return DataFrame with columns: group, horizon, mean, se, ci_lower, ci_upper, n."""
        rows = []
        for grp in sorted(self._sum.keys(), key=str):
            for c in self.value_cols:
                n = self._cnt[grp][c]
                if n == 0:
                    continue
                mean = self._sum[grp][c] / n
                var = self._ssq[grp][c] / n - mean ** 2
                se = np.sqrt(max(var, 0) / n) if n > 1 else np.nan
                rows.append({
                    "group": grp,
                    "column": c,
                    "mean": mean,
                    "se": se,
                    "ci_lower": mean - 1.96 * se,
                    "ci_upper": mean + 1.96 * se,
                    "n": n,
                })
        return pd.DataFrame(rows)


# ===================================================================
# A.  Signed markout curves
# ===================================================================

def compute_signed_markout_curves(exec_path=None, df=None, horizons=None):
    """Compute mean signed markouts (rev{x}s_bps) by isInt group.

    Parameters
    ----------
    exec_path : path – parquet file (chunked reading)
    df : DataFrame – if provided, use directly instead of chunked reading
    horizons : list[int] – markout horizons in seconds

    Returns
    -------
    DataFrame with columns: group, column, mean, se, ci_lower, ci_upper, n
    """
    cols = _rev_cols(horizons=horizons)
    acc = MarkoutAccumulator("isInt", cols)

    if df is not None:
        acc.add_chunk(df)
    else:
        for chunk in iter_execution_chunks(exec_path):
            acc.add_chunk(chunk)

    result = acc.result()
    # add horizon for convenience
    result["horizon_sec"] = result["column"].str.extract(r"rev(\d+)s_bps").astype(float)
    return result


# ===================================================================
# B.  Absolute markout curves
# ===================================================================

def compute_abs_markout_curves(exec_path=None, df=None, horizons=None):
    """Compute mean |rev{x}s_bps| by isInt group."""
    cols = _abs_rev_cols(horizons=horizons)
    acc = MarkoutAccumulator("isInt", cols)

    if df is not None:
        acc.add_chunk(df)
    else:
        for chunk in iter_execution_chunks(exec_path):
            acc.add_chunk(chunk)

    result = acc.result()
    result["horizon_sec"] = result["column"].str.extract(r"abs_rev(\d+)s_bps").astype(float)
    return result


# ===================================================================
# C.  Markout by intType
# ===================================================================

def compute_markout_by_inttype(exec_path=None, df=None, horizons=None):
    """Compute mean signed markouts grouped by intType.

    Non-CRB fills are labelled "non-CRB" in the intType column.
    """
    cols = _rev_cols(horizons=horizons)
    acc = MarkoutAccumulator("_inttype_label", cols)

    def _label_chunk(chunk):
        chunk = chunk.copy()
        chunk["_inttype_label"] = chunk["intType"].fillna("non-CRB")
        # for non-internalized fills, ensure label is "non-CRB"
        chunk.loc[chunk["isInt"] != True, "_inttype_label"] = "non-CRB"
        return chunk

    if df is not None:
        acc.add_chunk(_label_chunk(df))
    else:
        for chunk in iter_execution_chunks(exec_path):
            acc.add_chunk(_label_chunk(chunk))

    result = acc.result()
    result["horizon_sec"] = result["column"].str.extract(r"rev(\d+)s_bps").astype(float)
    return result


def compute_abs_markout_by_inttype(exec_path=None, df=None, horizons=None):
    """Compute mean |rev{x}s_bps| grouped by intType."""
    cols = _abs_rev_cols(horizons=horizons)
    acc = MarkoutAccumulator("_inttype_label", cols)

    def _label_chunk(chunk):
        chunk = chunk.copy()
        chunk["_inttype_label"] = chunk["intType"].fillna("non-CRB")
        chunk.loc[chunk["isInt"] != True, "_inttype_label"] = "non-CRB"
        return chunk

    if df is not None:
        acc.add_chunk(_label_chunk(df))
    else:
        for chunk in iter_execution_chunks(exec_path):
            acc.add_chunk(_label_chunk(chunk))

    result = acc.result()
    result["horizon_sec"] = result["column"].str.extract(r"abs_rev(\d+)s_bps").astype(float)
    return result


# ===================================================================
# D.  Within-order markout comparison
# ===================================================================

def compute_within_order_markouts(parent_df, exec_path=None, exec_df=None,
                                  horizons=None, sample_orders=200_000,
                                  seed=42):
    """Compare CRB vs non-CRB fill markouts within the same parent order.

    Strategy:
      1. Identify parent orders with mixed fills (both CRB and non-CRB).
      2. For those orders, compute mean markout separately for CRB and non-CRB fills.
      3. Compute the paired within-order difference.

    Parameters
    ----------
    parent_df : DataFrame – parent orders (used to identify mixed-fill orders)
    exec_path : path – executions parquet
    exec_df : DataFrame – pre-loaded executions (overrides exec_path)
    horizons : list[int]
    sample_orders : int – max parent orders to use (for memory)
    seed : int

    Returns
    -------
    dict:
      - "paired_diff": DataFrame with horizon, mean_diff, ci_lower, ci_upper
      - "within_order_stats": DataFrame with per-order mean CRB vs non-CRB markouts
    """
    rev_cols = _rev_cols(horizons=horizons)
    abs_cols = _abs_rev_cols(horizons=horizons)
    all_markout_cols = rev_cols + abs_cols

    # Step 1: find mixed-fill orders (hasCRB and CRBPct < 1)
    mixed = parent_df[(parent_df["hasCRB"]) & (parent_df["CRBPct"] < 1.0)]
    order_ids = mixed["AlgoOrderId"].values

    if sample_orders and len(order_ids) > sample_orders:
        rng = np.random.RandomState(seed)
        order_ids = rng.choice(order_ids, size=sample_orders, replace=False)

    order_set = set(order_ids)

    # Step 2: load executions for these orders
    if exec_df is not None:
        execs = exec_df[exec_df["AlgoOrderId"].isin(order_set)].copy()
    else:
        execs = load_executions_for_orders(order_set, exec_path)

    if execs.empty:
        return {"paired_diff": pd.DataFrame(), "within_order_stats": pd.DataFrame()}

    # Step 3: compute per-order, per-group means
    avail = _available(execs, all_markout_cols)
    grouped = (
        execs.groupby(["AlgoOrderId", "isInt"], observed=True)[avail]
        .mean()
        .reset_index()
    )

    crb_means = grouped[grouped["isInt"] == True].set_index("AlgoOrderId")
    non_crb_means = grouped[grouped["isInt"] != True].set_index("AlgoOrderId")

    common = crb_means.index.intersection(non_crb_means.index)
    if len(common) == 0:
        return {"paired_diff": pd.DataFrame(), "within_order_stats": pd.DataFrame()}

    diff = crb_means.loc[common, avail] - non_crb_means.loc[common, avail]

    # Step 4: summarize paired differences
    rows = []
    for col in avail:
        vals = diff[col].dropna().values
        if len(vals) == 0:
            continue
        mean_d, ci_lo, ci_hi = bootstrap_mean_ci(vals, n_boot=1000)
        is_abs = col.startswith("abs_")
        horizon = int(col.split("rev")[1].split("s_bps")[0])
        rows.append({
            "column": col,
            "horizon_sec": horizon,
            "is_absolute": is_abs,
            "mean_diff": mean_d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_orders": len(vals),
        })

    paired_diff = pd.DataFrame(rows)

    # Also return the per-order stats for plotting
    within_stats = pd.DataFrame({
        "AlgoOrderId": common,
        **{f"crb_{c}": crb_means.loc[common, c].values for c in avail},
        **{f"noncrb_{c}": non_crb_means.loc[common, c].values for c in avail},
    })

    return {"paired_diff": paired_diff, "within_order_stats": within_stats}


# ===================================================================
# E.  Markout by spread bucket
# ===================================================================

def compute_markout_by_spread(exec_path=None, df=None, horizons=None,
                              n_buckets=5):
    """Compute markout curves sliced by spread quintile.

    Returns
    -------
    DataFrame with columns: spread_bucket, isInt, column, horizon_sec, mean, se, ...
    """
    rev_cols = _rev_cols(horizons=horizons)
    acc = MarkoutAccumulator("_spread_isint", rev_cols)

    def _bucket_chunk(chunk, edges=None):
        chunk = chunk.copy()
        if edges is not None:
            chunk["_spread_bucket"] = pd.cut(chunk["spread"], bins=edges,
                                              include_lowest=True)
        else:
            chunk["_spread_bucket"] = pd.qcut(chunk["spread"], q=n_buckets,
                                               duplicates="drop")
        chunk["_spread_isint"] = (
            chunk["_spread_bucket"].astype(str) + " | "
            + chunk["isInt"].astype(str)
        )
        return chunk

    if df is not None:
        # compute global spread quintile edges
        edges = np.nanpercentile(df["spread"], np.linspace(0, 100, n_buckets + 1))
        edges = np.unique(edges)
        acc.add_chunk(_bucket_chunk(df, edges))
    else:
        # two-pass: first compute global spread edges, then accumulate
        spread_sample = []
        for chunk in iter_execution_chunks(exec_path):
            spread_sample.append(chunk["spread"].dropna().sample(
                n=min(100_000, len(chunk)), random_state=42
            ))
        spread_all = pd.concat(spread_sample)
        edges = np.nanpercentile(spread_all, np.linspace(0, 100, n_buckets + 1))
        edges = np.unique(edges)

        for chunk in iter_execution_chunks(exec_path):
            acc.add_chunk(_bucket_chunk(chunk, edges))

    result = acc.result()
    # parse composite key back into separate columns
    result[["spread_bucket", "isInt"]] = result["group"].str.split(" \\| ", expand=True)
    result["isInt"] = result["isInt"].map({"True": True, "False": False})
    result["horizon_sec"] = result["column"].str.extract(r"rev(\d+)s_bps").astype(float)
    return result


# ===================================================================
# Convenience: run everything
# ===================================================================

def run_full_execution_analysis(parent_df, exec_path=None, exec_df=None):
    """Run all execution-level analyses.

    Pass either exec_path (for chunked reading) or exec_df (pre-loaded).

    Returns
    -------
    dict of analysis results
    """
    kw = {"exec_path": exec_path} if exec_df is None else {"df": exec_df}

    print("  [1/6] Signed markout curves ...")
    signed = compute_signed_markout_curves(**kw)

    print("  [2/6] Absolute markout curves ...")
    absolute = compute_abs_markout_curves(**kw)

    print("  [3/6] Markout by intType ...")
    by_inttype = compute_markout_by_inttype(**kw)

    print("  [4/6] Absolute markout by intType ...")
    abs_by_inttype = compute_abs_markout_by_inttype(**kw)

    print("  [5/6] Within-order markout comparison ...")
    within = compute_within_order_markouts(parent_df, exec_path=exec_path,
                                           exec_df=exec_df)

    print("  [6/6] Markout by spread bucket ...")
    by_spread = compute_markout_by_spread(**kw)

    return {
        "signed_markouts": signed,
        "abs_markouts": absolute,
        "markout_by_inttype": by_inttype,
        "abs_markout_by_inttype": abs_by_inttype,
        "within_order": within,
        "markout_by_spread": by_spread,
    }
