"""
Data loading and preprocessing for internalization analysis.

Handles:
  - Parent order data: load, derive venue percentages, impact metrics, controls
  - Execution data: chunked loading, derived markout columns
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    CRB_BUCKET_EDGES,
    CRB_BUCKET_LABELS,
    EXEC_CHUNK_SIZE,
    EXECUTION_DATA_PATH,
    PARENT_DATA_PATH,
    REVERSION_HORIZONS_MIN,
    VENUE_QTY_COLS,
    get_no_auction_path,
)

# ===================================================================
# Auction filtering helper
# ===================================================================


def _filter_auctions(df):
    """Remove auction fills from execution DataFrame."""
    if "isAuction" in df.columns:
        return df[~df["isAuction"]].copy()
    return df


# ===================================================================
# Parent order data
# ===================================================================


def load_parent_data(path=None, exclude_auctions=False):
    """Load parent order parquet and derive all analysis columns."""
    path = path or PARENT_DATA_PATH
    if exclude_auctions:
        path = get_no_auction_path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    df = derive_parent_columns(df)
    return df


def derive_parent_columns(df):
    """Derive all analysis columns on the parent-order DataFrame.

    Assumes raw columns already exist (see config for expected names).
    Returns the same DataFrame with new columns appended.
    """
    df = df.copy()

    # -- Normalize isInt to boolean ----------------------------------------
    if "isInt" in df.columns:
        df["isInt"] = df["isInt"].astype(bool)

    # -- Venue percentages ------------------------------------------------
    safe_filled = df["FilledQty"].replace(0, np.nan)
    for col in VENUE_QTY_COLS:
        pct_col = col.replace("Qty", "Pct")
        df[pct_col] = (df[col] / safe_filled).fillna(0.0)

    # -- Treatment indicators ---------------------------------------------
    df["hasCRB"] = df["CRBPct"] > 0

    df["CRBPctBucket"] = pd.cut(
        df["CRBPct"],
        bins=CRB_BUCKET_EDGES,
        labels=CRB_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    # orders with exactly 0% internalization go to the "0%" bucket
    df.loc[df["CRBPct"] == 0, "CRBPctBucket"] = "0%"
    df["CRBPctBucket"] = pd.Categorical(
        df["CRBPctBucket"], categories=CRB_BUCKET_LABELS, ordered=True
    )
    df.loc[:, "log_qtyOverADV"] = np.log1p(df.loc[:, "qtyOverADV"])

    # -- Impact metrics ---------------------------------------------------
    # tempImpactBps = 1e4 * Side * (amid - emid) / amid
    #   positive = favorable (market moved with order)
    #   negative = adverse   (market moved against order)
    safe_amid = df["amid"].replace(0, np.nan)
    df["tempImpactBps"] = 1e4 * df["Side"] * (df["amid"] - df["emid"]) / safe_amid

    # Derive rev{x}m_mid from rev{x}m_bps:
    #   rev{x}m_bps ≈ 1e4 * Side * (emid - rev{x}m_mid) / emid
    #   => rev{x}m_mid = emid * (1 - Side * rev{x}m_bps / 1e4)
    #
    # permImpact{x}mBps = 1e4 * Side * (amid - rev{x}m_mid) / amid
    for x in REVERSION_HORIZONS_MIN:
        rev_col = f"rev{x}m_bps"
        mid_col = f"rev{x}m_mid"
        perm_col = f"permImpact{x}mBps"

        if rev_col not in df.columns:
            continue

        df[mid_col] = df["emid"] * (1 - df["Side"] * df[rev_col] / 1e4)
        df[perm_col] = 1e4 * df["Side"] * (df["amid"] - df[mid_col]) / safe_amid

    # -- Duration ---------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["EffectiveStartTime"]):
        df["EffectiveStartTime"] = pd.to_datetime(df["EffectiveStartTime"])
    if not pd.api.types.is_datetime64_any_dtype(df["EffectiveEndTime"]):
        df["EffectiveEndTime"] = pd.to_datetime(df["EffectiveEndTime"])
    df["duration_mins"] = (
        (df["EffectiveEndTime"] - df["EffectiveStartTime"]).dt.total_seconds() / 60
    ).clip(lower=0)
    df["start_mins"] = (df["EffectiveStartTime"].dt.hour * 60) + (
        df["EffectiveStartTime"].dt.minute
    )

    # -- Log transforms ---------------------------------------------------
    df["log_notional"] = np.log1p(df["Notional"].clip(lower=0))
    df["log_adv"] = np.log1p(df["adv"].clip(lower=0))

    return df


# ===================================================================
# Execution data
# ===================================================================


def load_execution_data(path=None, columns=None, exclude_auctions=False):
    """Load execution parquet (full load — use only if it fits in memory)."""
    path = path or EXECUTION_DATA_PATH
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    df = derive_execution_columns(df)
    if exclude_auctions:
        df = _filter_auctions(df)
    return df


def derive_execution_columns(df):
    """Derive analysis columns on execution DataFrame."""
    df = df.copy()

    # Normalize isInt to boolean
    if "isInt" in df.columns:
        df["isInt"] = df["isInt"].astype(bool)

    # Absolute markouts
    rev_cols = [c for c in df.columns if c.startswith("rev") and c.endswith("s_bps")]
    for col in rev_cols:
        df[f"abs_{col}"] = df[col].abs()

    return df


def iter_execution_chunks(
    path=None, chunksize=None, columns=None, exclude_auctions=False, filters=None
):
    """Yield execution data in chunks using pyarrow row groups.

    If the parquet file has row groups smaller than *chunksize*, each
    row group is yielded as a separate DataFrame.  Otherwise we read
    the whole file and split manually.

    Yields
    ------
    pd.DataFrame – one chunk at a time, with derived columns applied.
    """
    path = path or EXECUTION_DATA_PATH
    chunksize = chunksize or EXEC_CHUNK_SIZE
    df = pd.read_parquet(path, engine="pyarrow", columns=columns, filters=filters)
    if exclude_auctions:
        df = _filter_auctions(df)
    return df


# ===================================================================
# Sampling helpers
# ===================================================================


def sample_parent_orders(df, n=500_000, seed=42):
    """Sample parent orders for computationally expensive analyses."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def load_executions_for_orders(path, columns=None, exclude_auctions=False):
    """Load executions for a specific set of AlgoOrderIds.

    Uses predicate pushdown where possible (parquet filters).
    """
    path = path or EXECUTION_DATA_PATH
    df = iter_execution_chunks(path, columns=columns, exclude_auctions=exclude_auctions)
    return df
