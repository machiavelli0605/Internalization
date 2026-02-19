"""
Configuration for internalization market impact analysis.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data paths – update these to point at your actual parquet files
# ---------------------------------------------------------------------------
PARENT_DATA_PATH = Path("data/parent_orders.parquet")
EXECUTION_DATA_PATH = Path("data/executions.parquet")

OUTPUT_DIR = Path("output")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CRBPct bucket definitions
# ---------------------------------------------------------------------------
# First bin [0, 0.001) captures exact-zero orders; label overridden to "0%"
CRB_BUCKET_EDGES = [0, 0.001, 0.05, 0.15, 0.30, 0.50, 1.001]
CRB_BUCKET_LABELS = ["0%", "(0-5%]", "(5-15%]", "(15-30%]", "(30-50%]", "(50%+]"]

# ---------------------------------------------------------------------------
# Regression specification
# ---------------------------------------------------------------------------
CONTINUOUS_CONTROLS = [
    "qtyOverADV",
    "PcpRate",
    "IvlSpreadBps",
    "dailyvol",
    "ivlSpdVsAvgSpd",
    "log_notional",
    "log_adv",
    "duration_mins",
]

CATEGORICAL_FE = ["Strategy", "RiskAversion", "Side", "tickrule"]
ENTITY_FE = ["DeskId"]  # Account can be added but is high-cardinality

TREATMENT_COLS_ITT = ["isInt"]
TREATMENT_COLS_DOSE = ["CRBPct", "ATSPINPct"]

OUTCOME_VARS = [
    "tempImpactBps",
    "permImpact5mBps",
    "permImpact15mBps",
    "permImpact60mBps",
    "ArrivalSlippageBps",
]

# For PSM / matching
PSM_COVARIATES = [
    "qtyOverADV",
    "PcpRate",
    "IvlSpreadBps",
    "dailyvol",
    "ivlSpdVsAvgSpd",
    "log_notional",
    "log_adv",
    "duration_mins",
]

# ---------------------------------------------------------------------------
# Venue quantity columns (must sum to FilledQty)
# ---------------------------------------------------------------------------
VENUE_QTY_COLS = [
    "CRBQty",
    "ATSPINQty",
    "DarkQty",
    "LitQty",
    "InvertedQty",
    "ConditionalQty",
    "VenueTypeUnknownQty",
]

# ---------------------------------------------------------------------------
# Execution-level markout horizons (seconds)
# ---------------------------------------------------------------------------
MARKOUT_HORIZONS_SEC = [1, 5, 10, 30, 60, 120, 300]

# ---------------------------------------------------------------------------
# Parent-level reversion horizons (minutes)
# ---------------------------------------------------------------------------
REVERSION_HORIZONS_MIN = [5, 15, 60]

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
EXEC_CHUNK_SIZE = 5_000_000  # rows per chunk for execution processing

# Regression sample size (set to None to use full data)
REGRESSION_SAMPLE_SIZE = None

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
PLOT_DPI = 150
PLOT_FORMAT = "png"

# ---------------------------------------------------------------------------
# Auction exclusion helpers
# ---------------------------------------------------------------------------

def get_no_auction_path(path):
    """Derive the no-auction variant of a parquet file path."""
    p = Path(path)
    return p.with_name(p.stem + "_no_auction" + p.suffix)
COLOR_CRB = "#1f77b4"       # blue
COLOR_NON_CRB = "#ff7f0e"   # orange
COLOR_TREATED = "#2ca02c"    # green
COLOR_CONTROL = "#d62728"    # red
PALETTE_INTTYPE = "Set2"
