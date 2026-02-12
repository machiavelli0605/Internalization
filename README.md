# Principal Internalization Market Impact Analysis

Quasi-experimental framework to quantify whether principal internalization (CRB) reduces market impact for parent orders, using historical data only (no A/B test).

## Quick Start

```bash
# Install dependencies
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn pyarrow

# Run the smoke test with synthetic data
python test_pipeline.py

# Run on real data
python run_analysis.py --parent-data path/to/parents.parquet --exec-data path/to/executions.parquet
```

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Paths, constants, bucket definitions, control variable lists, plot settings |
| `utils.py` | Regression wrappers, propensity score matching/IPW, SMD, bootstrap CIs |
| `data_prep.py` | Data loading, column derivation, chunked execution reader (for 550M rows) |
| `parent_analysis.py` | Parent-level analyses: descriptive, OLS regression, PSM, dose-response |
| `execution_analysis.py` | Execution-level analyses: markout curves, within-order comparison |
| `plots.py` | All 15 visualization functions |
| `run_analysis.py` | Main CLI orchestration script |
| `test_pipeline.py` | End-to-end smoke test using synthetic data |

## Usage

### Full analysis (parent + execution)

```bash
python run_analysis.py --parent-data data/parents.parquet --exec-data data/executions.parquet
```

### Parent-level only (faster, no execution data needed)

```bash
python run_analysis.py --parent-only --parent-data data/parents.parquet
```

### Execution-level only

```bash
python run_analysis.py --exec-only --parent-data data/parents.parquet --exec-data data/executions.parquet
```

Note: `--parent-data` is still needed for `--exec-only` because the within-order analysis requires parent order metadata.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--parent-data` | `data/parent_orders.parquet` | Path to parent orders parquet file |
| `--exec-data` | `data/executions.parquet` | Path to executions parquet file |
| `--parent-only` | off | Skip execution-level analysis |
| `--exec-only` | off | Skip parent-level analysis |
| `--cluster-col` | `RIC` | Column for clustered standard errors in regressions |
| `--exec-sample` | `5000000` | Sample size for execution KDE plots (E6) |

## Configuration

Edit `config.py` to adjust:

- **`CRB_BUCKET_EDGES`** / **`CRB_BUCKET_LABELS`** — how CRBPct is binned for dose-response
- **`CONTINUOUS_CONTROLS`** — continuous covariates in regressions
- **`CATEGORICAL_FE`** / **`ENTITY_FE`** — fixed effects in regressions
- **`PSM_COVARIATES`** — variables used for propensity score estimation
- **`OUTCOME_VARS`** — which outcome metrics to analyze
- **`EXEC_CHUNK_SIZE`** — rows per chunk when streaming executions (tune for your memory)
- **`REGRESSION_SAMPLE_SIZE`** — set to an integer (e.g. `2_000_000`) to subsample for regressions; `None` uses all data

## Data Requirements

### Parent orders (parquet)

Expected columns:

| Column | Type | Description |
|---|---|---|
| `date` | date | Trade date |
| `AlgoOrderId` | int/str | Unique parent order identifier |
| `RIC` | str | Instrument identifier |
| `Side` | int | +1 for buy, -1 for sell |
| `EffectiveStartTime` | datetime | Order start time |
| `EffectiveEndTime` | datetime | Order end time |
| `Notional` | float | Order notional value |
| `qtyOverADV` | float | Order qty / average daily volume |
| `amid` | float | Arrival mid price |
| `emid` | float | End mid price |
| `rev5m_bps`, `rev15m_bps`, `rev60m_bps` | float | Post-trade reversion in bps (positive = price reverted) |
| `Strategy` | str | Algo strategy (categorical) |
| `PcpRate` | float | Realized participation rate |
| `TargetPcpRate` | float | Target participation rate (nullable) |
| `RiskAversion` | str | Urgency category |
| `Account` | str | Client account |
| `CRBQty` | float | Qty internalized principally |
| `ATSPINQty` | float | Qty internalized via other internal venue |
| `DarkQty` | float | Qty filled in dark venues |
| `LitQty` | float | Qty filled in lit venues |
| `InvertedQty` | float | Qty filled in inverted lit venues |
| `ConditionalQty` | float | Qty filled in conditional venues |
| `VenueTypeUnknownQty` | float | Qty filled in unknown venues |
| `FilledQty` | float | Total filled qty |
| `StartQty` | float | Original order qty |
| `AvgPx` | float | Share-weighted average fill price |
| `ArrivalSlippageBps` | float | `1e4 * Side * (amid - AvgPx) / amid` |
| `IvlSpreadBps` | float | Average spread during execution (bps) |
| `adv` | float | Average daily volume |
| `tickrule` | str | Tick size category |
| `dailyvol` | float | Average daily volatility |
| `ivlSpdVsAvgSpd` | float | Execution spread vs historical average |
| `DeskId` | str | Trading desk identifier |
| `isInt` | bool | Whether principal internalization was **enabled** for this order |

Venue quantities must satisfy: `CRBQty + ATSPINQty + DarkQty + LitQty + InvertedQty + ConditionalQty + VenueTypeUnknownQty = FilledQty`.

### Executions (parquet)

Expected columns:

| Column | Type | Description |
|---|---|---|
| `date` | date | Trade date |
| `AlgoOrderId` | int/str | Parent order identifier (join key) |
| `OrderId` | int/str | SOR child order identifier |
| `FillId` | int/str | Unique fill identifier |
| `RIC` | str | Instrument identifier |
| `Side` | int | +1 for buy, -1 for sell |
| `ExecTime` | datetime | Fill timestamp |
| `FilledQty` | float | Fill quantity |
| `FillPx` | float | Fill price |
| `LastLiquidity` | str | Liquidity type (added/removed/auction etc.) |
| `spread` | float | Spread in bps at time of fill |
| `imb` | float | Order book imbalance (-1 to 1) |
| `rev{x}s_bps` | float | Post-trade reversion after x seconds (1, 5, 10, 30, 60, 300) |
| `intType` | str/null | Internalization type (mid, far, dark, etc.; null for non-CRB) |
| `isInt` | bool | Whether this fill was principally internalized |

## Sign Conventions

All outcome metrics use a consistent convention: **positive = favorable for the order**.

| Metric | Formula | Positive means |
|---|---|---|
| `ArrivalSlippageBps` | `1e4 * Side * (amid - AvgPx) / amid` | Execution beat arrival price |
| `tempImpactBps` | `1e4 * Side * (amid - emid) / amid` | Market moved favorably during order |
| `rev{x}m_bps` | `~1e4 * Side * (emid - rev{x}m_mid) / emid` | Price reverted after order ended |
| `permImpact{x}mBps` | `1e4 * Side * (amid - rev{x}m_mid) / amid` | Favorable post-trade price vs arrival |
| `rev{x}s_bps` (exec) | signed post-trade reversion after x sec | Fill price was favorable |

Relationship: `permImpact{x}mBps ≈ tempImpactBps + rev{x}m_bps`

## Methodology

### Parent-Level Analyses

**1. ITT Regression (Intention-to-Treat)**

Compares orders where the feature was enabled (`isInt=True`) vs disabled, controlling for confounders:

```
Outcome_i = β₀ + β₁·isInt_i + γ·Controls_i + FixedEffects + ε_i
```

β₁ > 0 indicates enabling the feature is associated with less adverse impact.

**2. Dose-Response Regression**

Uses realized CRBPct (continuous 0–1) as treatment, controlling for ATSPINPct to isolate CRB from other internalization:

```
Outcome_i = β₀ + β₁·CRBPct_i + β₂·ATSPINPct_i + γ·Controls_i + FE + ε_i
```

β₁ represents the bps improvement for going from 0% to 100% CRB.

**3. Propensity Score Matching / IPW**

Estimates propensity scores via logistic regression, then:
- **IPW (full data)**: Inverse-propensity-weighted outcome means with trimming
- **Nearest-neighbor matching (subsample)**: 1:1 matching with caliper

Both include balance diagnostics (standardized mean differences before/after).

**4. Non-Parametric Dose-Response**

Bins CRBPct into quantiles, residualizes outcomes on controls, and computes adjusted means per bin with bootstrap CIs. This captures non-linear relationships.

### Execution-Level Analyses

**1. Markout Curves** — Mean signed and absolute `rev{x}s_bps` across horizons for CRB vs non-CRB fills.

**2. By intType** — Breaks down CRB markouts by internalization type (mid, far, dark).

**3. Within-Order Comparison** — For parent orders with both CRB and non-CRB fills, computes paired within-order markout differences. This controls for all parent-level confounders (stock, strategy, time, account).

**4. By Spread Bucket** — Tests whether the CRB benefit varies with spread (hypothesis: larger benefit in wider-spread names).

### Controls

Continuous: `qtyOverADV`, `PcpRate`, `IvlSpreadBps`, `dailyvol`, `ivlSpdVsAvgSpd`, `log(Notional)`, `log(adv)`, `duration_mins`

Fixed effects: `Strategy`, `RiskAversion`, `Side`, `tickrule`, `DeskId`

Standard errors are clustered by `RIC` (configurable via `--cluster-col`).

## Output

All plots are saved to `output/plots/` as PNG files.

### Parent-Level Plots

| Plot | Description |
|---|---|
| `P1_crbpct_distribution.png` | Distribution of CRBPct among enabled orders |
| `P2_covariate_balance.png` | Love plot of SMD before matching |
| `P3_regression_coefficients.png` | Forest plot of treatment coefficients across all outcomes |
| `P4_dose_response.png` | Adjusted mean impact by CRBPct bucket |
| `P5a_psm_balance_ipw.png` | Covariate balance after IPW reweighting |
| `P5b_psm_balance_nn.png` | Covariate balance after nearest-neighbor matching |
| `P6_psm_outcomes.png` | IPW-weighted and NN-matched outcome comparisons |
| `P7_impact_by_bucket.png` | Box plots of impact distributions by CRBPct bucket |

### Execution-Level Plots

| Plot | Description |
|---|---|
| `E1_signed_markout_curves.png` | Post-trade signed markout decay: CRB vs non-CRB |
| `E2_abs_markout_curves.png` | Post-trade absolute markout decay: CRB vs non-CRB |
| `E3a_markout_by_inttype.png` | Signed markout curves by internalization type |
| `E3b_abs_markout_by_inttype.png` | Absolute markout curves by internalization type |
| `E4_within_order_markouts.png` | Within-order paired difference (CRB - non-CRB) |
| `E5_markout_by_spread.png` | Markout curves faceted by spread quintile |
| `E6_markout_distribution.png` | KDE overlay of markout distributions |

## Caveats

1. **Mechanical effect on ArrivalSlippageBps**: CRB fills at mid have zero spread cost, which mechanically improves AvgPx. `tempImpactBps` and `permImpact{x}mBps` are the cleaner metrics for true market impact since they measure price displacement, not execution quality.

2. **Selection bias**: Feature enablement is largely at the Account/DeskId level. Account fixed effects partially address this, but between-account differences may remain. The within-enabled dose-response analysis avoids this by only comparing among enabled orders.

3. **Endogeneity of CRBPct**: Realized CRBPct is not randomly assigned — market conditions that allow more internalization may independently correlate with lower impact. The ITT analysis using `isInt` partially addresses this since the flag is predetermined.

4. **Venue substitution**: If CRB replaces fills that would have gone to dark pools, the net benefit is CRB minus counterfactual dark impact, not CRB minus lit impact. Including `ATSPINPct` and other venue controls helps address this.
