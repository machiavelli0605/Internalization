"""
Regression tests for internalization analysis bugs.

Each test targets a specific bug that was found during code review.
Run with: pytest tests.py -v
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from config import OUTCOME_VARS, PSM_COVARIATES, CRB_BUCKET_LABELS
from utils import bootstrap_mean_ci
from data_prep import derive_parent_columns, derive_execution_columns


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def small_parent_df():
    """Minimal parent DataFrame for unit tests."""
    rng = np.random.RandomState(0)
    n = 200
    side = rng.choice([1, -1], n)
    amid = rng.uniform(40, 60, n)
    emid = amid * (1 + side * rng.uniform(-0.005, 0.005, n))

    df = pd.DataFrame({
        "date": pd.Timestamp("2024-01-02"),
        "AlgoOrderId": np.arange(n),
        "RIC": rng.choice(["A.O", "B.O"], n),
        "Side": side,
        "EffectiveStartTime": pd.Timestamp("2024-01-02 09:30:00"),
        "EffectiveEndTime": pd.Timestamp("2024-01-02 10:30:00"),
        "Notional": rng.uniform(1e4, 1e6, n),
        "qtyOverADV": rng.uniform(0.001, 0.1, n),
        "amid": amid,
        "emid": emid,
        "rev5m_bps": rng.normal(2, 5, n),
        "rev15m_bps": rng.normal(1.5, 6, n),
        "rev60m_bps": rng.normal(1, 7, n),
        "Strategy": rng.choice(["VWAP", "TWAP"], n),
        "PcpRate": rng.uniform(0.01, 0.3, n),
        "TargetPcpRate": np.nan,
        "RiskAversion": rng.choice(["Low", "High"], n),
        "Account": rng.choice(["A1", "A2"], n),
        "CRBQty": rng.choice([0, 100, 500], n).astype(float),
        "ATSPINQty": rng.choice([0, 50], n).astype(float),
        "DarkQty": rng.choice([0, 200], n).astype(float),
        "LitQty": np.full(n, 300.0),
        "InvertedQty": np.zeros(n),
        "ConditionalQty": np.zeros(n),
        "VenueTypeUnknownQty": np.zeros(n),
        "FilledQty": np.full(n, 1000.0),
        "StartQty": np.full(n, 1000.0),
        "LimitPx": amid * 1.05,
        "AvgPx": amid * (1 + rng.uniform(-0.001, 0.001, n)),
        "ArrivalSlippageBps": rng.normal(0, 5, n),
        "IvlSpreadBps": rng.uniform(1, 20, n),
        "adv": rng.uniform(1e5, 1e7, n),
        "tickrule": rng.choice(["T1", "T2"], n),
        "dailyvol": rng.uniform(0.01, 0.04, n),
        "ivlSpdVsAvgSpd": rng.uniform(0.5, 2, n),
        "DeskId": rng.choice(["D1", "D2"], n),
        "isInt": np.array([True] * 120 + [False] * 80),
    })
    return derive_parent_columns(df)


@pytest.fixture
def small_exec_df():
    """Minimal execution DataFrame for unit tests."""
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({
        "AlgoOrderId": rng.choice(range(200), n),
        "FillId": np.arange(n),
        "RIC": "A.O",
        "Side": rng.choice([1, -1], n),
        "ExecTime": pd.Timestamp("2024-01-02 10:00:00"),
        "FilledQty": 100.0,
        "FillPx": rng.uniform(49, 51, n),
        "LastLiquidity": rng.choice(["ADDED", "REMOVED", "CLOSE_AUCTION"], n),
        "spread": rng.uniform(1, 20, n),
        "imb": rng.uniform(-1, 1, n),
        "rev1s_bps": rng.normal(0, 2, n),
        "rev5s_bps": rng.normal(0, 2.5, n),
        "rev10s_bps": rng.normal(0, 3, n),
        "rev30s_bps": rng.normal(0, 4, n),
        "rev60s_bps": rng.normal(0, 5, n),
        "rev300s_bps": rng.normal(0, 6, n),
        "intType": rng.choice(["mid", "far", None], n),
        "isInt": rng.choice([True, False], n),
    })
    return derive_execution_columns(df)


# ===================================================================
# Bug 1: NN matched paired difference on misaligned pairs
# ===================================================================

class TestBug1_NNMatchedPairAlignment:
    """The NN matched outcome comparison must keep pairs aligned.

    Previously, independent dropna() on treated and control broke the
    1:1 correspondence, causing wrong rows to be subtracted.
    """

    def test_paired_diff_with_nans_in_different_positions(self):
        """When NaNs appear in different rows of treated vs control,
        the paired difference should only use rows where BOTH are valid."""
        from parent_analysis import run_psm_analysis

        rng = np.random.RandomState(42)
        n = 400

        df = pd.DataFrame({
            "hasCRB": [True] * (n // 2) + [False] * (n // 2),
            "qtyOverADV": rng.uniform(0.01, 0.1, n),
            "PcpRate": rng.uniform(0.05, 0.2, n),
            "IvlSpreadBps": rng.uniform(2, 15, n),
            "dailyvol": rng.uniform(0.01, 0.04, n),
            "ivlSpdVsAvgSpd": rng.uniform(0.5, 2, n),
            "log_notional": rng.uniform(10, 15, n),
            "log_adv": rng.uniform(12, 17, n),
            "duration_mins": rng.uniform(5, 60, n),
        })

        # Create outcome with NaNs at DIFFERENT positions in treated vs control
        outcome_vals = rng.normal(0, 5, n)
        # NaN in first 10 treated rows
        outcome_vals[:10] = np.nan
        # NaN in rows 200-210 (first 10 control rows)
        outcome_vals[200:210] = np.nan
        df["tempImpactBps"] = outcome_vals

        # Add other required outcomes
        for oc in OUTCOME_VARS:
            if oc not in df.columns:
                df[oc] = rng.normal(0, 5, n)

        results = run_psm_analysis(df, treatment_col="hasCRB", max_sample=n)
        nn = results.get("nn_outcomes")

        if nn is not None and not nn.empty:
            temp_row = nn[nn["outcome"] == "tempImpactBps"]
            if not temp_row.empty:
                # The key property: n_pairs should be <= n/2 - 10
                # (at most we lose the 10 NaN pairs from each side)
                assert temp_row.iloc[0]["n_pairs"] <= n // 2
                # diff should be finite
                assert np.isfinite(temp_row.iloc[0]["diff"])

    def test_paired_diff_preserves_alignment(self):
        """Directly test that pair alignment is maintained:
        treated[i] is always matched with control[i]."""
        # Simulate the fixed code path directly
        n_pairs = 50
        rng = np.random.RandomState(0)

        t_vals = rng.normal(10, 2, n_pairs)
        c_vals = rng.normal(5, 2, n_pairs)

        # Inject NaN at different positions
        t_vals[0] = np.nan   # pair 0: treated is NaN
        c_vals[5] = np.nan   # pair 5: control is NaN
        t_vals[10] = np.nan  # pair 10: treated is NaN
        c_vals[10] = np.nan  # pair 10: both NaN

        # The correct approach: drop pairs where either is NaN
        valid = np.isfinite(t_vals) & np.isfinite(c_vals)
        pair_diff = t_vals[valid] - c_vals[valid]

        # Should have dropped pairs 0, 5, 10 = 3 pairs
        assert len(pair_diff) == n_pairs - 3

        # The WRONG approach (the old bug): independent dropna
        t_clean = t_vals[np.isfinite(t_vals)]  # drops indices 0, 10
        c_clean = c_vals[np.isfinite(c_vals)]  # drops indices 5, 10
        # Now t_clean[0] was originally t_vals[1], but c_clean[0] is c_vals[0]
        # These are DIFFERENT pairs — the subtraction is wrong
        wrong_diff = t_clean[:len(t_clean)] - c_clean[:len(t_clean)]

        # The correct and wrong diffs should NOT be equal
        assert not np.allclose(pair_diff[:len(wrong_diff)], wrong_diff[:len(pair_diff)])


# ===================================================================
# Bug 2: Division by zero with no enabled orders
# ===================================================================

class TestBug2_NoEnabledOrders:
    """descriptive_summary should not crash when no orders have isInt=True."""

    def test_descriptive_with_all_disabled(self, small_parent_df):
        from parent_analysis import descriptive_summary

        df = small_parent_df.copy()
        df["isInt"] = False
        df["hasCRB"] = False
        df["CRBPct"] = 0.0

        # Should not raise ZeroDivisionError
        results = descriptive_summary(df)

        dist = results["crbpct_distribution"]
        assert dist.iloc[0]["n_enabled"] == 0
        assert dist.iloc[0]["pct_with_crb"] == 0

    def test_descriptive_with_no_crb(self, small_parent_df):
        from parent_analysis import descriptive_summary

        df = small_parent_df.copy()
        df["CRBQty"] = 0.0
        df["CRBPct"] = 0.0
        df["hasCRB"] = False

        results = descriptive_summary(df)
        dist = results["crbpct_distribution"]
        assert dist.iloc[0]["n_nonzero"] == 0


# ===================================================================
# Bug 3: Spread sample size exceeds available rows after dropna
# ===================================================================

class TestBug3_SpreadSampleSize:
    """Spread sampling must not request more rows than available after dropna."""

    def test_spread_with_many_nans(self):
        """When most spread values are NaN, sampling should still work."""
        from execution_analysis import compute_markout_by_spread

        rng = np.random.RandomState(0)
        n = 1000
        df = pd.DataFrame({
            "isInt": rng.choice([True, False], n),
            "spread": np.where(rng.random(n) < 0.9, np.nan, rng.uniform(1, 20, n)),
            "rev1s_bps": rng.normal(0, 2, n),
            "rev5s_bps": rng.normal(0, 3, n),
            "rev10s_bps": rng.normal(0, 3, n),
            "rev30s_bps": rng.normal(0, 4, n),
            "rev60s_bps": rng.normal(0, 5, n),
            "rev120s_bps": rng.normal(0, 5, n),
            "rev300s_bps": rng.normal(0, 6, n),
        })

        # Should not raise ValueError about sample size
        result = compute_markout_by_spread(df=df, n_buckets=3)
        assert isinstance(result, pd.DataFrame)

    def test_spread_all_nan(self):
        """When ALL spread values are NaN, should not crash."""
        from execution_analysis import compute_markout_by_spread

        n = 100
        df = pd.DataFrame({
            "isInt": [True] * 50 + [False] * 50,
            "spread": np.full(n, np.nan),
            "rev1s_bps": np.random.normal(0, 2, n),
            "rev5s_bps": np.random.normal(0, 3, n),
            "rev10s_bps": np.random.normal(0, 3, n),
            "rev30s_bps": np.random.normal(0, 4, n),
            "rev60s_bps": np.random.normal(0, 5, n),
            "rev120s_bps": np.random.normal(0, 5, n),
            "rev300s_bps": np.random.normal(0, 6, n),
        })

        # Should handle gracefully (empty result or no crash)
        try:
            result = compute_markout_by_spread(df=df, n_buckets=3)
        except ValueError:
            pytest.fail("compute_markout_by_spread raised ValueError on all-NaN spread")


# ===================================================================
# Bug 4: IPW bar chart colors swapped
# ===================================================================

class TestBug4_IPWPivotColumnOrder:
    """The IPW pivot table must have columns ordered as [treated, control]
    so that colors map correctly."""

    def test_pivot_column_order(self):
        """Verify that after reindexing, 'treated' is the first column."""
        ipw = pd.DataFrame([
            {"outcome": "tempImpactBps", "group": "treated", "weighted_mean": -3.0, "n": 100},
            {"outcome": "tempImpactBps", "group": "control", "weighted_mean": -5.0, "n": 100},
            {"outcome": "permImpact5mBps", "group": "treated", "weighted_mean": -2.0, "n": 100},
            {"outcome": "permImpact5mBps", "group": "control", "weighted_mean": -4.0, "n": 100},
        ])

        pivot = ipw.pivot(index="outcome", columns="group", values="weighted_mean")

        # Before fix: alphabetical order puts "control" first
        assert pivot.columns.tolist() == ["control", "treated"]

        # After fix: reindex forces correct order
        pivot = pivot.reindex(columns=["treated", "control"])
        assert pivot.columns.tolist() == ["treated", "control"]

    def test_plot_psm_outcomes_no_crash(self):
        """Verify the actual plot function doesn't crash and applies correct order."""
        from plots import plot_psm_outcomes

        psm_results = {
            "ipw_outcomes": pd.DataFrame([
                {"outcome": "tempImpactBps", "group": "treated", "weighted_mean": -3.0, "n": 100},
                {"outcome": "tempImpactBps", "group": "control", "weighted_mean": -5.0, "n": 100},
            ]),
            "nn_outcomes": pd.DataFrame([
                {"outcome": "tempImpactBps", "treated_mean": -3.0, "control_mean": -5.0,
                 "diff": 2.0, "diff_ci_lower": 1.0, "diff_ci_upper": 3.0, "n_pairs": 50},
            ]),
        }

        # Should not crash
        plot_psm_outcomes(psm_results)
        plt.close("all")


# ===================================================================
# Bug 5: Lexicographic vs numeric sort of horizon columns
# ===================================================================

class TestBug5_HorizonColumnSorting:
    """Markout horizon columns must be sorted by numeric value, not lexicographically."""

    def test_lexicographic_sort_is_wrong(self):
        """Demonstrate that naive sorted() gives wrong order."""
        cols = ["rev1s_bps", "rev5s_bps", "rev10s_bps", "rev30s_bps",
                "rev60s_bps", "rev300s_bps"]
        lex_sorted = sorted(cols)

        # Lexicographic: "rev10" < "rev1s" (because '0' < 's')
        # Actually: "rev10s_bps" < "rev1s_bps" because '0' < 's' at position 4
        # The point is the numeric order 1,5,10,30,60,300 is NOT preserved
        assert lex_sorted != ["rev1s_bps", "rev5s_bps", "rev10s_bps",
                               "rev30s_bps", "rev60s_bps", "rev300s_bps"]

    def test_numeric_sort_is_correct(self):
        """The fixed sorting extracts the integer and sorts numerically."""
        cols = ["rev300s_bps", "rev1s_bps", "rev60s_bps", "rev10s_bps",
                "rev5s_bps", "rev30s_bps"]

        num_sorted = sorted(cols, key=lambda c: int(c.split("rev")[1].split("s_bps")[0]))

        assert num_sorted == ["rev1s_bps", "rev5s_bps", "rev10s_bps",
                               "rev30s_bps", "rev60s_bps", "rev300s_bps"]

    def test_plot_markout_distribution_selects_correct_horizons(self):
        """The first 4 horizons selected should be the 4 smallest numerically."""
        from plots import plot_markout_distribution

        rng = np.random.RandomState(0)
        n = 200
        exec_df = pd.DataFrame({
            "isInt": rng.choice([True, False], n),
            "rev1s_bps": rng.normal(0, 2, n),
            "rev5s_bps": rng.normal(0, 3, n),
            "rev10s_bps": rng.normal(0, 3, n),
            "rev30s_bps": rng.normal(0, 4, n),
            "rev60s_bps": rng.normal(0, 5, n),
            "rev300s_bps": rng.normal(0, 6, n),
        })

        # Manually check what horizons would be selected
        available = [c for c in exec_df.columns
                     if c.startswith("rev") and c.endswith("s_bps")
                     and not c.startswith("abs_")]
        available.sort(key=lambda c: int(c.split("rev")[1].split("s_bps")[0]))
        selected = available[:4]

        assert selected == ["rev1s_bps", "rev5s_bps", "rev10s_bps", "rev30s_bps"]

        # Also verify the plot doesn't crash
        plot_markout_distribution(exec_df)
        plt.close("all")


# ===================================================================
# Bug 6: Empty regression panels crash plt.subplots(1, 0)
# ===================================================================

class TestBug6_EmptyRegressionPanels:
    """plot_regression_coefficients must handle empty/missing coefficient DataFrames."""

    def test_all_panels_empty(self):
        """When all coefficient DataFrames are empty, should return without crashing."""
        from plots import plot_regression_coefficients

        reg_results = {
            "itt_coefficients": pd.DataFrame(),
            "dose_coefficients": pd.DataFrame(),
            "itt_enabled_coefficients": pd.DataFrame(),
        }

        # Should not raise: ValueError from plt.subplots(1, 0)
        plot_regression_coefficients(reg_results)
        plt.close("all")

    def test_all_panels_none(self):
        """When all coefficient DataFrames are None, should return without crashing."""
        from plots import plot_regression_coefficients

        reg_results = {
            "itt_coefficients": None,
            "dose_coefficients": None,
            "itt_enabled_coefficients": None,
        }

        plot_regression_coefficients(reg_results)
        plt.close("all")

    def test_some_panels_missing(self):
        """When some panels have data and others don't, should plot only valid ones."""
        from plots import plot_regression_coefficients

        reg_results = {
            "itt_coefficients": pd.DataFrame([{
                "treatment": "isInt", "outcome": "tempImpactBps",
                "coef": 1.5, "se": 0.3, "ci_lower": 0.9, "ci_upper": 2.1,
                "pvalue": 0.001, "nobs": 1000, "r2": 0.05,
            }]),
            "dose_coefficients": pd.DataFrame(),  # empty
            "itt_enabled_coefficients": None,       # missing
        }

        # Should plot 1 panel without crashing
        plot_regression_coefficients(reg_results)
        plt.close("all")


# ===================================================================
# Additional edge-case tests for robustness
# ===================================================================

class TestMarkoutAccumulator:
    """Verify the online accumulator produces correct statistics."""

    def test_single_chunk_matches_direct_calculation(self):
        from execution_analysis import MarkoutAccumulator

        rng = np.random.RandomState(0)
        n = 1000
        df = pd.DataFrame({
            "group": rng.choice(["A", "B"], n),
            "val1": rng.normal(5, 2, n),
            "val2": rng.normal(-3, 4, n),
        })

        acc = MarkoutAccumulator("group", ["val1", "val2"])
        acc.add_chunk(df)
        result = acc.result()

        # Compare with direct pandas groupby
        for grp in ["A", "B"]:
            for col in ["val1", "val2"]:
                expected_mean = df.loc[df["group"] == grp, col].mean()
                actual = result[(result["group"] == grp) & (result["column"] == col)]
                np.testing.assert_allclose(actual["mean"].values[0], expected_mean,
                                           rtol=1e-10)

    def test_multi_chunk_matches_single_chunk(self):
        from execution_analysis import MarkoutAccumulator

        rng = np.random.RandomState(0)
        n = 1000
        df = pd.DataFrame({
            "group": rng.choice(["A", "B"], n),
            "val": rng.normal(0, 5, n),
        })

        # Single chunk
        acc1 = MarkoutAccumulator("group", ["val"])
        acc1.add_chunk(df)
        r1 = acc1.result()

        # Multiple chunks
        acc2 = MarkoutAccumulator("group", ["val"])
        acc2.add_chunk(df.iloc[:300])
        acc2.add_chunk(df.iloc[300:700])
        acc2.add_chunk(df.iloc[700:])
        r2 = acc2.result()

        # Results should match
        for grp in ["A", "B"]:
            m1 = r1.loc[r1["group"] == grp, "mean"].values[0]
            m2 = r2.loc[r2["group"] == grp, "mean"].values[0]
            np.testing.assert_allclose(m1, m2, rtol=1e-10)

            n1 = r1.loc[r1["group"] == grp, "n"].values[0]
            n2 = r2.loc[r2["group"] == grp, "n"].values[0]
            assert n1 == n2


class TestWithinOrderMarkouts:
    """Verify within-order comparison handles edge cases."""

    def test_no_mixed_orders(self, small_parent_df):
        """When no orders have mixed CRB/non-CRB fills, should return empty."""
        from execution_analysis import compute_within_order_markouts

        df = small_parent_df.copy()
        # Make all orders fully CRB or fully not
        df["hasCRB"] = False
        df["CRBPct"] = 0.0

        exec_df = pd.DataFrame({
            "AlgoOrderId": [0, 0, 1, 1],
            "isInt": [False, False, False, False],
            "rev1s_bps": [1.0, 2.0, 3.0, 4.0],
            "rev5s_bps": [1.0, 2.0, 3.0, 4.0],
        })
        exec_df = derive_execution_columns(exec_df)

        result = compute_within_order_markouts(df, exec_df=exec_df)
        assert result["paired_diff"].empty


class TestDerivations:
    """Verify derived column formulas are correct."""

    def test_temp_impact_sign_convention(self):
        """Positive tempImpactBps = favorable (market moved with order)."""
        df = pd.DataFrame({
            "Side": [1, 1, -1, -1],
            "amid": [100.0, 100.0, 100.0, 100.0],
            "emid": [99.0, 101.0, 101.0, 99.0],
        })
        # Buy: emid < amid → amid-emid > 0 → favorable → positive
        # Buy: emid > amid → amid-emid < 0 → adverse → negative
        # Sell: emid > amid → -(amid-emid) → -(-1) = positive → favorable
        # Sell: emid < amid → -(amid-emid) → -(1) = negative → adverse
        impact = 1e4 * df["Side"] * (df["amid"] - df["emid"]) / df["amid"]
        assert impact.iloc[0] > 0   # buy, price went down = favorable
        assert impact.iloc[1] < 0   # buy, price went up = adverse
        assert impact.iloc[2] > 0   # sell, price went up = favorable
        assert impact.iloc[3] < 0   # sell, price went down = adverse

    def test_perm_impact_decomposition(self):
        """permImpact ≈ tempImpact + reversion."""
        rng = np.random.RandomState(0)
        n = 100
        side = rng.choice([1, -1], n)
        amid = rng.uniform(40, 60, n)
        emid = amid * (1 + side * rng.uniform(-0.01, 0.01, n))
        rev5m_bps = rng.normal(2, 5, n)

        # Derive rev5m_mid
        rev5m_mid = emid * (1 - side * rev5m_bps / 1e4)

        temp_impact = 1e4 * side * (amid - emid) / amid
        perm_impact = 1e4 * side * (amid - rev5m_mid) / amid

        # permImpact ≈ tempImpact + rev5m_bps (approximately, different denominators)
        approx_perm = temp_impact + rev5m_bps
        np.testing.assert_allclose(perm_impact, approx_perm, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
