"""
Visualization functions for internalization market impact study.

Parent-level plots (P1–P7):
  P1. CRBPct distribution
  P2. Covariate balance (Love plot)
  P3. Regression coefficient forest plot
  P4. Dose-response curve
  P5. PSM balance diagnostics
  P6. PSM / IPW outcome comparison
  P7. Impact distribution by CRBPct bucket

Execution-level plots (E1–E6):
  E1. Signed markout curves
  E2. Absolute markout curves
  E3. Markout by intType
  E4. Within-order paired comparison
  E5. Markout by spread bucket
  E6. Markout distribution (KDE)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    PLOT_DIR, PLOT_DPI, PLOT_FORMAT,
    COLOR_CRB, COLOR_NON_CRB, COLOR_TREATED, COLOR_CONTROL,
    CRB_BUCKET_LABELS, OUTCOME_VARS, PALETTE_INTTYPE,
)

# Consistent style
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": PLOT_DPI, "savefig.dpi": PLOT_DPI})


def _savefig(fig, name):
    path = PLOT_DIR / f"{name}.{PLOT_FORMAT}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


OUTCOME_LABELS = {
    "tempImpactBps": "Temp Impact (bps)",
    "permImpact5mBps": "Perm Impact 5m (bps)",
    "permImpact15mBps": "Perm Impact 15m (bps)",
    "permImpact60mBps": "Perm Impact 60m (bps)",
    "ArrivalSlippageBps": "Arrival Slippage (bps)",
}


# =====================================================================
# P1.  CRBPct distribution
# =====================================================================

def plot_crbpct_distribution(parent_df):
    """Histogram of CRBPct among isInt=True orders."""
    enabled = parent_df[parent_df["isInt"] == True]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) full distribution including zeros
    axes[0].hist(enabled["CRBPct"], bins=50, edgecolor="white", color=COLOR_CRB,
                 alpha=0.8)
    axes[0].set_xlabel("CRBPct (Principal Internalization %)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CRBPct Distribution (Enabled Orders)")

    # (b) non-zero only
    nz = enabled.loc[enabled["CRBPct"] > 0, "CRBPct"]
    axes[1].hist(nz, bins=50, edgecolor="white", color=COLOR_CRB, alpha=0.8)
    axes[1].set_xlabel("CRBPct (Principal Internalization %)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("CRBPct Distribution (Non-Zero Only)")

    fig.suptitle("P1: Distribution of Principal Internalization Rate", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P1_crbpct_distribution")


# =====================================================================
# P2.  Covariate balance (Love plot) — before matching
# =====================================================================

def plot_covariate_balance(smd_before, smd_after=None, title_suffix="",
                           filename="P2_covariate_balance"):
    """Love plot of standardized mean differences."""
    fig, ax = plt.subplots(figsize=(8, max(6, len(smd_before) * 0.5)))

    covariates = smd_before["covariate"].values
    y_pos = np.arange(len(covariates))

    ax.scatter(smd_before["smd"].abs(), y_pos, marker="o", s=60,
               color=COLOR_CONTROL, label="Before", zorder=3)

    if smd_after is not None and not smd_after.empty:
        ax.scatter(smd_after["smd"].abs(), y_pos, marker="D", s=60,
                   color=COLOR_TREATED, label="After", zorder=3)

    ax.axvline(0.1, color="grey", linestyle="--", alpha=0.6, label="SMD = 0.1 threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title(f"P2: Covariate Balance {title_suffix}")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    fig.tight_layout()
    _savefig(fig, filename)


# =====================================================================
# P3.  Regression coefficient forest plot
# =====================================================================

def plot_regression_coefficients(reg_results):
    """Forest plot showing treatment coefficients across outcomes.

    Expects reg_results dict with keys 'itt_coefficients', 'dose_coefficients',
    'itt_enabled_coefficients'.
    """
    panels = [
        ("ITT: isInt (Full Population)", reg_results.get("itt_coefficients")),
        ("Dose-Response: CRBPct (Full Population)", reg_results.get("dose_coefficients")),
        ("Dose-Response: CRBPct (Enabled Only)", reg_results.get("itt_enabled_coefficients")),
    ]
    panels = [(t, d) for t, d in panels if d is not None and not d.empty]

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (title, coef_df) in zip(axes, panels):
        # filter to CRBPct rows if dose-response (skip ATSPINPct for clarity)
        if "CRBPct" in coef_df["treatment"].values:
            plot_df = coef_df[coef_df["treatment"] == "CRBPct"].copy()
        else:
            plot_df = coef_df.copy()

        outcomes = plot_df["outcome"].values
        y_pos = np.arange(len(outcomes))

        ax.errorbar(
            plot_df["coef"], y_pos,
            xerr=[plot_df["coef"] - plot_df["ci_lower"],
                  plot_df["ci_upper"] - plot_df["coef"]],
            fmt="o", capsize=4, color=COLOR_CRB, markersize=7, linewidth=1.5,
        )
        ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in outcomes])
        ax.set_xlabel("Coefficient (bps)")
        ax.set_title(title, fontsize=11)

        # annotate with values
        for i, row in plot_df.iterrows():
            stars = ""
            if row["pvalue"] < 0.001:
                stars = "***"
            elif row["pvalue"] < 0.01:
                stars = "**"
            elif row["pvalue"] < 0.05:
                stars = "*"
            ax.annotate(
                f'{row["coef"]:.2f}{stars}',
                (row["coef"], y_pos[plot_df.index.get_loc(i)]),
                textcoords="offset points", xytext=(8, 0), fontsize=9,
            )

    fig.suptitle("P3: Regression Treatment Coefficients", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P3_regression_coefficients")


# =====================================================================
# P4.  Dose-response curve
# =====================================================================

def plot_dose_response(dose_response_results):
    """Plot confounder-adjusted mean outcomes by CRBPct bucket."""
    outcomes = [o for o in OUTCOME_VARS if o in dose_response_results]
    n = len(outcomes)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        dr = dose_response_results[outcome]

        x = np.arange(len(dr))
        ax.errorbar(
            x, dr["adj_mean"],
            yerr=[dr["adj_mean"] - dr["ci_lower"],
                  dr["ci_upper"] - dr["adj_mean"]],
            fmt="o-", capsize=4, color=COLOR_CRB, linewidth=2, markersize=7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(dr["bucket"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Adjusted Mean (bps)")
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)

        # annotate sample sizes
        for i, row in dr.iterrows():
            ax.annotate(f'n={row["n"]:,.0f}', (i, row["ci_lower"]),
                        textcoords="offset points", xytext=(0, -12),
                        fontsize=7, ha="center", color="grey")

    # hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("P4: Dose-Response — Adjusted Mean Impact by CRBPct Bucket",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P4_dose_response")


# =====================================================================
# P5.  PSM balance diagnostics (Love plot after matching)
# =====================================================================

def plot_psm_balance(psm_results):
    """Love plot comparing balance before and after PSM/IPW."""
    smd_before = psm_results.get("smd_before")
    smd_after_ipw = psm_results.get("smd_after_ipw")
    smd_after_nn = psm_results.get("smd_after_nn")

    if smd_before is not None:
        plot_covariate_balance(
            smd_before, smd_after_ipw,
            title_suffix="(IPW Reweighting)",
            filename="P5a_psm_balance_ipw",
        )
        if smd_after_nn is not None and not smd_after_nn.empty:
            plot_covariate_balance(
                smd_before, smd_after_nn,
                title_suffix="(Nearest-Neighbor Matching)",
                filename="P5b_psm_balance_nn",
            )


# =====================================================================
# P6.  PSM / IPW outcome comparison
# =====================================================================

def plot_psm_outcomes(psm_results):
    """Grouped bar chart of IPW-weighted and NN-matched outcome means."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- IPW outcomes ---
    ipw = psm_results.get("ipw_outcomes")
    if ipw is not None and not ipw.empty:
        ax = axes[0]
        pivot = ipw.pivot(index="outcome", columns="group", values="weighted_mean")
        pivot = pivot.reindex([o for o in OUTCOME_VARS if o in pivot.index])
        pivot.index = [OUTCOME_LABELS.get(o, o) for o in pivot.index]
        pivot.plot.barh(ax=ax, color=[COLOR_TREATED, COLOR_CONTROL])
        ax.set_xlabel("IPW-Weighted Mean (bps)")
        ax.set_title("IPW Outcome Comparison")
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        ax.legend(title="Group")

    # --- NN matched outcomes ---
    nn = psm_results.get("nn_outcomes")
    if nn is not None and not nn.empty:
        ax = axes[1]
        outcomes = nn["outcome"].values
        y_pos = np.arange(len(outcomes))
        ax.errorbar(
            nn["diff"], y_pos,
            xerr=[nn["diff"] - nn["diff_ci_lower"],
                  nn["diff_ci_upper"] - nn["diff"]],
            fmt="o", capsize=4, color=COLOR_CRB, markersize=7, linewidth=1.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in outcomes])
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Matched Difference: Treated − Control (bps)")
        ax.set_title("NN-Matched Outcome Difference")

    fig.suptitle("P6: Propensity Score Analysis — Outcome Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P6_psm_outcomes")


# =====================================================================
# P7.  Impact distribution by CRBPct bucket
# =====================================================================

def plot_impact_by_bucket(parent_df):
    """Violin / box plots of tempImpactBps and permImpact by CRBPct bucket."""
    outcomes_to_plot = ["tempImpactBps", "permImpact5mBps", "ArrivalSlippageBps"]
    outcomes_to_plot = [o for o in outcomes_to_plot if o in parent_df.columns]

    fig, axes = plt.subplots(1, len(outcomes_to_plot),
                             figsize=(6 * len(outcomes_to_plot), 6))
    if len(outcomes_to_plot) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes_to_plot):
        data = parent_df.dropna(subset=[outcome, "CRBPctBucket"])
        # winsorize for plotting
        lo, hi = data[outcome].quantile([0.01, 0.99])
        data = data[(data[outcome] >= lo) & (data[outcome] <= hi)]

        sns.boxplot(
            data=data, x="CRBPctBucket", y=outcome, ax=ax,
            color=COLOR_CRB, fliersize=0.5, linewidth=0.8,
            order=CRB_BUCKET_LABELS,
        )
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("CRBPct Bucket")
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome))
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("P7: Impact Distribution by CRBPct Bucket", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P7_impact_by_bucket")


# =====================================================================
# E1.  Signed markout curves
# =====================================================================

def plot_signed_markout_curves(signed_df):
    """Line plot of mean rev{x}s_bps over horizons, CRB vs non-CRB."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for is_int, label, color in [(True, "CRB Fills", COLOR_CRB),
                                  (False, "Non-CRB Fills", COLOR_NON_CRB)]:
        sub = signed_df[signed_df["group"] == is_int].sort_values("horizon_sec")
        if sub.empty:
            continue
        ax.plot(sub["horizon_sec"], sub["mean"], "o-", color=color, label=label,
                linewidth=2, markersize=6)
        ax.fill_between(sub["horizon_sec"], sub["ci_lower"], sub["ci_upper"],
                        alpha=0.15, color=color)

    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Mean Signed Markout (bps)")
    ax.set_title("E1: Post-Trade Signed Markout Curves — CRB vs Non-CRB")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "E1_signed_markout_curves")


# =====================================================================
# E2.  Absolute markout curves
# =====================================================================

def plot_abs_markout_curves(abs_df):
    """Line plot of mean |rev{x}s_bps| over horizons, CRB vs non-CRB."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for is_int, label, color in [(True, "CRB Fills", COLOR_CRB),
                                  (False, "Non-CRB Fills", COLOR_NON_CRB)]:
        sub = abs_df[abs_df["group"] == is_int].sort_values("horizon_sec")
        if sub.empty:
            continue
        ax.plot(sub["horizon_sec"], sub["mean"], "o-", color=color, label=label,
                linewidth=2, markersize=6)
        ax.fill_between(sub["horizon_sec"], sub["ci_lower"], sub["ci_upper"],
                        alpha=0.15, color=color)

    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Mean |Markout| (bps)")
    ax.set_title("E2: Post-Trade Absolute Markout Curves — CRB vs Non-CRB")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "E2_abs_markout_curves")


# =====================================================================
# E3.  Markout by intType
# =====================================================================

def plot_markout_by_inttype(inttype_df, absolute=False):
    """Multi-line markout plot broken down by intType."""
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette(PALETTE_INTTYPE,
                                n_colors=inttype_df["group"].nunique())

    for idx, (grp, sub) in enumerate(inttype_df.groupby("group", observed=True)):
        sub = sub.sort_values("horizon_sec")
        ax.plot(sub["horizon_sec"], sub["mean"], "o-", color=palette[idx],
                label=grp, linewidth=2, markersize=5)
        ax.fill_between(sub["horizon_sec"], sub["ci_lower"], sub["ci_upper"],
                        alpha=0.10, color=palette[idx])

    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Horizon (seconds)")

    if absolute:
        ax.set_ylabel("Mean |Markout| (bps)")
        ax.set_title("E3b: Absolute Markout by Internalization Type")
        fname = "E3b_abs_markout_by_inttype"
    else:
        ax.set_ylabel("Mean Signed Markout (bps)")
        ax.set_title("E3a: Signed Markout by Internalization Type")
        fname = "E3a_markout_by_inttype"

    ax.legend(title="intType", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _savefig(fig, fname)


# =====================================================================
# E4.  Within-order paired comparison
# =====================================================================

def plot_within_order_markouts(within_results):
    """Bar chart of mean within-order CRB − non-CRB markout difference."""
    paired = within_results.get("paired_diff")
    if paired is None or paired.empty:
        print("    [E4] No within-order data available, skipping.")
        return

    # --- Signed markouts ---
    signed = paired[~paired["is_absolute"]].sort_values("horizon_sec")
    absolute = paired[paired["is_absolute"]].sort_values("horizon_sec")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if not signed.empty:
        ax = axes[0]
        x = np.arange(len(signed))
        ax.bar(x, signed["mean_diff"], yerr=[
            signed["mean_diff"] - signed["ci_lower"],
            signed["ci_upper"] - signed["mean_diff"],
        ], capsize=4, color=COLOR_CRB, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(h)}s' for h in signed["horizon_sec"]])
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean Diff: CRB − Non-CRB (bps)")
        ax.set_title("Signed Markout (Within-Order)")

    if not absolute.empty:
        ax = axes[1]
        x = np.arange(len(absolute))
        ax.bar(x, absolute["mean_diff"], yerr=[
            absolute["mean_diff"] - absolute["ci_lower"],
            absolute["ci_upper"] - absolute["mean_diff"],
        ], capsize=4, color=COLOR_NON_CRB, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(h)}s' for h in absolute["horizon_sec"]])
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean Diff: CRB − Non-CRB (|bps|)")
        ax.set_title("Absolute Markout (Within-Order)")

    n_orders = signed["n_orders"].iloc[0] if not signed.empty else "?"
    fig.suptitle(
        f"E4: Within-Order Markout Comparison (n={n_orders:,} paired orders)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "E4_within_order_markouts")


# =====================================================================
# E5.  Markout by spread bucket
# =====================================================================

def plot_markout_by_spread(spread_df):
    """Faceted line plots: markout curves by spread quintile, CRB vs non-CRB."""
    spread_buckets = sorted(spread_df["spread_bucket"].unique(), key=str)
    n = len(spread_buckets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for idx, bucket in enumerate(spread_buckets):
        ax = axes[idx]
        sub = spread_df[spread_df["spread_bucket"] == bucket]

        for is_int, label, color in [(True, "CRB", COLOR_CRB),
                                      (False, "Non-CRB", COLOR_NON_CRB)]:
            s = sub[sub["isInt"] == is_int].sort_values("horizon_sec")
            if s.empty:
                continue
            ax.plot(s["horizon_sec"], s["mean"], "o-", color=color, label=label,
                    linewidth=1.5, markersize=4)
            ax.fill_between(s["horizon_sec"], s["ci_lower"], s["ci_upper"],
                            alpha=0.12, color=color)

        ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
        ax.set_title(f"Spread: {bucket}", fontsize=10)
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel("Markout (bps)")
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("E5: Markout by Spread Bucket — CRB vs Non-CRB", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "E5_markout_by_spread")


# =====================================================================
# E6.  Markout distribution (KDE)
# =====================================================================

def plot_markout_distribution(exec_df, horizons_to_plot=None):
    """KDE overlay of signed markout distribution for CRB vs non-CRB.

    Because the full execution dataset may not fit in memory, this function
    expects a DataFrame (possibly a sample).
    """
    if horizons_to_plot is None:
        # pick a few representative horizons
        available = [c for c in exec_df.columns
                     if c.startswith("rev") and c.endswith("s_bps")
                     and not c.startswith("abs_")]
        horizons_to_plot = sorted(available)[:4]  # first 4

    n = len(horizons_to_plot)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(horizons_to_plot):
        ax = axes[idx]
        for is_int, label, color in [(True, "CRB", COLOR_CRB),
                                      (False, "Non-CRB", COLOR_NON_CRB)]:
            vals = exec_df.loc[exec_df["isInt"] == is_int, col].dropna()
            # winsorize for plotting
            lo, hi = vals.quantile([0.01, 0.99])
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) > 500_000:
                vals = vals.sample(500_000, random_state=42)
            sns.kdeplot(vals, ax=ax, color=color, label=label, linewidth=1.5)

        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        horizon = col.split("rev")[1].split("s_bps")[0]
        ax.set_title(f"rev{horizon}s_bps")
        ax.set_xlabel("Markout (bps)")
        if idx == 0:
            ax.legend()

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("E6: Markout Distribution — CRB vs Non-CRB", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "E6_markout_distribution")


# =====================================================================
# Master plot functions
# =====================================================================

def generate_parent_plots(parent_df, parent_results):
    """Generate all parent-level plots."""
    print("\n  Generating parent-level plots ...")

    print("    P1: CRBPct distribution")
    plot_crbpct_distribution(parent_df)

    print("    P2: Covariate balance")
    desc = parent_results.get("descriptive", {})
    psm = parent_results.get("psm", {})
    if psm.get("smd_before") is not None:
        plot_covariate_balance(psm["smd_before"], title_suffix="(hasCRB)")

    print("    P3: Regression coefficients")
    reg = parent_results.get("regression", {})
    if reg:
        plot_regression_coefficients(reg)

    print("    P4: Dose-response curves")
    dr = parent_results.get("dose_response", {})
    if dr:
        plot_dose_response(dr)

    print("    P5: PSM balance diagnostics")
    if psm:
        plot_psm_balance(psm)

    print("    P6: PSM outcome comparison")
    if psm:
        plot_psm_outcomes(psm)

    print("    P7: Impact by CRBPct bucket")
    plot_impact_by_bucket(parent_df)


def generate_execution_plots(exec_results, exec_df_sample=None):
    """Generate all execution-level plots."""
    print("\n  Generating execution-level plots ...")

    print("    E1: Signed markout curves")
    if "signed_markouts" in exec_results:
        plot_signed_markout_curves(exec_results["signed_markouts"])

    print("    E2: Absolute markout curves")
    if "abs_markouts" in exec_results:
        plot_abs_markout_curves(exec_results["abs_markouts"])

    print("    E3: Markout by intType")
    if "markout_by_inttype" in exec_results:
        plot_markout_by_inttype(exec_results["markout_by_inttype"], absolute=False)
    if "abs_markout_by_inttype" in exec_results:
        plot_markout_by_inttype(exec_results["abs_markout_by_inttype"], absolute=True)

    print("    E4: Within-order markouts")
    if "within_order" in exec_results:
        plot_within_order_markouts(exec_results["within_order"])

    print("    E5: Markout by spread bucket")
    if "markout_by_spread" in exec_results:
        plot_markout_by_spread(exec_results["markout_by_spread"])

    print("    E6: Markout distribution")
    if exec_df_sample is not None:
        plot_markout_distribution(exec_df_sample)
    else:
        print("      (skipped — no execution sample provided)")
