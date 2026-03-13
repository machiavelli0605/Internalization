"""
Visualization functions for internalization market impact study.

Parent-level plots (P1–P7):
  P1. CRBPct distribution
  P2. Covariate balance (Love plot)
  P3. Regression coefficient forest plot
  P4. Dose-response PSM
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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    COLOR_CONTROL,
    COLOR_CRB,
    COLOR_NON_CRB,
    COLOR_TREATED,
    CRB_BUCKET_LABELS,
    OUTCOME_VARS,
    PALETTE_INTTYPE,
    PLOT_DIR,
    PLOT_DPI,
    PLOT_FORMAT,
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
    enabled = parent_df[parent_df["isInt"].astype(bool)]

    if len(enabled) == 0:
        print("    [P1] No enabled orders, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) full distribution including zeros
    axes[0].hist(
        enabled["CRBPct"].dropna(),
        bins=50,
        edgecolor="white",
        color=COLOR_CRB,
        alpha=0.8,
    )
    axes[0].set_xlabel("CRBPct (Principal Internalization %)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CRBPct Distribution (Enabled Orders)")

    # (b) non-zero only
    nz = enabled.loc[enabled["CRBPct"] > 0, "CRBPct"].dropna()
    if len(nz) == 0:
        axes[1].text(
            0.5,
            0.5,
            "No non-zero CRBPct",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=12,
            color="grey",
        )
    else:
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


def plot_covariate_balance(
    smd_before, smd_after=None, title_suffix="", filename="P2_covariate_balance"
):
    """Love plot of standardized mean differences."""
    if smd_before is None or smd_before.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(6, len(smd_before) * 0.5)))

    covariates = smd_before["covariate"].values
    y_pos = np.arange(len(covariates))

    ax.scatter(
        smd_before["smd"].abs(),
        y_pos,
        marker="o",
        s=60,
        color=COLOR_CONTROL,
        label="Before",
        zorder=3,
    )

    if smd_after is not None and not smd_after.empty:
        ax.scatter(
            smd_after["smd"].abs(),
            y_pos,
            marker="D",
            s=60,
            color=COLOR_TREATED,
            label="After",
            zorder=3,
        )

    ax.axvline(
        0.1, color="grey", linestyle="--", alpha=0.6, label="SMD = 0.1 threshold"
    )
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
        (
            "Dose-Response: CRBPct (Full Population)",
            reg_results.get("dose_coefficients"),
        ),
        (
            "Dose-Response: CRBPct (Enabled Only)",
            reg_results.get("itt_enabled_coefficients"),
        ),
    ]
    panels = [(t, d) for t, d in panels if d is not None and not d.empty]

    n_panels = len(panels)
    if n_panels == 0:
        return
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
            plot_df["coef"],
            y_pos,
            xerr=[
                plot_df["coef"] - plot_df["ci_lower"],
                plot_df["ci_upper"] - plot_df["coef"],
            ],
            fmt="o",
            capsize=4,
            color=COLOR_CRB,
            markersize=7,
            linewidth=1.5,
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
                f"{row['coef']:.2f}{stars}",
                (row["coef"], y_pos[plot_df.index.get_loc(i)]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=9,
            )

    fig.suptitle("P3: Regression Treatment Coefficients", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P3_regression_coefficients")


# =====================================================================
# P4.  Dose-response PSM
# =====================================================================


def plot_dose_response(dose_response_results):
    """Plot ATT by CRBPct bucket from dose-response PSM."""
    att = dose_response_results.get("att_by_bucket")
    if att is None or att.empty:
        return

    outcomes = [o for o in OUTCOME_VARS if o in att["outcome"].values]
    n = len(outcomes)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        dr = att[att["outcome"] == outcome].copy()
        if dr.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(dr))
        ax.errorbar(
            x,
            dr["att"].values,
            yerr=[
                dr["att"].values - dr["ci_lower"].values,
                dr["ci_upper"].values - dr["att"].values,
            ],
            fmt="o-",
            capsize=4,
            color=COLOR_CRB,
            linewidth=2,
            markersize=7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(dr["bucket"].values, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("ATT vs 0% CRB (bps)")
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)

        # annotate sample sizes
        for i, (_, row) in enumerate(dr.iterrows()):
            ax.annotate(
                f"n={row['n_treated']:,.0f}",
                (i, row["ci_lower"]),
                textcoords="offset points",
                xytext=(0, -12),
                fontsize=7,
                ha="center",
                color="grey",
            )

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("P4: Dose-Response PSM — ATT vs 0% CRB by Dose Level", fontsize=14)
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
            smd_before,
            smd_after_ipw,
            title_suffix="(IPW Reweighting)",
            filename="P5a_psm_balance_ipw",
        )
        if smd_after_nn is not None and not smd_after_nn.empty:
            plot_covariate_balance(
                smd_before,
                smd_after_nn,
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
    # ipw = psm_results.get("ipw_outcomes")
    # if ipw is not None and not ipw.empty:
    #     ax = axes[0]
    #     pivot = ipw.pivot(index="outcome", columns="group", values="weighted_mean")
    #     pivot = pivot.reindex([o for o in OUTCOME_VARS if o in pivot.index])
    #     # ensure treated comes before control so colors align
    #     avail_groups = [g for g in ["treated", "control"] if g in pivot.columns]
    #     pivot = pivot.reindex(columns=avail_groups)
    #     if not pivot.empty and len(avail_groups) > 0:
    #         pivot.index = [OUTCOME_LABELS.get(o, o) for o in pivot.index]
    #         colors = [COLOR_TREATED if g == "treated" else COLOR_CONTROL
    #                   for g in avail_groups]
    #         pivot.plot.barh(ax=ax, color=colors)
    #         ax.set_xlabel("IPW-Weighted Mean (bps)")
    #         ax.set_title("IPW Outcome Comparison")
    #         ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    #         ax.legend(title="Group")

    # --- NN matched outcomes ---
    nn = psm_results.get("nn_outcomes")
    if nn is not None and not nn.empty:
        ax = axes
        outcomes = nn["outcome"].values
        y_pos = np.arange(len(outcomes))
        ax.errorbar(
            nn["diff"],
            y_pos,
            xerr=[nn["diff"] - nn["diff_ci_lower"], nn["diff_ci_upper"] - nn["diff"]],
            fmt="o",
            capsize=4,
            color=COLOR_CRB,
            markersize=7,
            linewidth=1.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in outcomes])
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Treated (CRB orders) minus Control (non-CRB orders), bps")
        ax.set_title("NN-Matched CRB vs non-CRB benefits")

    fig.suptitle("Propensity Score Analysis", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P6_psm_outcomes")


# =====================================================================
# P7.  Impact distribution by CRBPct bucket
# =====================================================================


def plot_impact_by_bucket(parent_df):
    """Violin / box plots of tempImpactBps and permImpact by CRBPct bucket."""
    outcomes_to_plot = ["tempImpactBps", "permImpact5mBps", "ArrivalSlippageBps"]
    outcomes_to_plot = [o for o in outcomes_to_plot if o in parent_df.columns]

    if len(outcomes_to_plot) == 0:
        return

    fig, axes = plt.subplots(
        1, len(outcomes_to_plot), figsize=(6 * len(outcomes_to_plot), 6)
    )
    if len(outcomes_to_plot) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes_to_plot):
        data = parent_df.dropna(subset=[outcome, "CRBPctBucket"])
        if len(data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="grey",
            )
            continue
        # winsorize for plotting
        lo, hi = data[outcome].quantile([0.01, 0.99])
        data = data[(data[outcome] >= lo) & (data[outcome] <= hi)]
        if len(data) == 0:
            continue

        sns.boxplot(
            data=data,
            x="CRBPctBucket",
            y=outcome,
            ax=ax,
            color=COLOR_CRB,
            fliersize=0.5,
            linewidth=0.8,
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
    if signed_df is None or signed_df.empty:
        print("    [E1] No signed markout data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    for is_int, label, color in [
        (True, "CRB Fills", COLOR_CRB),
        (False, "Non-CRB Fills", COLOR_NON_CRB),
    ]:
        sub = signed_df[signed_df["group"] == is_int].sort_values("horizon_sec")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon_sec"],
            sub["mean"],
            "o-",
            color=color,
            label=label,
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            sub["horizon_sec"],
            sub["ci_lower"],
            sub["ci_upper"],
            alpha=0.15,
            color=color,
        )

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
    if abs_df is None or abs_df.empty:
        print("    [E2] No absolute markout data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    for is_int, label, color in [
        (True, "CRB Fills", COLOR_CRB),
        (False, "Non-CRB Fills", COLOR_NON_CRB),
    ]:
        sub = abs_df[abs_df["group"] == is_int].sort_values("horizon_sec")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon_sec"],
            sub["mean"],
            "o-",
            color=color,
            label=label,
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            sub["horizon_sec"],
            sub["ci_lower"],
            sub["ci_upper"],
            alpha=0.15,
            color=color,
        )

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
    if inttype_df is None or inttype_df.empty:
        print("    [E3] No intType markout data, skipping.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette(PALETTE_INTTYPE, n_colors=inttype_df["group"].nunique())

    for idx, (grp, sub) in enumerate(inttype_df.groupby("group", observed=True)):
        sub = sub.sort_values("horizon_sec")
        ax.plot(
            sub["horizon_sec"],
            sub["mean"],
            "o-",
            color=palette[idx],
            label=grp,
            linewidth=2,
            markersize=5,
        )
        ax.fill_between(
            sub["horizon_sec"],
            sub["ci_lower"],
            sub["ci_upper"],
            alpha=0.10,
            color=palette[idx],
        )

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
        ax.bar(
            x,
            signed["mean_diff"],
            yerr=[
                signed["mean_diff"] - signed["ci_lower"],
                signed["ci_upper"] - signed["mean_diff"],
            ],
            capsize=4,
            color=COLOR_CRB,
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}s" for h in signed["horizon_sec"]])
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean Diff: CRB − Non-CRB (bps)")
        ax.set_title("Signed Markout (Within-Order)")

    if not absolute.empty:
        ax = axes[1]
        x = np.arange(len(absolute))
        ax.bar(
            x,
            absolute["mean_diff"],
            yerr=[
                absolute["mean_diff"] - absolute["ci_lower"],
                absolute["ci_upper"] - absolute["mean_diff"],
            ],
            capsize=4,
            color=COLOR_NON_CRB,
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}s" for h in absolute["horizon_sec"]])
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean Diff: CRB − Non-CRB (|bps|)")
        ax.set_title("Absolute Markout (Within-Order)")

    if not signed.empty:
        n_orders = signed["n_orders"].iloc[0]
        title = f"E4: Within-Order Markout Comparison (n={n_orders:,} paired orders)"
    else:
        title = "E4: Within-Order Markout Comparison"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "E4_within_order_markouts")


# =====================================================================
# E5.  Markout by spread bucket
# =====================================================================


def plot_markout_by_spread(spread_df):
    """Faceted line plots: markout curves by spread quintile, CRB vs non-CRB."""
    if spread_df.empty or "spread_bucket" not in spread_df.columns:
        print("    [E5] No spread data available, skipping.")
        return
    spread_buckets = sorted(spread_df["spread_bucket"].dropna().unique(), key=str)
    n = len(spread_buckets)
    if n == 0:
        print("    [E5] No valid spread buckets, skipping.")
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, bucket in enumerate(spread_buckets):
        ax = axes[idx]
        sub = spread_df[spread_df["spread_bucket"] == bucket]

        for is_int, label, color in [
            (True, "CRB", COLOR_CRB),
            (False, "Non-CRB", COLOR_NON_CRB),
        ]:
            s = sub[sub["isInt"] == is_int].sort_values("horizon_sec")
            if s.empty:
                continue
            ax.plot(
                s["horizon_sec"],
                s["mean"],
                "o-",
                color=color,
                label=label,
                linewidth=1.5,
                markersize=4,
            )
            ax.fill_between(
                s["horizon_sec"], s["ci_lower"], s["ci_upper"], alpha=0.12, color=color
            )

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
        # pick a few representative horizons, sorted numerically
        available = [
            c
            for c in exec_df.columns
            if c.startswith("rev") and c.endswith("s_bps") and not c.startswith("abs_")
        ]
        available.sort(key=lambda c: int(c.split("rev")[1].split("s_bps")[0]))
        horizons_to_plot = available[:4]  # first 4 by numeric horizon

    n = len(horizons_to_plot)
    if n == 0:
        print("    [E6] No markout horizons available, skipping.")
        return
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(horizons_to_plot):
        ax = axes[idx]
        for is_int, label, color in [
            (True, "CRB", COLOR_CRB),
            (False, "Non-CRB", COLOR_NON_CRB),
        ]:
            vals = exec_df.loc[exec_df["isInt"] == is_int, col].dropna()
            if len(vals) < 2 or vals.nunique() < 2:
                continue
            # winsorize for plotting
            lo, hi = vals.quantile([0.01, 0.99])
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) < 2:
                continue
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

# =====================================================================
# P8.  PS overlap density
# =====================================================================


def plot_ps_overlap_density(df, psm_results, treatment_col="hasCRB", exact_cols=None):
    """KDE of propensity scores for treated vs control, overall and per stratum."""
    ps = psm_results.get("propensity_scores", pd.Series(dtype=float))
    if ps.empty or df.empty:
        return

    if "ps" not in df.columns:
        if len(ps) == len(df):
            df = df.copy()
            df["ps"] = ps.values
        else:
            return

    panels = [("Overall", df)]
    if exact_cols:
        for key, grp in df.groupby(exact_cols, observed=True):
            label = key if isinstance(key, str) else str(key)
            panels.append((label, grp))

    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (title, sub) in enumerate(panels):
        ax = axes[idx]
        for tval, label, color in [
            (True, "Treated", COLOR_TREATED),
            (False, "Control", COLOR_CONTROL),
        ]:
            vals = sub.loc[sub[treatment_col].astype(bool) == tval, "ps"].dropna()
            if len(vals) < 2:
                continue
            sns.kdeplot(vals, ax=ax, color=color, label=label, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Propensity Score")
        if idx == 0:
            ax.legend()

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("P8: Propensity Score Overlap Density", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "P8_ps_overlap_density")


# =====================================================================
# P9.  Per-stratum ATT waterfall
# =====================================================================


def plot_stratum_att_waterfall(diagnostics):
    """Horizontal bar chart: per-stratum ATT with contribution weights."""
    att_df = diagnostics.get("stratum_att", pd.DataFrame())
    if att_df.empty:
        return

    # Filter to tempImpactBps
    sub = att_df[att_df["outcome"] == "tempImpactBps"].copy()
    if sub.empty:
        sub = att_df[att_df["outcome"] == att_df["outcome"].iloc[0]].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.8)))
    y_pos = np.arange(len(sub))
    colors = [COLOR_TREATED if v >= 0 else COLOR_CONTROL for v in sub["att"]]

    ax.barh(y_pos, sub["att"], color=colors, alpha=0.8, edgecolor="white")
    ax.errorbar(
        sub["att"],
        y_pos,
        xerr=[sub["att"] - sub["ci_lower"], sub["ci_upper"] - sub["att"]],
        fmt="none",
        capsize=4,
        color="black",
        linewidth=1,
    )
    ax.set_yticks(y_pos)
    labels = [
        f"{s} (w={w:.0%}, n={n:,})"
        for s, w, n in zip(sub["stratum"], sub["contribution_weight"], sub["n_treated"])
    ]
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xlabel("ATT (bps)")
    ax.set_title("P9: Per-Stratum ATT Decomposition — tempImpactBps")

    fig.tight_layout()
    _savefig(fig, "P9_stratum_att_waterfall")


# =====================================================================
# P10.  Leave-one-out sensitivity
# =====================================================================


def plot_leave_one_out(diagnostics):
    """Forest plot: ATT when each stratum is excluded."""
    loo = diagnostics.get("leave_one_out", pd.DataFrame())
    if loo.empty:
        return

    sub = loo[loo["outcome"] == "tempImpactBps"].copy()
    if sub.empty:
        sub = loo[loo["outcome"] == loo["outcome"].iloc[0]].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.6 + 1)))
    y_pos = np.arange(len(sub) + 1)

    # Full ATT as first row
    att_full = sub["att_full"].iloc[0]
    ax.scatter(
        [att_full], [0], marker="D", s=80, color="black", zorder=5, label="Full ATT"
    )

    for i, (_, row) in enumerate(sub.iterrows()):
        color = COLOR_TREATED if row["att_without"] > att_full else COLOR_CONTROL
        ax.scatter(
            [row["att_without"]], [i + 1], marker="o", s=60, color=color, zorder=5
        )

    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
    ax.axvline(att_full, color="black", linestyle=":", alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Full (all strata)"] + list(sub["excluded_stratum"]))
    ax.set_xlabel("ATT (bps)")
    ax.set_title("P10: Leave-One-Out Stratum Sensitivity — tempImpactBps")
    ax.invert_yaxis()

    fig.tight_layout()
    _savefig(fig, "P10_leave_one_out")


# =====================================================================
# P11.  Prognostic covariate importance
# =====================================================================


def plot_prognostic_importance(diagnostics):
    """Bar chart of prognostic coefficients (outcome ~ covariates on controls)."""
    prog = diagnostics.get("prognostic", pd.DataFrame())
    if prog.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(prog) * 0.6)))
    y_pos = np.arange(len(prog))
    sorted_prog = prog.reindex(prog["coef"].abs().sort_values().index)
    colors = [COLOR_TREATED if c >= 0 else COLOR_CONTROL for c in sorted_prog["coef"]]

    ax.barh(y_pos, sorted_prog["coef"], color=colors, alpha=0.8)
    ax.errorbar(
        sorted_prog["coef"],
        y_pos,
        xerr=1.96 * sorted_prog["se"],
        fmt="none",
        capsize=3,
        color="black",
        linewidth=0.8,
    )
    ax.set_yticks(y_pos)
    labels = []
    for _, row in sorted_prog.iterrows():
        stars = (
            "***"
            if row["pvalue"] < 0.001
            else (
                "**" if row["pvalue"] < 0.01 else ("*" if row["pvalue"] < 0.05 else "")
            )
        )
        labels.append(f"{row['covariate']} {stars}")
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xlabel("Coefficient (predicting tempImpactBps in controls)")
    r2 = (
        sorted_prog["r_squared"].iloc[0] if "r_squared" in sorted_prog.columns else None
    )
    title = "P11: Prognostic Covariate Importance"
    if r2 is not None:
        title += f" (R²={r2:.3f})"
    ax.set_title(title)

    fig.tight_layout()
    _savefig(fig, "P11_prognostic_importance")


# =====================================================================
# P12.  Rosenbaum bounds
# =====================================================================


def plot_rosenbaum_bounds(diagnostics):
    """Line plot of Gamma vs p-value with significance threshold."""
    rb = diagnostics.get("rosenbaum_bounds", pd.DataFrame())
    if rb.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rb["gamma"], rb["p_upper"], "o-", color=COLOR_CRB, linewidth=2, markersize=6
    )
    ax.axhline(0.05, color=COLOR_CONTROL, linestyle="--", alpha=0.7, label="p = 0.05")
    ax.set_xlabel("Gamma (sensitivity parameter)")
    ax.set_ylabel("Upper-bound p-value")
    ax.set_title("P12: Rosenbaum Bounds — Sensitivity to Unmeasured Confounding")
    ax.legend()
    ax.set_ylim(bottom=0)

    # Annotate the breakpoint
    crossings = rb[rb["p_upper"] >= 0.05]
    if not crossings.empty:
        gamma_break = crossings["gamma"].iloc[0]
        ax.annotate(
            f"Breaks at Γ={gamma_break:.1f}",
            (gamma_break, 0.05),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
            color=COLOR_CONTROL,
        )

    fig.tight_layout()
    _savefig(fig, "P12_rosenbaum_bounds")


# =====================================================================
# P13.  PS specification sensitivity
# =====================================================================


def plot_spec_sensitivity(diagnostics):
    """Forest plot of ATT across different PS model specifications."""
    spec = diagnostics.get("spec_sensitivity", pd.DataFrame())
    if spec.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(spec) * 0.6)))
    y_pos = np.arange(len(spec))

    ax.errorbar(
        spec["att"],
        y_pos,
        xerr=[spec["att"] - spec["ci_lower"], spec["ci_upper"] - spec["att"]],
        fmt="o",
        capsize=4,
        color=COLOR_CRB,
        markersize=7,
        linewidth=1.5,
    )
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)

    # Highlight the base model
    base_idx = spec.index[spec["spec_name"] == "base"]
    if len(base_idx) > 0:
        bi = spec.index.get_loc(base_idx[0])
        ax.scatter(
            [spec.iloc[bi]["att"]], [bi], marker="D", s=100, color="black", zorder=5
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(spec["spec_name"])
    ax.set_xlabel("ATT — tempImpactBps (bps)")
    ax.set_title("P13: PS Specification Sensitivity")
    ax.invert_yaxis()

    fig.tight_layout()
    _savefig(fig, "P13_spec_sensitivity")


def generate_parent_plots(parent_df, parent_results):
    """Generate all parent-level plots."""
    print("\n  Generating parent-level plots ...")

    # print("    P1: CRBPct distribution")
    # plot_crbpct_distribution(parent_df)

    # print("    P2: Covariate balance")
    # desc = parent_results.get("descriptive", {})
    psm = parent_results.get("psm", {})
    # if psm.get("smd_before") is not None:
    #     plot_covariate_balance(psm["smd_before"], title_suffix="(hasCRB)")

    # print("    P3: Regression coefficients")
    # reg = parent_results.get("regression", {})
    # if reg:
    #     plot_regression_coefficients(reg)

    # print("    P4: Dose-response PSM")
    # dr = parent_results.get("dose_response", {})
    # if dr:
    #     plot_dose_response(dr)

    print("    P5: PSM balance diagnostics")
    if psm:
        plot_psm_balance(psm)

    print("    P6: PSM outcome comparison")
    if psm:
        plot_psm_outcomes(psm)

    print("    P7: Impact by CRBPct bucket")
    plot_impact_by_bucket(parent_df)

    # New diagnostic plots
    psm_diag = parent_results.get("psm_diagnostics", {})

    print("    P8: PS overlap density")
    if psm:
        plot_ps_overlap_density(
            parent_df,
            psm,
            treatment_col="hasCRB",
            exact_cols=["Strategy"] if "Strategy" in parent_df.columns else [],
        )

    if psm_diag:
        print("    P9: Stratum ATT waterfall")
        plot_stratum_att_waterfall(psm_diag)

        print("    P10: Leave-one-out sensitivity")
        plot_leave_one_out(psm_diag)

        print("    P11: Prognostic importance")
        plot_prognostic_importance(psm_diag)

        print("    P12: Rosenbaum bounds")
        plot_rosenbaum_bounds(psm_diag)

        print("    P13: PS specification sensitivity")
        plot_spec_sensitivity(psm_diag)


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
