# Author: Roger Ison   roger@miximum.info
"""Reusable plotting layer for NewLoop outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

METRIC_LABELS: Dict[str, str] = {
    "automation": "Automation (Whole Economy)",
    "automation_flow": "Automation Flow (Whole Economy, Δ/q)",
    "automation_info": "Automation (Info)",
    "automation_info_flow": "Automation Flow (Info, Δ/q)",
    "automation_phys": "Automation (Physical)",
    "automation_phys_flow": "Automation Flow (Physical, Δ/q)",
    "sector_tfp_mult_info": "TFP Multiplier (Info)",
    "sector_tfp_mult_physical": "TFP Multiplier (Physical)",
    "price_level": "Price Level",
    "price_level_deflated": "Real Price Level",
    "inflation": "Inflation",
    "real_avg_income": "Real Avg Income",
    "real_consumption": "Real Consumption",
    "gini_market": "Gini (Pre-Tax/Pre-Transfer)",
    "gini_disp": "Gini (Disposable)",
    "gini_wealth": "Gini (Wealth)",
    "private_eq_per_h": "Private Equity / Household",
    "hh_deposits_per_h": "Household Deposits / Household",
    "hh_housing_value_per_h": "Household Housing Value / Household",
    "hh_debt_per_h": "Household Debt / Household",
    "hh_mortgage_debt_per_h": "Household Mortgage Debt / Household",
    "hh_revolving_debt_per_h": "Household Revolving Debt / Household",
    "hh_mortgage_balance_total": "Outstanding Mortgage Balance",
    "hh_mortgage_orig_principal_total": "Outstanding Mortgage Original Principal",
    "hh_mortgage_active_count": "Active Mortgages",
    "private_roe_q": "Private Payout Yield / Quarter",
    "private_broad_roe_q": "Private Broad ROE (Annualized %)",
    "bank_broad_roe_q": "Bank Broad ROE (Annualized %)",
    "corporate_info_broad_roe_q": "Info Broad ROE (Annualized %)",
    "corporate_physical_broad_roe_q": "Physical Broad ROE (Annualized %)",
    "sector_op_margin_info": "Info Operating Margin (%)",
    "sector_op_margin_phys": "Physical Operating Margin (%)",
    "corporate_nonbank_broad_roe_q": "Non-Bank Corporate Broad ROE (Annualized %)",
    "corporate_broad_roe_q": "Total Corporate Broad ROE (Annualized %)",
    "private_inv_cov": "Investment Coverage",
    "pop_dti_med": "Mortgage Payment / Pre-Debt Disposable Income P50",
    "pop_dti_p90": "Mortgage Payment / Pre-Debt Disposable Income P90",
    "pop_dti_w_med": "Mortgage Payment / Wages P50",
    "pop_dti_w_p90": "Mortgage Payment / Wages P90",
    "corporate_eq_info_per_h": "Corporate Broad Equity (Info) / Household",
    "corporate_eq_physical_per_h": "Corporate Broad Equity (Physical) / Household",
    "corporate_eq_total_per_h": "Corporate Broad Equity (Total) / Household",
    "trust_equity_pct": "Trust Equity %",
    "uis_per_h": "Income Support / Household",
    "uis_from_fund_dep_per_h": "Income Support from FUND",
    "uis_from_gov_dep_per_h": "Income Support from GOV",
    "uis_issued_per_h": "Income Support Issued",
    "corp_tax_rate_eff": "Effective Corporate Tax Rate",
    "fund_dividend_inflow_per_h": "FUND Dividends / Household",
    "ums_drain_to_fund_per_h": "UMS -> FUND / Household",
    "fund_tracked_inflows_per_h": "Total FUND Inflows / Household",
    "ums_recycle_to_info_per_h": "UMS Recycled To IS / Household",
    "ums_recycle_to_phys_per_h": "UMS Recycled To PS / Household",
    "ums_recycle_total_per_h": "Total UMS Recycled / Household",
    "capex_per_h": "Capital Investment / Household",
    "sector_capacity_info_per_h": "Sector Capacity (Info) / Household",
    "sector_capacity_physical_per_h": "Sector Capacity (Physical) / Household",
    "sector_hh_util_info": "Household Utilization (Info)",
    "sector_hh_util_physical": "Household Utilization (Physical)",
    "sector_util_info": "Total Utilization (Info)",
    "sector_util_physical": "Total Utilization (Physical)",
    "sector_demand_info_per_h": "Sector Demand (Info) / Household",
    "sector_demand_physical_per_h": "Sector Demand (Physical) / Household",
    "unmet_demand_info_per_h": "Unmet HH Demand (Info) / Household",
    "unmet_demand_physical_per_h": "Unmet HH Demand (Physical) / Household",
    "wages_total": "Total Wage Base",
    "trust_value_per_h": "Trust Value / Household",
}

DEFAULT_LINE_METRICS: List[str] = [
    "inflation",
    "price_level_deflated",
]

SPLIT_ROE_EQUITY_FLOOR_PER_H = 500.0


def _require_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not rows:
        raise ValueError("No rows provided for plotting.")
    return list(rows)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def metric_options() -> Dict[str, str]:
    return dict(METRIC_LABELS)


def _series(rows: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
    def _annualize_quarterly_rate(value: float) -> float:
        q = float(value)
        if q <= -1.0:
            return float("nan")
        return ((1.0 + q) ** 4 - 1.0) * 100.0

    split_roe_equity_key = {
        "bank_broad_roe_q": None,
        "corporate_info_broad_roe_q": "corporate_eq_info_per_h",
        "corporate_physical_broad_roe_q": "corporate_eq_physical_per_h",
        "corporate_nonbank_broad_roe_q": "corporate_eq_total_per_h",
    }

    if metric == "price_level_deflated":
        values: List[float] = []
        for row in rows:
            p = float(row.get("price_level", 1.0))
            if p <= 0.0:
                p = 1e-9
            values.append(p)
        return values
    if metric == "trust_equity_pct":
        values: List[float] = []
        trust_started = False
        for row in rows:
            active = bool(row.get("trust_active", False))
            value = float(row.get(metric, 0.0))
            if active:
                trust_started = True
            values.append(value if trust_started else float("nan"))
        return values
    if metric in {
        "private_roe_q",
        "private_broad_roe_q",
        "bank_broad_roe_q",
        "corporate_info_broad_roe_q",
        "corporate_physical_broad_roe_q",
        "sector_op_margin_info",
        "sector_op_margin_phys",
        "corporate_nonbank_broad_roe_q",
        "corporate_broad_roe_q",
    }:
        values = [float(r.get(metric, 0.0)) for r in rows]
        equity_key = split_roe_equity_key.get(metric)
        if equity_key is not None:
            filtered: List[float] = []
            for idx, value in enumerate(values):
                if idx == 0:
                    filtered.append(float("nan"))
                else:
                    prev_equity_per_h = float(rows[idx - 1].get(equity_key, 0.0))
                    if prev_equity_per_h < SPLIT_ROE_EQUITY_FLOOR_PER_H:
                        filtered.append(float("nan"))
                    else:
                        filtered.append(value)
            values = filtered
        if metric in {
            "private_broad_roe_q",
            "bank_broad_roe_q",
            "corporate_info_broad_roe_q",
            "corporate_physical_broad_roe_q",
            "corporate_nonbank_broad_roe_q",
            "corporate_broad_roe_q",
        }:
            values = [_annualize_quarterly_rate(v) for v in values]
        elif metric in {"sector_op_margin_info", "sector_op_margin_phys"}:
            values = [100.0 * float(v) for v in values]
        if values:
            if metric in {
                "private_broad_roe_q",
                "bank_broad_roe_q",
                "corporate_info_broad_roe_q",
                "corporate_physical_broad_roe_q",
                "corporate_nonbank_broad_roe_q",
                "corporate_broad_roe_q",
            }:
                values[0] = float("nan")
        return values
    return [float(r.get(metric, 0.0)) for r in rows]


def _plot_points(
    rows: Sequence[Mapping[str, Any]],
    x: Sequence[float],
    metric: str,
) -> tuple[List[float], List[float]]:
    y = _series(rows, metric)
    if metric != "trust_equity_pct":
        return list(x), y

    launch_idx = None
    for i, row in enumerate(rows):
        if bool(row.get("trust_active", False)) and float(row.get("trust_equity_pct", 0.0)) > 0.0:
            launch_idx = i
            break

    if launch_idx is None:
        return list(x), y

    x_plot = list(x)
    y_plot = list(y)
    x_launch = float(x[launch_idx])
    y_launch = float(rows[launch_idx].get("trust_equity_pct", 0.0))

    # Insert an explicit zero-height point at the launch quarter so the line
    # shows the initial leveraged buy as a vertical jump rather than merely
    # starting at the post-buy level.
    x_plot.insert(launch_idx, x_launch)
    y_plot.insert(launch_idx, 0.0)
    if launch_idx > 0:
        y_plot[launch_idx - 1] = float("nan")
    y_plot[launch_idx + 1] = y_launch
    return x_plot, y_plot


def _line_style(metric: str, *, secondary: bool) -> Dict[str, Any]:
    style: Dict[str, Any] = {"linewidth": 2.0}
    if secondary:
        style["linestyle"] = "--"
    if metric == "capex_per_h":
        style["color"] = "tab:green"
    elif metric in {
        "sector_op_margin_info",
        "sector_capacity_info_per_h",
        "sector_hh_util_info",
        "sector_util_info",
        "sector_demand_info_per_h",
        "unmet_demand_info_per_h",
    }:
        style["color"] = "tab:blue"
    elif metric in {
        "sector_op_margin_phys",
        "sector_capacity_physical_per_h",
        "sector_hh_util_physical",
        "sector_util_physical",
        "sector_demand_physical_per_h",
        "unmet_demand_physical_per_h",
    }:
        style["color"] = "#ff7f0e"
    elif metric == "corporate_nonbank_broad_roe_q":
        style["color"] = "tab:red"
    if metric == "automation":
        style["linewidth"] = 2.6
        style["linestyle"] = "-"
    elif metric == "corporate_eq_total_per_h":
        style["linewidth"] = 2.6
        style["linestyle"] = "-"
    elif metric == "capex_per_h":
        style["linewidth"] = 2.6
        style["linestyle"] = "-"
    elif metric in {
        "sector_op_margin_info",
        "sector_op_margin_phys",
        "corporate_nonbank_broad_roe_q",
    }:
        style["linestyle"] = "-"
    elif metric == "automation_flow":
        style["linewidth"] = 2.6
        style["linestyle"] = ":"
    elif metric == "sector_demand_info_per_h":
        style["linewidth"] = 2.0
        style["linestyle"] = "-"
    elif metric == "sector_demand_physical_per_h":
        style["linewidth"] = 2.0
        style["linestyle"] = "-"
    elif metric == "sector_util_info":
        style["linewidth"] = 2.0
        style["linestyle"] = "-"
    elif metric == "sector_util_physical":
        style["linewidth"] = 2.0
        style["linestyle"] = "-"
    elif metric == "unmet_demand_info_per_h":
        style["linewidth"] = 2.0
        style["linestyle"] = "--"
    elif metric == "unmet_demand_physical_per_h":
        style["linewidth"] = 2.0
        style["linestyle"] = ":"
    return style


def _compact_tick_label(value: float) -> str:
    x = float(value)
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:.1f}T"
    if ax >= 1e9:
        return f"{x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.1f}k"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax == 0:
        return "0"
    return f"{x:.3f}"


def _apply_compact_y_ticks(ax: Any) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: _compact_tick_label(val)))


def _normalized_mode(support_mode: str | None) -> str | None:
    if support_mode is None:
        return None
    mode = str(support_mode).strip().upper()
    if mode in {"UIS", "UBI"}:
        return mode
    return None


def _title_with_mode(title: str, support_mode: str | None) -> str:
    mode = _normalized_mode(support_mode)
    if mode is None:
        return title
    return f"{title} ({mode})"


def _annotate_trust_launch(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    x: Sequence[float],
    y: Sequence[float],
) -> None:
    """Call out the initial leveraged trust buy on trust-equity plots."""
    launch_idx = None
    for i, row in enumerate(rows):
        if bool(row.get("trust_active", False)) and float(row.get("trust_equity_pct", 0.0)) > 0.0:
            launch_idx = i
            break

    if launch_idx is None:
        return

    x_launch = float(x[launch_idx])
    y_launch = float(y[launch_idx])
    if not np.isfinite(y_launch):
        return
    ax.axvline(x_launch, color="0.45", linewidth=1.2, linestyle=":", alpha=0.8)
    ax.scatter([x_launch], [y_launch], color="0.15", s=26, zorder=5)
    ax.annotate(
        "Launch buy",
        xy=(x_launch, y_launch),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=9,
        color="0.15",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.6", "alpha": 0.9},
    )


def plot_metric_lines(
    rows: Sequence[Mapping[str, Any]],
    metrics: Iterable[str],
    *,
    title: str = "NewLoop Time Series",
    ax: Any = None,
    primary_ylabel: str | None = None,
    secondary_metrics: Iterable[str] | None = None,
    secondary_ylabel: str = "Secondary Scale",
    legend_loc: str = "best",
    support_mode: str | None = None,
) -> Any:
    """Plot one or more line metrics over simulation quarter."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]

    metric_list = list(metrics)
    secondary_set = set(secondary_metrics or [])
    primary_list = [m for m in metric_list if m not in secondary_set]
    secondary_list = [m for m in metric_list if m in secondary_set]

    if (not primary_list) and secondary_list:
        primary_list = [secondary_list[0]]
        secondary_list = secondary_list[1:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    primary_lines = []
    for metric in primary_list:
        x_plot, y_plot = _plot_points(rows, x, metric)
        (line,) = ax.plot(x_plot, y_plot, label=metric_label(metric), **_line_style(metric, secondary=False))
        primary_lines.append(line)
        if metric == "trust_equity_pct":
            _annotate_trust_launch(ax, rows, x, _series(rows, metric))

    _apply_compact_y_ticks(ax)
    if primary_ylabel is None:
        if len(primary_list) == 1:
            primary_ylabel = metric_label(primary_list[0])
        else:
            primary_ylabel = "Primary Scale"
    ax.set_ylabel(primary_ylabel)

    secondary_lines = []
    if secondary_list:
        ax2 = ax.twinx()
        for metric in secondary_list:
            x_plot, y_plot = _plot_points(rows, x, metric)
            (line,) = ax2.plot(x_plot, y_plot, label=metric_label(metric), **_line_style(metric, secondary=True))
            secondary_lines.append(line)
        ax2.set_ylabel(secondary_ylabel)
        _apply_compact_y_ticks(ax2)
        ax2.grid(False)

    ax.set_title(_title_with_mode(title, support_mode))
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.25)

    all_lines = primary_lines + secondary_lines
    if all_lines:
        ax.legend(all_lines, [ln.get_label() for ln in all_lines], loc=legend_loc)

    return fig


def plot_income_support_funding_mix(
    rows: Sequence[Mapping[str, Any]],
    ax: Any = None,
    *,
    support_mode: str = "UBI",
) -> Any:
    """Stacked-area chart for income-support funding channels."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]
    mode = _normalized_mode(support_mode) or "UBI"

    fund = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "uis_from_fund_dep_per_h"), dtype=float), nan=0.0))
    gov = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "uis_from_gov_dep_per_h"), dtype=float), nan=0.0))
    issued = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "uis_issued_per_h"), dtype=float), nan=0.0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    layers = [fund, gov, issued]
    labels = [
        metric_label("uis_from_fund_dep_per_h"),
        metric_label("uis_from_gov_dep_per_h"),
        metric_label("uis_issued_per_h"),
    ]

    ax.stackplot(x, *layers, labels=labels, alpha=0.8)
    ax.set_title(_title_with_mode("Income Support Funding Mix", mode))
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Per-Household")
    _apply_compact_y_ticks(ax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def plot_fund_inflows(
    rows: Sequence[Mapping[str, Any]],
    ax: Any = None,
    *,
    support_mode: str = "UBI",
) -> Any:
    """Stacked-area chart for FUND inflow channels."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]
    mode = _normalized_mode(support_mode) or "UBI"

    fund_div = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "fund_dividend_inflow_per_h"), dtype=float), nan=0.0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    layers = [fund_div]
    labels = [metric_label("fund_dividend_inflow_per_h")]

    ax.stackplot(x, *layers, labels=labels, alpha=0.8)
    ax.set_title(_title_with_mode("Fund Inflows", mode))
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Per-Household")
    _apply_compact_y_ticks(ax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def plot_cumulative_income_support_funding(
    rows: Sequence[Mapping[str, Any]],
    ax: Any = None,
    *,
    support_mode: str = "UBI",
    household_count: int | None = None,
) -> Any:
    """Line chart for cumulative income-support funding channels over time."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]
    mode = _normalized_mode(support_mode) or "UBI"

    gov = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "uis_from_gov_dep_per_h"), dtype=float), nan=0.0))
    issued = np.maximum(0.0, np.nan_to_num(np.asarray(_series(rows, "uis_issued_per_h"), dtype=float), nan=0.0))

    n_hh = int(household_count) if household_count is not None else 1
    if n_hh <= 0:
        n_hh = 1
    scale = float(n_hh)

    gov_cum = np.cumsum(gov * scale)
    issued_cum = np.cumsum(issued * scale)
    public_total_cum = gov_cum + issued_cum

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    ax.plot(x, gov_cum, linewidth=2.0, label="Cumulative GOV Funding")
    ax.plot(x, issued_cum, linewidth=2.0, label="Cumulative Issuance Funding")
    ax.plot(x, public_total_cum, linewidth=2.4, linestyle="--", label="Cumulative Public Total")
    ax.set_title(_title_with_mode("Cumulative Income Support Funding", mode))
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Amount (Economy Total)")
    _apply_compact_y_ticks(ax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def plot_gini_series(
    rows: Sequence[Mapping[str, Any]],
    ax: Any = None,
    *,
    support_mode: str | None = None,
) -> Any:
    """Plot disposable and wealth Gini series on a dedicated 0-1 scale."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    metrics = ["gini_disp", "gini_wealth"]
    for metric in metrics:
        ax.plot(x, _series(rows, metric), linewidth=2.0, label=metric_label(metric))

    ax.set_title(_title_with_mode("Gini Metrics", support_mode))
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Gini (0-1)")
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.2f}"))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_default_dashboard(
    rows: Sequence[Mapping[str, Any]],
    support_mode: str = "UBI",
    *,
    household_count: int | None = None,
) -> Any:
    """Build a 2x2 dashboard for automation, gini, and funding diagnostics."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    mode = _normalized_mode(support_mode) or "UBI"
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    automation_ax = axs[0][0]
    plot_metric_lines(
        rows,
        [
            "automation_info",
            "automation_phys",
            "automation",
            "automation_info_flow",
            "automation_phys_flow",
            "automation_flow",
        ],
        title="Automation By Sector",
        ax=automation_ax,
        primary_ylabel="Automation Level",
        secondary_metrics=["automation_info_flow", "automation_phys_flow", "automation_flow"],
        secondary_ylabel="Automation Flow (Δ per quarter)",
        legend_loc="upper right",
        support_mode=mode,
    )
    automation_ax.set_ylim(-0.05, 1.0)
    plot_gini_series(rows, ax=axs[0][1], support_mode=mode)
    plot_income_support_funding_mix(rows, ax=axs[1][0], support_mode=mode)
    plot_cumulative_income_support_funding(
        rows,
        ax=axs[1][1],
        support_mode=mode,
        household_count=household_count,
    )

    return fig


def plot_uis_funding_mix(rows: Sequence[Mapping[str, Any]], ax: Any = None) -> Any:
    """Backward-compatible alias."""
    return plot_income_support_funding_mix(rows, ax=ax, support_mode="UIS")


def plot_distribution_compare(
    before: Sequence[float],
    after: Sequence[float],
    *,
    title: str,
    x_label: str,
    x_limits: tuple[float, float] | None = None,
    ax: Any = None,
    bins: int = 60,
) -> Any:
    """Overlay before/after cumulative distributions (ECDF) for one distribution."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    b = b[np.isfinite(b)]
    a = a[np.isfinite(a)]

    if b.size == 0 or a.size == 0:
        raise ValueError("Distribution plot requires non-empty before and after arrays.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    b_sorted = np.sort(b)
    a_sorted = np.sort(a)
    y_b = np.arange(1, b_sorted.size + 1, dtype=float) / float(b_sorted.size)
    y_a = np.arange(1, a_sorted.size + 1, dtype=float) / float(a_sorted.size)
    ax.step(b_sorted, y_b, where="post", linewidth=2.0, label="Before")
    ax.step(a_sorted, y_a, where="post", linewidth=2.0, label="After")

    med_b = float(np.median(b))
    med_a = float(np.median(a))
    ax.axvline(med_b, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(med_a, linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Share of Households")
    if x_limits is not None:
        x_lo, x_hi = float(x_limits[0]), float(x_limits[1])
        if x_hi > x_lo:
            ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.2f}%"))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def _anchored_distribution_range_and_edges(
    before: Sequence[float],
    after: Sequence[float],
    *,
    bins: int,
) -> tuple[tuple[float, float], np.ndarray]:
    """Build a histogram grid anchored to the before distribution, extended to cover both series."""
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    b = b[np.isfinite(b)]
    a = a[np.isfinite(a)]

    if b.size == 0 or a.size == 0:
        raise ValueError("Distribution plot requires non-empty before and after arrays.")

    anchor_lo = float(np.percentile(b, 1.0))
    anchor_hi = float(np.percentile(b, 99.0))
    if anchor_hi <= anchor_lo:
        anchor_lo = float(np.min(b))
        anchor_hi = float(np.max(b))
        if anchor_hi <= anchor_lo:
            anchor_hi = anchor_lo + 1.0

    n_bins_base = int(max(20, bins))
    bin_width = (anchor_hi - anchor_lo) / float(n_bins_base)
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        bin_width = max(1.0, abs(anchor_hi), abs(anchor_lo), 1.0) / float(n_bins_base)

    data_min = float(min(np.min(b), np.min(a)))
    data_max = float(max(np.max(b), np.max(a)))

    left_steps = int(max(0.0, np.ceil((anchor_lo - data_min) / bin_width)))
    right_steps = int(max(0.0, np.ceil((data_max - anchor_lo) / bin_width)))

    left_edge = anchor_lo - float(left_steps) * bin_width
    right_edge = anchor_lo + float(right_steps) * bin_width
    if right_edge <= left_edge:
        right_edge = left_edge + bin_width

    n_bins = int(max(1, round((right_edge - left_edge) / bin_width)))
    edges = left_edge + bin_width * np.arange(n_bins + 1, dtype=float)
    if edges.size < 2:
        edges = np.asarray([left_edge, right_edge], dtype=float)
    else:
        edges[-1] = right_edge
    return (float(left_edge), float(right_edge)), edges


def _series_percentile_window(
    values: Sequence[float],
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 95.0,
) -> tuple[float, float]:
    """Return a stable percentile window for one series."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Distribution plot requires a non-empty array.")

    q_lo = float(np.percentile(arr, lo_pct))
    q_hi = float(np.percentile(arr, hi_pct))
    if q_hi <= q_lo:
        q_lo = float(np.min(arr))
        q_hi = float(np.max(arr))
        if q_hi <= q_lo:
            q_hi = q_lo + 1.0
    return q_lo, q_hi


def plot_distribution_share(
    before: Sequence[float],
    after: Sequence[float],
    *,
    title: str,
    x_label: str,
    x_limits: tuple[float, float] | None = None,
    ax: Any = None,
    bins: int = 60,
    edges: Sequence[float] | None = None,
    after_edges: Sequence[float] | None = None,
) -> Any:
    """Overlay before/after share-per-bin histograms."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    b = b[np.isfinite(b)]
    a = a[np.isfinite(a)]

    if b.size == 0 or a.size == 0:
        raise ValueError("Distribution plot requires non-empty before and after arrays.")

    if edges is not None:
        edges_arr = np.asarray(edges, dtype=float)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("Histogram edges must be a one-dimensional sequence with at least two entries.")
        q_lo = float(edges_arr[0])
        q_hi = float(edges_arr[-1])
    elif x_limits is not None:
        q_lo = float(x_limits[0])
        q_hi = float(x_limits[1])
        edges_arr = np.linspace(q_lo, q_hi, int(max(20, bins)) + 1)
    else:
        q_lo = float(min(np.percentile(b, 1.0), np.percentile(a, 1.0)))
        q_hi = float(max(np.percentile(b, 99.0), np.percentile(a, 99.0)))
        edges_arr = np.linspace(q_lo, q_hi, int(max(20, bins)) + 1)
    if q_hi <= q_lo:
        q_lo = float(min(b.min(), a.min()))
        q_hi = float(max(b.max(), a.max()))
        if q_hi <= q_lo:
            q_hi = q_lo + 1.0
        edges_arr = np.linspace(q_lo, q_hi, int(max(20, bins)) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    after_edges_arr = edges_arr
    if after_edges is not None:
        after_edges_arr = np.asarray(after_edges, dtype=float)
        if after_edges_arr.ndim != 1 or after_edges_arr.size < 2:
            raise ValueError("After-series histogram edges must be a one-dimensional sequence with at least two entries.")

    # For zoomed views, fold observations beyond the visible window into the
    # edge bins instead of dropping them entirely. That preserves the presence
    # of the left/right tails while still focusing the x-axis on the requested
    # percentile band.
    if edges is not None:
        b = np.clip(b, float(edges_arr[0]), float(edges_arr[-1]))
    if after_edges is not None:
        a = np.clip(a, float(after_edges_arr[0]), float(after_edges_arr[-1]))
    elif edges is not None:
        a = np.clip(a, float(edges_arr[0]), float(edges_arr[-1]))

    if b.size == 0 or a.size == 0:
        raise ValueError("Truncated histogram window removed all observations from one series.")

    w_b = np.full(b.size, 1.0 / float(b.size), dtype=float)
    w_a = np.full(a.size, 1.0 / float(a.size), dtype=float)
    before_hist = ax.hist(b, bins=edges_arr, weights=w_b, histtype="step", linewidth=2.0, label="Before")
    after_hist = ax.hist(a, bins=after_edges_arr, weights=w_a, histtype="step", linewidth=2.0, label="After")
    before_color = before_hist[2][0].get_edgecolor()
    after_color = after_hist[2][0].get_edgecolor()

    ax.axvline(float(np.median(b)), color=before_color, linestyle=":", linewidth=1.8, alpha=0.9)
    ax.axvline(float(np.median(a)), color=after_color, linestyle=":", linewidth=1.8, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Share of Households")
    if x_limits is not None:
        ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
    else:
        ax.set_xlim(q_lo, q_hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.2f}%"))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_income_distribution(
    income_before: Sequence[float],
    income_after: Sequence[float],
    *,
    value_label: str,
    support_mode: str | None = None,
) -> Any:
    """Single-panel income before/after ECDF."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    plot_distribution_compare(
        income_before,
        income_after,
        title=_title_with_mode("Income Distribution (Before vs After)", support_mode),
        x_label=value_label,
        ax=ax,
    )
    return fig


def plot_income_distribution_dual(
    income_before: Sequence[float],
    income_after: Sequence[float],
    *,
    value_label: str,
    support_mode: str | None = None,
) -> Any:
    """Two-panel income distributions: cumulative + share-per-bin."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    b = np.asarray(income_before, dtype=float)
    a = np.asarray(income_after, dtype=float)
    b = b[np.isfinite(b)]
    a = a[np.isfinite(a)]
    if b.size == 0 or a.size == 0:
        raise ValueError("Distribution plot requires non-empty before and after arrays.")

    def _robust_sigma(arr: np.ndarray) -> float:
        q25 = float(np.percentile(arr, 25.0))
        q75 = float(np.percentile(arr, 75.0))
        return max(1e-9, (q75 - q25) / 1.349)

    med_b = float(np.median(b))
    med_a = float(np.median(a))
    sig_b = _robust_sigma(b)
    sig_a = _robust_sigma(a)
    x_lo = float(min(med_b - (3.0 * sig_b), med_a - (3.0 * sig_a)))
    x_hi = float(max(med_b + (3.0 * sig_b), med_a + (3.0 * sig_a)))
    x_lo = max(x_lo, float(min(np.min(b), np.min(a))))
    x_hi = min(x_hi, float(max(np.max(b), np.max(a))))
    if x_hi <= x_lo:
        x_lo = float(min(np.min(b), np.min(a)))
        x_hi = float(max(np.max(b), np.max(a)))
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0
    x_limits = (x_lo, x_hi)
    edges = np.linspace(x_lo, x_hi, 61, dtype=float)
    before_vals = b[(b >= x_lo) & (b <= x_hi)]
    after_vals = a[(a >= x_lo) & (a <= x_hi)]
    before_counts, _ = np.histogram(before_vals, bins=edges)
    after_counts, _ = np.histogram(after_vals, bins=edges)
    before_share = before_counts.astype(float) / max(1.0, float(b.size))
    after_share = after_counts.astype(float) / max(1.0, float(a.size))
    below_b = int(np.sum(b < x_lo))
    below_a = int(np.sum(a < x_lo))
    above_b = int(np.sum(b > x_hi))
    above_a = int(np.sum(a > x_hi))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_distribution_compare(
        income_before,
        income_after,
        title=_title_with_mode("Disposable Income Distribution (Cumulative)", support_mode),
        x_label=value_label,
        x_limits=x_limits,
        ax=axs[0],
    )
    axs[1].step(edges[:-1], before_share, where="post", color="#1f77b4", linewidth=2.0, label="Before")
    axs[1].step(edges[:-1], after_share, where="post", color="#ff7f0e", linewidth=2.0, label="After")
    axs[1].axvline(med_b, color="#1f77b4", linestyle=":", linewidth=1.8, alpha=0.9)
    axs[1].axvline(med_a, color="#ff7f0e", linestyle=":", linewidth=1.8, alpha=0.9)
    axs[1].set_title(_title_with_mode("Disposable Income Distribution (% per Bin, Median +/- 3 Robust Sigma)", support_mode))
    axs[1].set_xlabel(value_label)
    axs[1].set_ylabel("Share of Households")
    axs[1].set_xlim(x_lo, x_hi)
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.2f}%"))
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="best")
    note = (
        f"Clipped tails: below Before {below_b}, After {below_a}; "
        f"above Before {above_b}, After {above_a}"
    )
    axs[1].text(0.01, 0.99, note, transform=axs[1].transAxes, ha="left", va="top", fontsize=9, color="0.35")
    return fig


def plot_income_distribution_by_group(
    income_groups: Mapping[str, Sequence[float]],
    *,
    value_label: str,
    support_mode: str | None = None,
    overall_income: Sequence[float] | None = None,
    label_map: Mapping[str, str] | None = None,
    color_map: Mapping[str, str] | None = None,
    ordered_keys: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Show the after-income histogram split by a supplied household grouping."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    label_map = dict(label_map or {
        "renters": "Renters",
        "mortgagors": "Mortgagors",
        "outright_owners": "Outright Owners",
    })
    color_map = dict(color_map or {
        "renters": "#1f77b4",
        "mortgagors": "#ff7f0e",
        "outright_owners": "#2ca02c",
    })
    ordered_keys = tuple(ordered_keys or ("renters", "mortgagors", "outright_owners"))

    cleaned_groups: Dict[str, np.ndarray] = {}
    for key in ordered_keys:
        values = income_groups.get(key, [])
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            cleaned_groups[key] = arr

    if not cleaned_groups:
        raise ValueError("Income group plot requires at least one non-empty group.")

    if overall_income is None:
        overall_vals = np.concatenate(list(cleaned_groups.values()))
    else:
        overall_vals = np.asarray(overall_income, dtype=float)
        overall_vals = overall_vals[np.isfinite(overall_vals)]
        if overall_vals.size == 0:
            overall_vals = np.concatenate(list(cleaned_groups.values()))

    def _robust_sigma(arr: np.ndarray) -> float:
        q25 = float(np.percentile(arr, 25.0))
        q75 = float(np.percentile(arr, 75.0))
        return max(1e-9, (q75 - q25) / 1.349)

    med = float(np.median(overall_vals))
    sigma = _robust_sigma(overall_vals)
    x_lo = max(float(np.min(overall_vals)), med - (3.0 * sigma))
    x_hi = min(float(np.max(overall_vals)), med + (3.0 * sigma))
    if x_hi <= x_lo:
        x_lo = float(np.min(overall_vals))
        x_hi = float(np.max(overall_vals))
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0

    edges = np.linspace(x_lo, x_hi, 61, dtype=float)
    total_n = max(1.0, float(overall_vals.size))

    fig, ax = plt.subplots(figsize=figsize or (13, 4.5), constrained_layout=True)
    overall_clip = overall_vals[(overall_vals >= x_lo) & (overall_vals <= x_hi)]
    overall_counts, _ = np.histogram(overall_clip, bins=edges)
    overall_share = overall_counts.astype(float) / total_n
    ax.step(edges[:-1], overall_share, where="post", color="#ff7f0e", linewidth=2.4, linestyle="--", label="All Households")

    for key in ordered_keys:
        arr = cleaned_groups.get(key)
        if arr is None or arr.size == 0:
            continue
        clipped = arr[(arr >= x_lo) & (arr <= x_hi)]
        counts, _ = np.histogram(clipped, bins=edges)
        share = counts.astype(float) / total_n
        group_share = 100.0 * float(arr.size) / total_n
        ax.step(
            edges[:-1],
            share,
            where="post",
            color=color_map[key],
            linewidth=2.0,
            label=f"{label_map[key]} ({group_share:.1f}%)",
        )
        ax.axvline(float(np.median(arr)), color=color_map[key], linestyle=":", linewidth=1.4, alpha=0.85)

    chart_title = title or "Disposable Income Distribution By Group (After)"
    ax.set_title(_title_with_mode(chart_title, support_mode))
    ax.set_xlabel(value_label)
    ax.set_ylabel("Share of Households")
    ax.set_xlim(x_lo, x_hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.2f}%"))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_wealth_distributions_full_zoom(
    rows: Sequence[Mapping[str, Any]],
    wealth_before: Sequence[float],
    wealth_after: Sequence[float],
    *,
    value_label: str,
    zoom_lo_pct: float = 2.0,
    zoom_hi_pct: float = 98.0,
    support_mode: str | None = None,
) -> Any:
    """Wealth reservoirs over time + zoomed wealth histogram."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    w_b = np.asarray(wealth_before, dtype=float)
    w_a = np.asarray(wealth_after, dtype=float)
    w_b = w_b[np.isfinite(w_b)]
    w_a = w_a[np.isfinite(w_a)]
    if w_b.size == 0 or w_a.size == 0:
        raise ValueError("Wealth distribution plot requires non-empty before and after arrays.")

    lo = float(max(0.0, min(100.0, zoom_lo_pct)))
    hi = float(max(0.0, min(100.0, zoom_hi_pct)))
    if hi <= lo:
        hi = min(100.0, lo + 1.0)
        lo = max(0.0, hi - 1.0)

    all_w = np.concatenate([w_b, w_a])
    x_full_lo = float(np.min(all_w))
    x_full_hi = float(np.max(all_w))
    if x_full_hi <= x_full_lo:
        x_full_hi = x_full_lo + 1.0

    def _robust_sigma(arr: np.ndarray) -> float:
        q25 = float(np.percentile(arr, 25.0))
        q75 = float(np.percentile(arr, 75.0))
        return max(1e-9, (q75 - q25) / 1.349)

    med_b = float(np.median(w_b))
    med_a = float(np.median(w_a))
    sig_b = _robust_sigma(w_b)
    sig_a = _robust_sigma(w_a)
    x_lo = float(min(med_b - (3.0 * sig_b), med_a - (3.0 * sig_a)))
    x_hi = float(max(med_b + (3.0 * sig_b), med_a + (3.0 * sig_a)))
    x_lo = max(x_lo, x_full_lo)
    x_hi = min(x_hi, x_full_hi)
    if x_hi <= x_lo:
        x_lo = x_full_lo
        x_hi = x_full_hi

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    t = [int(row.get("t", idx)) for idx, row in enumerate(rows)]
    deposits = [float(row.get("hh_deposits_per_h", 0.0)) for row in rows]
    housing_value = [float(row.get("hh_housing_value_per_h", 0.0)) for row in rows]
    direct_equity = [float(row.get("private_eq_per_h", 0.0)) for row in rows]
    trust_value = [float(row.get("trust_value_per_h", 0.0)) for row in rows]
    debt = [-float(row.get("hh_debt_per_h", 0.0)) for row in rows]
    mortgage_debt = [-float(row.get("hh_mortgage_debt_per_h", 0.0)) for row in rows]
    revolving_debt = [-float(row.get("hh_revolving_debt_per_h", 0.0)) for row in rows]
    net_worth = [
        float(dep + hv + eq + trust + debt_val)
        for dep, hv, eq, trust, debt_val in zip(deposits, housing_value, direct_equity, trust_value, debt)
    ]

    ax_left = axs[0]
    ax_left.plot(t, deposits, label="Deposits", color="#1f77b4", linewidth=2.0)
    ax_left.plot(t, housing_value, label="Housing Value", color="#17becf", linewidth=2.0)
    ax_left.plot(t, direct_equity, label="Direct Equity", color="#ff7f0e", linewidth=2.0)
    ax_left.plot(t, trust_value, label="Trust Value", color="#2ca02c", linewidth=2.0)
    ax_left.plot(t, debt, label="Debt", color="#d62728", linewidth=2.0)
    ax_left.plot(t, mortgage_debt, label="Mortgage Debt", color="#d62728", linewidth=1.8, linestyle=":")
    ax_left.plot(t, revolving_debt, label="Revolving Debt", color="#8c564b", linewidth=1.8, linestyle=":")
    ax_left.plot(t, net_worth, label="Net Worth", color="#9467bd", linewidth=2.4, linestyle="--")
    ax_left.axhline(0.0, color="0.4", linewidth=1.0, alpha=0.6)
    ax_left.set_title(_title_with_mode("Household Wealth Reservoirs", support_mode))
    ax_left.set_xlabel("Quarter")
    ax_left.set_ylabel(value_label)
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(loc="best")

    ax_right = axs[1]
    edges = np.linspace(x_lo, x_hi, 81, dtype=float)
    before_vals = w_b[(w_b >= x_lo) & (w_b <= x_hi)]
    after_vals = w_a[(w_a >= x_lo) & (w_a <= x_hi)]
    before_counts, _ = np.histogram(before_vals, bins=edges)
    after_counts, _ = np.histogram(after_vals, bins=edges)
    before_share = before_counts.astype(float) / max(1.0, float(w_b.size))
    after_share = after_counts.astype(float) / max(1.0, float(w_a.size))

    ax_right.step(edges[:-1], before_share, where="post", color="#1f77b4", linewidth=2.0, label="Before")
    ax_right.step(edges[:-1], after_share, where="post", color="#ff7f0e", linewidth=2.0, label="After")
    ax_right.axvline(float(np.median(w_b)), color="#1f77b4", linestyle=":", linewidth=1.8, alpha=0.9)
    ax_right.axvline(float(np.median(w_a)), color="#ff7f0e", linestyle=":", linewidth=1.8, alpha=0.9)
    ax_right.set_title(_title_with_mode("Wealth Distribution (Bucketed Lines, Median +/- 3 Robust Sigma)", support_mode))
    ax_right.set_xlabel(value_label)
    ax_right.set_ylabel("Share of Households")
    ax_right.set_xlim(x_lo, x_hi)
    from matplotlib.ticker import FuncFormatter
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.2f}%"))
    ax_right.grid(alpha=0.25)
    ax_right.legend(loc="best")
    n_before_below = int(np.sum(w_b < x_lo))
    n_after_below = int(np.sum(w_a < x_lo))
    n_before_above = int(np.sum(w_b > x_hi))
    n_after_above = int(np.sum(w_a > x_hi))
    if n_before_below > 0 or n_after_below > 0 or n_before_above > 0 or n_after_above > 0:
        note = (
            f"Clipped tails: below Before {n_before_below}, After {n_after_below}; "
            f"above Before {n_before_above}, After {n_after_above}"
        )
        ax_right.text(0.01, 0.99, note, transform=ax_right.transAxes, ha="left", va="top", fontsize=9, color="0.35")
    return fig


def plot_mortgage_stock_over_time(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_label: str,
    support_mode: str | None = None,
) -> Any:
    """Plot mortgage stock values and active mortgage count over time."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    t = [int(row.get("t", idx)) for idx, row in enumerate(rows)]
    active_count = [float(row.get("hh_mortgage_active_count", 0.0)) for row in rows]
    balance_total = [float(row.get("hh_mortgage_balance_total", 0.0)) for row in rows]
    principal_total = [float(row.get("hh_mortgage_orig_principal_total", 0.0)) for row in rows]

    fig, ax_left = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    ax_right = ax_left.twinx()

    left_line = ax_left.plot(
        t,
        active_count,
        color="#2ca02c",
        linewidth=2.2,
        label="Active Mortgages",
    )[0]
    right_balance = ax_right.plot(
        t,
        balance_total,
        color="#1f77b4",
        linewidth=2.0,
        label="Outstanding Mortgage Value",
    )[0]
    right_principal = ax_right.plot(
        t,
        principal_total,
        color="#ff7f0e",
        linewidth=2.0,
        label="Outstanding Principal Value",
    )[0]

    ax_left.set_title(_title_with_mode("Mortgage Stock And Count", support_mode))
    ax_left.set_xlabel("Quarter")
    ax_left.set_ylabel("Active Mortgages")
    ax_right.set_ylabel(value_label)
    ax_left.grid(alpha=0.25)

    lines = [left_line, right_balance, right_principal]
    ax_left.legend(lines, [line.get_label() for line in lines], loc="best")
    return fig


def plot_income_wealth_distributions(
    income_before: Sequence[float],
    income_after: Sequence[float],
    wealth_before: Sequence[float],
    wealth_after: Sequence[float],
    *,
    value_label: str,
    support_mode: str | None = None,
) -> Any:
    """Build a 1x2 before/after distribution figure for income and wealth."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_distribution_compare(
        income_before,
        income_after,
        title=_title_with_mode("Income Distribution (Before vs After)", support_mode),
        x_label=value_label,
        ax=axs[0],
    )
    plot_distribution_share(
        wealth_before,
        wealth_after,
        title=_title_with_mode("Wealth Distribution (Histogram)", support_mode),
        x_label=value_label,
        ax=axs[1],
    )
    return fig
