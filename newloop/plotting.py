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
    "price_level": "Price Level",
    "inflation": "Inflation",
    "real_avg_income": "Real Avg Income",
    "real_consumption": "Real Consumption",
    "gini_market": "Gini (Pre-Tax/Pre-Transfer)",
    "gini_disp": "Gini (Disposable)",
    "gini_wealth": "Gini (Wealth)",
    "pop_dti_p90": "Debt-Service-to-Income (DTI) P90",
    "pop_dti_w_p90": "Debt-Service-to-Income (DTI) P90 (Wages)",
    "trust_equity_pct": "Trust Equity %",
    "uis_per_h": "Income Support / Household",
    "uis_from_fund_dep_per_h": "Income Support from FUND",
    "uis_from_gov_dep_per_h": "Income Support from GOV",
    "uis_issued_per_h": "Income Support Issued",
    "corp_tax_rate_eff": "Effective Corporate Tax Rate",
    "wages_total": "Total Wage Base",
    "trust_value_per_h": "Trust Value / Household",
}

DEFAULT_LINE_METRICS: List[str] = [
    "real_consumption",
    "trust_equity_pct",
]


def _require_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not rows:
        raise ValueError("No rows provided for plotting.")
    return list(rows)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def metric_options() -> Dict[str, str]:
    return dict(METRIC_LABELS)


def _series(rows: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
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
    if metric == "automation":
        style["linewidth"] = 2.6
        style["linestyle"] = "-"
    elif metric == "automation_flow":
        style["linewidth"] = 2.6
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
    support_mode: str = "UIS",
) -> Any:
    """Stacked-area chart for income-support funding channels."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]
    mode = _normalized_mode(support_mode) or "UIS"

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


def plot_cumulative_income_support_funding(
    rows: Sequence[Mapping[str, Any]],
    ax: Any = None,
    *,
    support_mode: str = "UIS",
    household_count: int | None = None,
) -> Any:
    """Line chart for cumulative income-support funding channels over time."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]
    mode = _normalized_mode(support_mode) or "UIS"

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
    """Plot pre-tax/pre-transfer, disposable, and wealth Gini series on a dedicated 0-1 scale."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    metrics = ["gini_market", "gini_disp", "gini_wealth"]
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
    support_mode: str = "UIS",
    *,
    household_count: int | None = None,
) -> Any:
    """Build a 2x2 dashboard for automation, gini, and funding diagnostics."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    mode = _normalized_mode(support_mode) or "UIS"
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


def plot_distribution_share(
    before: Sequence[float],
    after: Sequence[float],
    *,
    title: str,
    x_label: str,
    ax: Any = None,
    bins: int = 60,
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

    q_lo = float(min(np.percentile(b, 1.0), np.percentile(a, 1.0)))
    q_hi = float(max(np.percentile(b, 99.0), np.percentile(a, 99.0)))
    if q_hi <= q_lo:
        q_lo = float(min(b.min(), a.min()))
        q_hi = float(max(b.max(), a.max()))
        if q_hi <= q_lo:
            q_hi = q_lo + 1.0

    edges = np.linspace(q_lo, q_hi, int(max(20, bins)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    w_b = np.full(b.size, 1.0 / float(b.size), dtype=float)
    w_a = np.full(a.size, 1.0 / float(a.size), dtype=float)
    ax.hist(b, bins=edges, weights=w_b, histtype="step", linewidth=2.0, label="Before")
    ax.hist(a, bins=edges, weights=w_a, histtype="step", linewidth=2.0, label="After")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Share of Households")
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

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_distribution_compare(
        income_before,
        income_after,
        title=_title_with_mode("Income Distribution (Cumulative)", support_mode),
        x_label=value_label,
        ax=axs[0],
    )
    plot_distribution_share(
        income_before,
        income_after,
        title=_title_with_mode("Income Distribution (% per Bin)", support_mode),
        x_label=value_label,
        ax=axs[1],
    )
    return fig


def plot_wealth_distributions_full_zoom(
    wealth_before: Sequence[float],
    wealth_after: Sequence[float],
    *,
    value_label: str,
    zoom_lo_pct: float = 2.0,
    zoom_hi_pct: float = 98.0,
    support_mode: str | None = None,
) -> Any:
    """Two-panel wealth ECDF: full range + zoomed percentile window."""
    import matplotlib.pyplot as plt

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
    x_lo = float(np.percentile(all_w, lo))
    x_hi = float(np.percentile(all_w, hi))
    if x_hi <= x_lo:
        x_lo = float(np.min(all_w))
        x_hi = float(np.max(all_w))
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_distribution_compare(
        w_b,
        w_a,
        title=_title_with_mode("Wealth Distribution (Full Range)", support_mode),
        x_label=value_label,
        ax=axs[0],
    )
    plot_distribution_compare(
        w_b,
        w_a,
        title=_title_with_mode(f"Wealth Distribution (Zoomed p{int(round(lo))} to p{int(round(hi))})", support_mode),
        x_label=value_label,
        x_limits=(x_lo, x_hi),
        ax=axs[1],
    )
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
    plot_distribution_compare(
        wealth_before,
        wealth_after,
        title=_title_with_mode("Wealth Distribution (Before vs After)", support_mode),
        x_label=value_label,
        ax=axs[1],
    )
    return fig
