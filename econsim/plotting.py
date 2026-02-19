# Author: Roger Ison   roger@miximum.info
"""Reusable plotting layer for EconomySim outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

METRIC_LABELS: Dict[str, str] = {
    "automation": "Automation",
    "automation_info": "Automation (Info)",
    "automation_phys": "Automation (Physical)",
    "price_level": "Price Level",
    "inflation": "Inflation",
    "real_avg_income": "Real Avg Income",
    "real_consumption": "Real Consumption",
    "gini_market": "Gini (Market)",
    "gini_disp": "Gini (Disposable)",
    "gini_wealth": "Gini (Wealth)",
    "pop_dti_p90": "DTI P90",
    "pop_dti_w_p90": "DTI P90 (Wages)",
    "trust_equity_pct": "Trust Equity %",
    "ubi_per_h": "UBI / Household",
    "ubi_from_fund_dep_per_h": "UBI from FUND",
    "ubi_from_gov_dep_per_h": "UBI from GOV",
    "ubi_issued_per_h": "UBI Issued",
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
    return [float(r.get(metric, 0.0)) for r in rows]


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


def plot_metric_lines(
    rows: Sequence[Mapping[str, Any]],
    metrics: Iterable[str],
    *,
    title: str = "EconomySim Time Series",
    ax: Any = None,
    secondary_metrics: Iterable[str] | None = None,
    secondary_ylabel: str = "Secondary Scale",
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
        (line,) = ax.plot(x, _series(rows, metric), linewidth=2.0, label=metric_label(metric))
        primary_lines.append(line)

    _apply_compact_y_ticks(ax)
    secondary_lines = []
    if secondary_list:
        ax2 = ax.twinx()
        for metric in secondary_list:
            (line,) = ax2.plot(x, _series(rows, metric), linewidth=2.0, linestyle="--", label=metric_label(metric))
            secondary_lines.append(line)
        ax2.set_ylabel(secondary_ylabel)
        _apply_compact_y_ticks(ax2)
        ax2.grid(False)

    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.25)

    all_lines = primary_lines + secondary_lines
    if all_lines:
        ax.legend(all_lines, [ln.get_label() for ln in all_lines], loc="best")

    return fig


def plot_ubi_funding_mix(rows: Sequence[Mapping[str, Any]], ax: Any = None) -> Any:
    """Stacked-area chart for UBI funding channels."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    x = [int(r.get("t", i)) for i, r in enumerate(rows)]

    fund = _series(rows, "ubi_from_fund_dep_per_h")
    gov = _series(rows, "ubi_from_gov_dep_per_h")
    issued = _series(rows, "ubi_issued_per_h")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    ax.stackplot(
        x,
        fund,
        gov,
        issued,
        labels=[metric_label("ubi_from_fund_dep_per_h"), metric_label("ubi_from_gov_dep_per_h"), metric_label("ubi_issued_per_h")],
        alpha=0.8,
    )
    ax.set_title("UBI Funding Mix")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Per-Household")
    _apply_compact_y_ticks(ax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    return fig


def plot_gini_series(rows: Sequence[Mapping[str, Any]], ax: Any = None) -> Any:
    """Plot market, disposable, and wealth Gini series on a dedicated 0-1 scale."""
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

    ax.set_title("Gini Metrics")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Gini (0-1)")
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.2f}"))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_default_dashboard(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Build a 2x2 default dashboard figure for quick inspection."""
    import matplotlib.pyplot as plt

    rows = _require_rows(rows)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    plot_metric_lines(rows, ["automation_info", "automation_phys"], title="Automation By Sector", ax=axs[0][0])
    plot_gini_series(rows, ax=axs[0][1])
    plot_ubi_funding_mix(rows, ax=axs[1][0])
    plot_metric_lines(
        rows,
        ["real_consumption", "real_avg_income"],
        title="Real Household Outcomes",
        ax=axs[1][1],
        secondary_metrics=["real_avg_income"],
        secondary_ylabel="Real Avg Income",
    )

    return fig
