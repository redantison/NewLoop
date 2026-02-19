# Author: Roger Ison   roger@miximum.info
"""Streamlit app entrypoint for EconomySim.

Run with:
  streamlit run econsim/sl-econsim.py
"""

from __future__ import annotations

import copy
import csv
import io
import json
from typing import Any, Dict, List, Sequence

# Script mode:
#  streamlit run econsim/sl-econsim.py
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from econsim.plotting import (
        DEFAULT_LINE_METRICS,
        metric_options,
        plot_default_dashboard,
        plot_gini_series,
        plot_income_wealth_distributions,
        plot_metric_lines,
    )
    from econsim.config import get_default_config
    from econsim.results import run_simulation, summarize_rows
    from econsim.streamlit_params import (
        NON_GINI_METRIC_GROUPS,
        PARAMETER_CONTROLS,
        RUN_DEFAULT_QUARTERS,
        RUN_MAX_QUARTERS,
        RUN_MIN_QUARTERS,
        RUN_STEP_QUARTERS,
        SECTION_ORDER,
        control_widget_key,
        controls_by_section,
        get_by_path,
        set_by_path,
    )
else:
    from .plotting import (
        DEFAULT_LINE_METRICS,
        metric_options,
        plot_default_dashboard,
        plot_gini_series,
        plot_income_wealth_distributions,
        plot_metric_lines,
    )
    from .config import get_default_config
    from .results import run_simulation, summarize_rows
    from .streamlit_params import (
        NON_GINI_METRIC_GROUPS,
        PARAMETER_CONTROLS,
        RUN_DEFAULT_QUARTERS,
        RUN_MAX_QUARTERS,
        RUN_MIN_QUARTERS,
        RUN_STEP_QUARTERS,
        SECTION_ORDER,
        control_widget_key,
        controls_by_section,
        get_by_path,
        set_by_path,
    )


PERCENT_COLUMNS = {
    "automation",
    "automation_flow",
    "automation_info",
    "automation_info_flow",
    "automation_phys",
    "automation_phys_flow",
    "inflation",
    "gini",
    "gini_market",
    "gini_disp",
    "gini_wealth",
    "private_roe_q",
    "trust_equity_pct",
    "pop_dti_med",
    "pop_dti_p90",
    "pop_dti_w_med",
    "pop_dti_w_p90",
}

COMPACT_NUMBER_COLUMNS = {
    "private_eq_per_h",
    "vat_per_h",
    "inc_tax_per_h",
    "corp_tax_per_h",
    "vat_credit_per_h",
    "gov_dep_per_h",
    "fund_dep_per_h",
    "capex_per_h",
    "ubi_per_h",
    "ubi_from_fund_dep_per_h",
    "ubi_from_gov_dep_per_h",
    "ubi_issued_per_h",
    "trust_debt",
    "wages_total",
    "total_consumption",
    "real_avg_income",
    "real_consumption",
    "pop_inc_med",
    "pop_inc_p90",
}

DECIMAL_COLUMNS = {"price_level", "private_inv_cov"}

DISPLAY_VALUE_MODES: tuple[str, str] = ("nominal", "real")

# Columns to deflate when display mode is "real".
MONETARY_COLUMNS = {
    "private_eq_per_h",
    "vat_per_h",
    "inc_tax_per_h",
    "corp_tax_per_h",
    "vat_credit_per_h",
    "gov_dep_per_h",
    "fund_dep_per_h",
    "capex_per_h",
    "ubi_per_h",
    "ubi_from_fund_dep_per_h",
    "ubi_from_gov_dep_per_h",
    "ubi_issued_per_h",
    "trust_debt",
    "wages_total",
    "total_consumption",
    "pop_inc_med",
    "pop_inc_p90",
}


def _compact_number(value: float, decimals: int = 2) -> str:
    x = float(value)
    ax = abs(x)
    d = max(0, int(decimals))
    if ax >= 1e12:
        return f"{x / 1e12:.{d}f}T"
    if ax >= 1e9:
        return f"{x / 1e9:.{d}f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.{d}f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.{d}f}k"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax == 0:
        return "0"
    return f"{x:.3f}"


def _signed_compact(value: float) -> str:
    x = float(value)
    if x > 0:
        return f"+{_compact_number(x)}"
    return _compact_number(x)


def _float_format_from_step(step: float | int | None) -> str:
    if step is None:
        return "%.3f"
    s = f"{float(step):.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        return "%.0f"
    decimals = len(s.split(".", 1)[1])
    decimals = max(0, min(6, decimals))
    return f"%.{decimals}f"


def _build_styled_rows(rows: Sequence[Dict[str, Any]]) -> Any:
    try:
        import pandas as pd
    except Exception:
        return rows

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    formatters: Dict[str, Any] = {}

    for col in df.columns:
        if col == "t":
            formatters[col] = "{:.0f}"
        elif col == "trust_active":
            formatters[col] = lambda v: "Yes" if bool(v) else "No"
        elif col in PERCENT_COLUMNS:
            formatters[col] = lambda v: f"{100.0 * float(v):.2f}%"
        elif col in COMPACT_NUMBER_COLUMNS:
            formatters[col] = lambda v: _compact_number(float(v))
        elif col in DECIMAL_COLUMNS:
            formatters[col] = "{:.3f}"
    return df.style.format(formatters, na_rep="")


def _render_summary(summary: Dict[str, float], st: Any) -> None:
    if not summary:
        st.info("Run the simulation to populate results.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Automation", f"{summary['automation_end']:.2%}")
    c2.metric("Price Level", f"{summary['price_end']:.3f}")
    c3.metric(
        "Real Consumption",
        _compact_number(summary["real_consumption_end"]),
        delta=_signed_compact(summary["real_consumption_delta"]),
    )
    c4.metric("Trust Equity", f"{summary['trust_equity_end']:.2%}")


def _available_line_metric_ids() -> List[str]:
    ids: List[str] = []
    for _, metrics in NON_GINI_METRIC_GROUPS:
        for metric in metrics:
            if metric not in ids:
                ids.append(metric)
    return ids


def _apply_metric_defaults(st: Any) -> None:
    metric_ids = _available_line_metric_ids()
    default_non_gini = [m for m in DEFAULT_LINE_METRICS if m not in {"gini_market", "gini_disp", "gini_wealth"}]
    primary = default_non_gini[0] if default_non_gini else (metric_ids[0] if metric_ids else "real_consumption")
    secondary = default_non_gini[1] if len(default_non_gini) > 1 else "(None)"
    if primary not in metric_ids and metric_ids:
        primary = metric_ids[0]
    if secondary != "(None)" and secondary not in metric_ids:
        secondary = "(None)"
    st.session_state["line_metric_primary"] = primary
    st.session_state["line_metric_secondary"] = secondary


def _render_metric_selector(st: Any, metric_map: Dict[str, str]) -> List[str]:
    st.subheader("Selected Metrics")
    metric_ids = _available_line_metric_ids()
    if not metric_ids:
        return []

    primary = st.selectbox(
        "Primary (left axis)",
        options=metric_ids,
        key="line_metric_primary",
        format_func=lambda k: metric_map.get(k, k),
    )

    secondary_options = ["(None)"] + [m for m in metric_ids if m != primary]
    if st.session_state.get("line_metric_secondary") not in secondary_options:
        st.session_state["line_metric_secondary"] = "(None)"
    secondary = st.selectbox(
        "Secondary (right axis)",
        options=secondary_options,
        key="line_metric_secondary",
        format_func=lambda k: "None" if k == "(None)" else metric_map.get(k, k),
    )

    selected = [primary]
    if secondary != "(None)":
        selected.append(secondary)
    return selected


def _apply_control_defaults(st: Any, base_params: Dict[str, Any]) -> None:
    for control in PARAMETER_CONTROLS:
        key = control_widget_key(control)
        default_value = get_by_path(base_params, control.path, default=control.fallback_default)
        st.session_state[key] = default_value
    st.session_state["run__quarters"] = RUN_DEFAULT_QUARTERS
    raw_mode = str(base_params.get("dashboard_value_mode", "nominal")).strip().lower()
    st.session_state["view__value_mode"] = "real" if raw_mode in {"price_normalized", "price-normalized", "real"} else "nominal"
    _apply_metric_defaults(st)


def _ensure_control_defaults(st: Any, base_params: Dict[str, Any]) -> None:
    for control in PARAMETER_CONTROLS:
        key = control_widget_key(control)
        if key not in st.session_state:
            st.session_state[key] = get_by_path(base_params, control.path, default=control.fallback_default)
    if "run__quarters" not in st.session_state:
        st.session_state["run__quarters"] = RUN_DEFAULT_QUARTERS
    if "view__value_mode" not in st.session_state:
        raw_mode = str(base_params.get("dashboard_value_mode", "nominal")).strip().lower()
        st.session_state["view__value_mode"] = "real" if raw_mode in {"price_normalized", "price-normalized", "real"} else "nominal"
    if "line_metric_primary" not in st.session_state or "line_metric_secondary" not in st.session_state:
        _apply_metric_defaults(st)


def _coerce_value(raw: Any, kind: str) -> Any:
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return bool(raw)
    return raw


def _build_cfg_from_state(st: Any, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    params = cfg.get("parameters", {})
    for control in PARAMETER_CONTROLS:
        key = control_widget_key(control)
        raw = st.session_state.get(key, get_by_path(params, control.path, default=control.fallback_default))
        set_by_path(params, control.path, _coerce_value(raw, control.kind))
    cfg["parameters"] = params
    return cfg


def _cfg_json(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _rows_csv(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue()


def _rows_for_value_mode(rows: Sequence[Dict[str, Any]], value_mode: str) -> List[Dict[str, Any]]:
    mode = str(value_mode).strip().lower()
    if mode != "real":
        return [dict(r) for r in rows]

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        p = float(row.get("price_level", 1.0))
        if p <= 0.0:
            p = 1e-9
        out = dict(row)
        for key in MONETARY_COLUMNS:
            if key in out:
                out[key] = float(out[key]) / p
        out_rows.append(out)
    return out_rows


def _population_dist_for_value_mode(
    population_distributions: Dict[str, Dict[str, Any]] | None,
    value_mode: str,
    p0: float,
) -> Dict[str, Dict[str, Any]] | None:
    if not population_distributions:
        return None

    before = population_distributions.get("before")
    after = population_distributions.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None

    if str(value_mode).strip().lower() != "real":
        return {"before": dict(before), "after": dict(after)}

    p0_eff = float(p0) if float(p0) > 0.0 else 1e-9

    def _scaled(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        p = float(snapshot.get("price_level", 1.0))
        if p <= 0.0:
            p = 1e-9
        scale = p0_eff / p
        income = [float(v) * scale for v in snapshot.get("income", [])]
        wealth = [float(v) * scale for v in snapshot.get("wealth", [])]
        out = dict(snapshot)
        out["income"] = income
        out["wealth"] = wealth
        return out

    return {"before": _scaled(before), "after": _scaled(after)}


def _render_parameter_controls(st: Any, grouped_controls: Dict[str, List[Any]]) -> tuple[bool, bool, int]:
    with st.sidebar:
        st.header("Run Controls")
        col_run, col_reset = st.columns(2)
        run_clicked = col_run.button("Run Simulation", type="primary")
        reset_clicked = col_reset.button("Reset Defaults")

        quarters = st.slider(
            "Quarters",
            min_value=RUN_MIN_QUARTERS,
            max_value=RUN_MAX_QUARTERS,
            step=RUN_STEP_QUARTERS,
            key="run__quarters",
        )
        st.radio(
            "Display Values",
            options=list(DISPLAY_VALUE_MODES),
            key="view__value_mode",
            horizontal=True,
            format_func=lambda m: "Nominal" if m == "nominal" else "Real (Price-normalized)",
            help="Controls how monetary values are displayed in charts/tables. Simulation mechanics are unchanged.",
        )

        for section in SECTION_ORDER:
            controls = grouped_controls.get(section, [])
            if not controls:
                continue
            with st.expander(section, expanded=False):
                for control in controls:
                    key = control_widget_key(control)
                    if control.kind == "bool":
                        st.checkbox(control.label, key=key, help=control.help_text or None)
                    elif control.kind == "int":
                        kwargs: Dict[str, Any] = {"key": key, "help": control.help_text or None}
                        if control.min_value is not None:
                            kwargs["min_value"] = int(control.min_value)
                        if control.max_value is not None:
                            kwargs["max_value"] = int(control.max_value)
                        kwargs["step"] = int(control.step) if control.step is not None else 1
                        st.number_input(control.label, **kwargs)
                    elif control.kind == "select":
                        options = list(control.options or [])
                        st.selectbox(control.label, options=options, key=key, help=control.help_text or None)
                    else:
                        kwargs = {
                            "key": key,
                            "help": control.help_text or None,
                            "format": _float_format_from_step(control.step),
                        }
                        if control.min_value is not None:
                            kwargs["min_value"] = float(control.min_value)
                        if control.max_value is not None:
                            kwargs["max_value"] = float(control.max_value)
                        kwargs["step"] = float(control.step) if control.step is not None else 0.01
                        st.number_input(control.label, **kwargs)
    return run_clicked, reset_clicked, int(quarters)


def _cached_run_payload(n_quarters: int, cfg_json: str) -> Dict[str, Any]:
    cfg = json.loads(cfg_json)
    run = run_simulation(n_quarters=int(n_quarters), cfg=cfg)
    return {
        "rows": run.rows,
        "population_distributions": run.population_distributions or {},
    }


def main() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="EconomySim", layout="wide")
    st.title("EconomySim")
    st.caption("Interactive simulation with parameterized runs and reusable plotting.")

    base_cfg = copy.deepcopy(get_default_config())
    base_params = copy.deepcopy(base_cfg.get("parameters", {}))
    grouped_controls = controls_by_section()
    _ensure_control_defaults(st, base_params)

    run_clicked, reset_clicked, quarters = _render_parameter_controls(st, grouped_controls)
    if reset_clicked:
        _apply_control_defaults(st, base_params)
        st.rerun()

    metric_map = metric_options()
    selected_metrics = _render_metric_selector(st, metric_map)

    if "rows" not in st.session_state:
        st.session_state["rows"] = []
    if "population_distributions" not in st.session_state:
        st.session_state["population_distributions"] = {}
    if "last_run_cfg_json" not in st.session_state:
        st.session_state["last_run_cfg_json"] = ""
    if "last_run_quarters" not in st.session_state:
        st.session_state["last_run_quarters"] = 0

    current_cfg = _build_cfg_from_state(st, base_cfg)
    current_cfg_json = _cfg_json(current_cfg)
    cache_run = st.cache_data(show_spinner=False)(_cached_run_payload)

    should_run = run_clicked or (not st.session_state["rows"])
    if should_run:
        with st.spinner("Running simulation..."):
            payload = cache_run(quarters, current_cfg_json)
        st.session_state["rows"] = list(payload.get("rows", []))
        st.session_state["population_distributions"] = dict(payload.get("population_distributions", {}))
        st.session_state["last_run_cfg_json"] = current_cfg_json
        st.session_state["last_run_quarters"] = int(quarters)

    rows_raw: List[Dict[str, Any]] = list(st.session_state["rows"])
    if rows_raw:
        config_stale = (
            st.session_state.get("last_run_cfg_json", "") != current_cfg_json
            or int(st.session_state.get("last_run_quarters", 0)) != int(quarters)
        )
        if config_stale:
            st.warning("Parameters changed since last run. Click `Run Simulation` to generate new data.")

    display_value_mode = str(st.session_state.get("view__value_mode", "nominal")).strip().lower()
    rows = _rows_for_value_mode(rows_raw, display_value_mode)
    pop_dist = _population_dist_for_value_mode(
        st.session_state.get("population_distributions", {}),
        display_value_mode,
        float(current_cfg.get("parameters", {}).get("price_level_initial", 1.0)),
    )

    summary = summarize_rows(rows)
    _render_summary(summary, st)
    st.caption(
        "Displayed monetary values are "
        + ("price-normalized (real, base-period dollars)." if display_value_mode == "real" else "nominal.")
    )

    if not rows:
        return

    line_metrics = selected_metrics or [m for m in DEFAULT_LINE_METRICS if m not in {"gini_market", "gini_disp", "gini_wealth"}]
    line_metrics = line_metrics[:2]
    secondary_metrics = line_metrics[1:2] if len(line_metrics) > 1 else []

    if line_metrics:
        line_fig = plot_metric_lines(
            rows,
            line_metrics,
            title="Selected Metrics",
            secondary_metrics=secondary_metrics,
            secondary_ylabel="Trust Equity Fraction",
        )
        st.pyplot(line_fig, clear_figure=False)
        plt.close(line_fig)

    gini_fig = plot_gini_series(rows)
    st.pyplot(gini_fig, clear_figure=False)
    plt.close(gini_fig)

    dashboard_fig = plot_default_dashboard(rows)
    st.pyplot(dashboard_fig, clear_figure=False)
    plt.close(dashboard_fig)

    if pop_dist is not None:
        before = pop_dist.get("before", {})
        after = pop_dist.get("after", {})
        income_before = before.get("income", [])
        income_after = after.get("income", [])
        wealth_before = before.get("wealth", [])
        wealth_after = after.get("wealth", [])
        if income_before and income_after and wealth_before and wealth_after:
            st.subheader("Population Distributions")
            value_label = "Base-period dollars (real)" if display_value_mode == "real" else "Nominal dollars"
            dist_fig = plot_income_wealth_distributions(
                income_before,
                income_after,
                wealth_before,
                wealth_after,
                value_label=value_label,
            )
            st.pyplot(dist_fig, clear_figure=False)
            plt.close(dist_fig)

    st.subheader("Quarterly Data")
    limit_tail = st.checkbox("Show only tail rows", value=False)
    display_rows = rows
    if limit_tail:
        tail_n = st.slider("Tail size", min_value=5, max_value=max(5, len(rows)), value=min(40, len(rows)))
        display_rows = rows[-int(tail_n):]

    st.caption(f"Displaying {len(display_rows)} of {len(rows)} quarters.")
    st.dataframe(_build_styled_rows(display_rows), width="stretch")

    csv_data = _rows_csv(rows)
    st.download_button(
        "Download run CSV",
        data=csv_data,
        file_name=f"econsim_run_q{int(st.session_state.get('last_run_quarters', len(rows)))}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
