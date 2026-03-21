# Author: Roger Ison   roger@miximum.info
"""Streamlit app entrypoint for NewLoop.

Run with:
  streamlit run newloop/slnewloop.py
"""

from __future__ import annotations

import copy
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from plotting import (
    DEFAULT_LINE_METRICS,
    metric_options,
    plot_default_dashboard,
    plot_income_distribution_dual,
    plot_metric_lines,
    plot_wealth_distributions_full_zoom,
)
from config import get_default_config
from results import run_simulation, summarize_rows
from streamlit_params import (
    INCOME_SUPPORT_MODE_PATH,
    INCOME_SUPPORT_MODE_WIDGET_KEY,
    INCOME_SUPPORT_SECTION,
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
    "corp_tax_rate_eff",
    "sector_util_info",
    "sector_util_physical",
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
    "sector_capacity_info_per_h",
    "sector_capacity_physical_per_h",
    "unmet_demand_info_per_h",
    "unmet_demand_physical_per_h",
    "uis_per_h",
    "uis_from_fund_dep_per_h",
    "uis_from_gov_dep_per_h",
    "uis_issued_per_h",
    "trust_debt",
    "trust_value_per_h",
    "wages_total",
    "total_consumption",
    "real_avg_income",
    "real_consumption",
    "pop_inc_med",
    "pop_inc_p90",
}

DECIMAL_COLUMNS = {"price_level", "private_inv_cov"}

DISPLAY_VALUE_MODES: tuple[str, str] = ("nominal", "real")
CONTROL_DEFAULTS_VERSION = 8
UBI_PERCENTILE_PARAM_KEY = "param__ubi_target_percentile"
UBI_PERCENTILE_UI_KEY = "ui__ubi_target_percentile"
_TITLE_MODE_SUFFIX_RE = re.compile(r"\s+\((?:UIS|UBI|Stale)\)\s*$", re.IGNORECASE)
RECENT_IMPROVEMENTS_TEXT = (
    "- Added a top-level Policy Switches panel for baseline and no-policy diagnostics.\n"
    "- Added corporate-equity and investment-recycling charts to help explain baseline contraction dynamics.\n"
    "- Added an optional mortgage-turnover mechanism to replace amortized mortgage stock with new origination.\n"
    "- Added iterative debt-aware baseline calibration for startup household consumption and liquidity targets.\n"
    "- Added an income-tax disable switch for cleaner baseline diagnostics.\n"
    "- Added a mortgage-indexing disable switch for cleaner baseline diagnostics.\n"
    "- Added a mortgage-policy disable switch for cleaner baseline diagnostics.\n"
    "- Improved household consumption behavior to be forward-looking and maintain a liquidity buffer.\n"
    "- Income lines now plot disposable income."
)

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
    "uis_per_h",
    "uis_from_fund_dep_per_h",
    "uis_from_gov_dep_per_h",
    "uis_issued_per_h",
    "trust_debt",
    "trust_value_per_h",
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


def _render_startup_diagnostics(startup_diag: Dict[str, Any], baseline_calibration: Dict[str, Any], st: Any) -> None:
    if not startup_diag:
        return

    st.subheader("Startup Diagnostics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HH Below Buffer", f"{100.0 * float(startup_diag.get('share_below_runtime_buffer', 0.0)):.1f}%")
    c2.metric("Deposit / Target", f"{float(startup_diag.get('mean_deposit_to_target_ratio', 0.0)):.2f}x")
    c3.metric("Median Buffer Gap", _compact_number(float(startup_diag.get("median_buffer_gap", 0.0))))
    c4.metric("DTI P90 (Wages)", f"{100.0 * float(startup_diag.get('startup_dti_w_p90', 0.0)):.1f}%")
    c5, c6 = st.columns(2)
    c5.metric("Base Cons Uncovered", f"{100.0 * float(startup_diag.get('share_base_consumption_uncovered', 0.0)):.1f}%")
    c6.metric("Mean Base Gap", _compact_number(float(startup_diag.get("mean_base_consumption_gap", 0.0))))
    if baseline_calibration and bool(baseline_calibration.get("enabled", False)):
        iterations = int(baseline_calibration.get("iterations_completed", 0))
        max_change = 100.0 * float(baseline_calibration.get("max_target_change_pct", 0.0))
        converged = bool(baseline_calibration.get("converged", False))
        st.caption(
            f"Baseline calibration ran for {iterations} iteration(s) before visible Q0; final quintile-target change was {max_change:.2f}%."
        )
        st.caption("Visible Q0 is the calibrated startup state directly; no hidden startup burn-in is applied.")
        if not converged:
            st.caption("Baseline calibration has not fully converged yet, so some residual startup drift is still expected.")
    elif float(startup_diag.get("mean_deposit_to_target_ratio", 0.0)) > 0.95:
        st.caption("Startup buffer alignment appears active: visible Q0 household deposits have been re-seeded to the live runtime buffer target without changing the consumption ladder.")
    st.caption(f"Aggregate buffer shortfall to hit runtime targets: {_compact_number(float(startup_diag.get('buffer_shortfall_total', 0.0)))}.")
    st.caption(
        "Quarter-0 diagnostics compare initialized household deposits and baseline consumption needs to the solver's "
        "debt-aware disposable-income path before the first simulated quarter updates balances."
    )


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
    version_key = "app__control_defaults_version"
    if int(st.session_state.get(version_key, 0)) < CONTROL_DEFAULTS_VERSION:
        _apply_control_defaults(st, base_params)
        st.session_state[version_key] = CONTROL_DEFAULTS_VERSION
        return

    for control in PARAMETER_CONTROLS:
        key = control_widget_key(control)
        default_value = get_by_path(base_params, control.path, default=control.fallback_default)
        if key not in st.session_state or st.session_state.get(key) is None:
            st.session_state[key] = default_value
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

    # Guardrail: a zero UBI percentile anchors to zero in this population (some households have zero market income).
    # Keep run config sane even if a stale UI state slips through.
    mode = str(params.get("income_support_mode", "UIS")).strip().upper()
    if mode == "UBI":
        try:
            pct_val = float(params.get("ubi_target_percentile", 30.0))
        except Exception:
            pct_val = 30.0
        if pct_val <= 0.0:
            params["ubi_target_percentile"] = 30.0

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


def _mark_figure_stale(fig: Any) -> None:
    """Retitle figure axes as stale and dim plotted elements."""
    axes = list(fig.get_axes())
    for ax in axes:
        title = str(ax.get_title() or "").strip()
        if title:
            base = _TITLE_MODE_SUFFIX_RE.sub("", title).strip()
            ax.set_title(f"{base} (Stale)")

        for line in list(getattr(ax, "lines", [])):
            cur = line.get_alpha()
            line.set_alpha((float(cur) if cur is not None else 1.0) * 0.45)

        for coll in list(getattr(ax, "collections", [])):
            cur = coll.get_alpha()
            coll.set_alpha((float(cur) if cur is not None else 1.0) * 0.45)

        for patch in list(getattr(ax, "patches", [])):
            cur = patch.get_alpha()
            patch.set_alpha((float(cur) if cur is not None else 1.0) * 0.45)

        for img in list(getattr(ax, "images", [])):
            cur = img.get_alpha()
            img.set_alpha((float(cur) if cur is not None else 1.0) * 0.45)


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


def _inject_selectbox_chevron_fallback(st: Any) -> None:
    """Safari/iPad fallback when Material icon text appears in selectbox arrows."""
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] span[class*="material-symbols"],
        div[data-baseweb="select"] [data-testid="stIconMaterial"] {
            font-size: 0 !important;
            line-height: 1 !important;
        }

        div[data-baseweb="select"] span[class*="material-symbols"]::before,
        div[data-baseweb="select"] [data-testid="stIconMaterial"]::before {
            content: "▾";
            font-size: 0.95rem;
            line-height: 1 !important;
            color: currentColor;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_parameter_controls(st: Any, grouped_controls: Dict[str, List[Any]]) -> tuple[bool, bool, int]:
    def _request_reset_defaults() -> None:
        st.session_state["app__reset_requested"] = True
        st.session_state["app__force_stale_after_reset"] = True

    def _render_control(control: Any) -> None:
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
                "help": control.help_text or None,
                "format": _float_format_from_step(control.step),
            }
            if control.min_value is not None:
                kwargs["min_value"] = float(control.min_value)
            if control.max_value is not None:
                kwargs["max_value"] = float(control.max_value)
            kwargs["step"] = float(control.step) if control.step is not None else 0.01
            if key == UBI_PERCENTILE_PARAM_KEY:
                # Use a dedicated UI key to avoid stale hidden-widget state when switching modes.
                val = st.number_input(control.label, key=UBI_PERCENTILE_UI_KEY, **kwargs)
                st.session_state[UBI_PERCENTILE_PARAM_KEY] = float(val)
            else:
                kwargs["key"] = key
                st.number_input(control.label, **kwargs)

    with st.sidebar:
        st.header("Run Controls")
        col_run, col_reset = st.columns(2)
        run_clicked = col_run.button("Run Model", type="primary")
        reset_clicked = col_reset.button("Reset", on_click=_request_reset_defaults)

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
                if section == INCOME_SUPPORT_SECTION:
                    mode_control = next((c for c in controls if tuple(c.path) == INCOME_SUPPORT_MODE_PATH), None)
                    remaining_controls = [c for c in controls if mode_control is None or c is not mode_control]

                    if mode_control is not None:
                        _render_control(mode_control)
                    active_mode = str(st.session_state.get(INCOME_SUPPORT_MODE_WIDGET_KEY, "UIS")).strip().upper()
                    if active_mode not in {"UIS", "UBI"}:
                        active_mode = "UIS"

                    # Defensive reseed when switching into UBI mode.
                    # Hidden-widget state can occasionally revive stale values; in practice a 0.0
                    # percentile makes UBI anchor at zero for this population due zero-income households.
                    last_mode_key = "app__last_income_support_mode_ui"
                    if active_mode == "UBI":
                        for control in remaining_controls:
                            default_value = control.fallback_default
                            if default_value is None:
                                continue
                            key = control_widget_key(control)
                            if key not in st.session_state or st.session_state.get(key) is None:
                                st.session_state[key] = default_value

                        ubi_pct_control = next(
                            (c for c in remaining_controls if tuple(c.path) == ("ubi_target_percentile",)),
                            None,
                        )
                        if ubi_pct_control is not None:
                            ubi_pct_key = control_widget_key(ubi_pct_control)
                            try:
                                pct_val = float(st.session_state.get(ubi_pct_key, 0.0))
                            except Exception:
                                pct_val = 0.0
                            if pct_val <= 0.0:
                                st.session_state[ubi_pct_key] = 30.0
                            st.session_state[UBI_PERCENTILE_UI_KEY] = float(st.session_state.get(ubi_pct_key, 30.0))

                    st.session_state[last_mode_key] = active_mode

                    st.caption(f"Active Mode: {active_mode}")

                    for control in remaining_controls:
                        modes = getattr(control, "support_modes", None)
                        if modes is not None and active_mode not in {str(m).upper() for m in modes}:
                            continue
                        _render_control(control)
                elif section == "Population":
                    core_controls = [c for c in controls if not bool(getattr(c, "advanced", False))]
                    advanced_controls = [c for c in controls if bool(getattr(c, "advanced", False))]

                    for control in core_controls:
                        _render_control(control)

                    if advanced_controls:
                        show_advanced = st.checkbox(
                            "Show advanced population controls",
                            key="population__show_advanced_controls",
                            value=False,
                            help="Expose distribution-shape and debt-tail controls for population calibration.",
                        )
                        if show_advanced:
                            st.caption("Advanced controls can significantly change inequality and debt-stress tails.")
                            for control in advanced_controls:
                                _render_control(control)
                else:
                    for control in controls:
                        _render_control(control)
    return run_clicked, reset_clicked, int(quarters)


def _cached_run_payload(n_quarters: int, cfg_json: str) -> Dict[str, Any]:
    cfg = json.loads(cfg_json)
    run = run_simulation(n_quarters=int(n_quarters), cfg=cfg)
    support_debug = {
        "mode": str(run.sim.params.get("income_support_mode", "UIS")).strip().upper(),
        "disabled": bool(run.sim.params.get("disable_income_support", False)),
        "ubi_anchor_real_per_h": run.sim.state.get("ubi_anchor_real_per_h", None),
        "ubi_anchor_nominal_per_h_base": run.sim.state.get("ubi_anchor_nominal_per_h_base", None),
        "ubi_anchor_percentile": run.sim.state.get("ubi_anchor_percentile", None),
        "ubi_anchor_basis": run.sim.state.get("ubi_anchor_basis", None),
        "ubi_index_series": run.sim.params.get("ubi_index_series", None),
        "income_support_issuance_share": run.sim.params.get(
            "income_support_issuance_share",
            run.sim.params.get("uis_issuance_share", 0.0),
        ),
        "income_target_pool_real_pop": run.sim.state.get("income_target_pool_real_pop", None),
        "household_count": int(run.sim.hh.n) if getattr(run.sim, "hh", None) is not None else 0,
    }
    return {
        "rows": run.rows,
        "population_distributions": run.population_distributions or {},
        "startup_diagnostics": run.startup_diagnostics or {},
        "baseline_calibration": run.baseline_calibration or {},
        "support_debug": support_debug,
    }


def main() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="NewLoop", layout="wide")
    _inject_selectbox_chevron_fallback(st)
    st.title("NewLoop")
    st.caption("Interactive simulation with parameterized runs and reusable plotting.")
    st.text_area(
        "Recent improvements",
        value=RECENT_IMPROVEMENTS_TEXT,
        height=80,
        disabled=True,
    )

    base_cfg = copy.deepcopy(get_default_config())
    base_params = copy.deepcopy(base_cfg.get("parameters", {}))
    grouped_controls = controls_by_section()
    _ensure_control_defaults(st, base_params)

    if bool(st.session_state.pop("app__reset_requested", False)):
        _apply_control_defaults(st, base_params)

    run_clicked, _reset_clicked, quarters = _render_parameter_controls(st, grouped_controls)

    metric_map = metric_options()
    selected_metrics = _render_metric_selector(st, metric_map)

    if "rows" not in st.session_state:
        st.session_state["rows"] = []
    if "population_distributions" not in st.session_state:
        st.session_state["population_distributions"] = {}
    if "startup_diagnostics" not in st.session_state:
        st.session_state["startup_diagnostics"] = {}
    if "baseline_calibration" not in st.session_state:
        st.session_state["baseline_calibration"] = {}
    if "support_debug" not in st.session_state:
        st.session_state["support_debug"] = {}
    if "last_run_cfg_json" not in st.session_state:
        st.session_state["last_run_cfg_json"] = ""
    if "last_run_quarters" not in st.session_state:
        st.session_state["last_run_quarters"] = 0

    current_cfg = _build_cfg_from_state(st, base_cfg)
    current_cfg_json = _cfg_json(current_cfg)
    should_run = run_clicked or (not st.session_state["rows"])
    if should_run:
        with st.spinner("Running simulation..."):
            payload = _cached_run_payload(quarters, current_cfg_json)
        st.session_state["rows"] = list(payload.get("rows", []))
        st.session_state["population_distributions"] = dict(payload.get("population_distributions", {}))
        st.session_state["startup_diagnostics"] = dict(payload.get("startup_diagnostics", {}))
        st.session_state["baseline_calibration"] = dict(payload.get("baseline_calibration", {}))
        st.session_state["support_debug"] = dict(payload.get("support_debug", {}))
        st.session_state["last_run_cfg_json"] = current_cfg_json
        st.session_state["last_run_quarters"] = int(quarters)
        st.session_state["app__force_stale_after_reset"] = False

    rows_raw: List[Dict[str, Any]] = list(st.session_state["rows"])
    config_stale = bool(st.session_state.get("app__force_stale_after_reset", False)) or (
        st.session_state.get("last_run_cfg_json", "") != current_cfg_json
        or int(st.session_state.get("last_run_quarters", 0)) != int(quarters)
    )
    if rows_raw and config_stale:
        st.warning("Parameters changed since last run. Click `Run Model` to generate new data.")

    display_value_mode = str(st.session_state.get("view__value_mode", "nominal")).strip().lower()
    rows = _rows_for_value_mode(rows_raw, display_value_mode)
    pop_dist = _population_dist_for_value_mode(
        st.session_state.get("population_distributions", {}),
        display_value_mode,
        float(current_cfg.get("parameters", {}).get("price_level_initial", 1.0)),
    )

    summary = summarize_rows(rows)
    _render_summary(summary, st)
    _render_startup_diagnostics(
        dict(st.session_state.get("startup_diagnostics", {})),
        dict(st.session_state.get("baseline_calibration", {})),
        st,
    )
    support_debug = dict(st.session_state.get("support_debug", {}))
    support_mode_cfg = str(current_cfg.get("parameters", {}).get("income_support_mode", "UIS")).strip().upper()
    if support_mode_cfg not in {"UIS", "UBI"}:
        support_mode_cfg = "UIS"
    support_mode = str(support_debug.get("mode", support_mode_cfg)).strip().upper()
    if support_mode not in {"UIS", "UBI"}:
        support_mode = support_mode_cfg
    support_label = "Universal Income Stabilizer (UIS)" if support_mode == "UIS" else "Universal Basic Income (UBI)"
    support_disabled = bool(support_debug.get("disabled", False))
    if support_disabled:
        st.caption("Income support disabled for this run.")
    else:
        st.caption(f"Income support mode (last run): {support_label}.")
        if config_stale and support_mode_cfg != support_mode:
            pending_label = "Universal Income Stabilizer (UIS)" if support_mode_cfg == "UIS" else "Universal Basic Income (UBI)"
            st.caption(f"Pending mode in controls: {pending_label}.")

        issuance_share = float(support_debug.get("income_support_issuance_share", 0.0))
        st.caption(
            "Income-support funding order (all modes): "
            f"{issuance_share:.0%} issuance share first, then FUND deposits, then GOV deposits, then residual issuance."
        )

        if support_mode == "UBI":
            anchor_real = support_debug.get("ubi_anchor_real_per_h", None)
            anchor_nom = support_debug.get("ubi_anchor_nominal_per_h_base", None)
            anchor_pct = support_debug.get("ubi_anchor_percentile", None)
            anchor_basis = support_debug.get("ubi_anchor_basis", None)
            index_series = support_debug.get("ubi_index_series", None)
            if anchor_real is not None:
                pct_txt = f"{float(anchor_pct):.1f}" if anchor_pct is not None else "?"
                basis_txt = str(anchor_basis) if anchor_basis is not None else "?"
                idx_txt = str(index_series) if index_series is not None else "?"
                nom_txt = f"{float(anchor_nom):.3g}" if anchor_nom is not None else "?"
                st.caption(
                    "UBI anchor: "
                    f"P{pct_txt} of {basis_txt} at baseline; "
                    f"base nominal/HH={nom_txt}; "
                    f"real anchor/HH={float(anchor_real):.3g}; "
                    f"index={idx_txt}."
                )
                if float(anchor_real) <= 0.0:
                    st.warning(
                        "UBI anchor resolved to zero at baseline. This usually means "
                        "`UBI Target Percentile` is at or near 0 in a population with zero market-income households."
                    )
        else:
            target_pool = support_debug.get("income_target_pool_real_pop", None)
            if target_pool is not None:
                st.caption(f"UIS real target pool (population): {float(target_pool):.6g}.")
    st.caption(
        "Displayed monetary values are "
        + ("price-normalized (real, base-period dollars)." if display_value_mode == "real" else "nominal.")
    )

    if not rows:
        return

    line_metrics = selected_metrics or [m for m in DEFAULT_LINE_METRICS if m not in {"gini_market", "gini_disp", "gini_wealth"}]
    line_metrics = line_metrics[:2]
    secondary_metrics = line_metrics[1:2] if len(line_metrics) > 1 else []

    if rows:
        mismatch_quarters: List[int] = []
        for row in rows:
            support_per_h = float(row.get("uis_per_h", 0.0))
            support_components = (
                float(row.get("uis_from_fund_dep_per_h", 0.0))
                + float(row.get("uis_from_gov_dep_per_h", 0.0))
                + float(row.get("uis_issued_per_h", 0.0))
            )
            if abs(support_per_h - support_components) > 1e-6 * max(1.0, abs(support_per_h)):
                mismatch_quarters.append(int(row.get("t", 0)))

        if mismatch_quarters:
            sample = ", ".join(str(q) for q in mismatch_quarters[:6])
            if len(mismatch_quarters) > 6:
                sample += ", ..."
            st.warning(
                "Income-support accounting mismatch detected: "
                f"{len(mismatch_quarters)} quarter(s) with support != funding components "
                f"(quarters: {sample})."
            )

    if line_metrics:
        primary_ylabel = metric_map.get(line_metrics[0], line_metrics[0])
        if len(line_metrics) > 1:
            secondary_ylabel = metric_map.get(line_metrics[1], line_metrics[1])
        else:
            secondary_ylabel = "Secondary Scale"

        line_fig = plot_metric_lines(
            rows,
            line_metrics,
            title="Selected Metrics",
            primary_ylabel=primary_ylabel,
            secondary_metrics=secondary_metrics,
            secondary_ylabel=secondary_ylabel,
            support_mode=support_mode,
        )
        if config_stale:
            _mark_figure_stale(line_fig)
        st.pyplot(line_fig, clear_figure=False)
        plt.close(line_fig)

    st.caption(
        "Gini labels: Pre-Tax/Pre-Transfer is household wages plus household-distributed dividends, "
        "before income tax, VAT credit, income-support transfers, and debt-service deductions. Disposable is the model's "
        "post-policy household income measure. Wealth is deposits plus allocated household equity claims minus loans."
    )
    st.caption(
        "DTI means debt-service-to-income in this model: a household debt-burden ratio based on interest payments "
        "relative to income, not debt stock divided by income."
    )

    hh_count = int(support_debug.get("household_count", 0) or 0)
    dashboard_fig = plot_default_dashboard(rows, support_mode=support_mode, household_count=hh_count)
    if config_stale:
        _mark_figure_stale(dashboard_fig)
    st.pyplot(dashboard_fig, clear_figure=False)
    plt.close(dashboard_fig)
    st.caption(
        "Cumulative public funding equals cumulative GOV funding plus cumulative issuance funding "
        "(economy totals, in the currently selected nominal/real display mode)."
    )

    equity_fig, (ax_equity, ax_recycling) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_metric_lines(
        rows,
        [
            "corporate_eq_info_per_h",
            "corporate_eq_physical_per_h",
            "corporate_eq_total_per_h",
        ],
        title="Corporate Equity",
        primary_ylabel="Equity / Household",
        support_mode=support_mode,
        ax=ax_equity,
    )
    plot_metric_lines(
        rows,
        [
            "capex_per_h",
            "private_roe_q",
        ],
        title="Investment Recycling",
        primary_ylabel="CAPEX / Household",
        secondary_metrics=["private_roe_q"],
        secondary_ylabel="Private ROE / Quarter",
        support_mode=support_mode,
        ax=ax_recycling,
    )
    if config_stale:
        _mark_figure_stale(equity_fig)
    st.pyplot(equity_fig, clear_figure=False)
    plt.close(equity_fig)

    sector_fig, (ax_capacity, ax_shortfall) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_metric_lines(
        rows,
        [
            "sector_capacity_info_per_h",
            "sector_capacity_physical_per_h",
            "sector_util_info",
            "sector_util_physical",
        ],
        title="Sector Capacity And Utilization",
        primary_ylabel="Capacity / Household",
        secondary_metrics=["sector_util_info", "sector_util_physical"],
        secondary_ylabel="Utilization",
        support_mode=support_mode,
        ax=ax_capacity,
    )
    plot_metric_lines(
        rows,
        [
            "unmet_demand_info_per_h",
            "unmet_demand_physical_per_h",
        ],
        title="Unmet Household Demand",
        primary_ylabel="Real Units / Household",
        support_mode=support_mode,
        ax=ax_shortfall,
    )
    if config_stale:
        _mark_figure_stale(sector_fig)
    st.pyplot(sector_fig, clear_figure=False)
    plt.close(sector_fig)

    row_fig, (ax_outcomes, ax_corp_tax) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    plot_metric_lines(
        rows,
        ["real_consumption", "real_avg_income"],
        title="Real Household Outcomes",
        secondary_metrics=["real_avg_income"],
        secondary_ylabel="Real Avg Income",
        support_mode=support_mode,
        ax=ax_outcomes,
    )

    plot_metric_lines(
        rows,
        ["corp_tax_rate_eff", "wages_total"],
        title="Corporate Tax Rate Over Time",
        primary_ylabel="Rate",
        secondary_metrics=["wages_total"],
        secondary_ylabel="Total Wage Base",
        support_mode=support_mode,
        ax=ax_corp_tax,
    )
    _corp_axes = row_fig.get_axes()
    if _corp_axes:
        _corp_primary = _corp_axes[1]
        _corp_primary.set_ylim(0.0, 1.0)
        from matplotlib.ticker import FuncFormatter

        _corp_primary.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{100.0 * float(val):.1f}%"))

    if config_stale:
        _mark_figure_stale(row_fig)
    st.pyplot(row_fig, clear_figure=False)
    plt.close(row_fig)

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

            income_fig = plot_income_distribution_dual(
                income_before,
                income_after,
                value_label=value_label,
                support_mode=support_mode,
            )
            if config_stale:
                _mark_figure_stale(income_fig)
            st.pyplot(income_fig, clear_figure=False)
            plt.close(income_fig)

            zoom_window = st.slider(
                "Wealth zoom percentile window",
                min_value=0,
                max_value=100,
                value=(2, 98),
                step=1,
                help="Zooms the right-hand wealth panel to this percentile band while keeping a full-range panel for context.",
            )
            wealth_fig = plot_wealth_distributions_full_zoom(
                wealth_before,
                wealth_after,
                value_label=value_label,
                zoom_lo_pct=float(zoom_window[0]),
                zoom_hi_pct=float(zoom_window[1]),
                support_mode=support_mode,
            )
            if config_stale:
                _mark_figure_stale(wealth_fig)
            st.pyplot(wealth_fig, clear_figure=False)
            plt.close(wealth_fig)

    st.subheader("Quarterly Data")
    limit_tail = st.checkbox("Show only tail rows", value=False)
    display_rows = rows
    if limit_tail:
        tail_n = st.slider("Tail size", min_value=5, max_value=max(5, len(rows)), value=min(40, len(rows)))
        display_rows = rows[-int(tail_n):]

    st.caption(f"Displaying {len(display_rows)} of {len(rows)} quarters.")
    st.dataframe(_build_styled_rows(display_rows), use_container_width=True)

    csv_data = _rows_csv(rows)
    st.download_button(
        "Download run CSV",
        data=csv_data,
        file_name=f"newloop_run_q{int(st.session_state.get('last_run_quarters', len(rows)))}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
