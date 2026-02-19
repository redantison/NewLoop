# Author: Roger Ison   roger@miximum.info
"""Parameter schema and helpers for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence


ControlKind = Literal["float", "int", "bool", "select"]


@dataclass(frozen=True)
class ParamControl:
    """UI control specification for one model parameter."""

    path: tuple[str, ...]
    label: str
    section: str
    kind: ControlKind
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    options: tuple[str, ...] | None = None
    help_text: str = ""
    fallback_default: Any = None


SECTION_ORDER: tuple[str, ...] = (
    "Trust",
    "Taxes",
    "UBI",
    "Automation",
    "Price & Capital",
    "Population",
)


PARAMETER_CONTROLS: tuple[ParamControl, ...] = (
    ParamControl(("trust_trigger_dti",), "Trust Trigger DTI", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("trust_launch_loan",), "Trust Launch Loan", "Trust", "float", 0.0, 50000.0, 500.0),
    ParamControl(("trust_equity_cap",), "Trust Equity Cap", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("send_fund_residual_to_gov",), "Sweep Fund Residual To GOV", "Trust", "bool"),
    ParamControl(("vat_rate",), "VAT Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("vat_credit_cutoff_pct",), "VAT Credit Cutoff Percentile", "Taxes", "float", 0.0, 100.0, 0.5),
    ParamControl(("vat_poverty_cons_frac",), "VAT Poverty Consumption Fraction", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("income_tax_rate",), "Income Tax Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("income_tax_cutoff_pct",), "Income Tax Cutoff Percentile", "Taxes", "float", 0.0, 100.0, 0.5),
    ParamControl(("corporate_tax_rate",), "Corporate Tax Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_dynamic_with_wages",), "Dynamic Corporate Tax With Wages", "Taxes", "bool"),
    ParamControl(("corporate_tax_rate_base",), "Corporate Tax Base Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_wage_sensitivity",), "Corporate Tax Wage Sensitivity", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_rate_min",), "Corporate Tax Min Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_rate_max",), "Corporate Tax Max Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("gov_tax_rebate_rate",), "Government Tax Rebate Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("ubi_issuance_share",), "UBI Issuance Share", "UBI", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_path",), "Automation Path", "Automation", "select", options=("two_hump", "linear")),
    ParamControl(("automation_horizon_quarters",), "Automation Horizon Quarters", "Automation", "float", 4.0, 240.0, 1.0),
    ParamControl(("automation_w_info",), "Automation Weight: Info", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_info_cap",), "Automation Info Cap", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_phys_cap",), "Automation Physical Cap", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_ki",), "Automation ki", "Automation", "float", 0.01, 1.0, 0.01),
    ParamControl(("automation_ti",), "Automation ti", "Automation", "float", 0.0, 120.0, 0.5),
    ParamControl(("automation_bi",), "Automation bi", "Automation", "float", 0.0, 20.0, 0.1),
    ParamControl(("automation_kp",), "Automation kp", "Automation", "float", 0.01, 1.0, 0.01),
    ParamControl(("automation_tp",), "Automation tp", "Automation", "float", 0.0, 120.0, 0.5),
    ParamControl(("price_beta",), "Price Beta", "Price & Capital", "float", 0.0, 3.0, 0.01),
    ParamControl(("automation_markup_max",), "Automation Markup Max", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_markup_power",), "Automation Markup Power", "Price & Capital", "float", 0.1, 5.0, 0.1),
    ParamControl(("price_adjust_speed",), "Price Adjust Speed", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("capital_productivity_k",), "Capital Productivity k", "Price & Capital", "float", 0.0, 2.0, 0.01),
    ParamControl(("capital_productivity_scale",), "Capital Productivity Scale", "Price & Capital", "float", 100.0, 20000.0, 100.0),
    ParamControl(("capital_depr_rate_per_quarter",), "Capital Depreciation / Quarter", "Price & Capital", "float", 0.0, 1.0, 0.005),
    ParamControl(("population_config", "n_families"), "Population: Families", "Population", "int", 1000, 100000, 1000),
    ParamControl(("population_config", "seed"), "Population: Seed", "Population", "int", 1, 1000000, 1),
)


RUN_DEFAULT_QUARTERS = 80
RUN_MIN_QUARTERS = 20
RUN_MAX_QUARTERS = 240
RUN_STEP_QUARTERS = 4


NON_GINI_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Core", ("inflation", "trust_equity_pct")),
    (
        "Households",
        (
            "real_consumption",
            "real_avg_income",
            "ubi_per_h",
            "ubi_from_fund_dep_per_h",
            "ubi_from_gov_dep_per_h",
            "ubi_issued_per_h",
        ),
    ),
    ("Stress", ("pop_dti_p90", "pop_dti_w_p90")),
)


def control_widget_key(control: ParamControl) -> str:
    """Stable Streamlit widget key for a control."""
    return "param__" + "__".join(control.path)


def get_by_path(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    """Read a nested value from a dictionary by path."""
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping):
            return default
        if part not in current:
            return default
        current = current[part]
    return current


def set_by_path(mapping: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set a nested value in a dictionary by path, creating intermediate dicts if needed."""
    if not path:
        return
    current: Dict[str, Any] = mapping
    for part in path[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[path[-1]] = value


def controls_by_section() -> Dict[str, list[ParamControl]]:
    """Return controls grouped by section for UI rendering."""
    grouped: Dict[str, list[ParamControl]] = {section: [] for section in SECTION_ORDER}
    for control in PARAMETER_CONTROLS:
        grouped.setdefault(control.section, []).append(control)
    return grouped


def iter_non_gini_metrics() -> Iterable[str]:
    """Iterate through all non-gini metric ids used in the metric selector."""
    for _, metrics in NON_GINI_METRIC_GROUPS:
        for metric in metrics:
            yield metric
