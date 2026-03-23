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
    advanced: bool = False
    support_modes: tuple[str, ...] | None = None


POLICY_SWITCHES_SECTION = "Policy Switches"
STARTUP_SECTION = "Startup"
INCOME_SUPPORT_SECTION = "Income Support"
INCOME_SUPPORT_MODE_PATH: tuple[str, ...] = ("income_support_mode",)
INCOME_SUPPORT_MODE_WIDGET_KEY = "param__income_support_mode"


SECTION_ORDER: tuple[str, ...] = (
    POLICY_SWITCHES_SECTION,
    STARTUP_SECTION,
    "Trust",
    "Taxes",
    INCOME_SUPPORT_SECTION,
    "Automation",
    "Price & Capital",
    "Population",
)


PARAMETER_CONTROLS: tuple[ParamControl, ...] = (
    ParamControl(("disable_trust",), "Disable Trust", POLICY_SWITCHES_SECTION, "bool", help_text="Prevent trust activation, launch, and dilution."),
    ParamControl(("disable_mortgage_index",), "Disable Mortgage Indexing", POLICY_SWITCHES_SECTION, "bool", help_text="Force mortgage required payments back to the non-indexed contractual path while leaving mortgage balances in place."),
    ParamControl(("disable_mortgage_policy",), "Disable Mortgage Policy", POLICY_SWITCHES_SECTION, "bool", help_text="Turn off mortgage-gap neutralization transfers while leaving mortgage balances and required payments in place."),
    ParamControl(("mortgage_turnover_enabled",), "Enable Mortgage Turnover", POLICY_SWITCHES_SECTION, "bool", help_text="Re-originate mortgage credit to plausible households so amortized mortgage stock can turn over instead of shrinking away."),
    ParamControl(("disable_income_tax",), "Disable Income Tax", POLICY_SWITCHES_SECTION, "bool", help_text="Force household income taxes to zero while leaving other taxes and fiscal settings unchanged."),
    ParamControl(("disable_vat",), "Disable VAT", POLICY_SWITCHES_SECTION, "bool", help_text="Turn off VAT and VAT-credit effects while leaving other tax settings unchanged."),
    ParamControl(("disable_income_support",), "Disable Income Support", POLICY_SWITCHES_SECTION, "bool", help_text="Force income-support payments to zero in all quarters."),
    ParamControl(
        ("automation_disabled",),
        "Disable Automation Entirely",
        POLICY_SWITCHES_SECTION,
        "bool",
        help_text="Hold automation levels and flows at zero for all quarters. Useful for baseline-equilibrium checks.",
    ),
    ParamControl(
        ("startup_buffer_alignment_enabled",),
        "Align Startup Deposits To Runtime Buffer",
        STARTUP_SECTION,
        "bool",
        help_text="Re-seed household deposits to the solver's live liquidity-buffer target before visible Q0, without recalibrating the consumption ladder.",
    ),
    ParamControl(
        ("startup_buffer_alignment_max_iters",),
        "Buffer Alignment Iterations",
        STARTUP_SECTION,
        "int",
        1,
        20,
        1,
        help_text="Maximum startup iterations used when aligning deposits to the runtime buffer target before visible Q0.",
    ),
    ParamControl(
        ("baseline_calibration_enabled",),
        "Use Baseline Calibration",
        STARTUP_SECTION,
        "bool",
        help_text="Iteratively calibrate the startup consumption ladder against a hidden baseline regime before visible Q0.",
    ),
    ParamControl(
        ("baseline_calibration_max_iters",),
        "Calibration Iterations",
        STARTUP_SECTION,
        "int",
        1,
        20,
        1,
        help_text="Maximum fixed-point iterations used to calibrate startup household consumption anchors.",
    ),
    ParamControl(
        ("baseline_calibration_quarters",),
        "Calibration Quarters",
        STARTUP_SECTION,
        "int",
        1,
        40,
        1,
        help_text="Hidden baseline quarters simulated inside each calibration iteration.",
    ),
    ParamControl(
        ("baseline_calibration_alpha",),
        "Calibration Income Share",
        STARTUP_SECTION,
        "float",
        0.25,
        1.25,
        0.01,
        help_text="Target baseline real consumption as a share of sustainable real disposable income by quintile.",
    ),
    ParamControl(
        ("baseline_calibration_damping",),
        "Calibration Damping",
        STARTUP_SECTION,
        "float",
        0.05,
        1.0,
        0.05,
        help_text="How strongly each calibration iteration updates quintile consumption anchors.",
    ),
    ParamControl(
        ("baseline_calibration_reset_deposits_to_runtime_target",),
        "Reset Deposits To Runtime Target",
        STARTUP_SECTION,
        "bool",
        help_text="Re-seed household deposits to the solver's live liquidity-buffer target during baseline calibration.",
    ),
    ParamControl(("trust_trigger_dti",), "Trust Trigger Debt-Service-to-Income (DTI)", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("trust_launch_loan",), "Trust Launch Loan", "Trust", "float", 0.0, 50000.0, 500.0),
    ParamControl(("trust_launch_target_pct",), "Trust Launch Target %", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("trust_equity_cap",), "Trust Equity Cap", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("send_fund_residual_to_gov",), "Sweep Fund Residual To GOV", "Trust", "bool"),
    ParamControl(("fund_residual_to_gov_share",), "Fund Residual To GOV Share", "Trust", "float", 0.0, 1.0, 0.01),
    ParamControl(("vat_rate",), "VAT Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("vat_credit_phaseout_start_pct",), "VAT Credit Phaseout Start Percentile", "Taxes", "float", 0.0, 100.0, 0.5),
    ParamControl(("vat_credit_phaseout_end_pct",), "VAT Credit Phaseout End Percentile", "Taxes", "float", 0.0, 100.0, 0.5),
    ParamControl(("vat_poverty_cons_frac",), "VAT Poverty Consumption Fraction", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("income_tax_rate",), "Income Tax Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("income_tax_cutoff_pct",), "Income Tax Cutoff Percentile", "Taxes", "float", 0.0, 100.0, 0.5),
    ParamControl(("corporate_tax_rate",), "Corporate Tax Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_depr_rate_q",), "Corporate Tax Depreciation / Quarter", "Taxes", "float", 0.0, 0.10, 0.001),
    ParamControl(("dividend_payout_rate_firms",), "Firm Dividend Payout Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("dividend_payout_rate_bank",), "Bank Dividend Payout Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_dynamic_with_wages",), "Dynamic Corporate Tax With Wages", "Taxes", "bool"),
    ParamControl(("corporate_tax_rate_base",), "Corporate Tax Base Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_wage_sensitivity",), "Corporate Tax Wage Sensitivity", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_rate_min",), "Corporate Tax Min Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("corporate_tax_rate_max",), "Corporate Tax Max Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("gov_tax_rebate_rate",), "Government Surplus Rebate Rate", "Taxes", "float", 0.0, 1.0, 0.01),
    ParamControl(("gov_rebate_buffer_quarters",), "Government Rebate Buffer (quarters)", "Taxes", "int", 0, 16, 1),
    ParamControl(("gov_rebate_start_delay_quarters",), "Government Rebate Start Delay (quarters)", "Taxes", "int", 0, 40, 1),
    ParamControl(("gov_rebate_ramp_quarters",), "Government Rebate Ramp (quarters)", "Taxes", "int", 0, 80, 1),
    ParamControl(INCOME_SUPPORT_MODE_PATH, "Income Support Mode", INCOME_SUPPORT_SECTION, "select", options=("UIS", "UBI")),
    ParamControl(("income_support_issuance_share",), "Income Support Issuance Share", INCOME_SUPPORT_SECTION, "float", 0.0, 1.0, 0.01),
    ParamControl(("income_support_monotonic_floor",), "Income Support Monotonic Floor", INCOME_SUPPORT_SECTION, "bool"),
    ParamControl(("ubi_target_percentile",), "UBI Target Percentile", INCOME_SUPPORT_SECTION, "float", 0.0, 100.0, 0.5, support_modes=("UBI",)),
    ParamControl(("ubi_anchor_income_basis",), "UBI Anchor Income Basis", INCOME_SUPPORT_SECTION, "select", options=("market_income", "wages_only"), support_modes=("UBI",)),
    ParamControl(("ubi_index_series",), "UBI Index Series", INCOME_SUPPORT_SECTION, "select", options=("P_producer", "C_consumer"), support_modes=("UBI",)),
    ParamControl(("automation_path",), "Automation Path", "Automation", "select", options=("two_hump", "linear")),
    ParamControl(("hh_demand_info_share",), "HH Demand Share: Info", "Automation", "float", 0.0, 1.0, 0.01, help_text="Fixed household demand share allocated to the Info sector before any fulfillment rationing."),
    ParamControl(("automation_horizon_quarters",), "Automation Horizon Quarters", "Automation", "float", 4.0, 240.0, 1.0),
    ParamControl(("automation_w_info",), "Automation Weight: Info", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_info_cap",), "Automation Info Cap", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_phys_cap",), "Automation Physical Cap", "Automation", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_ki",), "Automation ki", "Automation", "float", 0.01, 1.0, 0.01),
    ParamControl(("automation_ti",), "Automation ti", "Automation", "float", 0.0, 120.0, 0.5),
    ParamControl(("automation_bi",), "Automation bi", "Automation", "float", 0.0, 20.0, 0.1),
    ParamControl(("automation_kp",), "Automation kp", "Automation", "float", 0.01, 1.0, 0.01),
    ParamControl(("automation_tp",), "Automation tp", "Automation", "float", 0.0, 120.0, 0.5),
    ParamControl(("sector_automation_capacity_bonus_info",), "Capacity Bonus: Info Automation", "Automation", "float", 0.0, 2.0, 0.05, advanced=True, help_text="How strongly Info-sector automation raises effective fulfillment capacity."),
    ParamControl(("sector_automation_capacity_bonus_phys",), "Capacity Bonus: Physical Automation", "Automation", "float", 0.0, 2.0, 0.05, advanced=True, help_text="How strongly Physical-sector automation raises effective fulfillment capacity."),
    ParamControl(("price_beta",), "Price Beta", "Price & Capital", "float", 0.0, 3.0, 0.01),
    ParamControl(("automation_markup_max",), "Automation Markup Max", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("automation_markup_power",), "Automation Markup Power", "Price & Capital", "float", 0.1, 5.0, 0.1),
    ParamControl(("price_adjust_speed",), "Price Adjust Speed", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("solver_relaxation",), "Solver Relaxation", "Price & Capital", "float", 0.05, 1.0, 0.05, advanced=True, help_text="Relaxation factor for the within-tick household fixed-point solver. Lower values add damping; 1.0 preserves the undamped update."),
    ParamControl(("policy_rate_rule_enabled",), "Policy Rate Rule Enabled", "Price & Capital", "bool"),
    ParamControl(("policy_rate_neutral_q",), "Policy Rate Neutral (q)", "Price & Capital", "float", 0.0, 0.05, 0.0005),
    ParamControl(("policy_rate_inflation_target_q",), "Policy Inflation Target (q)", "Price & Capital", "float", -0.02, 0.03, 0.0005),
    ParamControl(("policy_rate_phi_pi",), "Policy Phi Inflation", "Price & Capital", "float", 0.0, 3.0, 0.05),
    ParamControl(("policy_rate_phi_deflation",), "Policy Phi Deflation", "Price & Capital", "float", 0.0, 3.0, 0.05),
    ParamControl(("policy_rate_dti_target",), "Policy Debt-Service-to-Income (DTI) Target", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("policy_rate_phi_dti",), "Policy Phi Debt-Service-to-Income (DTI)", "Price & Capital", "float", 0.0, 3.0, 0.05),
    ParamControl(("policy_rate_smoothing",), "Policy Rate Smoothing", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("policy_rate_min_q",), "Policy Rate Min (q)", "Price & Capital", "float", 0.0, 0.05, 0.0005),
    ParamControl(("policy_rate_max_q",), "Policy Rate Max (q)", "Price & Capital", "float", 0.0, 0.10, 0.0005),
    ParamControl(("policy_rate_max_step_up_q",), "Policy Max Step Up (q)", "Price & Capital", "float", 0.0, 0.02, 0.0005),
    ParamControl(("policy_rate_max_step_down_q",), "Policy Max Step Down (q)", "Price & Capital", "float", 0.0, 0.02, 0.0005),
    ParamControl(("mort_index_enable",), "Mortgage Indexing Enabled", "Price & Capital", "bool"),
    ParamControl(("mort_index_weight_w",), "Mortgage Index Weight w", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("mort_index_price_series",), "Mortgage Price Series", "Price & Capital", "select", options=("P_producer", "C_consumer")),
    ParamControl(("mort_index_income_series",), "Mortgage Income Series", "Price & Capital", "select", options=("NominalHHIncome", "NominalWages", "NominalMarketIncome")),
    ParamControl(("mort_index_required_payment_mode",), "Mortgage Required Pay Mode", "Price & Capital", "select", options=("CurrentContractual", "AnchoredBase")),
    ParamControl(("mort_corridor_enable",), "Mortgage Corridor Enabled", "Price & Capital", "bool"),
    ParamControl(("mort_corridor_qtr_up",), "Mortgage Corridor Up (q)", "Price & Capital", "float", 0.0, 0.10, 0.005),
    ParamControl(("mort_corridor_qtr_dn",), "Mortgage Corridor Down (q)", "Price & Capital", "float", -0.50, 0.0, 0.005),
    ParamControl(("mort_index_ewma_lambda",), "Mortgage Index EWMA Lambda", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("mort_bank_neutralize_enable",), "Mortgage Bank Neutralization", "Price & Capital", "bool"),
    ParamControl(("mort_neutralize_trigger_mode",), "Mortgage Neutralize Trigger", "Price & Capital", "select", options=("StressOnly", "Always")),
    ParamControl(("mort_neutralize_trigger_threshold",), "Mortgage Neutralize Threshold", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("mort_neutralize_fund_allowed_if_debt_outstanding",), "Neutralize Use FUND While Debt", "Price & Capital", "bool"),
    ParamControl(("mort_neutralize_cap_mode",), "Mortgage Neutralize Cap Mode", "Price & Capital", "select", options=("None", "BankEquityFloor", "PctOfMortgageInterest", "PctOfMortgagePayment")),
    ParamControl(("mort_neutralize_cap_value",), "Mortgage Neutralize Cap Value", "Price & Capital", "float", 0.0, 5.0, 0.05),
    ParamControl(("startup_bootstrap_lagged_retained",), "Startup Bootstrap Lagged Retained", "Price & Capital", "bool"),
    ParamControl(("startup_bootstrap_retained_scale",), "Startup Bootstrap Retained Scale", "Price & Capital", "float", 0.0, 2.0, 0.05),
    ParamControl(("sector_capacity_initial_buffer",), "Initial Capacity Buffer", "Price & Capital", "float", 0.0, 0.50, 0.01, advanced=True, help_text="Extra sector capacity seeded at startup above the initial demand target."),
    ParamControl(("sector_capacity_per_k_info",), "Capacity / K: Info", "Price & Capital", "float", 0.0, 2.0, 0.01, advanced=True),
    ParamControl(("sector_capacity_per_k_phys",), "Capacity / K: Physical", "Price & Capital", "float", 0.0, 2.0, 0.01, advanced=True),
    ParamControl(("sector_supplier_share_info_for_info_capex",), "Info Supplier Share for Info CAPEX", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True, help_text="Share of Info-sector investment orders supplied by the Info sector."),
    ParamControl(("sector_supplier_share_info_for_phys_capex",), "Info Supplier Share for Physical CAPEX", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True, help_text="Share of Physical-sector investment orders supplied by the Info sector."),
    ParamControl(("sector_capex_share_min",), "Sector CAPEX Share Min", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True),
    ParamControl(("sector_capex_share_max",), "Sector CAPEX Share Max", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True),
    ParamControl(("sector_capex_gap_half_sat",), "Sector CAPEX Gap Half-Sat", "Price & Capital", "float", 0.01, 2.0, 0.01, advanced=True),
    ParamControl(("sector_capex_gap_close_rate",), "Sector CAPEX Gap Close Rate", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True),
    ParamControl(("sector_capex_growth_cap_rate_q",), "Sector CAPEX Growth Cap (q)", "Price & Capital", "float", 0.0, 0.50, 0.01, advanced=True),
    ParamControl(("sector_install_rate_q",), "Sector Install Rate (q)", "Price & Capital", "float", 0.0, 0.50, 0.01, advanced=True, help_text="Maximum share of current sector capacity that can be converted into installed new capital each quarter."),
    ParamControl(("sector_dividend_cash_buffer_q",), "Sector Dividend Cash Buffer", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True, help_text="Fraction of current-quarter revenue reserved before paying committed dividends."),
    ParamControl(("sector_dividend_service_floor",), "Sector Dividend Service Floor", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True, help_text="When household service falls below this share of desired sector demand, lagged dividend commitments are automatically reduced."),
    ParamControl(("firm_overhead_rate_info",), "Firm Overhead Rate: Info", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True),
    ParamControl(("firm_overhead_rate_phys",), "Firm Overhead Rate: Physical", "Price & Capital", "float", 0.0, 1.0, 0.01, advanced=True),
    ParamControl(("hh_buffer_spend_excess_rate_q",), "HH Spend Excess Buffer Rate (q)", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("hh_buffer_shortfall_conserve_rate_q",), "HH Conserve Shortfall Buffer Rate (q)", "Price & Capital", "float", 0.0, 1.0, 0.01),
    ParamControl(("capital_productivity_k",), "Capital Productivity k", "Price & Capital", "float", 0.0, 2.0, 0.01),
    ParamControl(("capital_productivity_scale",), "Capital Productivity Scale", "Price & Capital", "float", 100.0, 20000.0, 100.0),
    ParamControl(("capital_depr_rate_per_quarter",), "Capital Depreciation / Quarter", "Price & Capital", "float", 0.0, 1.0, 0.005),
    ParamControl(("population_config", "seed"), "Population: Seed", "Population", "int", 1, 1000000, 1),
    ParamControl(
        ("population_config", "median_wage_q"),
        "Population: Median Wage (q)",
        "Population",
        "float",
        50.0,
        5000.0,
        10.0,
    ),
    ParamControl(
        ("population_config", "employment_rate"),
        "Population: Employment Rate",
        "Population",
        "float",
        0.0,
        1.0,
        0.01,
    ),
    ParamControl(
        ("population_config", "median_deposits_q"),
        "Population: Median Deposits (q)",
        "Population",
        "float",
        100.0,
        20000.0,
        25.0,
    ),
    ParamControl(
        ("population_config", "mortgage_income_mult_median"),
        "Population: Mortgage Multiple (median)",
        "Population",
        "float",
        0.0,
        8.0,
        0.05,
    ),
    ParamControl(
        ("population_config", "revolving_income_mult_median"),
        "Population: Revolving Multiple (median)",
        "Population",
        "float",
        0.0,
        1.5,
        0.01,
    ),
    ParamControl(
        ("population_config", "sigma_wage_ln"),
        "Population: Wage Sigma (ln)",
        "Population",
        "float",
        0.10,
        2.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "sigma_deposits_ln"),
        "Population: Deposits Sigma (ln)",
        "Population",
        "float",
        0.10,
        3.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "wage_deposit_corr"),
        "Population: Wage-Deposits Correlation",
        "Population",
        "float",
        -0.95,
        0.95,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "tail_share"),
        "Population: Wealth Tail Share",
        "Population",
        "float",
        0.00,
        0.40,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "pareto_alpha"),
        "Population: Pareto Alpha",
        "Population",
        "float",
        1.01,
        5.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "mortgage_income_mult_sigma"),
        "Population: Mortgage Multiple Sigma",
        "Population",
        "float",
        0.00,
        2.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "revolving_income_mult_sigma"),
        "Population: Revolving Multiple Sigma",
        "Population",
        "float",
        0.00,
        2.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "revolving_cap_income_mult"),
        "Population: Revolving Cap (income multiple)",
        "Population",
        "float",
        0.00,
        2.00,
        0.01,
        advanced=True,
    ),
    ParamControl(
        ("population_config", "revolving_cap_deposits_mult"),
        "Population: Revolving Cap (deposits multiple)",
        "Population",
        "float",
        0.00,
        10.00,
        0.05,
        advanced=True,
    ),
)


RUN_DEFAULT_QUARTERS = 120
RUN_MIN_QUARTERS = 20
RUN_MAX_QUARTERS = 240
RUN_STEP_QUARTERS = 4


NON_GINI_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Core", ("inflation", "trust_equity_pct", "trust_value_per_h")),
    (
        "Households",
        (
            "real_consumption",
            "real_avg_income",
            "uis_per_h",
            "uis_from_fund_dep_per_h",
            "uis_from_gov_dep_per_h",
            "uis_issued_per_h",
        ),
    ),
    ("Stress", ("pop_dti_med", "pop_dti_p90", "pop_dti_w_med", "pop_dti_w_p90")),
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


def resolve_control_default(control: ParamControl, base_params: Mapping[str, Any]) -> Any:
    """Resolve a control default from config first, then optional UI fallback."""
    return get_by_path(base_params, control.path, default=control.fallback_default)


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
