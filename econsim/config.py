# Author: Roger Ison   roger@miximum.info
"""Default configuration for EconomySim."""

from __future__ import annotations

from typing import Any, Dict

config = {
    "parameters": {
        # Policy
        "trust_launch_loan": 12000.0,       # set to zero if no trust launch booster loan
        "trust_trigger_dti": 0.10,        # Trust triggers if last-quarter DTI metric exceeds this (population p90 when enabled; otherwise legacy proxy).
        "trust_equity_cap": 0.30,         # dilution target for FUND ownership share (avg per issuer)
        "debug_trust_trigger": True,
        "wealth_tax_rate": 0.0,            # wealth tax disabled (policy now uses VAT + income tax)

        # Consumption tax (VAT / sales tax) + low-income VAT credit ("prebate")
        "vat_rate": 0.18,                 # 18% tax-exclusive VAT on consumption
        "vat_credit_cutoff_pct": 31.0,    # eligibility cutoff percentile (approx "$50k" analogue)
        "vat_poverty_cons_frac": 0.15,    # poverty-line consumption as fraction of baseline real consumption
        # Dividend payout + reinvestment (simple capital stock K; no vintage queue)
        "dividend_payout_rate_firms": 0.75,
        "dividend_payout_rate_bank": 1.0,
        "reinvest_rate_of_retained": 1.0,
        "capital_depr_rate_per_quarter": 0.02,
        # Capital -> productivity feedback (A_eff = clamp(A + kappa*(K_per_h/K_scale)))
        "capital_productivity_k": 0.25,
        "capital_productivity_scale": 5000.0,
        # CAPEX supplier split: what fraction of nominal CAPEX demand is supplied by FA vs FH.
        # Default 0.0 means CAPEX goods/services are produced by the physical sector (FH).
        "capex_supply_share_fa": 0.0,

        # Income tax (marginal above threshold) on wages + dividends (excludes UBI)
        "income_tax_rate": 0.15,          # 15% marginal rate
        "income_tax_cutoff_pct": 31.0,    # threshold percentile (same analogue as above; can be tuned)
        # Corporate tax applies to retained earnings only (distributed dividends are not taxed at the corporate level).
        # Household dividend recipients are taxed via income tax; FUND recipients are untaxed.
        "corporate_tax_rate": 0.25,
        # Optional dynamic corporate tax policy (easy to disable/revert):
        # raise corporate tax as wages fall relative to baseline wages.
        "corporate_tax_dynamic_with_wages": True,
        "corporate_tax_rate_base": 0.25,
        "corporate_tax_wage_sensitivity": 0.20,  # +20pp when wage index falls from 1.0 to 0.0
        "corporate_tax_rate_min": 0.25,
        "corporate_tax_rate_max": 0.35,
        "loan_rate_per_quarter": 0.0125,   # 5% annual
        "revolving_principal_pay_rate_q": 0.05,  # 5%/q max paydown if cash available
        "mortgage_principal_pay_rate_q": 0.01,   # 1%/q max paydown if cash available
        "send_fund_residual_to_gov": False, # Trust belongs to citizens; do not sweep FUND deposits to GOV by default
        "ubi_issuance_share": 0.15,         # fixed share of each quarter's UBI paid via issuance
        "gov_tax_rebate_rate": 1.00,        # share of remaining GOV deposits rebated each tick by household tax-paid weights
        "hard_assert_sfc": False,          # set True to hard-fail on any mismatch

        # Population mode: when True, loads and generates a synthetic family population.
        "use_population": True,
        "population_dynamics": True,
        "population_print_baseline": False,
        "population_config": {
            "n_families": 20000,
            "seed": 7919,
            # Optional: tune correlation or other population knobs here
            # "wage_deposit_corr": 0.40,
        },

        # Price level
        # Competitive pass-through deflates the tax base as automation rises. To model corporate
        # capture of a share of automation gains, we allow an automation-dependent markup that
        # partially offsets deflation.
        "price_level_initial": 1.0,    # P0
        "price_beta": 1.0,             # 1.0 is default, strength of productivity->price decline vs automation
        "automation_markup_max": 0.25, # max markup factor at full automation (A=1); set 0.0 to disable
        "automation_markup_power": 1.0, # curvature: 1.0 linear in A; >1 back-loaded, <1 front-loaded
        "price_adjust_speed": 0.10,    # pass-through speed to target price (1.0=no stickiness, lower=more inertia)

        # Production structure
        "wage_share_of_revenue": {"FH": 0.65, "FA": 0.40},

        # Automation path ("two_hump" recommended; "linear" available as fallback)
        "automation_path": "two_hump",
        "automation_horizon_quarters": 60.0,  # used only if automation_path == "linear"

        # Two-hump parameters
        "automation_w_info": 0.65,
        # Per-sector automation caps (represents non-automatable share by sector)
        "automation_info_cap": 0.90,
        "automation_phys_cap": 0.70,
        "automation_ki": 0.18,
        "automation_ti": 12.0,
        "automation_bi": 4.0,
        "automation_kp": 0.14,
        "automation_tp": 32.0,
    },
    "nodes": {
        # Firms: deposit accounts used as transactional hubs
        "FA":   {"stocks": {"shares_outstanding": 10000.0, "deposits": 0.0, "K": 0.0}},
        "FH":   {"stocks": {"shares_outstanding": 10000.0, "deposits": 0.0, "K": 0.0}},

        # Bank: deposit issuer + equity issuer
        "BANK": {"stocks": {"shares_outstanding": 10000.0, "deposits": 0.0, "deposit_liab": 0.0, "loan_assets": 0.0, "reserves": 0.0, "equity": 0.0}},

        # Trust
        "FUND": {"stocks": {"deposits": 0.0, "loans": 0.0, "shares_FA": 0.0, "shares_FH": 0.0, "shares_BANK": 0.0}},

        # Government sink / spender
        "GOV":  {"stocks": {"deposits": 0.0}},

        # Aggregate households (population mode). Equity starts here.
        "HH": {"stocks": {
            "deposits": 0.0,
            "loans": 0.0,
            "shares_FA": 10000.0,
            "shares_FH": 10000.0,
            "shares_BANK": 10000.0,
        }},
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return the default config object (copy before mutating for scenario runs)."""
    return config
