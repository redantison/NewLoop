# Author: Roger Ison   roger@miximum.info
"""Default configuration for NewLoop."""

from __future__ import annotations

import copy
from typing import Any, Dict

config = {
    "parameters": {
        # Policy
        "disable_trust": False,
        "disable_mortgage_relief": False,
        "disable_mortgage_index": False,
        "disable_mortgage_policy": False,
        "disable_income_tax": False,
        "trust_launch_loan": 12000.0,       # positive value enables the leveraged launch sequence
        "trust_launch_target_pct": 0.10,    # initial trust purchase target as a share of each issuer's equity
        "trust_trigger_dti": 0.10,        # Trust triggers if last-quarter DTI metric exceeds this (population p90 when enabled; otherwise legacy proxy).
        "trust_equity_cap": 0.30,         # dilution target for FUND ownership share (avg per issuer)
        "debug_trust_trigger": True,

        # Consumption tax (VAT / sales tax) + low-income VAT credit ("prebate")
        "disable_vat": False,
        "vat_rate": 0.18,                 # 18% tax-exclusive VAT on consumption
        "vat_credit_phaseout_start_pct": 25.0,  # full credit through this eligibility-income percentile
        "vat_credit_phaseout_end_pct": 45.0,    # zero credit at and above this eligibility-income percentile
        "vat_poverty_cons_frac": 0.15,    # poverty-line consumption as fraction of baseline real consumption
        # Dividend payout + reinvestment (simple capital stock K; no vintage queue)
        "dividend_payout_rate_firms": 0.75,
        "dividend_payout_rate_firms_mature_max": 1.0,
        "dividend_payout_rate_bank": 1.0,
        "sector_dividend_maturity_gap_half_sat": 0.02,
        "sector_dividend_adjust_speed": 0.50,
        "sector_surplus_distribution_share": 0.50,
        "sector_surplus_cash_buffer_revenue_share": 0.10,
        "reinvest_rate_of_retained": 1.0,
        "solver_relaxation": 0.75,       # fixed-point relaxation for within-tick household solver (1.0 = none)
        # Startup bootstrap: seed lagged retained earnings and the initial broad-ROE
        # denominator at t=0 from the implied steady-state so early diagnostics do not
        # show an artificial jump. The heavier firm-capital bootstrap is kept as an
        # opt-in because it changes the simulated economy, not just the metric base.
        "startup_bootstrap_lagged_retained": True,
        "startup_bootstrap_retained_scale": 1.00,
        "startup_bootstrap_firm_capital": False,
        "startup_bootstrap_capital_scale": 1.00,
        "sector_input_cost_rate_info": 0.20,
        "sector_input_cost_rate_phys": 0.15,
        "ums_recycle_rate_q": 1.00,
        "capital_depr_rate_per_quarter": 0.02,
        # Sector-fulfillment pass 1: fixed household demand split plus
        # supplier-first capacity rationing with no new firm debt.
        "hh_demand_info_share": 0.30,
        "sector_capacity_initial_buffer": 0.05,
        "sector_capacity_per_k_info": 0.10,
        "sector_capacity_per_k_phys": 0.18,
        "sector_automation_capacity_bonus_info": 0.60,
        "sector_automation_capacity_bonus_phys": 0.65,
        "sector_supplier_share_info_for_info_capex": 0.55,
        "sector_supplier_share_info_for_phys_capex": 0.15,
        "sector_capex_share_min": 0.05,
        "sector_capex_share_max": 0.50,
        "sector_capex_gap_half_sat": 0.08,
        "sector_capex_gap_close_rate": 0.40,
        "sector_capex_growth_cap_rate_q": 0.08,
        "sector_capex_unmet_ewma_alpha": 0.50,
        "sector_install_rate_q": 0.07,
        "sector_maintenance_reserve_share": 0.75,
        "sector_dividend_cash_buffer_q": 0.00,
        "sector_dividend_service_floor": 0.95,
        "firm_overhead_rate_info": 0.15,
        "firm_overhead_rate_phys": 0.25,
        # Household consumption buffer behavior: spend only a fraction of deposits above
        # the target liquid buffer, and conserve when below target.
        "hh_buffer_spend_excess_rate_q": 0.10,
        "hh_buffer_shortfall_conserve_rate_q": 0.05,
        # Capital -> productivity feedback (A_eff = clamp(A + kappa*(K_per_h/K_scale)))
        "capital_productivity_k": 0.25,
        "capital_productivity_scale": 5000.0,
        # Income tax (marginal above threshold) on wages + dividends (excludes income support)
        "income_tax_rate": 0.15,          # 15% marginal rate
        "income_tax_cutoff_pct": 31.0,    # threshold percentile (same analogue as above; can be tuned)
        # Corporate tax applies to retained earnings only (distributed dividends are not taxed at the corporate level).
        # Household dividend recipients are taxed via income tax; FUND recipients are untaxed.
        "corporate_tax_rate": 0.25,
        "corporate_tax_depr_rate_q": 0.025, # ten-year straight-line depreciation allowance for the corporate tax base
        # Optional dynamic corporate tax policy (easy to disable/revert):
        # raise corporate tax as wages fall relative to baseline wages.
        "corporate_tax_dynamic_with_wages": True,
        "corporate_tax_rate_base": 0.25,
        "corporate_tax_wage_sensitivity": 0.20,  # +20pp when wage index falls from 1.0 to 0.0
        "corporate_tax_rate_min": 0.25,
        "corporate_tax_rate_max": 0.35,
        "loan_rate_per_quarter": 0.0125,   # 5% annual
        # Central-bank policy rate rule (quarterly). When enabled, loan interest uses policy_rate_q.
        "policy_rate_rule_enabled": False,
        "policy_rate_neutral_q": 0.0125,
        "policy_rate_inflation_target_q": 0.0050,  # ~2% annual
        "policy_rate_phi_pi": 0.40,
        "policy_rate_phi_deflation": 0.90,         # extra easing when inflation is negative
        "policy_rate_dti_target": 0.20,
        "policy_rate_phi_dti": 0.35,               # easing when borrower stress is high
        "policy_rate_smoothing": 0.15,             # partial-adjustment speed each quarter (lower = smoother policy path)
        "policy_rate_min_q": 0.0000,
        "policy_rate_max_q": 0.0300,
        "policy_rate_max_step_up_q": 0.0050,
        "policy_rate_max_step_down_q": 0.0050,
        # Mortgage macro-index module (hybrid price/income index anchored per household mortgage).
        "mort_index_enable": True,
        "mort_index_weight_w": 0.40,
        "mort_index_price_series": "P_producer",   # "P_producer" or "C_consumer"
        "mort_index_income_series": "NominalHHIncome",  # "NominalHHIncome"|"NominalWages"|"NominalMarketIncome"
        "mort_index_required_payment_mode": "CurrentContractual",  # "CurrentContractual"|"AnchoredBase"
        # Corridor/smoothing (corridor is always applied in log space).
        "mort_corridor_enable": True,
        "mort_corridor_qtr_up": 0.02,
        "mort_corridor_qtr_dn": -0.02,
        "mort_corridor_apply_in_logspace": True,
        "mort_index_ewma_lambda": 1.0,
        # Bank neutralization transfer for indexed-mortgage cashflow gaps.
        "mort_bank_neutralize_enable": True,
        "mort_neutralize_trigger_mode": "StressOnly",  # "Always"|"StressOnly"
        "mort_neutralize_trigger_threshold": 0.10,
        "mort_neutralize_funding_stack": ["GOV", "FUND", "ISSUANCE"],
        "mort_neutralize_fund_allowed_if_debt_outstanding": False,
        "mort_neutralize_cap_mode": "None",  # "None"|"BankEquityFloor"|"PctOfMortgageInterest"|"PctOfMortgagePayment"
        "mort_neutralize_cap_value": 0.0,
        "revolving_principal_pay_rate_q": 0.0,   # revolving principal may persist unless a later rule retires it
        "revolving_rollover_share": 0.0,        # share of revolving principal repayment immediately re-lent to the same household
        "mortgage_fixed_rate_q": 0.01125,       # 4.5% annual fixed coupon for new mortgages
        "mortgage_term_quarters": 60,           # 15-year fixed mortgage
        "mortgage_principal_pay_rate_q": 0.01,   # 1%/q max paydown if cash available
        "mortgage_turnover_enabled": True,      # replace amortized mortgage stock by issuing new mortgages to eligible households
        "mortgage_turnover_replace_share": 1.0,
        "mortgage_turnover_target_share": 0.55,  # desired share of households carrying a mortgage after turnover
        "mortgage_turnover_active_balance_floor_mult_min_desired": 0.0,  # set >0 to treat tiny balances as inactive for turnover targeting
        "mortgage_turnover_dti_cap": 0.40,
        "mortgage_turnover_income_mult_cap": 4.0,
        "mortgage_turnover_min_wage_q": 1.0,
        "send_fund_residual_to_gov": False, # legacy compatibility toggle for a full FUND residual sweep
        "fund_residual_to_gov_share": 0.0,  # optional share of residual FUND deposits sent to GOV after debt-first treatment
        "disable_income_support": False,
        "income_support_mode": "UBI",       # "UIS" | "UBI"
        "income_support_issuance_share": 0.15,  # fixed share of each quarter's income support paid via issuance
        "income_support_monotonic_floor": True, # if True, policy support cannot decline quarter-to-quarter
        "ubi_target_percentile": 30.0,      # nearest-rank percentile target used to anchor UBI per-household amount
        "ubi_anchor_income_basis": "market_income",  # "market_income" | "wages_only"
        "ubi_index_series": "P_producer",   # "P_producer" | "C_consumer"
        "gov_tax_rebate_rate": 0.25,        # share of GOV deposits above the stabilization buffer rebated each tick
        "gov_rebate_buffer_quarters": 4,    # keep one year of trailing GOV obligations before surplus recycling
        "gov_rebate_start_delay_quarters": 4, # wait one year before any GOV surplus rebate begins
        "gov_rebate_ramp_quarters": 20,     # linear ramp from zero to full rebate rate over five years
        "hard_assert_sfc": False,          # set True to hard-fail on any mismatch
        # Dashboard display mode for money columns: "nominal" or "price_normalized" (base-period dollars).
        "dashboard_value_mode": "price_normalized",

        # Population mode: when True, loads and generates a synthetic family population.
        "use_population": True,
        "population_dynamics": True,
        "population_print_baseline": False,
        "baseline_calibration_enabled": False,
        "baseline_calibration_max_iters": 12,
        "baseline_calibration_quarters": 8,
        "baseline_calibration_quintiles": 5,
        "baseline_calibration_alpha": 0.92,
        "baseline_calibration_damping": 0.30,
        "baseline_calibration_tol_pct": 0.02,
        "baseline_calibration_reset_deposits_to_runtime_target": True,
        "neutral_warmup_quarters": 2,
        "startup_buffer_alignment_enabled": True,
        "startup_buffer_alignment_max_iters": 8,
        "population_config": {
            "n_families": 20000,
            "seed": 7919,
            # Income
            "median_wage_q": 450.0,
            "sigma_wage_ln": 0.60,
            "employment_rate": 0.94,
            # Deposits / liquid-balance rule
            "deposit_generation_mode": "liquid_buffer_months",
            "base_real_cons_by_wealth_pct": (
                (20.0, 450.0),
                (50.0, 525.0),
                (80.0, 600.0),
                (95.0, 675.0),
                (100.0, 800.0),
            ),
            "liquid_buffer_months_by_wealth_pct": (
                (20.0, 1.5),
                (50.0, 3.0),
                (80.0, 5.0),
                (95.0, 8.0),
                (100.0, 12.0),
            ),
            "liquid_buffer_sigma_ln": 0.30,
            # Legacy deposit-mixture parameters retained for comparison scenarios
            "median_deposits_q": 1200.0,
            "sigma_deposits_ln": 1.05,
            "wage_deposit_corr": 0.40,
            "tail_share": 0.08,
            "pareto_alpha": 1.35,
            # Debt distribution
            "mortgage_income_mult_median": 3.25,
            "mortgage_income_mult_sigma": 0.55,
            "revolving_income_mult_median": 0.06,
            "revolving_income_mult_sigma": 0.80,
            "revolving_cap_income_mult": 0.50,
            "revolving_cap_deposits_mult": 2.0,
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
        "wage_share_of_revenue": {"FH": 0.50, "FA": 0.40},

        # Automation path ("two_hump" recommended; "linear" available as fallback)
        "automation_disabled": False,
        "automation_path": "two_hump",
        "automation_horizon_quarters": 60.0,  # used only if automation_path == "linear"

        # Two-hump parameters
        "automation_w_info": 0.65,
        # Per-sector automation asymptotes (represents non-automatable share by sector)
        "automation_info_cap": 0.90,
        "automation_phys_cap": 0.70,
        # Information automation is intentionally steeper/faster than physical automation.
        "automation_ki": 0.22,
        "automation_ti": 12.0,
        "automation_bi": 4.0,
        "automation_kp": 0.11,
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

        # Unmodeled sector reservoir for sector input-cost leakages.
        "UMS":  {"stocks": {"deposits": 0.0}},

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
    """Return a deep-copied default config safe for caller mutation."""
    return copy.deepcopy(config)
