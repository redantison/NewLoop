# Author: Roger Ison   roger@miximum.info
"""Core datatypes for NewLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class Node:
    """Economic entity with stock and memo ledgers."""

    node_id: str
    stocks: Dict[str, float] = field(default_factory=dict)
    memo: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.stocks.get(key, default))

    def set(self, key: str, value: float) -> None:
        self.stocks[key] = float(value)

    def add(self, key: str, delta: float) -> None:
        self.stocks[key] = float(self.stocks.get(key, 0.0) + delta)


@dataclass
class HouseholdState:
    """Vectorized household sector state (population mode)."""

    n: int
    wages0_q: np.ndarray
    deposits: np.ndarray
    housing_escrow: np.ndarray
    renter_rent_q: np.ndarray
    mortgage_loans: np.ndarray
    revolving_loans: np.ndarray
    mpc_q: np.ndarray
    base_real_cons_q: np.ndarray
    mort_rate_q: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_age_q: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_term_q: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_payment_sched_q: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_orig_principal: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    liquid_buffer_months_target: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    initial_tenure_code: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))

    prev_income: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    prev_uis: float = 0.0
    prev_wages_total: float = 0.0

    # Mortgage-index module state (one synthetic "mortgage record" per household).
    mort_P0: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_Y0: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_t0: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    mort_pay_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_index_prev: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    mort_dlnI_sm_prev: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    def ensure_memos(self) -> None:
        if (self.liquid_buffer_months_target.size == 0) or (self.liquid_buffer_months_target.shape[0] != self.n):
            self.liquid_buffer_months_target = np.zeros(self.n, dtype=float)
        if (self.prev_income.size == 0) or (self.prev_income.shape[0] != self.n):
            self.prev_income = np.asarray(self.wages0_q, dtype=float).copy()
        if (self.housing_escrow.size == 0) or (self.housing_escrow.shape[0] != self.n):
            self.housing_escrow = np.zeros(self.n, dtype=float)
        if (self.renter_rent_q.size == 0) or (self.renter_rent_q.shape[0] != self.n):
            self.renter_rent_q = np.zeros(self.n, dtype=float)
        if (self.mort_rate_q.size == 0) or (self.mort_rate_q.shape[0] != self.n):
            self.mort_rate_q = np.zeros(self.n, dtype=float)
        if (self.mort_age_q.size == 0) or (self.mort_age_q.shape[0] != self.n):
            self.mort_age_q = np.zeros(self.n, dtype=float)
        if (self.mort_term_q.size == 0) or (self.mort_term_q.shape[0] != self.n):
            self.mort_term_q = np.zeros(self.n, dtype=float)
        if (self.mort_payment_sched_q.size == 0) or (self.mort_payment_sched_q.shape[0] != self.n):
            self.mort_payment_sched_q = np.zeros(self.n, dtype=float)
        if (self.mort_orig_principal.size == 0) or (self.mort_orig_principal.shape[0] != self.n):
            self.mort_orig_principal = np.zeros(self.n, dtype=float)
        if (self.initial_tenure_code.size == 0) or (self.initial_tenure_code.shape[0] != self.n):
            initial_renters = (self.mortgage_loans <= 1e-12) & (self.housing_escrow <= 1e-12)
            initial_owners = (self.mortgage_loans <= 1e-12) & (self.housing_escrow > 1e-12)
            initial_codes = np.ones(self.n, dtype=int)
            initial_codes[initial_renters] = 0
            initial_codes[initial_owners] = 2
            self.initial_tenure_code = initial_codes
        if (self.mort_P0.size == 0) or (self.mort_P0.shape[0] != self.n):
            self.mort_P0 = np.zeros(self.n, dtype=float)
        if (self.mort_Y0.size == 0) or (self.mort_Y0.shape[0] != self.n):
            self.mort_Y0 = np.zeros(self.n, dtype=float)
        if (self.mort_t0.size == 0) or (self.mort_t0.shape[0] != self.n):
            self.mort_t0 = np.full(self.n, -1, dtype=int)
        if (self.mort_pay_base.size == 0) or (self.mort_pay_base.shape[0] != self.n):
            self.mort_pay_base = np.zeros(self.n, dtype=float)
        if (self.mort_index_prev.size == 0) or (self.mort_index_prev.shape[0] != self.n):
            self.mort_index_prev = np.ones(self.n, dtype=float)
        if (self.mort_dlnI_sm_prev.size == 0) or (self.mort_dlnI_sm_prev.shape[0] != self.n):
            self.mort_dlnI_sm_prev = np.zeros(self.n, dtype=float)

    def sum_deposits(self) -> float:
        return float(np.sum(self.deposits))

    def sum_loans(self) -> float:
        return float(np.sum(self.mortgage_loans) + np.sum(self.revolving_loans))


@dataclass
class TickResult:
    """Snapshot of the economy at the end of each quarter."""

    t: int
    automation: float
    automation_flow: float
    automation_info: float
    automation_info_flow: float
    automation_phys: float
    automation_phys_flow: float

    price_level: float
    inflation: float

    gini: float
    gini_market: float
    gini_disp: float
    gini_wealth: float
    private_eq_per_h: float
    hh_deposits_per_h: float
    hh_housing_value_per_h: float
    hh_debt_per_h: float
    hh_mortgage_debt_per_h: float
    hh_revolving_debt_per_h: float
    hh_mortgage_balance_total: float
    hh_mortgage_orig_principal_total: float
    hh_mortgage_active_count: float
    corporate_eq_info_per_h: float
    corporate_eq_physical_per_h: float
    corporate_eq_total_per_h: float
    private_roe_q: float
    private_broad_roe_q: float
    bank_broad_roe_q: float
    corporate_info_broad_roe_q: float
    corporate_physical_broad_roe_q: float
    sector_op_margin_info: float
    sector_op_margin_phys: float
    corporate_nonbank_broad_roe_q: float
    corporate_broad_roe_q: float
    private_inv_cov: float

    vat_per_h: float
    inc_tax_per_h: float
    corp_tax_per_h: float
    corp_tax_rate_eff: float
    vat_credit_per_h: float
    gov_dep_per_h: float
    fund_dep_per_h: float
    fund_dividend_inflow_per_h: float
    ums_drain_to_fund_per_h: float
    fund_tracked_inflows_per_h: float
    ums_recycle_to_info_per_h: float
    ums_recycle_to_phys_per_h: float
    ums_recycle_total_per_h: float
    capex_per_h: float
    sector_capacity_info_per_h: float
    sector_capacity_physical_per_h: float
    sector_hh_util_info: float
    sector_hh_util_physical: float
    sector_util_info: float
    sector_util_physical: float
    sector_demand_info_per_h: float
    sector_demand_physical_per_h: float
    unmet_demand_info_per_h: float
    unmet_demand_physical_per_h: float

    uis_per_h: float
    uis_from_fund_dep_per_h: float
    uis_from_gov_dep_per_h: float
    uis_issued_per_h: float
    trust_equity_pct: float
    trust_debt: float
    trust_value_per_h: float
    wages_total: float
    total_consumption: float

    real_avg_income: float
    real_consumption: float

    pop_gini: float
    pop_inc_med: float
    pop_inc_p90: float
    pop_dti_med: float
    pop_dti_p90: float
    pop_dti_w_med: float
    pop_dti_w_p90: float

    trust_active: bool
