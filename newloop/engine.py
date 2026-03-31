# Author: Roger Ison   roger@miximum.info
"""Economy model engine and stock-flow logic."""

from __future__ import annotations

import csv
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .mathutils import _as_np, _pct, _pct_np, automation_two_hump, calculate_gini_np
from .mortgage import (
    FixedRateMortgageSchedule,
    get_fixed_rate_mortgage_schedule,
    orig_principal_from_balance,
    payment_from_balance,
    remaining_term,
    scheduled_payment_components,
)
from .income_support import apply_income_support_payment, make_income_support_policy
from .newloop_types import HouseholdState, Node, TickResult

_WARNED_MORT_CORRIDOR_LOGSPACE_FALLBACK = False

class NewLoop:
    """
    Single-bank SFC model for experimenting with policies that could
    facilitate the AI/automation transition, with:
    - Households: synthetic population aggregated as HH; optional vectorized HouseholdState for distributional dynamics.
    - Firms: FA, FH
    - Bank: BANK (monetary issuer of deposits + equity issuer)
    - Trust: FUND
    - Government sink: GOV (optional)

    Deposit-money closure:
    - Loans create deposits (double-entry).
    - Principal repayment destroys deposits and loans (double-entry).
    - Deposit transfers conserve total deposits; they move balances between nodes.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._validate_config(config)
        self.params = config["parameters"]
        pop_cfg = self.params.get("population_config", {}) if isinstance(self.params.get("population_config", {}), dict) else {}
        rng_seed = int(pop_cfg.get("seed", 7919))
        self.rng = np.random.Generator(np.random.PCG64(rng_seed))
        self.income_support_policy = make_income_support_policy(self.params)
        p0 = float(config["parameters"].get("price_level_initial", 1.0))
        base_rate_q = max(0.0, float(config["parameters"].get("loan_rate_per_quarter", 0.0)))
        self._default_mortgage_schedule: FixedRateMortgageSchedule = get_fixed_rate_mortgage_schedule(
            max(0.0, float(config["parameters"].get("mortgage_fixed_rate_q", base_rate_q))),
            max(1, int(config["parameters"].get("mortgage_term_quarters", 60))),
        )
        self._mortgage_contract_state_dirty = True
        self._mortgage_contract_cache: Dict[str, np.ndarray] | None = None
        self._mortgage_contract_cache_rate_q: float | None = None
        self._mortgage_contract_cache_term_q: int | None = None
        payout_firms_base = max(0.0, min(1.0, float(config["parameters"].get("dividend_payout_rate_firms", 1.0))))
        self.state = {
            "t": 0,
            "automation": 0.0,
            "sector_tfp_mult_info": 1.0,
            "sector_tfp_mult_phys": 1.0,
            "trust_active": False,

            # Price level state
            "price_level": p0,
            "inflation": 0.0,

            # Per-tick income-support funding diagnostics (set in post_tick)
            "uis_from_fund_dep_total": 0.0,
            "uis_from_gov_dep_total": 0.0,
            "uis_issued_total": 0.0,
            "income_support_trigger_t": None,
            "tax_rebate_total": 0.0,
            # Lagged private equity stock used for private payout-yield proxy.
            "private_equity_prev_total": 0.0,
            "corporate_equity_prev_total": 0.0,
            "corporate_bank_equity_prev_total": 0.0,
            "corporate_info_equity_prev_total": 0.0,
            "corporate_physical_equity_prev_total": 0.0,
            "corporate_nonbank_equity_prev_total": 0.0,
            # One-time guard for startup lag bootstrap.
            "startup_bootstrap_done": False,
            # Sector-fulfillment diagnostics and lagged CAPEX planner inputs.
            "sector_capacity_info_real_prev": 0.0,
            "sector_capacity_phys_real_prev": 0.0,
            "sector_unmet_info_real_prev": 0.0,
            "sector_unmet_phys_real_prev": 0.0,
            "sector_unmet_info_real_sm_prev": 0.0,
            "sector_unmet_phys_real_sm_prev": 0.0,
            "sector_load_gap_info_real_prev": 0.0,
            "sector_load_gap_phys_real_prev": 0.0,
            "sector_load_gap_info_real_sm_prev": 0.0,
            "sector_load_gap_phys_real_sm_prev": 0.0,
            "sector_free_cash_info_prev": 0.0,
            "sector_free_cash_phys_prev": 0.0,
            "sector_capex_queue_info_nom": 0.0,
            "sector_capex_queue_phys_nom": 0.0,
            "sector_service_ratio_info_prev": 1.0,
            "sector_service_ratio_phys_prev": 1.0,
            "sector_payout_rate_info_prev": payout_firms_base,
            "sector_payout_rate_phys_prev": payout_firms_base,
            "fund_dividend_ownership_fa_prev": 0.0,
            "fund_dividend_ownership_fh_prev": 0.0,
            "fund_dividend_ownership_bk_prev": 0.0,
            # Central-bank policy-rate diagnostics (quarterly rate).
            "policy_rate_q": base_rate_q,
            "policy_rate_target_q": base_rate_q,
            "policy_rate_prev_q": base_rate_q,
            "policy_real_rate_lag_q": base_rate_q,
            "policy_inflation_input_q": 0.0,
            "policy_dti_input": 0.0,
            # Mortgage-index module diagnostics
            "mort_pay_req_total": 0.0,
            "mort_pay_ctr_total": 0.0,
            "mort_gap_total": 0.0,
            "mort_gap_paid_by_gov": 0.0,
            "mort_gap_paid_by_fund": 0.0,
            "mort_gap_paid_by_issuance": 0.0,
            "bank_mort_neutralize_inflow": 0.0,
            "bank_mort_neutralize_interest_inflow": 0.0,
            "bank_mort_neutralize_principal_inflow": 0.0,
            "mort_index_mean": 1.0,
            "mort_index_min": 1.0,
            "mort_index_max": 1.0,
            "mort_overdraft_due_to_payment_total": 0.0,
            "mort_overdraft_due_to_payment_count": 0.0,
            "sector_input_cost_info_total": 0.0,
            "sector_input_cost_phys_total": 0.0,
            "sector_input_cost_total": 0.0,
            "ums_drain_to_fund_total": 0.0,
            "ums_drain_to_gov_total": 0.0,
            "ums_recycle_to_info_total": 0.0,
            "ums_recycle_to_phys_total": 0.0,
            "ums_recycle_total": 0.0,
            "housing_financing_deposits_total": 0.0,
        }

        self.nodes: Dict[str, Node] = {
            nid: Node(nid, nd.get("stocks", {}).copy(), nd.get("memo", {}).copy())
            for nid, nd in config["nodes"].items()
        }

        # Ensure required nodes exist (population-mode core only)
        for required in ["BANK", "FUND", "FA", "FH", "GOV", "UMS", "HH", "HOUSING"]:
            if required not in self.nodes:
                self.nodes[required] = Node(required, stocks={})

        # Ensure deposit accounts exist
        for node in self.nodes.values():
            if "deposits" not in node.stocks:
                node.stocks["deposits"] = 0.0

        # -------------------------------------------------
        # Optional: synthetic population loader
        # -------------------------------------------------
        self.population = None
        self.population_cfg = None
        # Vectorized household sector (constructed when use_population is True)
        self.hh: Optional[HouseholdState] = None

        if bool(self.params.get("use_population", False)):
            from . import population as pop_mod
            # Allow config overrides via parameters["population_config"]
            overrides = self.params.get("population_config", {})
            if overrides is None:
                overrides = {}
            if not isinstance(overrides, dict):
                raise TypeError("population_config must be a dict of PopulationConfig overrides")

            cfg = pop_mod.PopulationConfig(**overrides)
            pop = pop_mod.generate_population(cfg)

            self.population = pop
            self.population_cfg = cfg

            # Build vectorized household state from the synthetic population.
            # Keep arrays as numpy for performance; avoid list<->array churn in the solver/tick loop.
            try:
                wages0_q_raw = getattr(pop, "wages_q")
                deposits_raw = getattr(pop, "deposits")
                housing_values_raw = getattr(pop, "housing_values")
                renter_rent_q_raw = getattr(pop, "renter_rent_q", [])
                mortgage_loans_raw = getattr(pop, "mortgage_loans")
                revolving_loans_raw = getattr(pop, "revolving_loans")
                mort_rate_q_raw = getattr(pop, "mortgage_rate_q")
                mort_age_q_raw = getattr(pop, "mortgage_age_q")
                mort_term_q_raw = getattr(pop, "mortgage_term_q")
                mort_payment_sched_q_raw = getattr(pop, "mortgage_payment_sched_q")
                mort_orig_principal_raw = getattr(pop, "mortgage_orig_principal")
                mpc_q_raw = getattr(pop, "mpc_q")
                base_real_cons_q_raw = getattr(pop, "base_real_cons_q")
            except Exception:
                wages0_q_raw = []
                deposits_raw = []
                housing_values_raw = []
                renter_rent_q_raw = []
                mortgage_loans_raw = []
                revolving_loans_raw = []
                mort_rate_q_raw = []
                mort_age_q_raw = []
                mort_term_q_raw = []
                mort_payment_sched_q_raw = []
                mort_orig_principal_raw = []
                mpc_q_raw = []
                base_real_cons_q_raw = []

            wages0_q = _as_np(wages0_q_raw, dtype=float)
            deposits = _as_np(deposits_raw, dtype=float)
            housing_values = _as_np(housing_values_raw, dtype=float)
            renter_rent_q = _as_np(renter_rent_q_raw, dtype=float)
            mortgage_loans = _as_np(mortgage_loans_raw, dtype=float)
            revolving_loans = _as_np(revolving_loans_raw, dtype=float)
            mort_rate_q = _as_np(mort_rate_q_raw, dtype=float)
            mort_age_q = _as_np(mort_age_q_raw, dtype=float)
            mort_term_q = _as_np(mort_term_q_raw, dtype=float)
            mort_payment_sched_q = _as_np(mort_payment_sched_q_raw, dtype=float)
            mort_orig_principal = _as_np(mort_orig_principal_raw, dtype=float)
            if housing_values.size:
                housing_escrow = housing_values.copy()
            else:
                housing_escrow = mort_orig_principal.copy() if mort_orig_principal.size else mortgage_loans.copy()
            initial_renters = (mortgage_loans <= 1e-12) & (housing_escrow <= 1e-12)
            initial_owners = (mortgage_loans <= 1e-12) & (housing_escrow > 1e-12)
            initial_tenure_code = np.ones(mortgage_loans.shape[0], dtype=int)
            initial_tenure_code[initial_renters] = 0
            initial_tenure_code[initial_owners] = 2
            mpc_q = _as_np(mpc_q_raw, dtype=float)
            base_real_cons_q = _as_np(base_real_cons_q_raw, dtype=float)
            liquid_buffer_months_target_raw = getattr(pop, "liquid_buffer_months_target", np.zeros(base_real_cons_q.shape[0], dtype=float))
            liquid_buffer_months_target = _as_np(liquid_buffer_months_target_raw, dtype=float)

            n = int(min(
                wages0_q.shape[0],
                deposits.shape[0],
                housing_escrow.shape[0],
                mortgage_loans.shape[0],
                revolving_loans.shape[0],
                mort_rate_q.shape[0] if mort_rate_q.size else mortgage_loans.shape[0],
                mort_age_q.shape[0] if mort_age_q.size else mortgage_loans.shape[0],
                mort_term_q.shape[0] if mort_term_q.size else mortgage_loans.shape[0],
                mort_payment_sched_q.shape[0] if mort_payment_sched_q.size else mortgage_loans.shape[0],
                mort_orig_principal.shape[0] if mort_orig_principal.size else mortgage_loans.shape[0],
                mpc_q.shape[0],
                base_real_cons_q.shape[0],
                liquid_buffer_months_target.shape[0],
                renter_rent_q.shape[0] if renter_rent_q.size else base_real_cons_q.shape[0],
            ))

            if n > 0:
                self.hh = HouseholdState(
                    n=n,
                    wages0_q=wages0_q[:n].copy(),
                    deposits=deposits[:n].copy(),
                    housing_escrow=housing_escrow[:n].copy(),
                    renter_rent_q=(renter_rent_q[:n].copy() if renter_rent_q.size else np.zeros(n, dtype=float)),
                    mortgage_loans=mortgage_loans[:n].copy(),
                    revolving_loans=revolving_loans[:n].copy(),
                    mpc_q=mpc_q[:n].copy(),
                    base_real_cons_q=base_real_cons_q[:n].copy(),
                    mort_rate_q=mort_rate_q[:n].copy(),
                    mort_age_q=mort_age_q[:n].copy(),
                    mort_term_q=mort_term_q[:n].copy(),
                    mort_payment_sched_q=mort_payment_sched_q[:n].copy(),
                    mort_orig_principal=mort_orig_principal[:n].copy(),
                    liquid_buffer_months_target=liquid_buffer_months_target[:n].copy(),
                    initial_tenure_code=initial_tenure_code[:n].copy(),
                )
                self.hh.ensure_memos()
            else:
                self.hh = None
            # -------------------------------------------------
            # Aggregate household node (HH) for population mode
            # -------------------------------------------------
            if "HH" not in self.nodes:
                self.nodes["HH"] = Node("HH", stocks={})

            if self.hh is not None:
                self.nodes["HH"].set("deposits", self.hh.sum_deposits())
                self.nodes["HOUSING"].set("deposits", 0.0)
                self.nodes["HH"].set("loans", self.hh.sum_loans())
                p_series0 = self._mort_price_series_value(float(self.state.get("price_level", p0)))
                y_series0 = self._mort_income_series_value(float(np.sum(self.hh.wages0_q)), 0.0, float(self.hh.prev_uis))
                self.state["mort_price_series_prev"] = float(p_series0)
                self.state["mort_income_series_prev"] = float(y_series0)
                self._ensure_mortgage_index_anchors(p_series0, y_series0, base_rate_q)
            else:
                self.nodes["HH"].set("deposits", 0.0)
                self.nodes["HOUSING"].set("deposits", 0.0)
                self.nodes["HH"].set("loans", 0.0)

            # Optional baseline printout for quick sanity checks
            if bool(self.params.get("population_print_baseline", False)) and hasattr(pop_mod, "baseline_report"):
                pop_mod.baseline_report(pop, cfg)

        # Initialize BANK aggregates to match current sector sums.
        # NOTE: config often seeds these keys with 0.0, which would otherwise prevent
        # correct initialization and can drive BANK.loan_assets negative on first repayments.
        bank = self.nodes["BANK"]
        bank.stocks["deposit_liab"] = float(self._sum_deposits_all())
        bank.stocks["loan_assets"] = float(self._sum_loans_borrowers())
        # --- Balance-sheet closure at t=0 ---
        # Ensure: loan_assets + reserves = deposit_liab + equity
        bank.stocks.setdefault("reserves", 0.0)
        bank.stocks.setdefault("equity", 0.0)

        dep_liab = bank.get("deposit_liab", 0.0)
        loan_assets = bank.get("loan_assets", 0.0)
        reserves = bank.get("reserves", 0.0)

        gap0 = dep_liab - (loan_assets + reserves)

        # If the non-bank sector begins with net deposits > net loans,
        # interpret the difference as pre-existing base reserves.
        if gap0 > 0:
            bank.add("reserves", gap0)

        # Set equity to exactly close at init
        bank.set("equity", bank.get("loan_assets", 0.0) + bank.get("reserves", 0.0) - bank.get("deposit_liab", 0.0))

        self.history: List[TickResult] = []
        self.bs_history: List[Dict[str, Any]] = []   # per-tick, per-node balance sheet snapshot
        self.inv_history: List[Dict[str, Any]] = []  # per-tick SFC invariant diagnostics
        # Per-tick income-support funding diagnostics captured *pre-payment* inside post_tick
        self.uis_debug_history: List[Dict[str, float]] = []
        # Prior-quarter GOV obligations used to retain a stabilization buffer before rebating surpluses.
        self.gov_obligation_history: List[float] = []

        self._assert_sfc_ok(context="init")

    def _bootstrap_startup_lagged_retained(self) -> None:
        """Seed startup lagged diagnostics before visible Q0."""
        if bool(self.state.get("startup_bootstrap_done", False)):
            return

        if int(self.state.get("t", 0)) != 0:
            self.state["startup_bootstrap_done"] = True
            return
        if not bool(self.params.get("startup_bootstrap_lagged_retained", True)):
            self.state["startup_bootstrap_done"] = True
            return
        if not bool(self.params.get("use_population", False)):
            self.state["startup_bootstrap_done"] = True
            return
        if not bool(self.params.get("population_dynamics", False)):
            self.state["startup_bootstrap_done"] = True
            return
        if self.hh is None or self.hh.n <= 0:
            self.state["startup_bootstrap_done"] = True
            return

        reinvest_rate = float(self.params.get("reinvest_rate_of_retained", 0.0))
        if reinvest_rate <= 0.0:
            self.state["startup_bootstrap_done"] = True
            return

        seed_sol = self.solve_within_tick_population(allow_income_support_trigger=False)
        if seed_sol is None:
            raise RuntimeError("Startup lagged-retained bootstrap expected a population solution but received None.")

        explicit_revenue = any(
            abs(float(self.nodes[node_id].memo.get("revenue_prev", 0.0))) > 1e-12
            for node_id in ("FA", "FH")
        )
        if not explicit_revenue:
            self.nodes["FA"].memo["revenue_prev"] = max(0.0, float(seed_sol.get("rev_fa", 0.0)))
            self.nodes["FH"].memo["revenue_prev"] = max(0.0, float(seed_sol.get("rev_fh", 0.0)))
            seed_sol = self.solve_within_tick_population(allow_income_support_trigger=False)
            if seed_sol is None:
                raise RuntimeError("Startup lagged-retained bootstrap expected a population solution after revenue seed.")

        self._bootstrap_startup_prev_equity(seed_sol)
        if bool(self.params.get("startup_bootstrap_firm_capital", False)):
            self._bootstrap_startup_firm_capital(seed_sol)
            # Re-solve after any true capital bootstrap so lagged retained earnings line
            # up with the visible quarter-0 balance sheet and price/productivity state.
            seed_sol = self.solve_within_tick_population(allow_income_support_trigger=False)
            if seed_sol is None:
                raise RuntimeError("Startup retained bootstrap expected a population solution after capital seed.")

        scale = float(self.params.get("startup_bootstrap_retained_scale", 1.0))
        scale = max(0.0, scale)

        explicit_retained = any(
            abs(float(self.nodes[node_id].memo.get("retained_prev", 0.0))) > 1e-12
            for node_id in ("FA", "FH", "BANK")
        )
        if not explicit_retained:
            self.nodes["FA"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_fa", 0.0)))
            self.nodes["FH"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_fh", 0.0)))
            self.nodes["BANK"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_bk", 0.0)))

        self.state["startup_bootstrap_done"] = True

    def _bootstrap_startup_prev_equity(self, seed_sol: Dict[str, Any]) -> None:
        """Seed the initial broad-ROE denominator without changing the economy state."""
        if any(
            abs(float(self.state.get(key, 0.0))) > 1e-12
            for key in (
                "corporate_equity_prev_total",
                "corporate_bank_equity_prev_total",
                "corporate_info_equity_prev_total",
                "corporate_physical_equity_prev_total",
                "corporate_nonbank_equity_prev_total",
            )
        ):
            return

        p_now = max(1e-9, float(self.state.get("price_level", self.params.get("price_level_initial", 1.0))))
        fa_broad_eq = self._firm_broad_equity_proxy("FA", p_now)
        fh_broad_eq = self._firm_broad_equity_proxy("FH", p_now)
        bank_eq = self._firm_balance_sheet_equity_proxy("BANK", p_now)

        self.state["corporate_info_equity_prev_total"] = float(fa_broad_eq)
        self.state["corporate_physical_equity_prev_total"] = float(fh_broad_eq)
        self.state["corporate_nonbank_equity_prev_total"] = float(fa_broad_eq + fh_broad_eq)
        self.state["corporate_bank_equity_prev_total"] = float(bank_eq)
        self.state["corporate_equity_prev_total"] = float(fa_broad_eq + fh_broad_eq + bank_eq)

    def _bootstrap_startup_firm_capital(self, seed_sol: Dict[str, Any]) -> None:
        """Seed a startup firm capital stock implied by retained-earnings capacity.

        The model starts firms with zero K and zero deposits, which makes visible Q0
        sector ROE explode because firms earn revenue immediately against a near-zero
        lagged equity base. We convert part of the existing startup sector capacity into
        installed capital, capped so total initial capacity stays unchanged.
        """
        if not bool(self.params.get("startup_bootstrap_firm_capital", True)):
            return

        # Respect explicit scenario-provided capital stocks.
        if any(abs(float(self.nodes[node_id].get("K", 0.0))) > 1e-12 for node_id in ("FA", "FH")):
            return

        p_now = max(1e-9, float(self.state.get("price_level", self.params.get("price_level_initial", 1.0))))
        depr_q = max(0.0, min(1.0, float(self.params.get("capital_depr_rate_per_quarter", 0.0))))
        reinvest_rate = max(0.0, float(self.params.get("reinvest_rate_of_retained", 0.0)))
        capital_scale = max(0.0, float(self.params.get("startup_bootstrap_capital_scale", 1.0)))
        if depr_q <= 1e-12 or reinvest_rate <= 0.0 or capital_scale <= 0.0:
            return

        for firm_id, retained_key, base_key in (
            ("FA", "retained_fa", "sector_base_capacity_info_real"),
            ("FH", "retained_fh", "sector_base_capacity_phys_real"),
        ):
            capacity_per_k = self._sector_capacity_per_k(firm_id)
            if capacity_per_k <= 1e-12:
                continue

            base_capacity = max(0.0, float(self.state.get(base_key, 0.0)))
            current_k = max(0.0, float(self.nodes[firm_id].get("K", 0.0)))
            embodied_capacity = base_capacity + (capacity_per_k * current_k)
            if embodied_capacity <= 1e-12:
                continue

            retained_nom = max(0.0, float(seed_sol.get(retained_key, 0.0)))
            if retained_nom <= 1e-12:
                continue

            k_target_by_retained = (capital_scale * reinvest_rate * retained_nom) / (depr_q * p_now)
            max_k_without_changing_capacity = embodied_capacity / capacity_per_k
            k_target = min(max_k_without_changing_capacity, k_target_by_retained)
            if k_target <= (current_k + 1e-12):
                continue

            self.nodes[firm_id].set("K", float(k_target))
            self.state[base_key] = float(
                max(0.0, embodied_capacity - (capacity_per_k * k_target))
            )

    def _firm_balance_sheet_equity_proxy(self, firm_id: str, price_level: float | None = None) -> float:
        p_now = float(self.state.get("price_level", 1.0) if price_level is None else price_level)
        if p_now <= 0.0:
            p_now = 1e-9
        if firm_id == "BANK":
            return float(max(0.0, float(self.nodes["BANK"].get("equity", 0.0))))
        return float(max(
            0.0,
            float(self.nodes[firm_id].get("deposits", 0.0))
            + (float(self.nodes[firm_id].get("K", 0.0)) * p_now)
            - float(self.nodes[firm_id].get("loans", 0.0)),
        ))

    def _firm_legacy_capacity_equity_proxy(self, firm_id: str, price_level: float | None = None) -> float:
        if firm_id not in ("FA", "FH"):
            return 0.0
        capacity_per_k = self._sector_capacity_per_k(firm_id)
        if capacity_per_k <= 1e-12:
            return 0.0
        p_now = float(self.state.get("price_level", 1.0) if price_level is None else price_level)
        if p_now <= 0.0:
            p_now = 1e-9
        base_key = "sector_base_capacity_info_real" if firm_id == "FA" else "sector_base_capacity_phys_real"
        base_capacity = max(0.0, float(self.state.get(base_key, 0.0)))
        return float(base_capacity * (p_now / capacity_per_k))

    def _firm_broad_equity_proxy(self, firm_id: str, price_level: float | None = None) -> float:
        return float(
            self._firm_balance_sheet_equity_proxy(firm_id, price_level)
            + self._firm_legacy_capacity_equity_proxy(firm_id, price_level)
        )

    def _update_policy_rate(self) -> None:
        """Update the quarterly policy rate using lagged observables (no same-tick circularity)."""
        fallback_rate = max(0.0, float(self.params.get("loan_rate_per_quarter", 0.0)))
        prev_rate = max(0.0, float(self.state.get("policy_rate_q", fallback_rate)))

        if self.history:
            prev = self.history[-1]
            pi_input = float(prev.inflation)
            dti_input = max(float(prev.pop_dti_p90), float(prev.pop_dti_w_p90))
        else:
            pi_input = float(self.state.get("inflation", 0.0))
            dti_input = 0.0

        enabled = bool(self.params.get("policy_rate_rule_enabled", False))
        if not enabled:
            rate_q = fallback_rate
            target_q = fallback_rate
        else:
            neutral_q = float(self.params.get("policy_rate_neutral_q", fallback_rate))
            pi_target_q = float(self.params.get("policy_rate_inflation_target_q", 0.0))
            phi_pi = float(self.params.get("policy_rate_phi_pi", 0.0))
            phi_defl = float(self.params.get("policy_rate_phi_deflation", 0.0))
            dti_target = float(self.params.get("policy_rate_dti_target", 0.0))
            phi_dti = float(self.params.get("policy_rate_phi_dti", 0.0))

            r_min = float(self.params.get("policy_rate_min_q", 0.0))
            r_max = float(self.params.get("policy_rate_max_q", 1.0))
            if r_max < r_min:
                r_max = r_min

            smooth = float(self.params.get("policy_rate_smoothing", 1.0))
            smooth = max(0.0, min(1.0, smooth))

            max_up = max(0.0, float(self.params.get("policy_rate_max_step_up_q", 1.0)))
            max_dn = max(0.0, float(self.params.get("policy_rate_max_step_down_q", 1.0)))

            deflation_gap = max(0.0, -pi_input)
            dti_gap = max(0.0, dti_input - dti_target)
            raw_target = (
                neutral_q
                + phi_pi * (pi_input - pi_target_q)
                - phi_defl * deflation_gap
                - phi_dti * dti_gap
            )
            target_q = max(r_min, min(r_max, raw_target))

            stepped = (1.0 - smooth) * prev_rate + smooth * target_q
            stepped = min(stepped, prev_rate + max_up)
            stepped = max(stepped, prev_rate - max_dn)
            rate_q = max(r_min, min(r_max, stepped))

        self.state["policy_rate_prev_q"] = float(prev_rate)
        self.state["policy_rate_q"] = float(rate_q)
        self.state["policy_rate_target_q"] = float(target_q)
        self.state["policy_inflation_input_q"] = float(pi_input)
        self.state["policy_dti_input"] = float(dti_input)
        self.state["policy_real_rate_lag_q"] = float(rate_q - pi_input)

    def _mort_price_series_value(self, producer_price_level: float) -> float:
        p = float(producer_price_level) if float(producer_price_level) > 0.0 else 1e-9
        price_series = str(self.params.get("mort_index_price_series", "P_producer")).strip()
        if price_series == "C_consumer":
            vat = self._effective_vat_rate()
            return max(1e-9, p * (1.0 + vat))
        return max(1e-9, p)

    def _trust_disabled(self) -> bool:
        return bool(self.params.get("disable_trust", False))

    def _mortgage_relief_disabled(self) -> bool:
        # Keep honoring the legacy split flags so older saved configs still work,
        # but treat either one as disabling the unified mortgage-relief regime.
        return (
            bool(self.params.get("disable_mortgage_relief", False))
            or bool(self.params.get("disable_mortgage_index", False))
            or bool(self.params.get("disable_mortgage_policy", False))
        )

    def _mortgage_index_disabled(self) -> bool:
        return self._mortgage_relief_disabled()

    def _mortgage_policy_disabled(self) -> bool:
        return self._mortgage_relief_disabled()

    def _income_tax_disabled(self) -> bool:
        return bool(self.params.get("disable_income_tax", False))

    def _income_support_disabled(self) -> bool:
        return bool(self.params.get("disable_income_support", False))

    def _income_support_start_delay_quarters(self) -> int:
        return max(0, int(self.params.get("income_support_start_delay_quarters", 0)))

    def _income_support_ramp_quarters(self) -> int:
        return max(0, int(self.params.get("income_support_ramp_quarters", 0)))

    def _apply_income_support_start_delay(self, support_per_h: float, *, allow_trigger: bool = True) -> float:
        support = max(0.0, float(support_per_h))
        delay_q = self._income_support_start_delay_quarters()
        current_t = int(self.state.get("t", 0))
        trigger_t = self.state.get("income_support_trigger_t", None)
        if support > 1e-12 and trigger_t is None and allow_trigger:
            trigger_t = current_t
            self.state["income_support_trigger_t"] = int(current_t)

        if trigger_t is None:
            return support if (delay_q <= 0 and self._income_support_ramp_quarters() <= 0) else 0.0
        release_t = int(trigger_t) + delay_q
        if current_t < release_t:
            return 0.0

        ramp_q = self._income_support_ramp_quarters()
        if ramp_q <= 0:
            return support

        ramp_elapsed_q = current_t - release_t
        ramp_frac = min(1.0, float(ramp_elapsed_q + 1) / float(ramp_q))
        return support * ramp_frac

    def _effective_income_tax_rate(self) -> float:
        if self._income_tax_disabled():
            return 0.0
        return max(0.0, float(self.params.get("income_tax_rate", 0.0)))

    def _effective_vat_rate(self) -> float:
        if bool(self.params.get("disable_vat", False)):
            return 0.0
        vat = float(self.params.get("vat_rate", 0.0))
        return max(0.0, vat)

    def _mort_income_series_value(self, wages_total: float, div_house_total: float, uis_per_h: float) -> float:
        n_hh = float(self.hh.n) if (self.hh is not None and self.hh.n > 0) else 1.0
        wages = max(0.0, float(wages_total))
        div_hh = max(0.0, float(div_house_total))
        uis_total = max(0.0, float(uis_per_h)) * n_hh

        income_series = str(self.params.get("mort_index_income_series", "NominalHHIncome")).strip()
        if income_series == "NominalWages":
            y = wages
        elif income_series == "NominalMarketIncome":
            y = wages + div_hh
        else:
            # NominalHHIncome (default): wages + household dividends + income support.
            y = wages + div_hh + uis_total
        return max(1e-9, float(y))

    def _mortgage_fixed_rate_q(self) -> float:
        return max(0.0, float(self.params.get("mortgage_fixed_rate_q", self.params.get("loan_rate_per_quarter", 0.0))))

    def _mortgage_term_quarters(self) -> int:
        return max(1, int(self.params.get("mortgage_term_quarters", 60)))

    def _invalidate_mortgage_contract_state(self) -> None:
        self._mortgage_contract_state_dirty = True
        self._mortgage_contract_cache = None
        self._mortgage_contract_cache_rate_q = None
        self._mortgage_contract_cache_term_q = None

    def _default_mortgage_product_mask(self, active_mask: np.ndarray) -> np.ndarray:
        if self.hh is None or self.hh.n <= 0:
            return np.zeros(0, dtype=bool)
        hh = self.hh
        default_rate_q = self._mortgage_fixed_rate_q()
        default_term_q = float(self._mortgage_term_quarters())
        return (
            active_mask
            & np.isclose(np.asarray(hh.mort_rate_q, dtype=float), default_rate_q, rtol=0.0, atol=1e-12)
            & np.isclose(np.asarray(hh.mort_term_q, dtype=float), default_term_q, rtol=0.0, atol=1e-12)
        )

    def _current_mortgage_schedule(self) -> FixedRateMortgageSchedule:
        rate_q = self._mortgage_fixed_rate_q()
        term_q = self._mortgage_term_quarters()
        if (
            abs(float(self._default_mortgage_schedule.rate_q) - float(rate_q)) > 1e-12
            or int(self._default_mortgage_schedule.term_q) != int(term_q)
        ):
            self._default_mortgage_schedule = get_fixed_rate_mortgage_schedule(rate_q, term_q)
        return self._default_mortgage_schedule

    def _contractual_mortgage_components(
        self,
        mort: np.ndarray,
        mort_rate_q: np.ndarray,
        mort_age_q: np.ndarray,
        mort_term_q: np.ndarray,
        mort_payment_sched_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(mort.shape[0])
        mort_remaining_q = remaining_term(mort_term_q, mort_age_q)
        mort_pay_ctr_i = np.zeros(n, dtype=float)
        mort_interest_due_i = np.zeros(n, dtype=float)
        mort_principal_ctr_i = np.zeros(n, dtype=float)

        active = mort > 1e-12
        if not np.any(active):
            return mort_remaining_q, mort_pay_ctr_i, mort_interest_due_i, mort_principal_ctr_i

        default_mask = self._default_mortgage_product_mask(active)
        if np.any(default_mask):
            schedule = self._current_mortgage_schedule()
            (
                mort_pay_ctr_i[default_mask],
                mort_interest_due_i[default_mask],
                mort_principal_ctr_i[default_mask],
            ) = schedule.contractual_components(
                mort[default_mask],
                mort_payment_sched_q[default_mask],
                mort_age_q[default_mask],
            )

        fallback_mask = active & (~default_mask)
        if np.any(fallback_mask):
            (
                mort_pay_ctr_i[fallback_mask],
                mort_interest_due_i[fallback_mask],
                mort_principal_ctr_i[fallback_mask],
            ) = scheduled_payment_components(
                mort[fallback_mask],
                mort_rate_q[fallback_mask],
                mort_payment_sched_q[fallback_mask],
                mort_remaining_q[fallback_mask],
            )

        return mort_remaining_q, mort_pay_ctr_i, mort_interest_due_i, mort_principal_ctr_i

    def _mortgage_contract_snapshot(self) -> Dict[str, np.ndarray]:
        self._refresh_mortgage_contract_state()
        if self._mortgage_contract_cache is None:
            raise RuntimeError("Mortgage contract cache is unavailable after refresh.")
        return self._mortgage_contract_cache

    def _refresh_mortgage_contract_state(self) -> None:
        if self.hh is None or self.hh.n <= 0:
            return
        current_rate_q = self._mortgage_fixed_rate_q()
        current_term_q = self._mortgage_term_quarters()
        if (
            not self._mortgage_contract_state_dirty
            and self._mortgage_contract_cache is not None
            and self._mortgage_contract_cache_rate_q is not None
            and self._mortgage_contract_cache_term_q is not None
            and abs(float(self._mortgage_contract_cache_rate_q) - float(current_rate_q)) <= 1e-12
            and int(self._mortgage_contract_cache_term_q) == int(current_term_q)
        ):
            return
        hh = self.hh
        hh.ensure_memos()
        mort = _as_np(hh.mortgage_loans, dtype=float)
        matured = (
            (mort > 1e-12)
            & (_as_np(hh.mort_term_q, dtype=float) > 1e-12)
            & (_as_np(hh.mort_age_q, dtype=float) >= (_as_np(hh.mort_term_q, dtype=float) - 1e-12))
        )
        if np.any(matured):
            matured_total = float(np.sum(np.maximum(0.0, mort[matured])))
            mort[matured] = 0.0
            if matured_total > 0.0:
                bank = self.nodes["BANK"]
                bank.add("loan_assets", -matured_total)
                bank.add("equity", -matured_total)
        active = mort > 1e-12
        inactive = ~active

        if np.any(active):
            default_rate_q = self._mortgage_fixed_rate_q()
            default_term_q = float(self._mortgage_term_quarters())
            hh.mort_rate_q[active] = np.where(hh.mort_rate_q[active] > 1e-12, hh.mort_rate_q[active], default_rate_q)
            hh.mort_term_q[active] = np.where(hh.mort_term_q[active] > 1e-12, hh.mort_term_q[active], default_term_q)
            hh.mort_age_q[active] = np.maximum(0.0, np.minimum(hh.mort_age_q[active], hh.mort_term_q[active] - 1.0))
            default_mask = self._default_mortgage_product_mask(active)
            if np.any(default_mask):
                schedule = self._current_mortgage_schedule()
                hh.mort_payment_sched_q[default_mask] = np.where(
                    hh.mort_payment_sched_q[default_mask] > 1e-12,
                    hh.mort_payment_sched_q[default_mask],
                    schedule.contract_payment_from_balance(mort[default_mask], hh.mort_age_q[default_mask]),
                )
                hh.mort_orig_principal[default_mask] = np.where(
                    hh.mort_orig_principal[default_mask] > 1e-12,
                    hh.mort_orig_principal[default_mask],
                    schedule.orig_principal_from_balance(mort[default_mask], hh.mort_age_q[default_mask]),
                )

            fallback_mask = active & (~default_mask)
            if np.any(fallback_mask):
                rem_q = remaining_term(hh.mort_term_q[fallback_mask], hh.mort_age_q[fallback_mask])
                hh.mort_payment_sched_q[fallback_mask] = np.where(
                    hh.mort_payment_sched_q[fallback_mask] > 1e-12,
                    hh.mort_payment_sched_q[fallback_mask],
                    payment_from_balance(mort[fallback_mask], hh.mort_rate_q[fallback_mask], rem_q),
                )
                hh.mort_orig_principal[fallback_mask] = np.where(
                    hh.mort_orig_principal[fallback_mask] > 1e-12,
                    hh.mort_orig_principal[fallback_mask],
                    orig_principal_from_balance(
                        mort[fallback_mask],
                        hh.mort_rate_q[fallback_mask],
                        hh.mort_term_q[fallback_mask],
                        hh.mort_age_q[fallback_mask],
                    ),
                )

        if np.any(inactive):
            hh.mort_rate_q[inactive] = 0.0
            hh.mort_age_q[inactive] = 0.0
            hh.mort_term_q[inactive] = 0.0
            hh.mort_payment_sched_q[inactive] = 0.0
            hh.mort_orig_principal[inactive] = 0.0
            hh.mort_P0[inactive] = 0.0
            hh.mort_Y0[inactive] = 0.0
            hh.mort_t0[inactive] = -1
            hh.mort_pay_base[inactive] = 0.0
            hh.mort_index_prev[inactive] = 1.0
            hh.mort_dlnI_sm_prev[inactive] = 0.0

        mort_rate_q = np.maximum(0.0, _as_np(hh.mort_rate_q, dtype=float))
        mort_age_q = np.maximum(0.0, _as_np(hh.mort_age_q, dtype=float))
        mort_term_q = np.maximum(0.0, _as_np(hh.mort_term_q, dtype=float))
        mort_payment_sched_q = np.maximum(0.0, _as_np(hh.mort_payment_sched_q, dtype=float))
        (
            mort_remaining_q,
            mort_pay_ctr_i,
            mort_interest_due_i,
            mort_principal_ctr_i,
        ) = self._contractual_mortgage_components(
            mort=mort,
            mort_rate_q=mort_rate_q,
            mort_age_q=mort_age_q,
            mort_term_q=mort_term_q,
            mort_payment_sched_q=mort_payment_sched_q,
        )
        self._mortgage_contract_cache = {
            "mort_rate_q": mort_rate_q.astype(float, copy=True),
            "mort_age_q": mort_age_q.astype(float, copy=True),
            "mort_term_q": mort_term_q.astype(float, copy=True),
            "mort_payment_sched_q": mort_payment_sched_q.astype(float, copy=True),
            "mort_remaining_q": mort_remaining_q.astype(float, copy=True),
            "mort_pay_ctr_i": mort_pay_ctr_i.astype(float, copy=True),
            "mort_interest_due_i": mort_interest_due_i.astype(float, copy=True),
            "mort_principal_ctr_i": mort_principal_ctr_i.astype(float, copy=True),
        }
        self._mortgage_contract_cache_rate_q = float(current_rate_q)
        self._mortgage_contract_cache_term_q = int(current_term_q)
        self._mortgage_contract_state_dirty = False

    def _ensure_mortgage_index_anchors(self, p_series_now: float, y_series_now: float, rL: float) -> None:
        if self.hh is None or self.hh.n <= 0:
            return
        hh = self.hh
        hh.ensure_memos()
        self._refresh_mortgage_contract_state()

        mort = _as_np(hh.mortgage_loans, dtype=float)
        active = mort > 1e-12
        inactive = ~active
        new_mask = active & (hh.mort_t0 < 0)

        if np.any(new_mask):
            hh.mort_P0[new_mask] = float(max(1e-9, p_series_now))
            hh.mort_Y0[new_mask] = float(max(1e-9, y_series_now))
            hh.mort_t0[new_mask] = int(self.state.get("t", 0))
            hh.mort_pay_base[new_mask] = np.maximum(0.0, hh.mort_payment_sched_q[new_mask])
            hh.mort_index_prev[new_mask] = 1.0
            hh.mort_dlnI_sm_prev[new_mask] = 0.0

        if np.any(inactive):
            hh.mort_P0[inactive] = 0.0
            hh.mort_Y0[inactive] = 0.0
            hh.mort_t0[inactive] = -1
            hh.mort_pay_base[inactive] = 0.0
            hh.mort_index_prev[inactive] = 1.0
            hh.mort_dlnI_sm_prev[inactive] = 0.0

    def _compute_mortgage_index_terms(
        self,
        *,
        mort: np.ndarray,
        rL: float,
        wages_total: float,
        div_house_total: float,
        uis_per_h: float,
        commit_state: bool,
    ) -> Dict[str, Any]:
        if self.hh is None or self.hh.n <= 0:
            n = int(mort.shape[0])
            zeros = np.zeros(n, dtype=float)
            return {
                "mort_pay_req_i": zeros,
                "mort_pay_ctr_i": zeros,
                "mort_interest_due_i": zeros,
                "mort_interest_paid_i": zeros,
                "mort_principal_paid_i": zeros,
                "mort_gap_i": zeros,
                "mort_gap_total": 0.0,
                "mort_pay_req_total": 0.0,
                "mort_pay_ctr_total": 0.0,
                "mort_index_mean": 1.0,
                "mort_index_min": 1.0,
                "mort_index_max": 1.0,
                "mort_index_i": zeros,
                "mort_dln_i": zeros,
                "mort_dln_sm_i": zeros,
                "p_series_now": float(max(1e-9, self.state.get("price_level", 1.0))),
                "y_series_now": 1.0,
            }

        hh = self.hh
        hh.ensure_memos()
        n = int(hh.n)
        mort_vec = _as_np(mort, dtype=float)
        snapshot = self._mortgage_contract_snapshot()
        mort_rate_q = snapshot["mort_rate_q"]
        mort_age_q = snapshot["mort_age_q"]
        mort_term_q = snapshot["mort_term_q"]
        mort_payment_sched_q = snapshot["mort_payment_sched_q"]
        mort_remaining_q = snapshot["mort_remaining_q"]
        use_cached_contract = np.shares_memory(mort_vec, _as_np(hh.mortgage_loans, dtype=float))
        if use_cached_contract:
            mort_pay_ctr_i = snapshot["mort_pay_ctr_i"]
            mort_interest_due_i = snapshot["mort_interest_due_i"]
            mort_principal_ctr_i = snapshot["mort_principal_ctr_i"]
        else:
            (
                mort_remaining_q,
                mort_pay_ctr_i,
                mort_interest_due_i,
                mort_principal_ctr_i,
            ) = self._contractual_mortgage_components(
                mort=mort_vec,
                mort_rate_q=mort_rate_q,
                mort_age_q=mort_age_q,
                mort_term_q=mort_term_q,
                mort_payment_sched_q=mort_payment_sched_q,
            )

        p_series_now = self._mort_price_series_value(float(self.state.get("price_level", 1.0)))
        y_series_now = self._mort_income_series_value(wages_total, div_house_total, uis_per_h)
        p_series_prev = float(self.state.get("mort_price_series_prev", p_series_now))
        y_series_prev = float(self.state.get("mort_income_series_prev", y_series_now))
        p_series_prev = max(1e-9, p_series_prev)
        y_series_prev = max(1e-9, y_series_prev)

        enabled = bool(self.params.get("mort_index_enable", False)) and (not self._mortgage_index_disabled())
        active = (mort_vec > 1e-12) & (hh.mort_t0 >= 0)
        current_t = int(self.state.get("t", 0))
        just_originated = active & (hh.mort_t0 == current_t)
        max_pay_i = mort_interest_due_i + np.maximum(0.0, mort_vec)

        if enabled:
            if not bool(self.params.get("mort_corridor_apply_in_logspace", True)):
                global _WARNED_MORT_CORRIDOR_LOGSPACE_FALLBACK
                if not _WARNED_MORT_CORRIDOR_LOGSPACE_FALLBACK:
                    warnings.warn(
                        "mort_corridor_apply_in_logspace=False is not currently implemented; "
                        "continuing with log-space corridor math.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _WARNED_MORT_CORRIDOR_LOGSPACE_FALLBACK = True
            w = max(0.0, min(1.0, float(self.params.get("mort_index_weight_w", 0.40))))
            dln_raw = (w * math.log(p_series_now / p_series_prev)) + ((1.0 - w) * math.log(y_series_now / y_series_prev))

            lam = float(self.params.get("mort_index_ewma_lambda", 1.0))
            lam = max(0.0, min(1.0, lam))
            dln_prev = _as_np(hh.mort_dlnI_sm_prev, dtype=float)
            dln_sm_i = (lam * dln_raw) + ((1.0 - lam) * dln_prev)

            if bool(self.params.get("mort_corridor_enable", True)):
                up = float(self.params.get("mort_corridor_qtr_up", 0.02))
                dn = float(self.params.get("mort_corridor_qtr_dn", -0.02))
                up = max(-0.999999, up)
                dn = max(-0.999999, dn)
                if up < dn:
                    up, dn = dn, up
                c_up = math.log1p(up)
                c_dn = math.log1p(dn)
                dln_i = np.clip(dln_sm_i, c_dn, c_up)
            else:
                dln_i = dln_sm_i

            i_prev = _as_np(hh.mort_index_prev, dtype=float)
            i_curr = i_prev.copy()
            i_curr[active] = i_prev[active] * np.exp(dln_i[active])
            if np.any(just_originated):
                # Freshly issued mortgages anchor at this quarter's series level. They
                # should start with index 1.0 rather than inheriting the pre-issuance
                # quarter-over-quarter move, and their EWMA state should start clean too.
                i_curr[just_originated] = 1.0
                dln_i[just_originated] = 0.0
                dln_sm_i[just_originated] = 0.0
            i_curr = np.maximum(0.0, i_curr)

            mort_pay_req_i = np.zeros(n, dtype=float)
            mort_pay_req_i[active] = np.maximum(0.0, _as_np(hh.mort_pay_base, dtype=float)[active] * i_curr[active])
        else:
            dln_sm_i = _as_np(hh.mort_dlnI_sm_prev, dtype=float).copy()
            i_curr = _as_np(hh.mort_index_prev, dtype=float).copy()
            mort_pay_req_i = mort_pay_ctr_i.copy()
            dln_i = np.zeros(n, dtype=float)

        req_mode = str(self.params.get("mort_index_required_payment_mode", "CurrentContractual")).strip()
        if enabled and req_mode == "AnchoredBase":
            base_vec = np.maximum(0.0, _as_np(hh.mort_pay_base, dtype=float))
            mort_pay_req_i = base_vec * i_curr
        elif enabled:
            mort_pay_req_i = mort_pay_ctr_i * i_curr

        if enabled:
            # Borrower-protection ceiling: the indexed nominal payment may move down with the
            # index, but it cannot exceed the original contractual burden in real terms.
            # Convert the origination real burden back into current nominal terms using the
            # configured mortgage price series so inflation never raises required payments above
            # the origination real burden.
            base_vec = np.maximum(0.0, _as_np(hh.mort_pay_base, dtype=float))
            p0_vec = np.maximum(1e-9, _as_np(hh.mort_P0, dtype=float))
            real_burden_cap_i = base_vec * (float(p_series_now) / p0_vec)
            mort_pay_req_i = np.minimum(mort_pay_req_i, real_burden_cap_i)
            # The indexed path is strictly a relief mechanism. It must never require more
            # than the contemporaneous non-indexed contractual payment.
            mort_pay_req_i = np.minimum(mort_pay_req_i, mort_pay_ctr_i)

        mort_pay_req_i = np.minimum(np.maximum(0.0, mort_pay_req_i), max_pay_i)
        mort_interest_paid_i = np.minimum(mort_interest_due_i, mort_pay_req_i)
        mort_interest_paid_i = np.maximum(0.0, mort_interest_paid_i)
        mort_principal_paid_i = np.maximum(0.0, mort_pay_req_i - mort_interest_paid_i)
        mort_principal_paid_i = np.minimum(mort_principal_paid_i, np.maximum(0.0, mort_vec))
        mort_interest_gap_i = np.maximum(0.0, mort_interest_due_i - mort_interest_paid_i)
        mort_principal_gap_i = np.maximum(0.0, mort_principal_ctr_i - mort_principal_paid_i)

        mort_gap_i = np.maximum(0.0, mort_pay_ctr_i - mort_pay_req_i) if enabled else np.zeros(n, dtype=float)

        if np.any(active):
            mort_index_mean = float(np.mean(i_curr[active]))
            mort_index_min = float(np.min(i_curr[active]))
            mort_index_max = float(np.max(i_curr[active]))
        else:
            mort_index_mean = 1.0
            mort_index_min = 1.0
            mort_index_max = 1.0

        if commit_state:
            hh.mort_index_prev = i_curr.astype(float, copy=True)
            hh.mort_dlnI_sm_prev = dln_sm_i.astype(float, copy=True)
            self.state["mort_price_series_prev"] = float(p_series_now)
            self.state["mort_income_series_prev"] = float(y_series_now)

        return {
            "mort_pay_req_i": mort_pay_req_i,
            "mort_pay_ctr_i": mort_pay_ctr_i,
            "mort_interest_due_i": mort_interest_due_i,
            "mort_interest_paid_i": mort_interest_paid_i,
            "mort_principal_paid_i": mort_principal_paid_i,
            "mort_interest_gap_i": mort_interest_gap_i,
            "mort_principal_gap_i": mort_principal_gap_i,
            "mort_gap_i": mort_gap_i,
            "mort_interest_gap_total": float(np.sum(mort_interest_gap_i)),
            "mort_principal_gap_total": float(np.sum(mort_principal_gap_i)),
            "mort_gap_total": float(np.sum(mort_gap_i)),
            "mort_pay_req_total": float(np.sum(mort_pay_req_i)),
            "mort_pay_ctr_total": float(np.sum(mort_pay_ctr_i)),
            "mort_index_mean": mort_index_mean,
            "mort_index_min": mort_index_min,
            "mort_index_max": mort_index_max,
            "mort_index_i": i_curr.astype(float, copy=True),
            "mort_dln_i": dln_i.astype(float, copy=True),
            "mort_dln_sm_i": dln_sm_i.astype(float, copy=True),
            "p_series_now": float(p_series_now),
            "y_series_now": float(y_series_now),
        }

    def _sector_hh_demand_share_fa(self) -> float:
        return max(0.0, min(1.0, float(self.params.get("hh_demand_info_share", 0.5))))

    def _sector_supplier_share_info(self, investor_id: str) -> float:
        if investor_id == "FA":
            key = "sector_supplier_share_info_for_info_capex"
        else:
            key = "sector_supplier_share_info_for_phys_capex"
        return max(0.0, min(1.0, float(self.params.get(key, 0.5))))

    def _sector_capacity_per_k(self, firm_id: str) -> float:
        key = "sector_capacity_per_k_info" if firm_id == "FA" else "sector_capacity_per_k_phys"
        return max(0.0, float(self.params.get(key, 0.0)))

    def _sector_tfp_alpha(self, firm_id: str) -> float:
        key = "sector_tfp_alpha_info" if firm_id == "FA" else "sector_tfp_alpha_phys"
        return max(0.0, float(self.params.get(key, 0.0)))

    def _sector_tfp_multiplier(self, firm_id: str) -> float:
        if firm_id == "FA":
            auto = max(0.0, float(self.state.get("automation_info", self.state.get("automation", 0.0))))
        else:
            auto = max(0.0, float(self.state.get("automation_phys", self.state.get("automation", 0.0))))
        return max(1.0, 1.0 + (self._sector_tfp_alpha(firm_id) * auto))

    def _sector_capacity_multiplier(self, firm_id: str) -> float:
        if firm_id == "FA":
            auto = max(0.0, float(self.state.get("automation_info", self.state.get("automation", 0.0))))
            bonus = max(0.0, float(self.params.get("sector_automation_capacity_bonus_info", 0.0)))
        else:
            auto = max(0.0, float(self.state.get("automation_phys", self.state.get("automation", 0.0))))
            bonus = max(0.0, float(self.params.get("sector_automation_capacity_bonus_phys", 0.0)))
        return max(1.0, (1.0 + (bonus * auto)) * self._sector_tfp_multiplier(firm_id))

    def _ensure_sector_capacity_anchors(
        self,
        hh_demand_fa_real: float,
        hh_demand_fh_real: float,
        supplier_fa_real: float = 0.0,
        supplier_fh_real: float = 0.0,
        ums_fa_real: float = 0.0,
        ums_fh_real: float = 0.0,
    ) -> None:
        if ("sector_base_capacity_info_real" in self.state) and ("sector_base_capacity_phys_real" in self.state):
            return

        buffer = max(0.0, float(self.params.get("sector_capacity_initial_buffer", 0.05)))
        target_fa = max(0.0, float(hh_demand_fa_real) + float(supplier_fa_real) + float(ums_fa_real))
        target_fh = max(0.0, float(hh_demand_fh_real) + float(supplier_fh_real) + float(ums_fh_real))
        if target_fa <= 1e-12 and target_fh <= 1e-12 and self.hh is not None and self.hh.n > 0:
            base_real = _as_np(self.hh.base_real_cons_q, dtype=float)
            target_total = float(np.sum(np.maximum(0.0, base_real)))
            share_fa = self._sector_hh_demand_share_fa()
            target_fa = share_fa * target_total
            target_fh = (1.0 - share_fa) * target_total

        target_fa *= (1.0 + buffer)
        target_fh *= (1.0 + buffer)

        mult_fa = max(1e-9, self._sector_capacity_multiplier("FA"))
        mult_fh = max(1e-9, self._sector_capacity_multiplier("FH"))
        k_fa = max(0.0, float(self.nodes["FA"].get("K", 0.0)))
        k_fh = max(0.0, float(self.nodes["FH"].get("K", 0.0)))
        base_fa = max(0.0, (target_fa / mult_fa) - (self._sector_capacity_per_k("FA") * k_fa))
        base_fh = max(0.0, (target_fh / mult_fh) - (self._sector_capacity_per_k("FH") * k_fh))
        self.state["sector_base_capacity_info_real"] = float(base_fa)
        self.state["sector_base_capacity_phys_real"] = float(base_fh)

    def _sector_capacity_real(self, firm_id: str) -> float:
        if firm_id == "FA":
            base_key = "sector_base_capacity_info_real"
        else:
            base_key = "sector_base_capacity_phys_real"
        base_capacity = max(0.0, float(self.state.get(base_key, 0.0)))
        capital = max(0.0, float(self.nodes[firm_id].get("K", 0.0)))
        return self._sector_capacity_multiplier(firm_id) * (
            base_capacity + (self._sector_capacity_per_k(firm_id) * capital)
        )

    def _sector_overhead_nom(self, firm_id: str) -> float:
        if firm_id == "FA":
            rate = max(0.0, float(self.params.get("firm_overhead_rate_info", 0.0)))
        else:
            rate = max(0.0, float(self.params.get("firm_overhead_rate_phys", 0.0)))
        revenue_prev = max(0.0, float(self.nodes[firm_id].memo.get("revenue_prev", 0.0)))
        return rate * revenue_prev

    def _sector_input_cost_rate(self, firm_id: str) -> float:
        if firm_id == "FA":
            return max(0.0, float(self.params.get("sector_input_cost_rate_info", 0.0)))
        return max(0.0, float(self.params.get("sector_input_cost_rate_phys", 0.0)))

    def _lagged_dividend_commit_nom(self, issuer: str) -> float:
        commit = max(0.0, float(self.nodes[issuer].memo.get("dividend_commit_prev", 0.0)))
        if commit > 0.0:
            return commit
        if issuer == "BANK":
            payout_rate = max(0.0, min(1.0, float(self.params.get("dividend_payout_rate_bank", 1.0))))
        elif issuer == "FA":
            payout_rate = max(0.0, min(1.0, float(self.state.get("sector_payout_rate_info_prev", self.params.get("dividend_payout_rate_firms", 1.0)))))
        elif issuer == "FH":
            payout_rate = max(0.0, min(1.0, float(self.state.get("sector_payout_rate_phys_prev", self.params.get("dividend_payout_rate_firms", 1.0)))))
        else:
            payout_rate = max(0.0, min(1.0, float(self.params.get("dividend_payout_rate_firms", 1.0))))
        retained_prev = max(0.0, float(self.nodes[issuer].memo.get("retained_prev", 0.0)))
        return payout_rate * retained_prev

    def _sector_maintenance_capex_nom(self, firm_id: str, price_level: float) -> float:
        p_now = max(1e-9, float(price_level))
        depr_q = max(0.0, min(1.0, float(self.params.get("capital_depr_rate_per_quarter", 0.0))))
        capacity_per_k = self._sector_capacity_per_k(firm_id)
        capital = max(0.0, float(self.nodes[firm_id].get("K", 0.0)))
        if firm_id == "FA":
            base_capacity = max(0.0, float(self.state.get("sector_base_capacity_info_real", 0.0)))
        else:
            base_capacity = max(0.0, float(self.state.get("sector_base_capacity_phys_real", 0.0)))
        if capacity_per_k > 1e-12:
            maintenance_stock = capital + (base_capacity / capacity_per_k)
        else:
            maintenance_stock = capital
        return float(depr_q * maintenance_stock * p_now)

    def _sector_maturity_signal(self, firm_id: str) -> float:
        half_sat = max(1e-9, float(self.params.get("sector_dividend_maturity_gap_half_sat", 0.02)))
        if firm_id == "FA":
            prev_capacity = max(0.0, float(self.state.get("sector_capacity_info_real_prev", 0.0)))
            prev_unmet = max(0.0, float(self.state.get("sector_unmet_info_real_prev", 0.0)))
        else:
            prev_capacity = max(0.0, float(self.state.get("sector_capacity_phys_real_prev", 0.0)))
            prev_unmet = max(0.0, float(self.state.get("sector_unmet_phys_real_prev", 0.0)))
        gap_ratio = (prev_unmet / max(prev_capacity, 1e-9)) if prev_capacity > 1e-12 else 0.0
        return float(half_sat / (gap_ratio + half_sat))

    def _sector_target_payout_rate(self, firm_id: str) -> float:
        payout_base = max(0.0, min(1.0, float(self.params.get("dividend_payout_rate_firms", 1.0))))
        payout_max = max(payout_base, min(1.0, float(self.params.get("dividend_payout_rate_firms_mature_max", payout_base))))
        maturity_signal = self._sector_maturity_signal(firm_id)
        return float(payout_base + ((payout_max - payout_base) * maturity_signal))

    def _update_sector_payout_rate(self, firm_id: str) -> float:
        adjust_speed = max(0.0, min(1.0, float(self.params.get("sector_dividend_adjust_speed", 0.50))))
        target_rate = self._sector_target_payout_rate(firm_id)
        if firm_id == "FA":
            key = "sector_payout_rate_info_prev"
        else:
            key = "sector_payout_rate_phys_prev"
        prev_rate = max(0.0, min(1.0, float(self.state.get(key, self.params.get("dividend_payout_rate_firms", 1.0)))))
        next_rate = prev_rate + (adjust_speed * (target_rate - prev_rate))
        next_rate = max(0.0, min(1.0, next_rate))
        self.state[key] = float(next_rate)
        return float(next_rate)

    def _sector_surplus_distribution_nom(self, firm_id: str, price_level: float) -> float:
        sweep_share = max(0.0, min(1.0, float(self.params.get("sector_surplus_distribution_share", 0.0))))
        if sweep_share <= 0.0:
            return 0.0
        revenue_buffer_share = max(0.0, float(self.params.get("sector_surplus_cash_buffer_revenue_share", 0.0)))
        maturity_signal = self._sector_maturity_signal(firm_id)
        deposits = max(0.0, float(self.nodes[firm_id].get("deposits", 0.0)))
        revenue_prev = max(0.0, float(self.nodes[firm_id].memo.get("revenue_prev", 0.0)))
        reserve_nom = self._sector_maintenance_capex_nom(firm_id, price_level) + (revenue_buffer_share * revenue_prev)
        surplus_cash = max(0.0, deposits - reserve_nom)
        return float(sweep_share * maturity_signal * surplus_cash)

    def _sector_maintenance_reserve_nom(self, firm_id: str, price_level: float) -> float:
        reserve_share = max(0.0, min(1.0, float(self.params.get("sector_maintenance_reserve_share", 1.0))))
        return float(reserve_share * self._sector_maintenance_capex_nom(firm_id, price_level))

    def _sector_profit_distributable_nom(self, firm_id: str, after_tax_profit_nom: float, price_level: float) -> float:
        maintenance_nom = self._sector_maintenance_reserve_nom(firm_id, price_level)
        return float(max(0.0, float(after_tax_profit_nom) - maintenance_nom))

    def _sector_capex_plan_nom(self, firm_id: str, price_level: float) -> float:
        p_now = max(1e-9, float(price_level))
        half_sat = max(1e-9, float(self.params.get("sector_capex_gap_half_sat", 0.15)))
        share_min = max(0.0, min(1.0, float(self.params.get("sector_capex_share_min", 0.0))))
        share_max = max(share_min, min(1.0, float(self.params.get("sector_capex_share_max", share_min))))
        gap_close = max(0.0, min(1.0, float(self.params.get("sector_capex_gap_close_rate", 0.25))))
        growth_cap_rate = max(0.0, float(self.params.get("sector_capex_growth_cap_rate_q", 0.08)))
        capacity_per_k = self._sector_capacity_per_k(firm_id)

        if firm_id == "FA":
            prev_capacity = max(0.0, float(self.state.get("sector_capacity_info_real_prev", 0.0)))
            prev_unmet = max(
                0.0,
                float(
                    self.state.get(
                        "sector_unmet_info_real_sm_prev",
                        self.state.get("sector_unmet_info_real_prev", 0.0),
                    )
                ),
            )
            prev_free_cash = max(0.0, float(self.state.get("sector_free_cash_info_prev", 0.0)))
        else:
            prev_capacity = max(0.0, float(self.state.get("sector_capacity_phys_real_prev", 0.0)))
            prev_unmet = max(
                0.0,
                float(
                    self.state.get(
                        "sector_unmet_phys_real_sm_prev",
                        self.state.get("sector_unmet_phys_real_prev", 0.0),
                    )
                ),
            )
            prev_free_cash = max(0.0, float(self.state.get("sector_free_cash_phys_prev", 0.0)))

        maintenance_nom = self._sector_maintenance_capex_nom(firm_id, p_now)
        maintenance_reserve_nom = self._sector_maintenance_reserve_nom(firm_id, p_now)
        gap_ratio = prev_unmet / max(prev_capacity, 1e-9) if prev_capacity > 1e-12 else 0.0
        gap_signal = gap_ratio / (gap_ratio + half_sat) if gap_ratio > 0.0 else 0.0
        capex_share = share_min + ((share_max - share_min) * gap_signal)
        maintenance_budget_nom = min(prev_free_cash, maintenance_reserve_nom)
        expansion_cash_nom = max(0.0, prev_free_cash - maintenance_budget_nom)
        capex_budget_nom = maintenance_budget_nom + (expansion_cash_nom * capex_share)

        if capacity_per_k <= 1e-12:
            expand_need_nom = 0.0
            growth_cap_nom = maintenance_nom
        else:
            expand_need_nom = gap_close * prev_unmet * (p_now / capacity_per_k)
            growth_cap_nom = maintenance_nom + (growth_cap_rate * max(0.0, prev_capacity) * (p_now / capacity_per_k))

        capex_need_nom = maintenance_nom + expand_need_nom
        return float(max(0.0, min(capex_budget_nom, capex_need_nom, growth_cap_nom)))

    def _sector_installation_limit_nom(self, firm_id: str, price_level: float, capacity_real: float) -> float:
        install_rate = max(0.0, float(self.params.get("sector_install_rate_q", 0.05)))
        capacity_per_k = self._sector_capacity_per_k(firm_id)
        if capacity_per_k <= 1e-12:
            return 0.0
        p_now = max(1e-9, float(price_level))
        installable_capacity_real = install_rate * max(0.0, float(capacity_real))
        return float(max(0.0, installable_capacity_real * (p_now / capacity_per_k)))

    def _sector_fulfillment_step(self, hh_demand_total_real: float, price_level: float) -> Dict[str, float]:
        p_now = max(1e-9, float(price_level))
        hh_share_fa = self._sector_hh_demand_share_fa()
        hh_demand_fa_real = hh_share_fa * hh_demand_total_real
        hh_demand_fh_real = (1.0 - hh_share_fa) * hh_demand_total_real
        ums_recycle_rate = max(0.0, min(1.0, float(self.params.get("ums_recycle_rate_q", 0.0))))
        ums_recycle_total_nom = max(0.0, float(self.nodes["UMS"].get("deposits", 0.0))) * ums_recycle_rate
        rev_prev_fa = max(0.0, float(self.nodes["FA"].memo.get("revenue_prev", 0.0)))
        rev_prev_fh = max(0.0, float(self.nodes["FH"].memo.get("revenue_prev", 0.0)))
        rev_prev_total = rev_prev_fa + rev_prev_fh
        if rev_prev_total > 1e-12:
            ums_share_fa = rev_prev_fa / rev_prev_total
        else:
            ums_share_fa = hh_share_fa
        ums_share_fa = max(0.0, min(1.0, float(ums_share_fa)))
        ums_recycle_fa_nom = ums_recycle_total_nom * ums_share_fa
        ums_recycle_fh_nom = ums_recycle_total_nom - ums_recycle_fa_nom
        ums_recycle_fa_real = ums_recycle_fa_nom / p_now
        ums_recycle_fh_real = ums_recycle_fh_nom / p_now

        capex_fa_request_nom = self._sector_capex_plan_nom("FA", p_now)
        capex_fh_request_nom = self._sector_capex_plan_nom("FH", p_now)
        supplier_share_info_for_info = self._sector_supplier_share_info("FA")
        supplier_share_info_for_phys = self._sector_supplier_share_info("FH")
        supplier_fa_seed_nom = (
            (supplier_share_info_for_info * capex_fa_request_nom)
            + (supplier_share_info_for_phys * capex_fh_request_nom)
        )
        supplier_fh_seed_nom = (
            ((1.0 - supplier_share_info_for_info) * capex_fa_request_nom)
            + ((1.0 - supplier_share_info_for_phys) * capex_fh_request_nom)
        )

        self._ensure_sector_capacity_anchors(
            hh_demand_fa_real,
            hh_demand_fh_real,
            supplier_fa_real=(supplier_fa_seed_nom / p_now),
            supplier_fh_real=(supplier_fh_seed_nom / p_now),
            ums_fa_real=ums_recycle_fa_real,
            ums_fh_real=ums_recycle_fh_real,
        )
        capacity_fa_real = self._sector_capacity_real("FA")
        capacity_fh_real = self._sector_capacity_real("FH")

        capex_queue_info_prev = max(0.0, float(self.state.get("sector_capex_queue_info_nom", 0.0)))
        capex_queue_phys_prev = max(0.0, float(self.state.get("sector_capex_queue_phys_nom", 0.0)))

        install_limit_fa_nom = self._sector_installation_limit_nom("FA", p_now, capacity_fa_real)
        install_limit_fh_nom = self._sector_installation_limit_nom("FH", p_now, capacity_fh_real)
        capex_fa_nom = min(capex_queue_info_prev + capex_fa_request_nom, install_limit_fa_nom)
        capex_fh_nom = min(capex_queue_phys_prev + capex_fh_request_nom, install_limit_fh_nom)

        supplier_sales_fa_nom = (
            (supplier_share_info_for_info * capex_fa_nom)
            + (supplier_share_info_for_phys * capex_fh_nom)
        )
        supplier_sales_fh_nom = (
            ((1.0 - supplier_share_info_for_info) * capex_fa_nom)
            + ((1.0 - supplier_share_info_for_phys) * capex_fh_nom)
        )
        supplier_sales_fa_real = supplier_sales_fa_nom / p_now
        supplier_sales_fh_real = supplier_sales_fh_nom / p_now

        hh_sales_fa_real = min(hh_demand_fa_real, capacity_fa_real)
        hh_sales_fh_real = min(hh_demand_fh_real, capacity_fh_real)
        hh_fulfilled_total_real = hh_sales_fa_real + hh_sales_fh_real
        hh_fulfillment_ratio = (
            min(1.0, hh_fulfilled_total_real / hh_demand_total_real)
            if hh_demand_total_real > 1e-12 else 1.0
        )

        return {
            "hh_demand_fa_real": float(hh_demand_fa_real),
            "hh_demand_fh_real": float(hh_demand_fh_real),
            "capacity_fa_real": float(capacity_fa_real),
            "capacity_fh_real": float(capacity_fh_real),
            "install_limit_fa_nom": float(install_limit_fa_nom),
            "install_limit_fh_nom": float(install_limit_fh_nom),
            "capex_fa_nom": float(capex_fa_nom),
            "capex_fh_nom": float(capex_fh_nom),
            "capex_total_nom": float(capex_fa_nom + capex_fh_nom),
            "capex_fa_request_nom": float(capex_fa_request_nom),
            "capex_fh_request_nom": float(capex_fh_request_nom),
            "capex_queue_info_next": float(max(0.0, (capex_queue_info_prev + capex_fa_request_nom) - capex_fa_nom)),
            "capex_queue_phys_next": float(max(0.0, (capex_queue_phys_prev + capex_fh_request_nom) - capex_fh_nom)),
            "supplier_share_info_for_info_capex": float(supplier_share_info_for_info),
            "supplier_share_info_for_phys_capex": float(supplier_share_info_for_phys),
            "supplier_sales_fa_nom": float(supplier_sales_fa_nom),
            "supplier_sales_fh_nom": float(supplier_sales_fh_nom),
            "supplier_sales_fa_real": float(supplier_sales_fa_real),
            "supplier_sales_fh_real": float(supplier_sales_fh_real),
            "ums_recycle_fa_nom": float(ums_recycle_fa_nom),
            "ums_recycle_fh_nom": float(ums_recycle_fh_nom),
            "ums_recycle_total_nom": float(ums_recycle_total_nom),
            "hh_sales_fa_real": float(hh_sales_fa_real),
            "hh_sales_fh_real": float(hh_sales_fh_real),
            "hh_fulfillment_ratio": float(hh_fulfillment_ratio),
            "rev_fa": float((p_now * hh_sales_fa_real) + supplier_sales_fa_nom + ums_recycle_fa_nom),
            "rev_fh": float((p_now * hh_sales_fh_real) + supplier_sales_fh_nom + ums_recycle_fh_nom),
        }

    def _neutralize_stress_active(self) -> bool:
        mode = str(self.params.get("mort_neutralize_trigger_mode", "StressOnly")).strip()
        if mode == "Always":
            return True
        threshold = float(self.params.get("mort_neutralize_trigger_threshold", self.params.get("trust_trigger_dti", 0.10)))
        dti_ratio = 0.0
        if self.history:
            try:
                dti_ratio = max(float(self.history[-1].pop_dti_w_p90), float(self.history[-1].pop_dti_p90))
            except Exception:
                dti_ratio = 0.0
        return dti_ratio >= threshold

    def _pay_bank_income_from_payer(self, payer: str, amount: float) -> float:
        amt = max(0.0, float(amount))
        if amt <= 0.0:
            return 0.0
        avail = max(0.0, float(self.nodes[payer].get("deposits", 0.0)))
        pay = min(amt, avail)
        if pay <= 0.0:
            return 0.0
        self.nodes[payer].add("deposits", -pay)
        bank = self.nodes["BANK"]
        bank.add("deposit_liab", -pay)
        bank.add("equity", +pay)
        return float(pay)

    def _pay_bank_principal_from_payer(self, payer: str, amount: float) -> float:
        amt = max(0.0, float(amount))
        if amt <= 0.0:
            return 0.0
        avail = max(0.0, float(self.nodes[payer].get("deposits", 0.0)))
        pay = min(amt, avail)
        if pay <= 0.0:
            return 0.0
        self.nodes[payer].add("deposits", -pay)
        bank = self.nodes["BANK"]
        bank.add("deposit_liab", -pay)
        bank.add("loan_assets", -pay)
        return float(pay)

    def _apply_mortgage_gap_neutralization(
        self,
        *,
        interest_gap_i: np.ndarray,
        principal_gap_i: np.ndarray,
        mort_interest_due_total: float,
        mort_pay_ctr_total: float,
    ) -> Dict[str, float]:
        interest_gap_vec = np.maximum(0.0, _as_np(interest_gap_i, dtype=float))
        principal_gap_vec = np.maximum(0.0, _as_np(principal_gap_i, dtype=float))
        gap_total_raw = float(np.sum(interest_gap_vec) + np.sum(principal_gap_vec))
        if gap_total_raw <= 0.0:
            zeros = np.zeros_like(principal_gap_vec, dtype=float)
            return {
                "gap_total": 0.0,
                "paid_gov": 0.0,
                "paid_fund": 0.0,
                "paid_issuance": 0.0,
                "paid_total": 0.0,
                "paid_interest_total": 0.0,
                "paid_principal_total": 0.0,
                "paid_principal_i": zeros,
            }

        if self._mortgage_policy_disabled():
            zeros = np.zeros_like(principal_gap_vec, dtype=float)
            return {
                "gap_total": gap_total_raw,
                "paid_gov": 0.0,
                "paid_fund": 0.0,
                "paid_issuance": 0.0,
                "paid_total": 0.0,
                "paid_interest_total": 0.0,
                "paid_principal_total": 0.0,
                "paid_principal_i": zeros,
            }
        if not bool(self.params.get("mort_bank_neutralize_enable", True)):
            zeros = np.zeros_like(principal_gap_vec, dtype=float)
            return {
                "gap_total": gap_total_raw,
                "paid_gov": 0.0,
                "paid_fund": 0.0,
                "paid_issuance": 0.0,
                "paid_total": 0.0,
                "paid_interest_total": 0.0,
                "paid_principal_total": 0.0,
                "paid_principal_i": zeros,
            }
        if not self._neutralize_stress_active():
            zeros = np.zeros_like(principal_gap_vec, dtype=float)
            return {
                "gap_total": gap_total_raw,
                "paid_gov": 0.0,
                "paid_fund": 0.0,
                "paid_issuance": 0.0,
                "paid_total": 0.0,
                "paid_interest_total": 0.0,
                "paid_principal_total": 0.0,
                "paid_principal_i": zeros,
            }

        cap_mode = str(self.params.get("mort_neutralize_cap_mode", "None")).strip()
        cap_val = float(self.params.get("mort_neutralize_cap_value", 0.0))
        cap_total = gap_total_raw
        if cap_mode == "BankEquityFloor":
            bank_eq = float(self.nodes["BANK"].get("equity", 0.0))
            cap_total = max(0.0, float(cap_val) - bank_eq)
        elif cap_mode == "PctOfMortgageInterest":
            cap_total = max(0.0, float(cap_val)) * max(0.0, float(mort_interest_due_total))
        elif cap_mode == "PctOfMortgagePayment":
            cap_total = max(0.0, float(cap_val)) * max(0.0, float(mort_pay_ctr_total))
        gap_total = min(gap_total_raw, cap_total)
        if gap_total <= 0.0:
            zeros = np.zeros_like(principal_gap_vec, dtype=float)
            return {
                "gap_total": gap_total,
                "paid_gov": 0.0,
                "paid_fund": 0.0,
                "paid_issuance": 0.0,
                "paid_total": 0.0,
                "paid_interest_total": 0.0,
                "paid_principal_total": 0.0,
                "paid_principal_i": zeros,
            }

        stack_raw = self.params.get("mort_neutralize_funding_stack", ["GOV", "FUND", "ISSUANCE"])
        if isinstance(stack_raw, (list, tuple)):
            stack = [str(x).strip().upper() for x in stack_raw]
        else:
            stack = ["GOV", "FUND", "ISSUANCE"]

        remaining = float(gap_total)
        paid_gov = 0.0
        paid_fund = 0.0
        paid_iss = 0.0
        paid_interest_total = 0.0
        paid_principal_total = 0.0

        def _pay_from_source(src: str, amount: float) -> float:
            amt = max(0.0, float(amount))
            if amt <= 0.0:
                return 0.0
            if src == "GOV":
                return self._pay_bank_income_from_payer("GOV", amt)
            if src == "FUND":
                return self._pay_bank_income_from_payer("FUND", amt)
            if src == "ISSUANCE":
                self.nodes["BANK"].add("deposit_liab", +amt)
                self.nodes["BANK"].add("reserves", +amt)
                self.nodes["GOV"].add("deposits", +amt)
                self.nodes["GOV"].add("money_issued", +amt)
                return self._pay_bank_income_from_payer("GOV", amt)
            return 0.0

        def _pay_principal_from_source(src: str, amount: float) -> float:
            amt = max(0.0, float(amount))
            if amt <= 0.0:
                return 0.0
            if src == "GOV":
                return self._pay_bank_principal_from_payer("GOV", amt)
            if src == "FUND":
                return self._pay_bank_principal_from_payer("FUND", amt)
            if src == "ISSUANCE":
                self.nodes["BANK"].add("deposit_liab", +amt)
                self.nodes["BANK"].add("reserves", +amt)
                self.nodes["GOV"].add("deposits", +amt)
                self.nodes["GOV"].add("money_issued", +amt)
                return self._pay_bank_principal_from_payer("GOV", amt)
            return 0.0

        for src in stack:
            if remaining <= 0.0:
                break
            if src == "GOV":
                source_paid = 0.0
                interest_need = max(0.0, float(np.sum(interest_gap_vec)) - paid_interest_total)
                paid = _pay_from_source("GOV", min(remaining, interest_need))
                paid_interest_total += paid
                source_paid += paid
                remaining -= paid
                principal_need = max(0.0, float(np.sum(principal_gap_vec)) - paid_principal_total)
                paid = _pay_principal_from_source("GOV", min(remaining, principal_need))
                paid_principal_total += paid
                source_paid += paid
                remaining -= paid
                paid_gov += source_paid
            elif src == "FUND":
                allow_if_debt = bool(self.params.get("mort_neutralize_fund_allowed_if_debt_outstanding", False))
                fund_debt = float(self.nodes["FUND"].get("loans", 0.0))
                if (fund_debt <= 1e-12) or allow_if_debt:
                    source_paid = 0.0
                    interest_need = max(0.0, float(np.sum(interest_gap_vec)) - paid_interest_total)
                    paid = _pay_from_source("FUND", min(remaining, interest_need))
                    paid_interest_total += paid
                    source_paid += paid
                    remaining -= paid
                    principal_need = max(0.0, float(np.sum(principal_gap_vec)) - paid_principal_total)
                    paid = _pay_principal_from_source("FUND", min(remaining, principal_need))
                    paid_principal_total += paid
                    source_paid += paid
                    remaining -= paid
                    paid_fund += source_paid
            elif src == "ISSUANCE":
                source_paid = 0.0
                interest_need = max(0.0, float(np.sum(interest_gap_vec)) - paid_interest_total)
                paid = _pay_from_source("ISSUANCE", min(remaining, interest_need))
                paid_interest_total += paid
                source_paid += paid
                remaining -= paid
                principal_need = max(0.0, float(np.sum(principal_gap_vec)) - paid_principal_total)
                paid = _pay_principal_from_source("ISSUANCE", min(remaining, principal_need))
                paid_principal_total += paid
                source_paid += paid
                remaining -= paid
                paid_iss += source_paid

        paid_total = float(paid_gov + paid_fund + paid_iss)
        principal_gap_total = float(np.sum(principal_gap_vec))
        if paid_principal_total > 0.0 and principal_gap_total > 1e-12:
            paid_principal_i = principal_gap_vec * (paid_principal_total / principal_gap_total)
        else:
            paid_principal_i = np.zeros_like(principal_gap_vec, dtype=float)
        return {
            "gap_total": float(gap_total),
            "paid_gov": float(paid_gov),
            "paid_fund": float(paid_fund),
            "paid_issuance": float(paid_iss),
            "paid_total": float(paid_total),
            "paid_interest_total": float(paid_interest_total),
            "paid_principal_total": float(paid_principal_total),
            "paid_principal_i": paid_principal_i.astype(float, copy=True),
        }

    # ------------------------
    # Core double-entry primitives
    # ------------------------

    def _xfer_deposits(self, payer: str, receiver: str, amount: float) -> None:
        """Transfer deposits between two holders (does not change BANK.deposit_liab)."""
        if amount <= 0:
            return
        self.nodes[payer].add("deposits", -amount)
        self.nodes[receiver].add("deposits", +amount)

    def _xfer_deposits_to_households(self, payer: str, amount: float) -> None:
        """Transfer deposits from a payer to households (population-aware).

        In population mode we must credit the household vector, not only the aggregate HH node,
        otherwise HH aggregate deposits will be overwritten at sync and break SFC identities.
        """
        if amount <= 0:
            return
        self.nodes[payer].add("deposits", -amount)

        if bool(self.params.get("use_population", False)) and (self.hh is not None) and (self.hh.n > 0):
            w0 = _as_np(self.hh.wages0_q, dtype=float)
            w0_sum = float(w0.sum()) if w0.size == self.hh.n else 0.0
            if w0_sum > 0.0:
                weights = w0 / w0_sum
            else:
                weights = np.full(self.hh.n, 1.0 / float(self.hh.n), dtype=float)
            self.hh.deposits[:] = self.hh.deposits + (weights * amount)
            self.nodes["HH"].set("deposits", self.hh.sum_deposits())
        else:
            self.nodes["HH"].add("deposits", +amount)

    def _create_loan(self, borrower: str, amount: float, memo_tag: Optional[str] = None) -> None:
        """
        Bank creates a loan:
          borrower.loans    += amount
          borrower.deposits += amount
          BANK.loan_assets  += amount
          BANK.deposit_liab += amount
        """
        if amount <= 0:
            return

        b = self.nodes[borrower]
        bank = self.nodes["BANK"]

        b.add("loans", +amount)
        b.add("deposits", +amount)

        bank.add("loan_assets", +amount)
        bank.add("deposit_liab", +amount)

        if memo_tag:
            b.memo[memo_tag] = b.memo.get(memo_tag, 0.0) + amount

    def _repay_loan(self, borrower: str, amount: float) -> None:
        """
        Principal repayment destroys deposits and reduces loan principal:
          borrower.deposits -= pay
          borrower.loans    -= pay
          BANK.loan_assets  -= pay
          BANK.deposit_liab -= pay
        """
        if amount <= 0:
            return

        b = self.nodes[borrower]
        bank = self.nodes["BANK"]

        principal = b.get("loans", 0.0)
        pay = min(amount, principal)
        if pay <= 0:
            return

        b.add("deposits", -pay)
        b.add("loans", -pay)

        bank.add("loan_assets", -pay)
        bank.add("deposit_liab", -pay)


    def _gov_credit_deposits(self, receiver: str, amount: float) -> None:
        """
        Government/CB credits deposits to a receiver (money issuance).
          receiver.deposits += amount
          BANK.deposit_liab += amount
        Optional tracking: GOV.money_issued += amount
        """
        if amount <= 0:
            return
        self.nodes[receiver].add("deposits", +amount)
        bank = self.nodes["BANK"]
        bank.add("deposit_liab", +amount)
        bank.add("reserves", +amount)

        gov = self.nodes["GOV"]
        gov.add("money_issued", +amount)

    # ------------------------
    # Sums & SFC checks
    # ------------------------

    def _sum_deposits_all(self) -> float:
        return sum(node.get("deposits", 0.0) for node in self.nodes.values())

    def _sum_loans_borrowers(self) -> float:
        total = 0.0
        for nid, node in self.nodes.items():
            if nid == "BANK":
                continue
            total += node.get("loans", 0.0)
        return total

    def _assert_sfc_ok(self, context: str = "") -> None:
        """
        Optional hard assertions of SFC invariants.
        """
        if not self.params.get("hard_assert_sfc", False):
            return

        bank = self.nodes["BANK"]
        dep_liab = bank.get("deposit_liab", 0.0)
        dep_sum = self._sum_deposits_all()

        loan_assets = bank.get("loan_assets", 0.0)
        loan_sum = self._sum_loans_borrowers()

        eps = 1e-6
        if abs(dep_liab - dep_sum) > eps:
            raise AssertionError(f"SFC FAIL ({context}): deposit_liab={dep_liab} vs sum_deposits={dep_sum}")
        if abs(loan_assets - loan_sum) > eps:
            raise AssertionError(f"SFC FAIL ({context}): loan_assets={loan_assets} vs sum_loans={loan_sum}")
        if bool(self.params.get("use_population", False)) and (self.hh is not None):
            hh_dep_node = self.nodes["HH"].get("deposits", 0.0)
            hh_dep_vec = self.hh.sum_deposits()
            hh_loan_node = self.nodes["HH"].get("loans", 0.0)
            hh_loan_vec = self.hh.sum_loans()
            if abs(hh_dep_node - hh_dep_vec) > eps:
                raise AssertionError(
                    f"SFC FAIL ({context}): HH.deposits node={hh_dep_node} vs vector_sum={hh_dep_vec}"
                )
            if abs(hh_loan_node - hh_loan_vec) > eps:
                raise AssertionError(
                    f"SFC FAIL ({context}): HH.loans node={hh_loan_node} vs vector_sum={hh_loan_vec}"
                )

    # ---------------------------------------------------------
    # Balance sheet / diagnostics
    # ---------------------------------------------------------

    def _record_balance_sheets(self, t: int) -> None:
        """Record per-node stocks and SFC invariant diagnostics for this tick."""
        bank = self.nodes["BANK"]

        dep_liab = bank.get("deposit_liab", 0.0)
        dep_sum = self._sum_deposits_all()
        loan_assets = bank.get("loan_assets", 0.0)
        loan_sum = self._sum_loans_borrowers()

        self.inv_history.append({
            "t": t,
            "bank_deposit_liab": dep_liab,
            "sum_deposits": dep_sum,
            "deposit_gap": dep_liab - dep_sum,
            "bank_loan_assets": loan_assets,
            "sum_loans": loan_sum,
            "loan_gap": loan_assets - loan_sum,
        })

        for nid, node in self.nodes.items():
            row: Dict[str, Any] = {"t": t, "node": nid}

            # Export *all* stock keys so future state variables automatically appear in CSV
            for k, v in node.stocks.items():
                if k in ("t", "node"):
                    continue
                row[k] = float(v)

            self.bs_history.append(row)

    def write_balance_sheets_csv(
        self,
        filename_nodes: str = "balance_sheets.csv",
        filename_invariants: str = "sfc_invariants.csv",
    ) -> None:
        """Write per-node balance sheet snapshots and SFC invariant diagnostics."""
        if self.bs_history:
            keys = set()
            for r in self.bs_history:
                keys.update(r.keys())
            fieldnames = ["t", "node"] + sorted(k for k in keys if k not in ("t", "node"))

            with open(filename_nodes, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.bs_history:
                    writer.writerow(r)

        if self.inv_history:
            fieldnames = [
                "t",
                "bank_deposit_liab",
                "sum_deposits",
                "deposit_gap",
                "bank_loan_assets",
                "sum_loans",
                "loan_gap",
            ]
            with open(filename_invariants, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.inv_history:
                    writer.writerow(r)

    def _issuer_equity_proxy_value(self, issuer: str) -> float:
        """Nominal equity proxy used for share-transfer sizing and diagnostics."""
        if issuer == "BANK":
            return max(0.0, float(self.nodes["BANK"].get("equity", 0.0)))

        p_now = float(self.state.get("price_level", self.params.get("price_level_initial", 1.0)))
        if p_now <= 0.0:
            p_now = 1e-9

        node = self.nodes[issuer]
        return max(
            0.0,
            float(node.get("deposits", 0.0))
            + float(node.get("K", 0.0)) * p_now
            - float(node.get("loans", 0.0)),
        )

    def _gov_rebate_buffer_amount(self) -> float | None:
        """Return the GOV deposit buffer implied by trailing obligations, or None if not yet available."""
        buffer_quarters = int(self.params.get("gov_rebate_buffer_quarters", 4))
        buffer_quarters = max(0, buffer_quarters)
        if buffer_quarters == 0:
            return 0.0
        if len(self.gov_obligation_history) < buffer_quarters:
            return None

        trailing = self.gov_obligation_history[-buffer_quarters:]
        avg_obligation = sum(float(v) for v in trailing) / float(len(trailing))
        return max(0.0, float(buffer_quarters) * avg_obligation)

    def _gov_rebate_ramp_multiplier(self) -> float:
        """Return a 0-1 multiplier for phased GOV surplus recycling."""
        start_delay = int(self.params.get("gov_rebate_start_delay_quarters", 4))
        ramp_quarters = int(self.params.get("gov_rebate_ramp_quarters", 20))
        start_delay = max(0, start_delay)
        ramp_quarters = max(0, ramp_quarters)

        t = int(self.state.get("t", 0))
        if t < start_delay:
            return 0.0
        if ramp_quarters == 0:
            return 1.0

        elapsed = (t - start_delay) + 1
        return max(0.0, min(1.0, float(elapsed) / float(ramp_quarters)))

    # ---------------------------------------------------------
    # Phase A: Trigger logic + leveraged launch
    # ---------------------------------------------------------

    def apply_policy_logic(self) -> None:
        """
        Trigger: if DTI metric exceeds threshold, activate FUND.

        Launch sequence:
        - BANK makes a loan to FUND (creates FUND deposits).
        - FUND uses those deposits to buy 10% of FA/FH/BANK shares from HH.
        """
        if self._trust_disabled():
            self.state["trust_active"] = False
            return
        if self.state["trust_active"]:
            return

        # Allow one baseline quarter before any trigger logic (needed for baseline targeting)
        if int(self.state["t"]) == 0:
            return

        # Trigger metric: use last-quarter mortgage-payment stress.
        # We take the maximum of:
        #   - pop_dti_w_p90: required mortgage payment / wage among mortgagors
        #   - pop_dti_p90:   required mortgage payment / disposable income among mortgagors
        # This prevents “never trigger” behavior if one of the two sub-metrics is empty or near-zero.
        dti_ratio = 0.0
        dti_w_p90 = 0.0
        dti_p90 = 0.0
        if self.history:
            try:
                dti_w_p90 = float(self.history[-1].pop_dti_w_p90)
                dti_p90 = float(self.history[-1].pop_dti_p90)
                dti_ratio = max(dti_w_p90, dti_p90)
            except Exception:
                dti_ratio = 0.0
                dti_w_p90 = 0.0
                dti_p90 = 0.0

        trigger_threshold = float(self.params.get("trust_trigger_dti", 0.10))


        if dti_ratio <= trigger_threshold:
            return

        self.state["trust_active"] = True
        t = self.state["t"]
        launch_loan = float(self.params.get("trust_launch_loan", 15000.0))

        # Optional leveraged launch: BANK lends to FUND and FUND buys an initial equity block from HH.
        # The launch targets a configured ownership share per issuer, defaulting to 10%.
        # If the configured seed loan is insufficient to fund that purchase at current equity proxies,
        # the FUND borrows the additional cash needed so the visible launch block matches the target.
        if launch_loan > 0:
            seller = "HH"
            issuers = [("FA", "shares_FA"), ("FH", "shares_FH"), ("BANK", "shares_BANK")]
            launch_target_pct = float(self.params.get("trust_launch_target_pct", 0.10))
            launch_target_pct = max(0.0, min(1.0, launch_target_pct))
            purchasable: List[tuple[str, str, float, float, float, float]] = []
            total_required_cash = 0.0

            for issuer, key in issuers:
                shares_out = float(self.nodes[issuer].get("shares_outstanding", 0.0))
                hh_shares = max(0.0, float(self.nodes[seller].get(key, 0.0)))
                issuer_equity_value = self._issuer_equity_proxy_value(issuer)
                if shares_out <= 0.0 or hh_shares <= 0.0 or issuer_equity_value <= 0.0:
                    continue

                target_shares = min(hh_shares, launch_target_pct * shares_out)
                if target_shares <= 0.0:
                    continue

                price_per_share = issuer_equity_value / shares_out
                if price_per_share <= 0.0:
                    continue

                cash_required = target_shares * price_per_share
                purchasable.append((issuer, key, shares_out, hh_shares, issuer_equity_value, target_shares))
                total_required_cash += cash_required

            if total_required_cash > 0.0:
                self._create_loan("FUND", launch_loan, memo_tag="launch_loan_created")
                extra_cash_needed = max(0.0, total_required_cash - float(self.nodes["FUND"].get("deposits", 0.0)))
                if extra_cash_needed > 0.0:
                    self._create_loan("FUND", extra_cash_needed, memo_tag="launch_topup_loan_created")

                for issuer, key, shares_out, hh_shares, issuer_equity_value, target_shares in purchasable:
                    price_per_share = issuer_equity_value / shares_out
                    shares_to_buy = min(hh_shares, target_shares)
                    if price_per_share <= 0.0 or shares_to_buy <= 0.0:
                        continue

                    cash_spent = shares_to_buy * price_per_share
                    self.nodes[seller].add(key, -shares_to_buy)
                    self.nodes["FUND"].add(key, +shares_to_buy)
                    self._xfer_deposits_to_households("FUND", cash_spent)

            for _, key in issuers:
                if self.nodes[seller].get(key, 0.0) < -1e-9:
                    raise ValueError(f"{seller} has negative {key} after launch. Check initial ownership.")

        self._assert_sfc_ok(context=f"after_trigger_t{t}")

    # ---------------------------------------------------------
    # Phase B: Dilution (2% annual) to FUND until policy cap
    # ---------------------------------------------------------

    def issue_social_shares(self) -> None:
        if self._trust_disabled():
            return
        if not self.state["trust_active"]:
            return

        phi_qtr = 1.0 - (0.98 ** 0.25)  # exact quarterly issuance for 2% annual
        own_cap = float(self.params.get("trust_equity_cap", 0.25))
        own_cap = max(0.0, min(1.0, own_cap))

        for issuer, key in [("FA", "shares_FA"), ("FH", "shares_FH"), ("BANK", "shares_BANK")]:
            shares_out = self.nodes[issuer].get("shares_outstanding", 0.0)
            if shares_out <= 0:
                continue

            fund_shares = self.nodes["FUND"].get(key, 0.0)
            own = fund_shares / shares_out

            if own >= own_cap:
                continue

            new_shares = phi_qtr * shares_out
            self.nodes[issuer].add("shares_outstanding", +new_shares)
            self.nodes["FUND"].add(key, +new_shares)


    # ---------------------------------------------------------
    # Population-mode solver (Step A): vectorized household sector
    # ---------------------------------------------------------

    def solve_within_tick_population(
        self,
        max_iter: int = 200,
        tol: float = 1e-8,
        *,
        allow_income_support_trigger: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Vectorized within-tick solver for synthetic population households.
        Implements VAT, VAT credit, and marginal income tax (population mode).
        """
        if not bool(self.params.get("use_population", False)):
            return None
        if self.hh is None or self.hh.n <= 0:
            return None

        hh = self.hh
        hh.ensure_memos()

        auto = float(self.state["automation"])
        auto_eff = float(self.state.get("automation_eff", auto))
        auto_info = float(self.state.get("automation_info", auto_eff))
        auto_phys = float(self.state.get("automation_phys", auto_eff))
        rL = float(self.state.get("policy_rate_q", self.params["loan_rate_per_quarter"]))

        ws_fh_base = float(self.params["wage_share_of_revenue"]["FH"])
        ws_fa_base = float(self.params["wage_share_of_revenue"]["FA"])
        ws_fh = ws_fh_base * (1.0 - max(0.0, min(1.0, auto_phys)))
        ws_fa = ws_fa_base * (1.0 - max(0.0, min(1.0, auto_info)))

        # Price level (nominal $ per unit real consumption). Consumption rule is applied in real terms.
        P = float(self.state.get("price_level", 1.0))
        if P <= 0:
            P = 1e-9
        # VAT wedge: consumer price includes VAT as a markup
        vat_rate = self._effective_vat_rate()
        P_cons = P * (1.0 + vat_rate)  # tax-exclusive VAT treated as higher consumer price

        # Baseline wage weights used to distribute wages and (temporarily) dividends.
        w0 = hh.wages0_q
        w0_sum = float(w0.sum())
        if w0_sum <= 0:
            return None
        w_weights = w0 / w0_sum

        # Taxable-income and eligibility-income ranks are invariant within the
        # fixed-point loop because both are affine transforms of baseline wages.
        it_pct = float(self.params.get("income_tax_cutoff_pct", 100.0))
        if it_pct < 0:
            it_pct = 0.0
        if it_pct > 100:
            it_pct = 100.0
        k_it = int(math.ceil((it_pct / 100.0) * w0.size)) - 1 if w0.size > 0 else 0
        k_it = max(0, min(w0.size - 1, k_it)) if w0.size > 0 else 0

        vc_start_pct = float(self.params.get("vat_credit_phaseout_start_pct", 25.0))
        vc_end_pct = float(self.params.get("vat_credit_phaseout_end_pct", 45.0))
        vc_start_pct = max(0.0, min(100.0, vc_start_pct))
        vc_end_pct = max(vc_start_pct, min(100.0, vc_end_pct))
        k_vc_start = int(math.ceil((vc_start_pct / 100.0) * w0.size)) - 1 if w0.size > 0 else 0
        k_vc_start = max(0, min(w0.size - 1, k_vc_start)) if w0.size > 0 else 0
        k_vc_end = int(math.ceil((vc_end_pct / 100.0) * w0.size)) - 1 if w0.size > 0 else 0
        k_vc_end = max(0, min(w0.size - 1, k_vc_end)) if w0.size > 0 else 0

        if w0.size > 0:
            anchor_idx = sorted({k_it, k_vc_start, k_vc_end})
            wage_partition = np.partition(w0.copy(), anchor_idx)
            it_anchor_w0 = float(wage_partition[k_it])
            vc_start_anchor_w0 = float(wage_partition[k_vc_start])
            vc_end_anchor_w0 = float(wage_partition[k_vc_end])
        else:
            it_anchor_w0 = 0.0
            vc_start_anchor_w0 = 0.0
            vc_end_anchor_w0 = 0.0

        # Static household vectors are already float NumPy arrays on HouseholdState.
        base_real = hh.base_real_cons_q
        mpc = hh.mpc_q
        mort = hh.mortgage_loans
        rev = hh.revolving_loans
        renter_rent_q = np.maximum(0.0, _as_np(hh.renter_rent_q, dtype=float))
        if renter_rent_q.shape[0] != hh.n:
            renter_rent_q = np.zeros(hh.n, dtype=float)
        liquid_buffer_months_target = hh.liquid_buffer_months_target
        if liquid_buffer_months_target.shape[0] != hh.n:
            liquid_buffer_months_target = np.zeros(hh.n, dtype=float)

        # Beginning-of-tick deposits (nominal). Use a copy so the solver is not affected by post-tick mutations.
        dep0 = hh.deposits.copy()

        # Initial income guess: previous-quarter income memo.
        y_guess = hh.prev_income
        if y_guess.shape[0] != hh.n:
            y_guess = w0.copy()
        solver_relax = float(self.params.get("solver_relaxation", 1.0))
        solver_relax = max(1e-3, min(1.0, solver_relax))

        # Current trust ownership fractions for balance-sheet and equity diagnostics.
        def own_frac(issuer: str, key: str) -> float:
            so = self.nodes[issuer].get("shares_outstanding", 1.0)
            return self.nodes["FUND"].get(key, 0.0) / so if so > 0 else 0.0

        # Dividends are lagged, so newly acquired shares should not receive the
        # current tick's payout. Use prior-quarter ownership for dividend splits.
        f_fa = max(0.0, min(1.0, float(self.state.get("fund_dividend_ownership_fa_prev", 0.0))))
        f_fh = max(0.0, min(1.0, float(self.state.get("fund_dividend_ownership_fh_prev", 0.0))))
        f_bk = max(0.0, min(1.0, float(self.state.get("fund_dividend_ownership_bk_prev", 0.0))))

        fund_loan = float(self.nodes["FUND"].get("loans", 0.0))
        fa_loan = float(self.nodes["FA"].get("loans", 0.0))
        fh_loan = float(self.nodes["FH"].get("loans", 0.0))

        fa_interest = fa_loan * rL
        fh_interest = fh_loan * rL

        overhead_fa = self._sector_overhead_nom("FA")
        overhead_fh = self._sector_overhead_nom("FH")
        div_commit_fa = self._lagged_dividend_commit_nom("FA")
        div_commit_fh = self._lagged_dividend_commit_nom("FH")
        div_commit_bk = self._lagged_dividend_commit_nom("BANK")
        div_cash_buffer_share = max(0.0, min(1.0, float(self.params.get("sector_dividend_cash_buffer_q", 0.0))))
        div_house_total_est = 0.0
        spend_excess_rate = float(self.params.get("hh_buffer_spend_excess_rate_q", 0.10))
        conserve_shortfall_rate = float(self.params.get("hh_buffer_shortfall_conserve_rate_q", 0.35))
        spend_excess_rate = max(0.0, min(1.0, spend_excess_rate))
        conserve_shortfall_rate = max(0.0, min(1.0, conserve_shortfall_rate))

        max_delta = float("inf")
        for iter_idx in range(1, max_iter + 1):
            # 1) Household consumption (nominal), vectorized
            # Consumption decision uses the consumer price (includes VAT wedge)
            y_real = y_guess / P_cons

            # Desired real consumption: an income-driven core moderated by a precautionary
            # liquid-buffer rule. Households spend only a fraction of buffer excess and
            # begin conserving before the hard cash constraint binds.
            c_real_core = np.maximum(0.0, base_real + mpc * y_real)
            c_hh_nom_core = P_cons * c_real_core
            target_buffer_nom = (liquid_buffer_months_target / 3.0) * c_hh_nom_core
            buffer_gap_nom = dep0 - target_buffer_nom
            c_hh_nom_des = (
                c_hh_nom_core
                + (spend_excess_rate * np.maximum(0.0, buffer_gap_nom))
                - (conserve_shortfall_rate * np.maximum(0.0, -buffer_gap_nom))
            )
            c_hh_nom_des = np.maximum(0.0, c_hh_nom_des)

            # Cash-in-advance constraint (no new borrowing for consumption inside the solver):
            # available = beginning deposits + current-quarter disposable income guess.
            # If disposable income is negative, available is floored at 0.
            avail_nom = np.maximum(0.0, dep0 + y_guess)
            c_hh_nom_budgeted = np.minimum(c_hh_nom_des, avail_nom)

            # Split desired household demand by fixed sector shares, then ration it only by
            # household-serving capacity. Installed CAPEX uses a separate bounded installation lane.
            c_real_budgeted = c_hh_nom_budgeted / P_cons
            hh_demand_total_real = float(np.sum(c_real_budgeted))
            sector_step = self._sector_fulfillment_step(hh_demand_total_real, P)
            hh_demand_fa_real = float(sector_step["hh_demand_fa_real"])
            hh_demand_fh_real = float(sector_step["hh_demand_fh_real"])
            capacity_fa_real = float(sector_step["capacity_fa_real"])
            capacity_fh_real = float(sector_step["capacity_fh_real"])
            install_limit_fa_nom = float(sector_step["install_limit_fa_nom"])
            install_limit_fh_nom = float(sector_step["install_limit_fh_nom"])
            capex_fa_nom = float(sector_step["capex_fa_nom"])
            capex_fh_nom = float(sector_step["capex_fh_nom"])
            capex_total_nom = float(sector_step["capex_total_nom"])
            capex_fa_request_nom = float(sector_step["capex_fa_request_nom"])
            capex_fh_request_nom = float(sector_step["capex_fh_request_nom"])
            capex_queue_info_next = float(sector_step["capex_queue_info_next"])
            capex_queue_phys_next = float(sector_step["capex_queue_phys_next"])
            supplier_share_info_for_info = float(sector_step["supplier_share_info_for_info_capex"])
            supplier_share_info_for_phys = float(sector_step["supplier_share_info_for_phys_capex"])
            supplier_sales_fa_nom = float(sector_step["supplier_sales_fa_nom"])
            supplier_sales_fh_nom = float(sector_step["supplier_sales_fh_nom"])
            supplier_sales_fa_real = float(sector_step["supplier_sales_fa_real"])
            supplier_sales_fh_real = float(sector_step["supplier_sales_fh_real"])
            ums_recycle_fa_nom = float(sector_step["ums_recycle_fa_nom"])
            ums_recycle_fh_nom = float(sector_step["ums_recycle_fh_nom"])
            ums_recycle_total_nom = float(sector_step["ums_recycle_total_nom"])
            hh_sales_fa_real = float(sector_step["hh_sales_fa_real"])
            hh_sales_fh_real = float(sector_step["hh_sales_fh_real"])
            hh_fulfillment_ratio = float(sector_step["hh_fulfillment_ratio"])

            c_hh_nom = c_hh_nom_budgeted * hh_fulfillment_ratio
            c_real = c_hh_nom / P_cons
            c_firm_nom = P * c_real
            c_total = float(c_firm_nom.sum())

            rev_fa = float(sector_step["rev_fa"])
            rev_fh = float(sector_step["rev_fh"])
            input_cost_fa = rev_fa * self._sector_input_cost_rate("FA")
            input_cost_fh = rev_fh * self._sector_input_cost_rate("FH")

            w_fa = rev_fa * ws_fa
            w_fh = rev_fh * ws_fh
            w_total = float(w_fa + w_fh)

            # profits pre-tax (capex is not expensed; it's a cash outflow later)
            p_fa_pre_tax = max(0.0, rev_fa - w_fa - fa_interest - overhead_fa - input_cost_fa)
            p_fh_pre_tax = max(0.0, rev_fh - w_fh - fh_interest - overhead_fh - input_cost_fh)

            mort_interest_due = mort * rL
            rev_interest = rev * rL
            interest_hh = mort_interest_due + rev_interest
            trust_interest = fund_loan * rL
            bank_interest_ex_mort = float(rev_interest.sum() + trust_interest + fa_interest + fh_interest)

            # Corporate income tax policy:
            # - Distributed dividends are NOT taxed at the corporate level.
            # - Retained earnings are taxed after a depreciation allowance on the capital stock.
            # Household recipients are still taxed via income_tax_i; FUND is untaxed.
            # Optional policy: raise tax rate as wages fall relative to baseline.
            corp_tax_rate = float(self.params.get("corporate_tax_rate", 0.0))
            corp_tax_rate = max(0.0, min(1.0, corp_tax_rate))
            corp_tax_depr_rate_q = float(self.params.get("corporate_tax_depr_rate_q", 0.025))
            corp_tax_depr_rate_q = max(0.0, min(1.0, corp_tax_depr_rate_q))

            if bool(self.params.get("corporate_tax_dynamic_with_wages", False)):
                wage_baseline = float(self.state.get("baseline_wages_total_pop", 0.0))
                if wage_baseline > 0.0:
                    wage_index = max(0.0, min(1.0, float(w_total) / wage_baseline))
                else:
                    wage_index = 1.0

                base_rate = float(self.params.get("corporate_tax_rate_base", corp_tax_rate))
                slope = float(self.params.get("corporate_tax_wage_sensitivity", 0.0))
                tax_min = float(self.params.get("corporate_tax_rate_min", 0.0))
                tax_max = float(self.params.get("corporate_tax_rate_max", 1.0))
                tax_min = max(0.0, min(1.0, tax_min))
                tax_max = max(tax_min, min(1.0, tax_max))

                corp_tax_rate = base_rate + slope * (1.0 - wage_index)
                corp_tax_rate = max(tax_min, min(tax_max, corp_tax_rate))

            # 4) Income-support policy (mode selected by parameters)
            if self._income_support_disabled():
                uis = 0.0
            else:
                raw_uis = self.income_support_policy.compute_per_household(
                    wages_total=float(w_total),
                    div_house_total=float(div_house_total_est),
                    price_level=float(P),
                    n_households=int(hh.n),
                    previous_support_per_h=float(hh.prev_uis),
                    state=self.state,
                )
                uis = self._apply_income_support_start_delay(
                    raw_uis,
                    allow_trigger=allow_income_support_trigger,
                )

            # Mortgage index module: compute indexed required payment per household mortgage.
            mort_index_enable = bool(self.params.get("mort_index_enable", False)) and (not self._mortgage_index_disabled())
            p_series_now = self._mort_price_series_value(P)
            y_series_now = self._mort_income_series_value(w_total, float(div_house_total_est), float(uis))
            self._ensure_mortgage_index_anchors(p_series_now, y_series_now, rL)
            mort_terms = self._compute_mortgage_index_terms(
                mort=mort,
                rL=rL,
                wages_total=w_total,
                div_house_total=float(div_house_total_est),
                uis_per_h=float(uis),
                commit_state=False,
            )
            mort_zero_vec = np.zeros(hh.n, dtype=float)
            mort_one_vec = np.ones(hh.n, dtype=float)
            mort_pay_req_i = _as_np(mort_terms.get("mort_pay_req_i", mort_zero_vec), dtype=float)
            mort_pay_ctr_i = _as_np(mort_terms.get("mort_pay_ctr_i", mort_zero_vec), dtype=float)
            mort_interest_due_i = _as_np(mort_terms.get("mort_interest_due_i", mort_zero_vec), dtype=float)
            mort_interest_paid_i = _as_np(mort_terms.get("mort_interest_paid_i", mort_zero_vec), dtype=float)
            mort_principal_paid_i = _as_np(mort_terms.get("mort_principal_paid_i", mort_zero_vec), dtype=float)
            mort_interest_gap_i = _as_np(mort_terms.get("mort_interest_gap_i", mort_zero_vec), dtype=float)
            mort_principal_gap_i = _as_np(mort_terms.get("mort_principal_gap_i", mort_zero_vec), dtype=float)
            mort_gap_i = _as_np(mort_terms.get("mort_gap_i", mort_zero_vec), dtype=float)
            mort_index_i = _as_np(mort_terms.get("mort_index_i", mort_one_vec), dtype=float)
            mort_dln_i = _as_np(mort_terms.get("mort_dln_i", mort_zero_vec), dtype=float)
            mort_dln_sm_i = _as_np(mort_terms.get("mort_dln_sm_i", mort_zero_vec), dtype=float)
            bank_profit_pre_tax = float(bank_interest_ex_mort + np.sum(np.maximum(0.0, mort_interest_paid_i)))

            corp_tax_depr_fa = max(0.0, float(self.nodes["FA"].get("K", 0.0))) * P * corp_tax_depr_rate_q
            corp_tax_depr_fh = max(0.0, float(self.nodes["FH"].get("K", 0.0))) * P * corp_tax_depr_rate_q
            corp_tax_base_fa = max(0.0, p_fa_pre_tax - corp_tax_depr_fa)
            corp_tax_base_fh = max(0.0, p_fh_pre_tax - corp_tax_depr_fh)
            corp_tax_fa = corp_tax_rate * corp_tax_base_fa
            corp_tax_fh = corp_tax_rate * corp_tax_base_fh
            corp_tax_bk = corp_tax_rate * max(0.0, bank_profit_pre_tax)

            after_tax_profit_fa = max(0.0, p_fa_pre_tax - corp_tax_fa)
            after_tax_profit_fh = max(0.0, p_fh_pre_tax - corp_tax_fh)
            after_tax_profit_bk = max(0.0, bank_profit_pre_tax - corp_tax_bk)

            div_cash_buffer_fa = div_cash_buffer_share * rev_fa
            div_cash_buffer_fh = div_cash_buffer_share * rev_fh
            div_fa_total = min(
                div_commit_fa,
                max(
                    0.0,
                    float(self.nodes["FA"].get("deposits", 0.0))
                    + rev_fa
                    - capex_fa_nom
                    - w_fa
                    - fa_interest
                    - overhead_fa
                    - input_cost_fa
                    - corp_tax_fa
                    - div_cash_buffer_fa,
                ),
            )
            div_fh_total = min(
                div_commit_fh,
                max(
                    0.0,
                    float(self.nodes["FH"].get("deposits", 0.0))
                    + rev_fh
                    - capex_fh_nom
                    - w_fh
                    - fh_interest
                    - overhead_fh
                    - input_cost_fh
                    - corp_tax_fh
                    - div_cash_buffer_fh,
                ),
            )
            bank_dividend_capacity = max(
                0.0,
                float(self.nodes["BANK"].get("equity", 0.0)) + after_tax_profit_bk,
            )
            div_bk_total = min(div_commit_bk, bank_dividend_capacity)

            div_house_firms = (div_fa_total * (1.0 - f_fa)) + (div_fh_total * (1.0 - f_fh))
            div_fund_firms = (div_fa_total * f_fa) + (div_fh_total * f_fh)
            div_fund = div_fund_firms + (div_bk_total * f_bk)
            div_house_total = div_house_firms + (div_bk_total * (1.0 - f_bk))
            div_house_total_est = float(div_house_total)

            retained_fa = max(0.0, after_tax_profit_fa - div_fa_total)
            retained_fh = max(0.0, after_tax_profit_fh - div_fh_total)
            retained_bk = max(0.0, after_tax_profit_bk - div_bk_total)

            # Keep after-tax profit diagnostics consistent with accounting identity:
            # after_tax_profit = distributed_dividends + retained_after_tax
            p_fa = div_fa_total + retained_fa
            p_fh = div_fh_total + retained_fh
            bank_profit = div_bk_total + retained_bk

            # 5) Allocate wages and (temporary) household dividends by baseline wage weights
            wage_scale = w_total / w0_sum
            wages_i = w0 * wage_scale
            div_i = w_weights * float(div_house_total)

            # --- Taxes & VAT credit (computed endogenously inside the solver) ---
            taxable_income = wages_i + div_i  # excludes income support by policy

            # Income tax: 15% marginal above a percentile threshold (nearest-rank)
            it_rate = self._effective_income_tax_rate()
            taxable_scale = float((w_total + float(div_house_total)) / w0_sum)
            it_thr = it_anchor_w0 * taxable_scale
            income_tax_i = it_rate * np.maximum(0.0, taxable_income - it_thr)

            # VAT credit ("prebate"): vat_rate * poverty-line consumption, with a linear
            # phaseout over the configured eligibility-income percentile band.
            elig_income = taxable_income + float(uis)  # user policy: eligibility uses taxable income + income support
            vc_thr_start = (vc_start_anchor_w0 * taxable_scale) + float(uis)
            vc_thr_end = (vc_end_anchor_w0 * taxable_scale) + float(uis)

            if vc_thr_end <= (vc_thr_start + 1e-12):
                vat_credit_weight_i = (elig_income <= vc_thr_start).astype(float)
            else:
                vat_credit_weight_i = np.ones_like(elig_income, dtype=float)
                hi_mask = elig_income >= vc_thr_end
                mid_mask = (elig_income > vc_thr_start) & (~hi_mask)
                vat_credit_weight_i[hi_mask] = 0.0
                vat_credit_weight_i[mid_mask] = (
                    (vc_thr_end - elig_income[mid_mask]) / (vc_thr_end - vc_thr_start)
                )

            # Poverty-line consumption in real units is anchored to baseline average real consumption per household.
            # If baseline is not yet stored, initialize it from the current iteration (baseline quarter t==0).
            base_real_avg = self.state.get("baseline_real_cons_per_h", None)
            if base_real_avg is None:
                base_real_avg = float(np.mean(c_real)) if c_real.size else 0.0
                self.state["baseline_real_cons_per_h"] = float(base_real_avg)

            pov_frac = float(self.params.get("vat_poverty_cons_frac", 0.0))
            if pov_frac < 0:
                pov_frac = 0.0
            pov_real = float(pov_frac) * float(base_real_avg)
            pov_nom = P * pov_real
            vat_credit_per_h = vat_rate * pov_nom
            vat_credit_i = vat_credit_per_h * vat_credit_weight_i

            # Disposable income used by the consumption rule.
            # Households service the full required mortgage payment plus revolving interest
            # in both indexed and non-indexed mortgage regimes so solver income matches
            # the actual cash-settlement path.
            y_new = wages_i + float(uis) + div_i + vat_credit_i - rev_interest - mort_pay_req_i - renter_rent_q - income_tax_i
            max_delta = float(np.max(np.abs(y_new - y_guess)))

            if max_delta < tol:
                return {
                    "c_firm_nom": c_firm_nom,
                    "c_hh_nom": c_hh_nom,
                    "c_total": c_total,
                    "rev_fa": rev_fa,
                    "rev_fh": rev_fh,
                    "w_fa": w_fa,
                    "w_fh": w_fh,
                    "w_total": w_total,
                    "p_fa": p_fa,
                    "p_fh": p_fh,
                    "interest_hh": interest_hh,
                    "trust_interest": trust_interest,
                    "fa_interest": float(fa_interest),
                    "fh_interest": float(fh_interest),
                    "bank_profit": bank_profit,
                    "corp_tax_rate": float(corp_tax_rate),
                    "corp_tax_depr_rate_q": float(corp_tax_depr_rate_q),
                    "corp_tax_depr_fa": float(corp_tax_depr_fa),
                    "corp_tax_depr_fh": float(corp_tax_depr_fh),
                    "corp_tax_base_fa": float(corp_tax_base_fa),
                    "corp_tax_base_fh": float(corp_tax_base_fh),
                    "corp_tax_fa": float(corp_tax_fa),
                    "corp_tax_fh": float(corp_tax_fh),
                    "corp_tax_bk": float(corp_tax_bk),
                    "div_fa_total": float(div_fa_total),
                    "div_fh_total": float(div_fh_total),
                    "div_bk_total": float(div_bk_total),
                    "retained_fa": float(retained_fa),
                    "retained_fh": float(retained_fh),
                    "retained_bk": float(retained_bk),
                    "capex_fa_nom": float(capex_fa_nom),
                    "capex_fh_nom": float(capex_fh_nom),
                    "capex_total_nom": float(capex_total_nom),
                    "capex_fa_request_nom": float(capex_fa_request_nom),
                    "capex_fh_request_nom": float(capex_fh_request_nom),
                    "capex_queue_info_next": float(capex_queue_info_next),
                    "capex_queue_phys_next": float(capex_queue_phys_next),
                    "install_limit_fa_nom": float(install_limit_fa_nom),
                    "install_limit_fh_nom": float(install_limit_fh_nom),
                    "supplier_share_info_for_info_capex": float(supplier_share_info_for_info),
                    "supplier_share_info_for_phys_capex": float(supplier_share_info_for_phys),
                    "supplier_sales_fa_nom": float(supplier_sales_fa_nom),
                    "supplier_sales_fh_nom": float(supplier_sales_fh_nom),
                    "supplier_sales_fa_real": float(supplier_sales_fa_real),
                    "supplier_sales_fh_real": float(supplier_sales_fh_real),
                    "ums_recycle_fa_nom": float(ums_recycle_fa_nom),
                    "ums_recycle_fh_nom": float(ums_recycle_fh_nom),
                    "ums_recycle_total_nom": float(ums_recycle_total_nom),
                    "hh_sales_fa_real": float(hh_sales_fa_real),
                    "hh_sales_fh_real": float(hh_sales_fh_real),
                    "hh_demand_fa_real": float(hh_demand_fa_real),
                    "hh_demand_fh_real": float(hh_demand_fh_real),
                    "capacity_fa_real": float(capacity_fa_real),
                    "capacity_fh_real": float(capacity_fh_real),
                    "hh_fulfillment_ratio": float(hh_fulfillment_ratio),
                    "overhead_fa": float(overhead_fa),
                    "overhead_fh": float(overhead_fh),
                    "input_cost_fa": float(input_cost_fa),
                    "input_cost_fh": float(input_cost_fh),
                    "div_commit_fa": float(div_commit_fa),
                    "div_commit_fh": float(div_commit_fh),
                    "div_commit_bk": float(div_commit_bk),
                    "f_fa": f_fa,
                    "f_fh": f_fh,
                    "f_bk": f_bk,
                    "div_fund": div_fund,
                    "div_house_total": float(div_house_total),
                    "uis": float(uis),
                    "wages_i": wages_i,
                    "div_i": div_i,
                    "y": y_new,
                    "buffer_target_total": float(target_buffer_nom.sum()),
                    "buffer_gap_total": float(buffer_gap_nom.sum()),
                    "buffer_gap_positive_total": float(np.maximum(0.0, buffer_gap_nom).sum()),
                    "buffer_gap_shortfall_total": float(np.maximum(0.0, -buffer_gap_nom).sum()),
                    "mort_index_enable": bool(mort_index_enable),
                    "mort_pay_req_i": mort_pay_req_i,
                    "renter_rent_q": renter_rent_q,
                    "mort_pay_ctr_i": mort_pay_ctr_i,
                    "mort_interest_due_i": mort_interest_due_i,
                    "mort_interest_paid_i": mort_interest_paid_i,
                    "mort_principal_paid_i": mort_principal_paid_i,
                    "mort_interest_gap_i": mort_interest_gap_i,
                    "mort_principal_gap_i": mort_principal_gap_i,
                    "mort_gap_i": mort_gap_i,
                    "mort_gap_total": float(mort_terms.get("mort_gap_total", 0.0)),
                    "mort_pay_req_total": float(mort_terms.get("mort_pay_req_total", 0.0)),
                    "mort_pay_ctr_total": float(mort_terms.get("mort_pay_ctr_total", 0.0)),
                    "mort_index_mean": float(mort_terms.get("mort_index_mean", 1.0)),
                    "mort_index_min": float(mort_terms.get("mort_index_min", 1.0)),
                    "mort_index_max": float(mort_terms.get("mort_index_max", 1.0)),
                    "mort_index_i": mort_index_i,
                    "mort_dln_i": mort_dln_i,
                    "mort_dln_sm_i": mort_dln_sm_i,
                    "p_series_now": float(mort_terms.get("p_series_now", P)),
                    "y_series_now": float(mort_terms.get("y_series_now", max(1e-9, w_total + div_house_total + float(uis) * float(hh.n)))),
                    "rev_interest_i": rev_interest,
                    "taxable_income": taxable_income,
                    "income_tax_i": income_tax_i,
                    "vat_credit_i": vat_credit_i,
                    "it_threshold": float(it_thr),
                    "vc_phaseout_start_threshold": float(vc_thr_start),
                    "vc_phaseout_end_threshold": float(vc_thr_end),
                    "solver_iterations": int(iter_idx),
                    "solver_max_delta": float(max_delta),
                }

            y_guess = ((1.0 - solver_relax) * y_guess) + (solver_relax * y_new)

        raise RuntimeError(
            "Population solver failed to converge "
            f"at t={int(self.state.get('t', 0))} after {int(max_iter)} iterations "
            f"(max_delta={float(max_delta):.3e}, tol={float(tol):.3e})."
        )


    def post_tick_population(self, sol: Dict[str, Any]) -> None:
        if self.hh is None or self.hh.n <= 0:
            return

        hh = self.hh
        hh.ensure_memos()

        auto = float(self.state["automation"])
        auto_eff = float(self.state.get("automation_eff", auto))

        # Convenience
        n = int(hh.n)
        deposits = hh.deposits
        mort = hh.mortgage_loans
        rev = hh.revolving_loans

        # Solver vectors (accept list or ndarray)
        c_firm_nom = _as_np(sol.get("c_firm_nom", []), dtype=float)
        c_hh_nom = _as_np(sol.get("c_hh_nom", []), dtype=float)
        wages_i = _as_np(sol.get("wages_i", []), dtype=float)
        div_i = _as_np(sol.get("div_i", []), dtype=float)
        interest_hh = _as_np(sol.get("interest_hh", []), dtype=float)
        rev_interest_i = _as_np(sol.get("rev_interest_i", []), dtype=float)
        mort_pay_req_i = _as_np(sol.get("mort_pay_req_i", []), dtype=float)
        renter_rent_q = _as_np(sol.get("renter_rent_q", []), dtype=float)
        mort_pay_ctr_i = _as_np(sol.get("mort_pay_ctr_i", []), dtype=float)
        mort_interest_due_i = _as_np(sol.get("mort_interest_due_i", []), dtype=float)
        mort_interest_paid_i = _as_np(sol.get("mort_interest_paid_i", []), dtype=float)
        mort_principal_paid_i = _as_np(sol.get("mort_principal_paid_i", []), dtype=float)
        mort_interest_gap_i = _as_np(sol.get("mort_interest_gap_i", []), dtype=float)
        mort_principal_gap_i = _as_np(sol.get("mort_principal_gap_i", []), dtype=float)
        mort_gap_i = _as_np(sol.get("mort_gap_i", []), dtype=float)
        mort_index_i = _as_np(sol.get("mort_index_i", []), dtype=float)
        mort_dln_i = _as_np(sol.get("mort_dln_i", []), dtype=float)
        mort_dln_sm_i = _as_np(sol.get("mort_dln_sm_i", []), dtype=float)
        mort_index_enable = bool(sol.get("mort_index_enable", False))
        y_vec = _as_np(sol.get("y", []), dtype=float)
        self.state["hh_buffer_target_total"] = float(sol.get("buffer_target_total", 0.0))
        self.state["hh_buffer_gap_total"] = float(sol.get("buffer_gap_total", 0.0))
        self.state["hh_buffer_gap_positive_total"] = float(sol.get("buffer_gap_positive_total", 0.0))
        self.state["hh_buffer_gap_shortfall_total"] = float(sol.get("buffer_gap_shortfall_total", 0.0))
        # HOUSING is only used as an accounting aid in this build; turnover closes through
        # direct balance-sheet adjustments rather than a persistent escrow stock.
        self.state["housing_financing_deposits_total"] = 0.0

        if c_firm_nom.shape[0] != n:
            raise ValueError("population solver returned c_firm_nom of wrong length")
        if c_hh_nom.shape[0] != n:
            raise ValueError("population solver returned c_hh_nom of wrong length")
        if wages_i.shape[0] != n:
            raise ValueError("population solver returned wages_i of wrong length")
        if div_i.shape[0] != n:
            raise ValueError("population solver returned div_i of wrong length")
        if interest_hh.shape[0] != n:
            raise ValueError("population solver returned interest_hh of wrong length")
        if rev_interest_i.shape[0] != n:
            rev_interest_i = np.maximum(0.0, interest_hh - np.maximum(0.0, mort_interest_due_i if mort_interest_due_i.shape[0] == n else np.zeros(n, dtype=float)))
        if mort_pay_req_i.shape[0] != n:
            mort_pay_req_i = np.zeros(n, dtype=float)
        if renter_rent_q.shape[0] != n:
            renter_rent_q = np.zeros(n, dtype=float)
        if mort_pay_ctr_i.shape[0] != n:
            mort_pay_ctr_i = np.zeros(n, dtype=float)
        if mort_interest_due_i.shape[0] != n:
            mort_interest_due_i = np.zeros(n, dtype=float)
        if mort_interest_paid_i.shape[0] != n:
            mort_interest_paid_i = np.zeros(n, dtype=float)
        if mort_principal_paid_i.shape[0] != n:
            mort_principal_paid_i = np.zeros(n, dtype=float)
        if mort_interest_gap_i.shape[0] != n:
            mort_interest_gap_i = np.zeros(n, dtype=float)
        if mort_principal_gap_i.shape[0] != n:
            mort_principal_gap_i = np.zeros(n, dtype=float)
        if mort_gap_i.shape[0] != n:
            mort_gap_i = np.zeros(n, dtype=float)
        if mort_index_i.shape[0] != n:
            mort_index_i = np.ones(n, dtype=float)
        if mort_dln_i.shape[0] != n:
            mort_dln_i = np.zeros(n, dtype=float)
        if mort_dln_sm_i.shape[0] != n:
            mort_dln_sm_i = np.zeros(n, dtype=float)

        c_total = float(sol.get("c_total", 0.0))
        supplier_sales_fa_nom = float(sol.get("supplier_sales_fa_nom", 0.0))
        supplier_sales_fh_nom = float(sol.get("supplier_sales_fh_nom", 0.0))
        ums_recycle_fa_nom = float(sol.get("ums_recycle_fa_nom", 0.0))
        ums_recycle_fh_nom = float(sol.get("ums_recycle_fh_nom", 0.0))
        ums_recycle_total_nom = float(sol.get("ums_recycle_total_nom", ums_recycle_fa_nom + ums_recycle_fh_nom))
        capacity_fa_real = float(sol.get("capacity_fa_real", 0.0))
        capacity_fh_real = float(sol.get("capacity_fh_real", 0.0))
        hh_demand_fa_real = float(sol.get("hh_demand_fa_real", 0.0))
        hh_demand_fh_real = float(sol.get("hh_demand_fh_real", 0.0))
        hh_sales_fa_real = float(sol.get("hh_sales_fa_real", 0.0))
        hh_sales_fh_real = float(sol.get("hh_sales_fh_real", 0.0))

        # -------------------------------------------------
        # 1) Consumption: households -> firms, VAT remitted to GOV
        # -------------------------------------------------
        vat_rate = self._effective_vat_rate()

        deposits[:] = deposits - c_hh_nom

        # Firms receive only the household sales actually fulfilled this quarter.
        p_now = float(self.state.get("price_level", 1.0))
        self.nodes["FA"].add("deposits", p_now * hh_sales_fa_real)
        self.nodes["FH"].add("deposits", p_now * hh_sales_fh_real)

        # GOV receives VAT receipts
        vat_total = float(np.sum(c_hh_nom - c_firm_nom))
        if vat_total > 0:
            self.nodes["GOV"].add("deposits", vat_total)
        # Diagnostics: store per-tick VAT receipts (nominal total)
        self.state["vat_receipts_total"] = float(max(0.0, vat_total))

        # -------------------------------------------------
        # CAPEX: firm investment demand paid to suppliers + capital accumulation
        # -------------------------------------------------
        capex_fa_nom = float(sol.get("capex_fa_nom", 0.0))
        capex_fh_nom = float(sol.get("capex_fh_nom", 0.0))
        capex_total_nom = float(sol.get("capex_total_nom", capex_fa_nom + capex_fh_nom))

        # Depreciate existing capital (real units, non-cash)
        depr_q = float(self.params.get("capital_depr_rate_per_quarter", 0.0))
        depr_q = max(0.0, min(1.0, depr_q))
        for firm in ["FA", "FH"]:
            k0 = float(self.nodes[firm].get("K", 0.0))
            if k0 > 0 and depr_q > 0:
                self.nodes[firm].set("K", max(0.0, k0 * (1.0 - depr_q)))
            if firm == "FA":
                base_key = "sector_base_capacity_info_real"
            else:
                base_key = "sector_base_capacity_phys_real"
            base_capacity = float(self.state.get(base_key, 0.0))
            if base_capacity > 0 and depr_q > 0:
                self.state[base_key] = max(0.0, base_capacity * (1.0 - depr_q))

        # Settle CAPEX cash flows: investors pay; suppliers receive according to the lagged
        # supplier matrix embedded in the solver.
        if capex_total_nom > 0:
            capex_to_fa = supplier_sales_fa_nom
            capex_to_fh = supplier_sales_fh_nom

            # Pay from investors (cash outflow)
            self.nodes["FA"].add("deposits", -capex_fa_nom)
            self.nodes["FH"].add("deposits", -capex_fh_nom)

            # Receive by suppliers (cash inflow; net across firms = 0)
            self.nodes["FA"].add("deposits", +capex_to_fa)
            self.nodes["FH"].add("deposits", +capex_to_fh)

            # Capital formation (store K in REAL units)
            P_cap = float(self.state.get("price_level", 1.0))
            if P_cap <= 0:
                P_cap = 1e-9
            self.nodes["FA"].add("K", capex_fa_nom / P_cap)
            self.nodes["FH"].add("K", capex_fh_nom / P_cap)

        # Diagnostics (nominal flow)
        self.state["capex_total"] = float(max(0.0, capex_total_nom))
        self.state["ums_recycle_to_info_total"] = float(max(0.0, ums_recycle_fa_nom))
        self.state["ums_recycle_to_phys_total"] = float(max(0.0, ums_recycle_fh_nom))
        self.state["ums_recycle_total"] = float(max(0.0, ums_recycle_total_nom))

        if ums_recycle_total_nom > 0.0:
            self.nodes["UMS"].add("deposits", -ums_recycle_total_nom)
            self.nodes["FA"].add("deposits", +ums_recycle_fa_nom)
            self.nodes["FH"].add("deposits", +ums_recycle_fh_nom)

        # -------------------------------------------------
        # 2) Wages: firms -> households
        # -------------------------------------------------
        w_fa = float(sol.get("w_fa", 0.0))
        w_fh = float(sol.get("w_fh", 0.0))
        self.nodes["FA"].add("deposits", -w_fa)
        self.nodes["FH"].add("deposits", -w_fh)
        deposits[:] = deposits + wages_i

        # -------------------------------------------------
        # 2a) Sector overhead: firms -> GOV sink
        # -------------------------------------------------
        overhead_fa = float(sol.get("overhead_fa", 0.0))
        overhead_fh = float(sol.get("overhead_fh", 0.0))
        overhead_total = 0.0
        if overhead_fa > 0.0:
            self.nodes["FA"].add("deposits", -overhead_fa)
            overhead_total += overhead_fa
        if overhead_fh > 0.0:
            self.nodes["FH"].add("deposits", -overhead_fh)
            overhead_total += overhead_fh
        if overhead_total > 0.0:
            self.nodes["GOV"].add("deposits", overhead_total)

        # -------------------------------------------------
        # 2aa) Sector input costs: firms -> UMS reservoir
        # -------------------------------------------------
        input_cost_fa = float(sol.get("input_cost_fa", 0.0))
        input_cost_fh = float(sol.get("input_cost_fh", 0.0))
        input_cost_total = 0.0
        if input_cost_fa > 0.0:
            self.nodes["FA"].add("deposits", -input_cost_fa)
            input_cost_total += input_cost_fa
        if input_cost_fh > 0.0:
            self.nodes["FH"].add("deposits", -input_cost_fh)
            input_cost_total += input_cost_fh
        if input_cost_total > 0.0:
            self.nodes["UMS"].add("deposits", input_cost_total)
        self.state["sector_input_cost_info_total"] = float(max(0.0, input_cost_fa))
        self.state["sector_input_cost_phys_total"] = float(max(0.0, input_cost_fh))
        self.state["sector_input_cost_total"] = float(max(0.0, input_cost_total))

        # -------------------------------------------------
        # 2b) Firm interest: firms -> BANK
        # -------------------------------------------------
        fa_interest = float(sol.get("fa_interest", 0.0))
        fh_interest = float(sol.get("fh_interest", 0.0))

        if fa_interest > 0:
            self.nodes["FA"].add("deposits", -fa_interest)
            bank = self.nodes["BANK"]
            bank.add("deposit_liab", -fa_interest)
            bank.add("equity", +fa_interest)

        if fh_interest > 0:
            self.nodes["FH"].add("deposits", -fh_interest)
            bank = self.nodes["BANK"]
            bank.add("deposit_liab", -fh_interest)
            bank.add("equity", +fh_interest)

        # -------------------------------------------------
        # 2c) Corporate tax: corporations -> GOV (before dividends / reinvestment)
        # -------------------------------------------------
        corp_tax_fa = float(sol.get("corp_tax_fa", 0.0))
        corp_tax_fh = float(sol.get("corp_tax_fh", 0.0))
        corp_tax_bk = float(sol.get("corp_tax_bk", 0.0))

        corp_tax_total = max(0.0, corp_tax_fa) + max(0.0, corp_tax_fh) + max(0.0, corp_tax_bk)
        if corp_tax_total > 0:
            # Transfer cash from corporate deposits to GOV deposits.
            if corp_tax_fa > 0:
                self.nodes["FA"].add("deposits", -corp_tax_fa)
            if corp_tax_fh > 0:
                self.nodes["FH"].add("deposits", -corp_tax_fh)
            if corp_tax_bk > 0:
                bank = self.nodes["BANK"]
                bank.add("equity", -corp_tax_bk)
                bank.add("deposit_liab", +corp_tax_bk)
            self.nodes["GOV"].add("deposits", corp_tax_total)

        # Diagnostics: store per-tick corporate-tax receipts (nominal total)
        self.state["corp_tax_total"] = float(max(0.0, corp_tax_total))
        corp_tax_rate_eff = float(sol.get("corp_tax_rate", self.params.get("corporate_tax_rate", 0.0)))
        self.state["corp_tax_rate_eff"] = float(max(0.0, min(1.0, corp_tax_rate_eff)))

        # -------------------------------------------------
        # 3) Dividends: issuers -> FUND and households
        # -------------------------------------------------
        p_fa = float(sol.get("p_fa", 0.0))
        p_fh = float(sol.get("p_fh", 0.0))
        bank_profit = float(sol.get("bank_profit", 0.0))

        f_fa = float(sol.get("f_fa", 0.0))
        f_fh = float(sol.get("f_fh", 0.0))
        f_bk = float(sol.get("f_bk", 0.0))

        div_fa_total = float(sol.get("div_fa_total", p_fa))
        div_fh_total = float(sol.get("div_fh_total", p_fh))
        div_bk_total = float(sol.get("div_bk_total", bank_profit))

        # Cash to FUND
        self.nodes["FA"].add("deposits", -(div_fa_total * f_fa))
        self.nodes["FH"].add("deposits", -(div_fh_total * f_fh))
        self.nodes["FUND"].add("deposits", (div_fa_total * f_fa) + (div_fh_total * f_fh) + (div_bk_total * f_bk))

        # Cash to households (distributed via solver weights)
        self.nodes["FA"].add("deposits", -(div_fa_total * (1.0 - f_fa)))
        self.nodes["FH"].add("deposits", -(div_fh_total * (1.0 - f_fh)))
        deposits[:] = deposits + div_i

        # Bank dividends are paid out of equity and create new deposits for recipients
        if div_bk_total > 0:
            bank = self.nodes["BANK"]
            bank.add("equity", -div_bk_total)
            bank.add("deposit_liab", +div_bk_total)

        # -------------------------------------------------
        # 4) Interest: households and FUND -> BANK
        # -------------------------------------------------
        bank = self.nodes["BANK"]
        self.state["mort_pay_req_total"] = 0.0
        self.state["mort_pay_ctr_total"] = 0.0
        self.state["mort_gap_total"] = 0.0
        self.state["mort_gap_paid_by_gov"] = 0.0
        self.state["mort_gap_paid_by_fund"] = 0.0
        self.state["mort_gap_paid_by_issuance"] = 0.0
        self.state["bank_mort_neutralize_inflow"] = 0.0
        self.state["bank_mort_neutralize_interest_inflow"] = 0.0
        self.state["bank_mort_neutralize_principal_inflow"] = 0.0
        self.state["mort_index_mean"] = 1.0
        self.state["mort_index_min"] = 1.0
        self.state["mort_index_max"] = 1.0
        self.state["mort_overdraft_due_to_payment_total"] = 0.0
        self.state["mort_overdraft_due_to_payment_count"] = 0.0
        self.state["mort_principal_paid_total"] = 0.0

        if mort_index_enable:
            # Commit mortgage index recursion state for this tick using converged solver outputs.
            active_mort = (mort > 1e-12) & (hh.mort_t0 >= 0)
            curr_i = np.maximum(0.0, mort_index_i)
            dln_sm_curr = mort_dln_sm_i.copy()
            if np.any(~active_mort):
                curr_i[~active_mort] = 1.0
                dln_sm_curr[~active_mort] = 0.0
            hh.mort_index_prev = curr_i.astype(float, copy=True)
            hh.mort_dlnI_sm_prev = dln_sm_curr.astype(float, copy=True)
            self.state["mort_price_series_prev"] = float(sol.get("p_series_now", self.state.get("mort_price_series_prev", self.state.get("price_level", 1.0))))
            self.state["mort_income_series_prev"] = float(sol.get("y_series_now", self.state.get("mort_income_series_prev", 1.0)))

            self.state["mort_pay_req_total"] = float(np.sum(np.maximum(0.0, mort_pay_req_i)))
            self.state["mort_pay_ctr_total"] = float(np.sum(np.maximum(0.0, mort_pay_ctr_i)))
            self.state["mort_gap_total"] = float(np.sum(np.maximum(0.0, mort_gap_i)))
            if np.any(active_mort):
                self.state["mort_index_mean"] = float(np.mean(curr_i[active_mort]))
                self.state["mort_index_min"] = float(np.min(curr_i[active_mort]))
                self.state["mort_index_max"] = float(np.max(curr_i[active_mort]))

        self.state["mort_pay_req_total"] = float(np.sum(np.maximum(0.0, mort_pay_req_i)))
        self.state["mort_pay_ctr_total"] = float(np.sum(np.maximum(0.0, mort_pay_ctr_i)))
        self.state["mort_gap_total"] = float(np.sum(np.maximum(0.0, mort_gap_i)))

        if mort_index_enable:
            if np.any(active_mort):
                self.state["mort_index_mean"] = float(np.mean(curr_i[active_mort]))
                self.state["mort_index_min"] = float(np.min(curr_i[active_mort]))
                self.state["mort_index_max"] = float(np.max(curr_i[active_mort]))

        pop_cfg = self.params.get("population_config", {}) if isinstance(self.params.get("population_config", {}), dict) else {}
        rev_income_cap_mult = self.params.get("revolving_credit_limit_income_mult", pop_cfg.get("revolving_cap_income_mult", 0.0))
        rev_income_cap_mult = max(0.0, float(rev_income_cap_mult))
        annual_wage_i = 4.0 * np.maximum(0.0, wages_i)
        if rev_income_cap_mult > 0.0:
            rev_limit_i = rev_income_cap_mult * annual_wage_i
        else:
            rev_limit_i = np.full(n, np.inf, dtype=float)
        rev_headroom_i = np.maximum(0.0, rev_limit_i - rev)
        mort_unpaid_cash_shortfall_i = np.zeros(n, dtype=float)

        # Revolving interest remains contractual in all mortgage regimes.
        deposits[:] = deposits - rev_interest_i
        rev_int_total = float(np.sum(np.maximum(0.0, rev_interest_i)))
        if rev_int_total > 0.0:
            bank.add("deposit_liab", -rev_int_total)
            bank.add("equity", +rev_int_total)

        # Mortgage required payment is the single household mortgage cash-flow path.
        # Households may bridge only a capped portion via revolving credit; any
        # residual shortfall remains unpaid on the mortgage side instead of
        # turning into effectively unlimited revolving debt.
        active_mort_start = mort > 1e-12
        dep_before_mort = deposits.copy()
        cash_available_for_mort_i = np.maximum(0.0, dep_before_mort)
        mort_overdraft_need = np.maximum(0.0, mort_pay_req_i - cash_available_for_mort_i)
        mort_revolving_bridge_i = np.minimum(mort_overdraft_need, rev_headroom_i)
        actual_mort_payment_i = np.minimum(mort_pay_req_i, cash_available_for_mort_i + mort_revolving_bridge_i)
        mort_unpaid_cash_shortfall_i = np.maximum(0.0, mort_pay_req_i - actual_mort_payment_i)
        deposits[:] = deposits - actual_mort_payment_i
        mort_bridge_total = float(np.sum(np.maximum(0.0, mort_revolving_bridge_i)))
        if mort_bridge_total > 0.0:
            deposits[:] = deposits + mort_revolving_bridge_i
            rev[:] = rev + mort_revolving_bridge_i
            bank.add("loan_assets", +mort_bridge_total)
            bank.add("deposit_liab", +mort_bridge_total)
        actual_mort_interest_paid_i = np.minimum(np.maximum(0.0, mort_interest_paid_i), actual_mort_payment_i)
        actual_mort_principal_paid_i = np.minimum(
            np.maximum(0.0, mort_principal_paid_i),
            np.maximum(0.0, actual_mort_payment_i - actual_mort_interest_paid_i),
        )
        self.state["mort_overdraft_due_to_payment_total"] = float(np.sum(mort_overdraft_need))
        self.state["mort_overdraft_due_to_payment_count"] = float(np.sum(mort_overdraft_need > 1e-12))
        self.state["mort_revolving_bridge_total"] = float(mort_bridge_total)
        self.state["mort_unpaid_cash_shortfall_total"] = float(np.sum(mort_unpaid_cash_shortfall_i))

        mort_int_paid_total = float(np.sum(np.maximum(0.0, actual_mort_interest_paid_i)))
        mort_prin_paid_total = float(np.sum(np.maximum(0.0, actual_mort_principal_paid_i)))
        self.state["mort_principal_paid_total"] = float(mort_prin_paid_total)

        if mort_int_paid_total > 0.0:
            bank.add("deposit_liab", -mort_int_paid_total)
            bank.add("equity", +mort_int_paid_total)
        if mort_prin_paid_total > 0.0:
            mort[:] = np.maximum(0.0, mort - actual_mort_principal_paid_i)
            bank.add("loan_assets", -mort_prin_paid_total)
            bank.add("deposit_liab", -mort_prin_paid_total)

        if mort_index_enable:
            # Optional bank neutralization transfer for reduced mortgage cashflow.
            neutral = self._apply_mortgage_gap_neutralization(
                interest_gap_i=np.maximum(0.0, mort_interest_gap_i),
                principal_gap_i=np.maximum(0.0, mort_principal_gap_i),
                mort_interest_due_total=float(np.sum(np.maximum(0.0, mort_interest_due_i))),
                mort_pay_ctr_total=float(np.sum(np.maximum(0.0, mort_pay_ctr_i))),
            )
            self.state["mort_gap_total"] = float(neutral["gap_total"])
            self.state["mort_gap_paid_by_gov"] = float(neutral["paid_gov"])
            self.state["mort_gap_paid_by_fund"] = float(neutral["paid_fund"])
            self.state["mort_gap_paid_by_issuance"] = float(neutral["paid_issuance"])
            self.state["bank_mort_neutralize_inflow"] = float(neutral["paid_total"])
            self.state["bank_mort_neutralize_interest_inflow"] = float(neutral["paid_interest_total"])
            self.state["bank_mort_neutralize_principal_inflow"] = float(neutral["paid_principal_total"])
            neutral_principal_i = _as_np(neutral.get("paid_principal_i", np.zeros(n, dtype=float)), dtype=float)
            neutral_principal_total = float(np.sum(np.maximum(0.0, neutral_principal_i)))
            if neutral_principal_total > 0.0:
                mort[:] = np.maximum(0.0, mort - neutral_principal_i)
                self.state["mort_principal_paid_total"] = float(
                    float(self.state.get("mort_principal_paid_total", 0.0)) + neutral_principal_total
                )
        if np.any(active_mort_start):
            hh.mort_age_q[active_mort_start] = np.minimum(
                hh.mort_term_q[active_mort_start],
                hh.mort_age_q[active_mort_start] + 1.0,
            )

        trust_interest = float(sol.get("trust_interest", 0.0))
        if trust_interest > 0:
            short = max(0.0, trust_interest - self.nodes["FUND"].get("deposits"))
            if short > 0:
                self._create_loan("FUND", short, memo_tag="interest_capitalized")
            self.nodes["FUND"].add("deposits", -trust_interest)
            bank = self.nodes["BANK"]
            bank.add("deposit_liab", -trust_interest)
            bank.add("equity", +trust_interest)

        # -------------------------------------------------
        # 4a) Rent: renter households -> HOUSING reservoir
        # -------------------------------------------------
        rent_total = float(np.sum(np.maximum(0.0, renter_rent_q)))
        if rent_total > 0.0:
            deposits[:] = deposits - renter_rent_q
            rent_to_fa = rent_total * self._sector_hh_demand_share_fa()
            rent_to_fh = rent_total - rent_to_fa
            self.nodes["FA"].add("deposits", rent_to_fa)
            self.nodes["FH"].add("deposits", rent_to_fh)
            self.state["renter_rent_to_info_total"] = float(max(0.0, rent_to_fa))
            self.state["renter_rent_to_phys_total"] = float(max(0.0, rent_to_fh))
        self.state["renter_rent_total"] = float(max(0.0, rent_total))

        # -------------------------------------------------
        # 5) Taxes + VAT credit (before income support)
        # -------------------------------------------------
        income_tax_i = _as_np(sol.get("income_tax_i", []), dtype=float)
        vat_credit_i = _as_np(sol.get("vat_credit_i", []), dtype=float)

        if income_tax_i.shape[0] == n:
            income_tax_total = float(np.sum(np.maximum(0.0, income_tax_i)))
            if income_tax_total > 0:
                deposits[:] = deposits - income_tax_i
                self.nodes["GOV"].add("deposits", income_tax_total)
        else:
            income_tax_total = 0.0
        # Diagnostics: store per-tick income-tax receipts (nominal total)
        self.state["income_tax_total"] = float(max(0.0, income_tax_total))

        # VAT credit is a transfer from GOV (or issuance if needed) to eligible households
        vat_credit_total = float(np.sum(np.maximum(0.0, vat_credit_i))) if (vat_credit_i.shape[0] == n) else 0.0
        vat_credit_total_initial = float(vat_credit_total)
        vat_credit_paid_from_gov = 0.0
        vat_credit_issued = 0.0

        if vat_credit_total > 0:
            gov_dep = max(0.0, self.nodes["GOV"].get("deposits"))
            pay_gov = min(vat_credit_total, gov_dep)
            if pay_gov > 0:
                self.nodes["GOV"].add("deposits", -pay_gov)
                vat_credit_paid_from_gov += pay_gov
                self._distribute_household_transfer_by_weights(deposits, vat_credit_i, pay_gov)
                vat_credit_total -= pay_gov

            if vat_credit_total > 0:
                # Issuance to complete the credit
                vat_credit_issued += vat_credit_total
                self._distribute_household_transfer_by_weights(deposits, vat_credit_i, vat_credit_total)
                self.nodes["BANK"].add("deposit_liab", vat_credit_total)
                self.nodes["BANK"].add("reserves", vat_credit_total)
                self.nodes["GOV"].add("money_issued", vat_credit_total)

        # Diagnostics: store per-tick VAT credit totals
        self.state["vat_credit_paid_total"] = float(max(0.0, vat_credit_paid_from_gov))
        self.state["vat_credit_issued_total"] = float(max(0.0, vat_credit_issued))
        self.state["vat_credit_total"] = float(max(0.0, vat_credit_total_initial))

        # -------------------------------------------------
        # 5a) Trust amortization (before income support)
        # -------------------------------------------------
        fund_loan = float(self.nodes["FUND"].get("loans", 0.0))
        if fund_loan > 0:
            fund_dep = float(self.nodes["FUND"].get("deposits", 0.0))
            repay_amt = min(fund_dep, fund_loan)
            if repay_amt > 0:
                self._repay_loan("FUND", repay_amt)

        fund_residual_share = self.params.get("fund_residual_to_gov_share", None)
        if fund_residual_share is None:
            fund_residual_share = 1.0 if self.params.get("send_fund_residual_to_gov", False) else 0.0
        fund_residual_share = max(0.0, min(1.0, float(fund_residual_share)))
        if fund_residual_share <= 0.0 and self.params.get("send_fund_residual_to_gov", False):
            fund_residual_share = 1.0

        if fund_residual_share > 0.0:
            residual = max(0.0, self.nodes["FUND"].get("deposits"))
            transfer = residual * fund_residual_share
            if transfer > 0:
                self._xfer_deposits("FUND", "GOV", transfer)

        self.state["ums_drain_to_fund_total"] = 0.0
        self.state["ums_drain_to_gov_total"] = 0.0

        # -------------------------------------------------
        # 6) Income-support payments: issuance share -> FUND dep -> GOV dep -> extra issuance
        # -------------------------------------------------
        uis = float(sol.get("uis", 0.0))
        funding = apply_income_support_payment(
            support_per_household=float(uis),
            n_households=int(n),
            issue_share=float(
                self.params.get(
                    "income_support_issuance_share",
                    self.params.get("uis_issuance_share", 0.0),
                )
            ),
            deposits=deposits,
            nodes=self.nodes,
        )

        # Store diagnostics for this tick (totals across all households)
        self.state["uis_from_fund_dep_total"] = float(funding.from_fund_dep_total)
        self.state["uis_from_gov_dep_total"] = float(funding.from_gov_dep_total)
        self.state["uis_issued_total"] = float(funding.issued_total)

        # -------------------------------------------------
        # 6a) Optional tax refund: rebate a share of remaining GOV deposits
        # in proportion to each household's tax paid (VAT + income tax).
        # -------------------------------------------------
        current_gov_obligation = (
            float(self.state.get("uis_from_gov_dep_total", 0.0))
            + float(self.state.get("vat_credit_paid_total", 0.0))
            + float(self.state.get("mort_gap_paid_by_gov", 0.0))
        )
        tax_rebate_total = 0.0
        rebate_rate_target = float(self.params.get("gov_tax_rebate_rate", 0.0))
        rebate_rate_target = max(0.0, min(1.0, rebate_rate_target))
        rebate_rate = rebate_rate_target * self._gov_rebate_ramp_multiplier()
        if rebate_rate > 0.0:
            gov_dep_avail = max(0.0, self.nodes["GOV"].get("deposits", 0.0))
            gov_buffer_target = self._gov_rebate_buffer_amount()
            if gov_buffer_target is None:
                rebate_base = 0.0
            else:
                rebate_base = max(0.0, gov_dep_avail - gov_buffer_target)
            tax_rebate_total = rebate_rate * rebate_base
            if tax_rebate_total > 0.0:
                # Per-household VAT paid is the VAT wedge on each household's consumption.
                vat_paid_i = np.maximum(0.0, c_hh_nom - c_firm_nom)
                if income_tax_i.shape[0] == n:
                    income_tax_paid_i = np.maximum(0.0, income_tax_i)
                else:
                    income_tax_paid_i = np.zeros(n, dtype=float)

                tax_paid_i = vat_paid_i + income_tax_paid_i
                tax_paid_total = float(np.sum(tax_paid_i))

                self.nodes["GOV"].add("deposits", -tax_rebate_total)
                if tax_paid_total > 0.0:
                    deposits[:] = deposits + tax_paid_i * (tax_rebate_total / tax_paid_total)
                else:
                    deposits[:] = deposits + (tax_rebate_total / float(n))

        self.state["tax_rebate_total"] = float(max(0.0, tax_rebate_total))
        self.state["gov_obligation_total"] = float(max(0.0, current_gov_obligation))
        self.state["gov_rebate_rate_eff"] = float(max(0.0, rebate_rate))
        buffer_target = self._gov_rebate_buffer_amount()
        self.state["gov_rebate_buffer_target"] = float(buffer_target if buffer_target is not None else 0.0)
        self.gov_obligation_history.append(float(max(0.0, current_gov_obligation)))
        buffer_quarters = max(0, int(self.params.get("gov_rebate_buffer_quarters", 4)))
        keep = max(1, buffer_quarters + 1)
        if len(self.gov_obligation_history) > keep:
            self.gov_obligation_history = self.gov_obligation_history[-keep:]

        # -------------------------------------------------
        # 6b) Household principal repayment (revolving first, then mortgage)
        # -------------------------------------------------
        rev_pay_rate = float(self.params.get("revolving_principal_pay_rate_q", 0.0))
        rL = float(self.state.get("policy_rate_q", self.params.get("loan_rate_per_quarter", 0.0)))
        rev_rollover_share = float(self.params.get("revolving_rollover_share", 0.0))
        mort_turnover_enabled = bool(self.params.get("mortgage_turnover_enabled", False))
        mort_turnover_active_min_remaining_q = float(self.params.get("mortgage_turnover_active_min_remaining_q", 3.0))
        mort_turnover_target_payment_floor_share = float(
            self.params.get("mortgage_turnover_target_payment_floor_share", 1.0)
        )
        mort_turnover_dti_cap = float(self.params.get("mortgage_turnover_dti_cap", 0.40))
        mort_turnover_income_mult_cap = float(self.params.get("mortgage_turnover_income_mult_cap", 4.0))
        mort_turnover_support_income_weight = float(
            self.params.get("mortgage_turnover_support_income_weight", 0.0)
        )
        mort_turnover_min_wage_q = float(self.params.get("mortgage_turnover_min_wage_q", 1.0))
        housing_turnover_rate_mortgagor_q = float(self.params.get("housing_turnover_rate_mortgagor_q", 0.0))
        housing_turnover_rate_owner_q = float(self.params.get("housing_turnover_rate_owner_q", 0.0))
        housing_turnover_owner_mortgage_share = float(self.params.get("housing_turnover_owner_mortgage_share", 0.0))
        rev_pay_rate = max(0.0, min(1.0, rev_pay_rate))
        rev_rollover_share = max(0.0, min(1.0, rev_rollover_share))
        mort_turnover_active_min_remaining_q = max(0.0, mort_turnover_active_min_remaining_q)
        mort_turnover_target_payment_floor_share = max(0.0, mort_turnover_target_payment_floor_share)
        mort_turnover_dti_cap = max(0.0, mort_turnover_dti_cap)
        mort_turnover_income_mult_cap = max(0.0, mort_turnover_income_mult_cap)
        mort_turnover_support_income_weight = max(0.0, mort_turnover_support_income_weight)
        mort_turnover_min_wage_q = max(0.0, mort_turnover_min_wage_q)
        housing_turnover_rate_mortgagor_q = max(0.0, min(1.0, housing_turnover_rate_mortgagor_q))
        housing_turnover_rate_owner_q = max(0.0, min(1.0, housing_turnover_rate_owner_q))
        housing_turnover_owner_mortgage_share = max(0.0, min(1.0, housing_turnover_owner_mortgage_share))

        principal_paid_total = 0.0
        rev_rollover_total = 0.0
        mort_principal_paid_total = float(self.state.get("mort_principal_paid_total", 0.0))
        mort_turnover_total = 0.0
        self.state["rev_principal_paid_total"] = 0.0
        self.state["rev_rollover_total"] = 0.0
        self.state["mort_turnover_total"] = 0.0
        self.state["mort_turnover_households"] = 0.0
        self.state["mortgage_turnover_active_count"] = 0.0
        self.state["mortgage_turnover_payment_gap_total"] = 0.0
        self.state["mortgage_turnover_payment_gap_remaining_total"] = 0.0
        self.state["mortgage_turnover_payment_capacity_total"] = 0.0
        self.state["mortgage_turnover_nonmort_count"] = 0.0
        self.state["mortgage_turnover_outright_owner_excluded_count"] = 0.0
        self.state["mortgage_turnover_new_eligible_count"] = 0.0
        self.state["mortgage_turnover_dti_binding_count"] = 0.0
        self.state["mortgage_turnover_income_binding_count"] = 0.0
        self.state["mortgage_turnover_zero_dti_room_count"] = 0.0
        self.state["mortgage_turnover_zero_income_room_count"] = 0.0
        self.state["mortgage_turnover_renter_entry_count"] = 0.0
        self.state["mortgage_turnover_supply_released_count"] = 0.0

        if rev_pay_rate > 0.0:
            # desired paydown is a fraction of outstanding revolver
            desired_rev_pay = rev * rev_pay_rate
            # cannot pay more than deposits available
            rev_pay = np.minimum(desired_rev_pay, deposits)
            rev_pay = np.maximum(0.0, rev_pay)

            pay_sum = float(rev_pay.sum())
            if pay_sum > 0.0:
                deposits[:] = deposits - rev_pay
                rev[:] = rev - rev_pay
                principal_paid_total += pay_sum
                self.state["rev_principal_paid_total"] = float(pay_sum)

                if rev_rollover_share > 0.0:
                    reissue = rev_pay * rev_rollover_share
                    reissue_sum = float(np.sum(np.maximum(0.0, reissue)))
                    if reissue_sum > 0.0:
                        deposits[:] = deposits + reissue
                        rev[:] = rev + reissue
                        rev_rollover_total += reissue_sum
                        self.state["rev_rollover_total"] = float(reissue_sum)

        if principal_paid_total > 0.0:
            # Principal repayment destroys deposits and loans (double-entry)
            self.nodes["BANK"].add("loan_assets", -principal_paid_total)
            self.nodes["BANK"].add("deposit_liab", -principal_paid_total)

        if rev_rollover_total > 0.0:
            # Same-household revolving credit rollover recreates deposits and loan assets.
            self.nodes["BANK"].add("loan_assets", rev_rollover_total)
            self.nodes["BANK"].add("deposit_liab", rev_rollover_total)

        if mort_turnover_enabled:
            hh_deposits_before_turnover = float(deposits.sum())
            support_income_q_i = np.full_like(wages_i, mort_turnover_support_income_weight * max(0.0, uis), dtype=float)
            underwriting_income_q_i = np.maximum(0.0, wages_i) + support_income_q_i
            underwriting_income_annual_i = 4.0 * underwriting_income_q_i
            mort_cap_i = mort_turnover_income_mult_cap * underwriting_income_annual_i
            active_mort_i = mort > 1e-9
            old_housing_value_i = np.maximum(0.0, _as_np(hh.housing_escrow, dtype=float)).copy()
            remaining_q_i = remaining_term(
                np.asarray(hh.mort_term_q, dtype=float),
                np.asarray(hh.mort_age_q, dtype=float),
            )

            current_rev_interest_i = np.maximum(0.0, rev * rL)
            housing_value_i = old_housing_value_i.copy()
            owner_mask_i = (~active_mort_i) & (housing_value_i > 1e-12)
            renter_mask_i = (~active_mort_i) & (housing_value_i <= 1e-12)
            mort_turnover_candidate_i = active_mort_i & (remaining_q_i > mort_turnover_active_min_remaining_q)
            owner_turnover_candidate_i = owner_mask_i
            mort_turnover_event_i = mort_turnover_candidate_i & (self.rng.random(hh.n) < housing_turnover_rate_mortgagor_q)
            owner_turnover_event_i = owner_turnover_candidate_i & (self.rng.random(hh.n) < housing_turnover_rate_owner_q)
            owner_finance_event_i = owner_turnover_event_i & (self.rng.random(hh.n) < housing_turnover_owner_mortgage_share)
            turnover_event_i = mort_turnover_event_i | owner_turnover_event_i
            turnover_buyer_i = mort_turnover_event_i | owner_finance_event_i

            dti_room_nom = np.maximum(
                0.0,
                (mort_turnover_dti_cap * underwriting_income_q_i) - current_rev_interest_i,
            )
            new_rate_q = self._mortgage_fixed_rate_q()
            new_term_q = float(self._mortgage_term_quarters())
            new_schedule = self._current_mortgage_schedule()
            unit_payment_q = float(new_schedule.contract_payment_factor)
            income_limit_payment_i = mort_cap_i * max(0.0, unit_payment_q)
            desired_mult = float(pop_cfg.get("mortgage_income_mult_median", mort_turnover_income_mult_cap))
            desired_mult = max(0.0, desired_mult)
            turnover_ltv_cap = float(pop_cfg.get("mortgage_startup_ltv_max", 1.0))
            turnover_ltv_cap = max(0.0, turnover_ltv_cap)
            housing_limit_principal_i = housing_value_i * turnover_ltv_cap
            income_limit_principal_i = np.minimum(
                mort_cap_i,
                np.minimum(desired_mult * underwriting_income_annual_i, housing_limit_principal_i),
            )
            income_limit_payment_i = np.minimum(
                income_limit_payment_i,
                income_limit_principal_i * max(0.0, unit_payment_q),
            )
            desired_payment_i = np.minimum(dti_room_nom, income_limit_payment_i)
            desired_principal_i = np.minimum(
                income_limit_principal_i,
                desired_payment_i / max(1e-9, unit_payment_q),
            )
            min_desired_payment_i = (0.25 * underwriting_income_annual_i) * max(0.0, unit_payment_q)

            allocation = np.zeros_like(mort, dtype=float)
            acquired_house_value = np.zeros_like(mort, dtype=float)
            matched_seller_mask = np.zeros(hh.n, dtype=bool)
            matched_renter_downpayment_total = 0.0
            matched_seller_net_total = 0.0
            self.state["mortgage_turnover_payment_gap_total"] = 0.0
            self.state["mortgage_turnover_payment_gap_remaining_total"] = 0.0
            self.state["mortgage_turnover_active_count"] = float(np.sum(mort_turnover_candidate_i) + np.sum(owner_turnover_candidate_i))
            nonmort_mask = ~active_mort_i
            base_new_pool = turnover_buyer_i & (underwriting_income_q_i >= mort_turnover_min_wage_q)
            self.state["mortgage_turnover_nonmort_count"] = float(np.sum(nonmort_mask))
            self.state["mortgage_turnover_outright_owner_excluded_count"] = 0.0
            self.state["mortgage_turnover_zero_dti_room_count"] = float(np.sum(base_new_pool & (dti_room_nom <= 1e-9)))
            self.state["mortgage_turnover_zero_income_room_count"] = float(np.sum(base_new_pool & (income_limit_principal_i <= 1e-9)))

            new_eligible = (
                turnover_buyer_i
                & (underwriting_income_q_i >= mort_turnover_min_wage_q)
                & (desired_payment_i > 1e-9)
            )
            self.state["mortgage_turnover_new_eligible_count"] = float(np.sum(new_eligible))
            self.state["mortgage_turnover_dti_binding_count"] = float(
                np.sum(new_eligible & (dti_room_nom <= income_limit_payment_i))
            )
            self.state["mortgage_turnover_income_binding_count"] = float(
                np.sum(new_eligible & (income_limit_payment_i < dti_room_nom))
            )
            self.state["mortgage_turnover_payment_capacity_total"] = float(
                np.sum(np.maximum(0.0, desired_payment_i[new_eligible]))
            )

            if np.any(new_eligible):
                candidate_idx = np.where(new_eligible)[0]
                for idx in candidate_idx.tolist():
                    desired_payment = float(desired_payment_i[idx])
                    min_desired_payment = float(min_desired_payment_i[idx])
                    desired_principal = float(desired_principal_i[idx])
                    if desired_payment <= 1e-9 or desired_principal <= 1e-9:
                        continue
                    if desired_payment < max(1e-9, min_desired_payment):
                        continue
                    allocation[idx] = desired_principal
                    acquired_house_value[idx] = housing_value_i[idx]

            turnover_exit_mask = turnover_event_i & ~(allocation > 1e-9)
            supply_idx = np.where(turnover_exit_mask & (housing_value_i > 1e-9))[0]
            supply_values = housing_value_i[supply_idx].astype(float) if supply_idx.size else np.asarray([], dtype=float)
            rent_mult_median = float(pop_cfg.get("renter_rent_payment_mult_median", 1.0))
            rent_mult_median = max(0.0, rent_mult_median)
            renter_entry_count = 0

            if supply_values.size > 0:
                eligible_renter_idx = np.where(renter_mask_i & (underwriting_income_q_i >= mort_turnover_min_wage_q))[0]
                if eligible_renter_idx.size > 0:
                    remaining_renters = eligible_renter_idx[np.argsort(underwriting_income_q_i[eligible_renter_idx])].tolist()
                    sorted_supply_idx = supply_idx[np.argsort(supply_values)]
                    for seller_idx in sorted_supply_idx.tolist():
                        house_value = float(housing_value_i[seller_idx])
                        matched_pos = None
                        matched_principal = 0.0
                        matched_downpayment = 0.0
                        for pos, renter_idx in enumerate(remaining_renters):
                            income_principal = min(
                                float(mort_cap_i[renter_idx]),
                                float(desired_mult * underwriting_income_annual_i[renter_idx]),
                            )
                            principal_cap = min(income_principal, house_value * turnover_ltv_cap)
                            if principal_cap <= 1e-9:
                                continue
                            desired_payment = min(
                                float(dti_room_nom[renter_idx]),
                                principal_cap * max(0.0, unit_payment_q),
                            )
                            min_desired_payment = 0.25 * float(underwriting_income_annual_i[renter_idx]) * max(0.0, unit_payment_q)
                            if desired_payment <= 1e-9 or desired_payment < max(1e-9, min_desired_payment):
                                continue
                            principal = min(principal_cap, desired_payment / max(1e-9, unit_payment_q))
                            downpayment = max(0.0, house_value - principal)
                            if float(deposits[renter_idx]) + 1e-9 < downpayment:
                                continue
                            matched_pos = pos
                            matched_principal = principal
                            matched_downpayment = downpayment
                            break
                        if matched_pos is None:
                            continue
                        renter_idx = remaining_renters.pop(matched_pos)
                        allocation[renter_idx] = matched_principal
                        acquired_house_value[renter_idx] = house_value
                        deposits[renter_idx] = deposits[renter_idx] - matched_downpayment
                        seller_old_mort = float(max(0.0, mort[seller_idx]))
                        deposits[seller_idx] = deposits[seller_idx] + (house_value - seller_old_mort)
                        matched_renter_downpayment_total += float(matched_downpayment)
                        matched_seller_net_total += float(house_value - seller_old_mort)
                        matched_seller_mask[seller_idx] = True
                        renter_entry_count += 1

            self_turnover_mask = turnover_event_i & (allocation > 1e-9)
            self_turnover_net_total = 0.0
            if np.any(self_turnover_mask):
                sale_value_i = np.maximum(0.0, housing_value_i[self_turnover_mask])
                downpayment_i = np.maximum(0.0, acquired_house_value[self_turnover_mask] - allocation[self_turnover_mask])
                self_turnover_net_total = float(np.sum(
                    sale_value_i
                    - np.maximum(0.0, mort[self_turnover_mask])
                    - downpayment_i
                ))
                deposits[self_turnover_mask] = (
                    deposits[self_turnover_mask]
                    + sale_value_i
                    - np.maximum(0.0, mort[self_turnover_mask])
                    - downpayment_i
                )

            closed_sale_mask = self_turnover_mask | matched_seller_mask

            if np.any(matched_seller_mask):
                exit_idx = np.where(matched_seller_mask)[0]
                sold_house_values = housing_value_i[exit_idx]
                sold_house_payments = new_schedule.payment_from_orig_principal(sold_house_values)
                hh.housing_escrow[exit_idx] = 0.0
                renter_rent_q[exit_idx] = np.maximum(0.0, sold_house_payments * rent_mult_median)

            old_mort_turnover_total = float(np.sum(np.maximum(0.0, mort[closed_sale_mask])))
            if old_mort_turnover_total > 0.0:
                self.nodes["BANK"].add("loan_assets", -old_mort_turnover_total)
            if np.any(closed_sale_mask):
                mort[closed_sale_mask] = 0.0
                hh.mort_rate_q[closed_sale_mask] = 0.0
                hh.mort_term_q[closed_sale_mask] = 0.0
                hh.mort_age_q[closed_sale_mask] = 0.0
                hh.mort_payment_sched_q[closed_sale_mask] = 0.0
                hh.mort_orig_principal[closed_sale_mask] = 0.0
                hh.mort_t0[closed_sale_mask] = -1

            mort_turnover_total = float(np.sum(np.maximum(0.0, allocation)))
            if mort_turnover_total > 0.0:
                new_mask = allocation > 1e-9
                hh.housing_escrow[new_mask] = acquired_house_value[new_mask]
                mort[new_mask] = allocation[new_mask]
                hh.mort_rate_q[new_mask] = new_rate_q
                hh.mort_term_q[new_mask] = new_term_q
                hh.mort_age_q[new_mask] = 0.0
                # Turnover creates a fresh mortgage on the next housing-finance event.
                hh.mort_payment_sched_q[new_mask] = new_schedule.payment_from_orig_principal(allocation[new_mask])
                hh.mort_orig_principal[new_mask] = allocation[new_mask]
                hh.mort_t0[new_mask] = -1
                renter_rent_q[new_mask] = 0.0
                self.nodes["BANK"].add("loan_assets", mort_turnover_total)
                self.state["mort_turnover_total"] = float(mort_turnover_total)
                self.state["mort_turnover_households"] = float(np.sum(allocation > 1e-9))
            turnover_deposit_delta = float(deposits.sum()) - hh_deposits_before_turnover
            if abs(turnover_deposit_delta) > 1e-12:
                # In the bookkeeping-only housing-finance build, turnover settlement closes
                # through direct household deposit mutations. Mirror that exact net change on
                # the bank liability side instead of assuming full principal passes through escrow.
                self.nodes["BANK"].add("deposit_liab", turnover_deposit_delta)
            self.state["mort_turnover_old_payoff_total"] = float(old_mort_turnover_total)
            self.state["mort_turnover_deposit_delta_total"] = float(turnover_deposit_delta)
            self.state["mort_turnover_self_net_dep_total"] = float(self_turnover_net_total)
            self.state["mort_turnover_renter_downpayment_total"] = float(matched_renter_downpayment_total)
            self.state["mort_turnover_seller_net_total"] = float(matched_seller_net_total)
            self.state["mortgage_turnover_renter_entry_count"] = float(renter_entry_count)
            self.state["mortgage_turnover_supply_released_count"] = float(supply_values.size)

        self._invalidate_mortgage_contract_state()
        self._refresh_mortgage_contract_state()

        # 7) Household overdrafts -> revolving loans
        # -------------------------------------------------
        neg_mask = deposits < 0.0
        overdraft_total = 0.0
        if np.any(neg_mask):
            need = -deposits[neg_mask]
            deposits[neg_mask] = 0.0
            rev[neg_mask] = rev[neg_mask] + need
            overdraft_total = float(need.sum())
        self.state["hh_overdraft_total"] = float(overdraft_total)

        if overdraft_total > 0:
            self.nodes["BANK"].add("loan_assets", overdraft_total)
            self.nodes["BANK"].add("deposit_liab", overdraft_total)

        # FUND overdraft (rare)
        fund_dep = float(self.nodes["FUND"].get("deposits", 0.0))
        if fund_dep < 0:
            self._create_loan("FUND", -fund_dep, memo_tag="fund_overdraft_credit")

        # -------------------------------------------------
        # 8) Sync aggregate HH node + store memos
        # -------------------------------------------------
        hh_total_dep = float(deposits.sum())
        hh_total_loan = float(mort.sum() + rev.sum())

        self.nodes["HH"].set("deposits", hh_total_dep)
        self.nodes["HOUSING"].set(
            "deposits",
            float(self.state.get("housing_financing_deposits_total", 0.0)),
        )
        self.nodes["HH"].set("loans", hh_total_loan)

        if y_vec.shape[0] == n:
            hh.prev_income = (y_vec + mort_unpaid_cash_shortfall_i).astype(float, copy=True)
        hh.prev_uis = float(uis)
        hh.prev_wages_total = float(sol.get("w_total", 0.0))

        # 8b) Store lagged firm/bank earnings, installation queues, and next-quarter dividend commitments.
        self.nodes["FA"].memo["retained_prev"] = float(sol.get("retained_fa", 0.0))
        self.nodes["FH"].memo["retained_prev"] = float(sol.get("retained_fh", 0.0))
        bank_neutralize_interest_inflow = float(max(0.0, self.state.get("bank_mort_neutralize_interest_inflow", 0.0)))
        bank_retained_total = float(sol.get("retained_bk", 0.0)) + bank_neutralize_interest_inflow
        self.nodes["BANK"].memo["retained_prev"] = float(bank_retained_total)
        self.state["sector_capex_queue_info_nom"] = float(sol.get("capex_queue_info_next", 0.0))
        self.state["sector_capex_queue_phys_nom"] = float(sol.get("capex_queue_phys_next", 0.0))
        p_now = max(1e-9, float(self.state.get("price_level", 1.0)))
        payout_firms_info = self._update_sector_payout_rate("FA")
        payout_firms_phys = self._update_sector_payout_rate("FH")
        payout_bank = max(0.0, min(1.0, float(self.params.get("dividend_payout_rate_bank", 1.0))))
        service_floor = max(1e-9, float(self.params.get("sector_dividend_service_floor", 0.95)))
        service_ratio_info = min(1.0, float(sol.get("hh_sales_fa_real", 0.0)) / max(1e-9, float(sol.get("hh_demand_fa_real", 0.0)))) if float(sol.get("hh_demand_fa_real", 0.0)) > 1e-12 else 1.0
        service_ratio_phys = min(1.0, float(sol.get("hh_sales_fh_real", 0.0)) / max(1e-9, float(sol.get("hh_demand_fh_real", 0.0)))) if float(sol.get("hh_demand_fh_real", 0.0)) > 1e-12 else 1.0
        stress_scale_info = max(0.0, min(1.0, service_ratio_info / service_floor))
        stress_scale_phys = max(0.0, min(1.0, service_ratio_phys / service_floor))
        profit_dividend_base_info = self._sector_profit_distributable_nom("FA", float(sol.get("p_fa", 0.0)), p_now)
        profit_dividend_base_phys = self._sector_profit_distributable_nom("FH", float(sol.get("p_fh", 0.0)), p_now)
        surplus_dist_info = self._sector_surplus_distribution_nom("FA", p_now)
        surplus_dist_phys = self._sector_surplus_distribution_nom("FH", p_now)
        self.nodes["FA"].memo["dividend_commit_prev"] = (payout_firms_info * profit_dividend_base_info * stress_scale_info) + surplus_dist_info
        self.nodes["FH"].memo["dividend_commit_prev"] = (payout_firms_phys * profit_dividend_base_phys * stress_scale_phys) + surplus_dist_phys
        self.nodes["BANK"].memo["dividend_commit_prev"] = payout_bank * float(sol.get("bank_profit", 0.0))
        self.nodes["FA"].memo["revenue_prev"] = float(sol.get("rev_fa", 0.0))
        self.nodes["FH"].memo["revenue_prev"] = float(sol.get("rev_fh", 0.0))

        self.state["sector_capacity_info_real_prev"] = float(capacity_fa_real)
        self.state["sector_capacity_phys_real_prev"] = float(capacity_fh_real)
        self.state["fund_dividend_ownership_fa_prev"] = float(max(0.0, min(1.0, self.nodes["FUND"].get("shares_FA", 0.0) / max(1e-9, self.nodes["FA"].get("shares_outstanding", 0.0)))))
        self.state["fund_dividend_ownership_fh_prev"] = float(max(0.0, min(1.0, self.nodes["FUND"].get("shares_FH", 0.0) / max(1e-9, self.nodes["FH"].get("shares_outstanding", 0.0)))))
        self.state["fund_dividend_ownership_bk_prev"] = float(max(0.0, min(1.0, self.nodes["FUND"].get("shares_BANK", 0.0) / max(1e-9, self.nodes["BANK"].get("shares_outstanding", 0.0)))))
        unmet_info_real = float(max(0.0, hh_demand_fa_real - hh_sales_fa_real))
        unmet_phys_real = float(max(0.0, hh_demand_fh_real - hh_sales_fh_real))
        load_info_real = (
            float(sol.get("hh_demand_fa_real", 0.0))
            + float(sol.get("supplier_sales_fa_real", 0.0))
            + (float(sol.get("ums_recycle_fa_nom", 0.0)) / p_now)
        )
        load_phys_real = (
            float(sol.get("hh_demand_fh_real", 0.0))
            + float(sol.get("supplier_sales_fh_real", 0.0))
            + (float(sol.get("ums_recycle_fh_nom", 0.0)) / p_now)
        )
        load_gap_info_real = float(max(0.0, load_info_real - capacity_fa_real))
        load_gap_phys_real = float(max(0.0, load_phys_real - capacity_fh_real))
        self.state["sector_unmet_info_real_prev"] = float(unmet_info_real)
        self.state["sector_unmet_phys_real_prev"] = float(unmet_phys_real)
        self.state["sector_load_gap_info_real_prev"] = float(load_gap_info_real)
        self.state["sector_load_gap_phys_real_prev"] = float(load_gap_phys_real)
        unmet_alpha = max(0.0, min(1.0, float(self.params.get("sector_capex_unmet_ewma_alpha", 1.0))))
        prev_unmet_info_sm = float(self.state.get("sector_unmet_info_real_sm_prev", unmet_info_real))
        prev_unmet_phys_sm = float(self.state.get("sector_unmet_phys_real_sm_prev", unmet_phys_real))
        prev_load_gap_info_sm = float(self.state.get("sector_load_gap_info_real_sm_prev", load_gap_info_real))
        prev_load_gap_phys_sm = float(self.state.get("sector_load_gap_phys_real_sm_prev", load_gap_phys_real))
        if int(self.state.get("t", 0)) <= 0:
            next_unmet_info_sm = unmet_info_real
            next_unmet_phys_sm = unmet_phys_real
            next_load_gap_info_sm = load_gap_info_real
            next_load_gap_phys_sm = load_gap_phys_real
        else:
            next_unmet_info_sm = (unmet_alpha * unmet_info_real) + ((1.0 - unmet_alpha) * prev_unmet_info_sm)
            next_unmet_phys_sm = (unmet_alpha * unmet_phys_real) + ((1.0 - unmet_alpha) * prev_unmet_phys_sm)
            next_load_gap_info_sm = (unmet_alpha * load_gap_info_real) + ((1.0 - unmet_alpha) * prev_load_gap_info_sm)
            next_load_gap_phys_sm = (unmet_alpha * load_gap_phys_real) + ((1.0 - unmet_alpha) * prev_load_gap_phys_sm)
        self.state["sector_unmet_info_real_sm_prev"] = float(max(0.0, next_unmet_info_sm))
        self.state["sector_unmet_phys_real_sm_prev"] = float(max(0.0, next_unmet_phys_sm))
        self.state["sector_load_gap_info_real_sm_prev"] = float(max(0.0, next_load_gap_info_sm))
        self.state["sector_load_gap_phys_real_sm_prev"] = float(max(0.0, next_load_gap_phys_sm))
        self.state["sector_service_ratio_info_prev"] = float(service_ratio_info)
        self.state["sector_service_ratio_phys_prev"] = float(service_ratio_phys)
        self.state["sector_free_cash_info_prev"] = float(max(0.0, self.nodes["FA"].get("deposits", 0.0)))
        self.state["sector_free_cash_phys_prev"] = float(max(0.0, self.nodes["FH"].get("deposits", 0.0)))

        deposit_tol = 1e-6
        for firm_id in ("FA", "FH"):
            dep = float(self.nodes[firm_id].get("deposits", 0.0))
            if abs(dep) < deposit_tol:
                self.nodes[firm_id].set("deposits", 0.0)
            elif dep < 0.0:
                raise ValueError(f"{firm_id} ended tick with negative deposits under no-new-debt sector rules: {dep:.6f}")

        self._assert_sfc_ok(context=f"post_tick_population_t{self.state['t']}")

    # ---------------------------------------------------------
    # Phase E: One tick execution
    # ---------------------------------------------------------

    def step(self) -> None:
        # Set this quarter's policy rate from lagged inflation/DTI observables.
        self._update_policy_rate()

        # Automation path (levels + per-quarter flow for visualization)
        t = int(self.state["t"])
        automation_disabled = bool(self.params.get("automation_disabled", False))
        path = str(self.params.get("automation_path", "two_hump")).lower()

        if automation_disabled:
            self.state["automation"] = 0.0
            self.state["automation_flow"] = 0.0
            self.state["automation_info"] = 0.0
            self.state["automation_info_flow"] = 0.0
            self.state["automation_phys"] = 0.0
            self.state["automation_phys_flow"] = 0.0
        elif path == "linear":
            horizon_q = float(self.params.get("automation_horizon_quarters", 60.0))
            a = min(1.0, t / horizon_q) if horizon_q > 0 else 1.0
            a_prev = min(1.0, (t - 1) / horizon_q) if (horizon_q > 0 and t > 0) else 0.0
            self.state["automation"] = float(a)
            self.state["automation_flow"] = float(a - a_prev)
            self.state["automation_info"] = 0.0
            self.state["automation_info_flow"] = 0.0
            self.state["automation_phys"] = 0.0
            self.state["automation_phys_flow"] = 0.0
        else:
            # Two-hump defaults tuned for a ~15-year (60-quarter) horizon.
            w_info = float(self.params.get("automation_w_info", 0.65))
            ki = float(self.params.get("automation_ki", 0.18))
            ti = float(self.params.get("automation_ti", 12.0))
            bi = float(self.params.get("automation_bi", 4.0))
            kp = float(self.params.get("automation_kp", 0.14))
            tp = float(self.params.get("automation_tp", 32.0))

            info_cap = float(self.params.get("automation_info_cap", 1.0))
            phys_cap = float(self.params.get("automation_phys_cap", 1.0))
            res = automation_two_hump(t, w_info=w_info, ki=ki, ti=ti, bi=bi, kp=kp, tp=tp, info_cap=info_cap, phys_cap=phys_cap)
            self.state["automation"] = float(res["level"])
            self.state["automation_flow"] = float(res["flow"])
            self.state["automation_info"] = float(res["info_level"])
            self.state["automation_info_flow"] = float(res["info_flow"])
            self.state["automation_phys"] = float(res["phys_level"])
            self.state["automation_phys_flow"] = float(res["phys_flow"])
        self.state["sector_tfp_mult_info"] = float(self._sector_tfp_multiplier("FA"))
        self.state["sector_tfp_mult_phys"] = float(self._sector_tfp_multiplier("FH"))

        # Price level
        # Competitive pass-through: P_comp = P0 / (1 + beta*A)
        # Corporate capture via markup: P = P_comp * (1 + mu(A))
        p0 = float(self.params.get("price_level_initial", 1.0))
        beta = float(self.params.get("price_beta", 1.0))

        mu_max = float(self.params.get("automation_markup_max", 0.0))
        mu_pow = float(self.params.get("automation_markup_power", 1.0))
        if mu_max < 0:
            mu_max = 0.0
        if mu_pow <= 0:
            mu_pow = 1.0

        P_prev = float(self.state.get("price_level", p0))
        A = float(self.state["automation"])

        # Capital deepening: separate from the automation *share*.
        # Automation caps represent an irreducible non-automatable share.
        # Capital accumulation should raise productivity (lower prices / raise real output) but
        # should NOT push the automation *share* beyond those caps.
        denom = float(self.hh.n) if (self.hh is not None and bool(self.params.get("use_population", False))) else 1.0
        K_total = float(self.nodes["FA"].get("K", 0.0)) + float(self.nodes["FH"].get("K", 0.0))
        K_per_h = (K_total / denom) if denom > 0 else 0.0

        kappa = float(self.params.get("capital_productivity_k", 0.0))
        K_scale = float(self.params.get("capital_productivity_scale", 1.0))
        if K_scale <= 0:
            K_scale = 1.0

        # Productivity multiplier (>=1). Capital deepening and the optional sector TFP experiment
        # both feed the competitive price channel, so extra productivity can show up as lower prices
        # rather than only as unused slack capacity.
        prod_mult = 1.0 + (kappa * (K_per_h / K_scale))
        tfp_mult_info = float(self.state.get("sector_tfp_mult_info", 1.0))
        tfp_mult_phys = float(self.state.get("sector_tfp_mult_phys", 1.0))
        w_info = max(0.0, min(1.0, float(self.params.get("automation_w_info", 0.65))))
        tfp_mult_agg = (w_info * tfp_mult_info) + ((1.0 - w_info) * tfp_mult_phys)
        if tfp_mult_agg < 1.0:
            tfp_mult_agg = 1.0
        prod_mult *= tfp_mult_agg
        if prod_mult < 1.0:
            prod_mult = 1.0
        self.state["capital_productivity_mult"] = float(prod_mult)

        # Effective automation used for sector split and wage-share rules remains the automation share A.
        # (A is already capped by sector caps in the two-hump path.)
        self.state["automation_eff"] = float(A)

        # Price competition channel uses automation *and* capital deepening.
        # Interpretation: higher A and higher prod_mult both lower the competitive price level.
        # A price-adjustment speed < 1.0 adds short-run stickiness in pass-through.
        A_eff = float(self.state.get("automation_eff", A))
        P_comp = p0 / (1.0 + beta * (A_eff * prod_mult))
        mu = mu_max * (A_eff ** mu_pow)
        P_target = P_comp * (1.0 + mu)

        price_adjust_speed = float(self.params.get("price_adjust_speed", 1.0))
        price_adjust_speed = max(0.0, min(1.0, price_adjust_speed))
        P = (1.0 - price_adjust_speed) * P_prev + price_adjust_speed * P_target
        if P <= 0:
            P = 1e-9

        self.state["price_level"] = float(P)
        self.state["inflation"] = float((P / P_prev - 1.0) if P_prev > 0 else 0.0)
        self.state["automation_markup"] = float(mu)
        self.state["price_target"] = float(P_target)
        self.state["capital_productivity_mult"] = float(self.state.get("capital_productivity_mult", 1.0))

        # Startup warm initialization for lagged CAPEX state before first recorded quarter.
        self._bootstrap_startup_lagged_retained()

        # Trigger + dilution at start of tick
        self.apply_policy_logic()
        self.issue_social_shares()

        use_pop_dyn = bool(self.params.get("use_population", False)) and bool(self.params.get("population_dynamics", False)) and (self.hh is not None)

        if use_pop_dyn:
            self._refresh_mortgage_contract_state()
            # Allow policy modules to initialize first-tick anchors before solving.
            if not self._income_support_disabled():
                self.income_support_policy.warm_start_anchor_if_needed(
                    state=self.state,
                    baseline_wages_i=self.hh.wages0_q,
                    price_level=float(self.state.get("price_level", 1.0)),
                )

            solp = self.solve_within_tick_population()
            if solp is None:
                raise RuntimeError("Population dynamics are enabled but solve_within_tick_population() returned None.")
            self.post_tick_population(solp)

            # Let the active income-support policy initialize any one-time baseline anchor.
            if not self._income_support_disabled():
                self.income_support_policy.initialize_anchor_if_needed(
                    state=self.state,
                    wages_total=float(solp["w_total"]),
                    support_per_h=float(solp["uis"]),
                    n_households=int(self.hh.n),
                    price_level=float(self.state.get("price_level", 1.0)),
                    wages_i=solp.get("wages_i"),
                    div_i=solp.get("div_i"),
                )

            # Population inequality diagnostics (vectorized)
            y_vec = _as_np(solp.get("y", []), dtype=float)             # disposable income
            wages_i = _as_np(solp.get("wages_i", []), dtype=float)     # market component
            div_i = _as_np(solp.get("div_i", []), dtype=float)         # market component

            market_inc = wages_i + div_i if (wages_i.size and div_i.size and wages_i.size == div_i.size) else np.asarray([], dtype=float)

            gini_market = calculate_gini_np(market_inc) if market_inc.size else 0.0
            gini_disp = calculate_gini_np(y_vec) if y_vec.size else 0.0

            # Net-wealth Gini proxy:
            #   wealth_i = deposits_i + allocated_hh_equity_i + allocated_trust_value_i - loans_i
            # Direct household equity claims are allocated by baseline wage weights because
            # ownership is tracked at HH aggregate. Trust value is split equally per household.
            dep_i = _as_np(self.hh.deposits, dtype=float)
            housing_i = _as_np(self.hh.housing_escrow, dtype=float)
            mort_i = _as_np(self.hh.mortgage_loans, dtype=float)
            rev_i = _as_np(self.hh.revolving_loans, dtype=float)
            loan_i = mort_i + rev_i
            active_mort_i = mort_i > 1e-9
            mort_orig_principal_i = _as_np(self.hh.mort_orig_principal, dtype=float)
            active_mort_orig_principal_total = (
                float(np.sum(mort_orig_principal_i[active_mort_i]))
                if mort_orig_principal_i.shape[0] == mort_i.shape[0]
                else float(np.sum(mort_i[active_mort_i]))
            )

            private_equity_total = 0.0
            if dep_i.size and (dep_i.size == loan_i.size):
                w0_wealth = _as_np(self.hh.wages0_q, dtype=float)
                w0_wealth_sum = float(w0_wealth.sum()) if w0_wealth.size == dep_i.size else 0.0
                if w0_wealth_sum > 0.0:
                    wealth_weights = w0_wealth / w0_wealth_sum
                else:
                    wealth_weights = np.full(dep_i.size, 1.0 / float(dep_i.size), dtype=float)

                P_wealth = float(self.state.get("price_level", 1.0))
                if P_wealth <= 0:
                    P_wealth = 1e-9

                def node_share_frac(holder: str, issuer: str, key: str) -> float:
                    so = float(self.nodes[issuer].get("shares_outstanding", 0.0))
                    if so <= 0.0:
                        return 0.0
                    frac = float(self.nodes[holder].get(key, 0.0)) / so
                    return max(0.0, min(1.0, frac))

                fa_equity_proxy = self._firm_balance_sheet_equity_proxy("FA", P_wealth)
                fh_equity_proxy = self._firm_balance_sheet_equity_proxy("FH", P_wealth)
                bank_equity_proxy = self._firm_balance_sheet_equity_proxy("BANK", P_wealth)

                hh_equity_total = (
                    node_share_frac("HH", "FA", "shares_FA") * fa_equity_proxy
                    + node_share_frac("HH", "FH", "shares_FH") * fh_equity_proxy
                    + node_share_frac("HH", "BANK", "shares_BANK") * bank_equity_proxy
                )
                trust_equity_total = (
                    node_share_frac("FUND", "FA", "shares_FA") * fa_equity_proxy
                    + node_share_frac("FUND", "FH", "shares_FH") * fh_equity_proxy
                    + node_share_frac("FUND", "BANK", "shares_BANK") * bank_equity_proxy
                )
                trust_value_total = (
                    float(self.nodes["FUND"].get("deposits", 0.0))
                    + trust_equity_total
                    - float(self.nodes["FUND"].get("loans", 0.0))
                )
                private_equity_total = float(max(0.0, hh_equity_total))
                equity_i = wealth_weights * hh_equity_total
                trust_i = np.full(dep_i.size, trust_value_total / float(dep_i.size), dtype=float)
                wealth_i = dep_i + housing_i + equity_i + trust_i - loan_i
                gini_wealth = calculate_gini_np(wealth_i)
            else:
                gini_wealth = 0.0

            # Private-capital diagnostics (all nominal, population totals unless noted).
            priv_payout_total = float(max(0.0, float(solp.get("div_house_total", 0.0))))
            prev_priv_eq_total = float(self.state.get("private_equity_prev_total", 0.0))
            private_roe_q = (priv_payout_total / prev_priv_eq_total) if prev_priv_eq_total > 1e-9 else 0.0

            retained_fa = float(solp.get("retained_fa", 0.0))
            retained_fh = float(solp.get("retained_fh", 0.0))
            bank_neutralize_interest_inflow = float(max(0.0, self.state.get("bank_mort_neutralize_interest_inflow", 0.0)))
            retained_bk = float(solp.get("retained_bk", 0.0)) + bank_neutralize_interest_inflow
            f_fa = float(solp.get("f_fa", 0.0))
            f_fh = float(solp.get("f_fh", 0.0))
            f_bk = float(solp.get("f_bk", 0.0))
            private_retained_total = (
                retained_fa * (1.0 - f_fa)
                + retained_fh * (1.0 - f_fh)
                + retained_bk * (1.0 - f_bk)
            )
            private_broad_roe_q = (
                (priv_payout_total + private_retained_total) / prev_priv_eq_total
                if prev_priv_eq_total > 1e-9 else 0.0
            )

            capex_total_nom = float(solp.get("capex_total_nom", 0.0))
            private_inv_cov = (private_retained_total / capex_total_nom) if capex_total_nom > 1e-9 else 0.0
            self.state["private_equity_prev_total"] = float(max(0.0, private_equity_total))

            def frac(issuer: str, key: str) -> float:
                so = self.nodes[issuer].get("shares_outstanding", 1.0)
                return self.nodes["FUND"].get(key, 0.0) / so if so > 0 else 0.0

            own_avg = (frac("FA", "shares_FA") + frac("FH", "shares_FH") + frac("BANK", "shares_BANK")) / 3.0

            # Population mortgage-burden percentiles directly from solver components (vectorized)
            uis = float(solp.get("uis", 0.0))
            dti_sol = solp
            if int(self.state.get("t", 0)) == 0:
                sol_diag = self.solve_within_tick_population(allow_income_support_trigger=False)
                if sol_diag is not None:
                    dti_sol = sol_diag

            dti_wages_i = _as_np(dti_sol.get("wages_i", []), dtype=float)
            dti_div_i = _as_np(dti_sol.get("div_i", []), dtype=float)
            interest_hh = _as_np(dti_sol.get("interest_hh", []), dtype=float)
            mort_pay_req_i = _as_np(dti_sol.get("mort_pay_req_i", []), dtype=float)
            income_tax_i = _as_np(dti_sol.get("income_tax_i", []), dtype=float)
            vat_credit_i = _as_np(dti_sol.get("vat_credit_i", []), dtype=float)
            dti_uis = float(dti_sol.get("uis", 0.0))

            if interest_hh.size and dti_wages_i.size and interest_hh.size == dti_wages_i.size:
                gross = dti_wages_i + dti_uis

                # Income percentiles: everyone with gross income
                incs = (gross - interest_hh)[gross > 0]

                if (
                    mort_pay_req_i.size
                    and mort_pay_req_i.size == dti_wages_i.size
                    and income_tax_i.size
                    and income_tax_i.size == dti_wages_i.size
                    and vat_credit_i.size
                    and vat_credit_i.size == dti_wages_i.size
                ):
                    disp_pre_debt_i = dti_wages_i + float(dti_uis) + dti_div_i + vat_credit_i - income_tax_i
                    # IMPORTANT: apply the mask *before* dividing to avoid divide-by-zero / invalid warnings.
                    dti_mask = (mort_pay_req_i > 0) & (disp_pre_debt_i > 0)
                    dtis = (
                        mort_pay_req_i[dti_mask] / disp_pre_debt_i[dti_mask]
                    ) if np.any(dti_mask) else np.asarray([], dtype=float)

                    wage_dti_mask = (mort_pay_req_i > 0) & (dti_wages_i > 0)
                    dtis_w = (
                        mort_pay_req_i[wage_dti_mask] / dti_wages_i[wage_dti_mask]
                    ) if np.any(wage_dti_mask) else np.asarray([], dtype=float)
                else:
                    dtis = np.asarray([], dtype=float)
                    dtis_w = np.asarray([], dtype=float)

                pop_dti_med = _pct_np(dtis, 50.0) if dtis.size else 0.0
                pop_dti_p90 = _pct_np(dtis, 90.0) if dtis.size else 0.0
                pop_dti_w_med = _pct_np(dtis_w, 50.0) if dtis_w.size else 0.0
                pop_dti_w_p90 = _pct_np(dtis_w, 90.0) if dtis_w.size else 0.0
                pop_inc_med = _pct_np(incs, 50.0) if incs.size else 0.0
                pop_inc_p90 = _pct_np(incs, 90.0) if incs.size else 0.0
            else:
                pop_dti_med = 0.0
                pop_dti_p90 = 0.0
                pop_dti_w_med = 0.0
                pop_dti_w_p90 = 0.0
                pop_inc_med = 0.0
                pop_inc_p90 = 0.0

            P_now = float(self.state.get("price_level", 1.0))
            if P_now <= 0:
                P_now = 1e-9

            # Trust value proxy (nominal): FUND deposits + FUND equity claims - FUND debt.
            fa_equity_proxy_hist = self._firm_balance_sheet_equity_proxy("FA", P_now)
            fh_equity_proxy_hist = self._firm_balance_sheet_equity_proxy("FH", P_now)
            bank_equity_proxy_hist = self._firm_balance_sheet_equity_proxy("BANK", P_now)
            fa_broad_equity_proxy_hist = self._firm_broad_equity_proxy("FA", P_now)
            fh_broad_equity_proxy_hist = self._firm_broad_equity_proxy("FH", P_now)
            trust_equity_value_total = (
                frac("FA", "shares_FA") * fa_equity_proxy_hist
                + frac("FH", "shares_FH") * fh_equity_proxy_hist
                + frac("BANK", "shares_BANK") * bank_equity_proxy_hist
            )
            trust_value_total = (
                float(self.nodes["FUND"].get("deposits", 0.0))
                + float(trust_equity_value_total)
                - float(self.nodes["FUND"].get("loans", 0.0))
            )

            total_corporate_equity_total = float(fa_broad_equity_proxy_hist + fh_broad_equity_proxy_hist + bank_equity_proxy_hist)
            prev_corporate_eq_total = float(self.state.get("corporate_equity_prev_total", 0.0))
            prev_bank_eq_total = float(self.state.get("corporate_bank_equity_prev_total", 0.0))
            prev_info_eq_total = float(self.state.get("corporate_info_equity_prev_total", 0.0))
            prev_physical_eq_total = float(self.state.get("corporate_physical_equity_prev_total", 0.0))
            prev_nonbank_eq_total = float(self.state.get("corporate_nonbank_equity_prev_total", 0.0))
            total_corporate_payout_total = (
                float(solp.get("div_fa_total", 0.0))
                + float(solp.get("div_fh_total", 0.0))
                + float(solp.get("div_bk_total", 0.0))
            )
            info_corporate_payout_total = float(solp.get("div_fa_total", 0.0))
            physical_corporate_payout_total = float(solp.get("div_fh_total", 0.0))
            bank_corporate_payout_total = float(solp.get("div_bk_total", 0.0))
            nonbank_corporate_payout_total = info_corporate_payout_total + physical_corporate_payout_total
            total_corporate_retained_total = retained_fa + retained_fh + retained_bk
            info_corporate_retained_total = retained_fa
            physical_corporate_retained_total = retained_fh
            bank_corporate_retained_total = retained_bk
            nonbank_corporate_retained_total = info_corporate_retained_total + physical_corporate_retained_total
            bank_broad_roe_q = (
                (bank_corporate_payout_total + bank_corporate_retained_total) / prev_bank_eq_total
                if prev_bank_eq_total > 1e-9 else 0.0
            )
            rev_info = float(solp.get("rev_fa", 0.0))
            rev_phys = float(solp.get("rev_fh", 0.0))
            sector_op_margin_info = (
                (
                    rev_info
                    - float(solp.get("w_fa", 0.0))
                    - float(solp.get("overhead_fa", 0.0))
                    - float(solp.get("input_cost_fa", 0.0))
                ) / rev_info
                if rev_info > 1e-9 else 0.0
            )
            sector_op_margin_phys = (
                (
                    rev_phys
                    - float(solp.get("w_fh", 0.0))
                    - float(solp.get("overhead_fh", 0.0))
                    - float(solp.get("input_cost_fh", 0.0))
                ) / rev_phys
                if rev_phys > 1e-9 else 0.0
            )
            corporate_info_broad_roe_q = (
                (info_corporate_payout_total + info_corporate_retained_total) / prev_info_eq_total
                if prev_info_eq_total > 1e-9 else 0.0
            )
            corporate_physical_broad_roe_q = (
                (physical_corporate_payout_total + physical_corporate_retained_total) / prev_physical_eq_total
                if prev_physical_eq_total > 1e-9 else 0.0
            )
            corporate_nonbank_broad_roe_q = (
                (nonbank_corporate_payout_total + nonbank_corporate_retained_total) / prev_nonbank_eq_total
                if prev_nonbank_eq_total > 1e-9 else 0.0
            )
            corporate_broad_roe_q = (
                (total_corporate_payout_total + total_corporate_retained_total) / prev_corporate_eq_total
                if prev_corporate_eq_total > 1e-9 else 0.0
            )
            self.state["corporate_equity_prev_total"] = float(max(0.0, total_corporate_equity_total))
            self.state["corporate_bank_equity_prev_total"] = float(max(0.0, bank_equity_proxy_hist))
            self.state["corporate_info_equity_prev_total"] = float(max(0.0, fa_broad_equity_proxy_hist))
            self.state["corporate_physical_equity_prev_total"] = float(max(0.0, fh_broad_equity_proxy_hist))
            self.state["corporate_nonbank_equity_prev_total"] = float(max(0.0, fa_broad_equity_proxy_hist + fh_broad_equity_proxy_hist))

            wages_total = float(solp["w_total"])
            c_total = float(solp["c_total"])

            real_avg_income = float(np.mean(y_vec) / P_now) if y_vec.size else float(((wages_total / float(self.hh.n)) + float(uis)) / P_now)

            self.history.append(TickResult(
                t=self.state["t"],
                automation=float(self.state["automation"]),
                automation_flow=float(self.state.get("automation_flow", 0.0)),
                automation_info=float(self.state.get("automation_info", 0.0)),
                automation_info_flow=float(self.state.get("automation_info_flow", 0.0)),
                automation_phys=float(self.state.get("automation_phys", 0.0)),
                automation_phys_flow=float(self.state.get("automation_phys_flow", 0.0)),
                sector_tfp_mult_info=float(self.state.get("sector_tfp_mult_info", 1.0)),
                sector_tfp_mult_physical=float(self.state.get("sector_tfp_mult_phys", 1.0)),
                price_level=float(self.state.get("price_level", 1.0)),
                inflation=float(self.state.get("inflation", 0.0)),
                gini=float(gini_disp),
                gini_market=float(gini_market),
                gini_disp=float(gini_disp),
                gini_wealth=float(gini_wealth),
                private_eq_per_h=float(private_equity_total) / float(self.hh.n),
                hh_deposits_per_h=float(np.sum(dep_i)) / float(self.hh.n),
                hh_housing_value_per_h=float(np.sum(housing_i)) / float(self.hh.n),
                hh_debt_per_h=float(np.sum(loan_i)) / float(self.hh.n),
                hh_mortgage_debt_per_h=float(np.sum(mort_i)) / float(self.hh.n),
                hh_revolving_debt_per_h=float(np.sum(rev_i)) / float(self.hh.n),
                hh_mortgage_balance_total=float(np.sum(mort_i)),
                hh_mortgage_orig_principal_total=float(active_mort_orig_principal_total),
                hh_mortgage_active_count=float(np.sum(active_mort_i)),
                corporate_eq_info_per_h=float(fa_broad_equity_proxy_hist) / float(self.hh.n),
                corporate_eq_physical_per_h=float(fh_broad_equity_proxy_hist) / float(self.hh.n),
                corporate_eq_total_per_h=float(fa_broad_equity_proxy_hist + fh_broad_equity_proxy_hist) / float(self.hh.n),
                private_roe_q=float(private_roe_q),
                private_broad_roe_q=float(private_broad_roe_q),
                bank_broad_roe_q=float(bank_broad_roe_q),
                corporate_info_broad_roe_q=float(corporate_info_broad_roe_q),
                corporate_physical_broad_roe_q=float(corporate_physical_broad_roe_q),
                sector_op_margin_info=float(sector_op_margin_info),
                sector_op_margin_phys=float(sector_op_margin_phys),
                corporate_nonbank_broad_roe_q=float(corporate_nonbank_broad_roe_q),
                corporate_broad_roe_q=float(corporate_broad_roe_q),
                private_inv_cov=float(private_inv_cov),
                # --- Fiscal / funding diagnostics (per household) ---
                vat_per_h=float(self.state.get("vat_receipts_total", 0.0)) / float(self.hh.n),
                inc_tax_per_h=float(self.state.get("income_tax_total", 0.0)) / float(self.hh.n),
                corp_tax_per_h=float(self.state.get("corp_tax_total", 0.0)) / float(self.hh.n),
                corp_tax_rate_eff=float(self.state.get("corp_tax_rate_eff", self.params.get("corporate_tax_rate", 0.0))),
                vat_credit_per_h=float(self.state.get("vat_credit_total", 0.0)) / float(self.hh.n),
                gov_dep_per_h=float(self.nodes["GOV"].get("deposits", 0.0)) / float(self.hh.n),
                fund_dep_per_h=float(self.nodes["FUND"].get("deposits", 0.0)) / float(self.hh.n),
                fund_dividend_inflow_per_h=float(solp.get("div_fund", 0.0)) / float(self.hh.n),
                ums_drain_to_fund_per_h=float(self.state.get("ums_drain_to_fund_total", 0.0)) / float(self.hh.n),
                fund_tracked_inflows_per_h=(
                    float(solp.get("div_fund", 0.0)) + float(self.state.get("ums_drain_to_fund_total", 0.0))
                ) / float(self.hh.n),
                ums_recycle_to_info_per_h=float(self.state.get("ums_recycle_to_info_total", 0.0)) / float(self.hh.n),
                ums_recycle_to_phys_per_h=float(self.state.get("ums_recycle_to_phys_total", 0.0)) / float(self.hh.n),
                ums_recycle_total_per_h=float(self.state.get("ums_recycle_total", 0.0)) / float(self.hh.n),
                capex_per_h=float(self.state.get("capex_total", 0.0)) / float(self.hh.n),
                sector_capacity_info_per_h=float(solp.get("capacity_fa_real", 0.0)) / float(self.hh.n),
                sector_capacity_physical_per_h=float(solp.get("capacity_fh_real", 0.0)) / float(self.hh.n),
                sector_hh_util_info=(
                    float(solp.get("hh_sales_fa_real", 0.0))
                    / max(1e-9, float(solp.get("capacity_fa_real", 0.0)))
                ),
                sector_hh_util_physical=(
                    float(solp.get("hh_sales_fh_real", 0.0))
                    / max(1e-9, float(solp.get("capacity_fh_real", 0.0)))
                ),
                sector_util_info=(
                    (
                        float(solp.get("hh_sales_fa_real", 0.0))
                        + float(solp.get("supplier_sales_fa_real", 0.0))
                        + (float(solp.get("ums_recycle_fa_nom", 0.0)) / max(1e-9, float(self.state.get("price_level", 1.0))))
                    )
                    / max(1e-9, float(solp.get("capacity_fa_real", 0.0)))
                ),
                sector_util_physical=(
                    (
                        float(solp.get("hh_sales_fh_real", 0.0))
                        + float(solp.get("supplier_sales_fh_real", 0.0))
                        + (float(solp.get("ums_recycle_fh_nom", 0.0)) / max(1e-9, float(self.state.get("price_level", 1.0))))
                    )
                    / max(1e-9, float(solp.get("capacity_fh_real", 0.0)))
                ),
                sector_demand_info_per_h=float(solp.get("hh_demand_fa_real", 0.0)) / float(self.hh.n),
                sector_demand_physical_per_h=float(solp.get("hh_demand_fh_real", 0.0)) / float(self.hh.n),
                unmet_demand_info_per_h=max(
                    0.0,
                    float(solp.get("hh_demand_fa_real", 0.0)) - float(solp.get("hh_sales_fa_real", 0.0)),
                ) / float(self.hh.n),
                unmet_demand_physical_per_h=max(
                    0.0,
                    float(solp.get("hh_demand_fh_real", 0.0)) - float(solp.get("hh_sales_fh_real", 0.0)),
                ) / float(self.hh.n),
                uis_per_h=float(uis),
                uis_from_fund_dep_per_h=float(self.state.get("uis_from_fund_dep_total", 0.0)) / float(self.hh.n),
                uis_from_gov_dep_per_h=float(self.state.get("uis_from_gov_dep_total", 0.0)) / float(self.hh.n),
                uis_issued_per_h=float(self.state.get("uis_issued_total", 0.0)) / float(self.hh.n),
                trust_equity_pct=float(own_avg),
                trust_debt=float(self.nodes["FUND"].get("loans", 0.0)),
                trust_value_per_h=float(trust_value_total) / float(self.hh.n),
                wages_total=wages_total,
                total_consumption=c_total,

                real_avg_income=real_avg_income,
                real_consumption=float(c_total / P_now),

                pop_gini=float(gini_disp),
                pop_inc_med=pop_inc_med,
                pop_inc_p90=pop_inc_p90,
                pop_dti_med=pop_dti_med,
                pop_dti_p90=pop_dti_p90,
                pop_dti_w_med=pop_dti_w_med,
                pop_dti_w_p90=pop_dti_w_p90,

                trust_active=bool(self.state["trust_active"]),
            ))

            self._record_balance_sheets(self.state["t"])
            # Advance simulation time (quarter counter)
            self.state["t"] = int(self.state["t"]) + 1

        else:
            raise ValueError(
                "Population-dynamics execution is disabled. "
                "Enable parameters['use_population']=True and parameters['population_dynamics']=True."
            )

    def _distribute_household_transfer_by_weights(
        self,
        deposits: np.ndarray,
        weights: np.ndarray,
        total_amount: float,
    ) -> None:
        amount = float(total_amount)
        if amount <= 0.0:
            return
        denom = float(np.sum(weights))
        if denom <= 0.0:
            return
        deposits[:] = deposits + (weights * (amount / denom))

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        params = config.get("parameters")
        nodes = config.get("nodes")
        if not isinstance(params, dict):
            raise TypeError("config['parameters'] must be a dict")
        if not isinstance(nodes, dict):
            raise TypeError("config['nodes'] must be a dict")
        if not bool(params.get("use_population", False)):
            raise ValueError("This build requires parameters['use_population']=True.")
        if not bool(params.get("population_dynamics", False)):
            raise ValueError("This build requires parameters['population_dynamics']=True.")
        automation_path = params.get("automation_path", "two_hump")
        if not isinstance(automation_path, str):
            raise TypeError("parameters['automation_path'] must be a string.")
        if automation_path.lower() not in {"two_hump", "linear"}:
            raise ValueError("parameters['automation_path'] must be 'two_hump' or 'linear'.")

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    def write_csv(self, filename: str = "run_history.csv") -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TickResult.__dataclass_fields__.keys())
            writer.writeheader()
            for r in self.history:
                writer.writerow(r.__dict__)
