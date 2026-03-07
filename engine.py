# Author: Roger Ison   roger@miximum.info
"""Economy model engine and stock-flow logic."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from mathutils import _as_np, _pct, _pct_np, automation_two_hump, calculate_gini_np
from newloop_types import HouseholdState, Node, TickResult

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
        self.params = config["parameters"]
        p0 = float(config["parameters"].get("price_level_initial", 1.0))
        base_rate_q = max(0.0, float(config["parameters"].get("loan_rate_per_quarter", 0.0)))
        self.state = {
            "t": 0,
            "automation": 0.0,
            "trust_active": False,

            # Price level state
            "price_level": p0,
            "inflation": 0.0,

            # Per-tick UBI funding diagnostics (set in post_tick)
            "ubi_from_fund_dep_total": 0.0,
            "ubi_from_gov_dep_total": 0.0,
            "ubi_issued_total": 0.0,
            "tax_rebate_total": 0.0,
            # Lagged private equity stock used for private payout-yield proxy.
            "private_equity_prev_total": 0.0,
            # One-time guard for startup lag bootstrap.
            "startup_bootstrap_done": False,
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
            "mort_index_mean": 1.0,
            "mort_index_min": 1.0,
            "mort_index_max": 1.0,
            "mort_overdraft_due_to_payment_total": 0.0,
            "mort_overdraft_due_to_payment_count": 0.0,
        }

        self.nodes: Dict[str, Node] = {
            nid: Node(nid, nd.get("stocks", {}).copy(), nd.get("memo", {}).copy())
            for nid, nd in config["nodes"].items()
        }

        # Ensure required nodes exist (population-mode core only)
        for required in ["BANK", "FUND", "FA", "FH", "GOV", "HH"]:
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
            import population as pop_mod
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
                mortgage_loans_raw = getattr(pop, "mortgage_loans")
                revolving_loans_raw = getattr(pop, "revolving_loans")
                mpc_q_raw = getattr(pop, "mpc_q")
                base_real_cons_q_raw = getattr(pop, "base_real_cons_q")
            except Exception:
                wages0_q_raw = []
                deposits_raw = []
                mortgage_loans_raw = []
                revolving_loans_raw = []
                mpc_q_raw = []
                base_real_cons_q_raw = []

            wages0_q = _as_np(wages0_q_raw, dtype=float)
            deposits = _as_np(deposits_raw, dtype=float)
            mortgage_loans = _as_np(mortgage_loans_raw, dtype=float)
            revolving_loans = _as_np(revolving_loans_raw, dtype=float)
            mpc_q = _as_np(mpc_q_raw, dtype=float)
            base_real_cons_q = _as_np(base_real_cons_q_raw, dtype=float)

            n = int(min(
                wages0_q.shape[0],
                deposits.shape[0],
                mortgage_loans.shape[0],
                revolving_loans.shape[0],
                mpc_q.shape[0],
                base_real_cons_q.shape[0],
            ))

            if n > 0:
                self.hh = HouseholdState(
                    n=n,
                    wages0_q=wages0_q[:n].copy(),
                    deposits=deposits[:n].copy(),
                    mortgage_loans=mortgage_loans[:n].copy(),
                    revolving_loans=revolving_loans[:n].copy(),
                    mpc_q=mpc_q[:n].copy(),
                    base_real_cons_q=base_real_cons_q[:n].copy(),
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
                self.nodes["HH"].set("loans", self.hh.sum_loans())
                p_series0 = self._mort_price_series_value(float(self.state.get("price_level", p0)))
                y_series0 = self._mort_income_series_value(float(np.sum(self.hh.wages0_q)), 0.0, float(self.hh.prev_ubi))
                self.state["mort_price_series_prev"] = float(p_series0)
                self.state["mort_income_series_prev"] = float(y_series0)
                self._ensure_mortgage_index_anchors(p_series0, y_series0, base_rate_q)
            else:
                self.nodes["HH"].set("deposits", 0.0)
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
        # Per-tick UBI funding diagnostics captured *pre-UBI payment* inside post_tick
        self.ubi_debug_history: List[Dict[str, float]] = []

        self._assert_sfc_ok(context="init")

    def _bootstrap_startup_lagged_retained(self) -> None:
        """Seed lagged retained earnings at startup to avoid a one-quarter CAPEX jump."""
        if bool(self.state.get("startup_bootstrap_done", False)):
            return
        self.state["startup_bootstrap_done"] = True

        if int(self.state.get("t", 0)) != 0:
            return
        if not bool(self.params.get("startup_bootstrap_lagged_retained", True)):
            return
        if not bool(self.params.get("use_population", False)):
            return
        if not bool(self.params.get("population_dynamics", False)):
            return
        if self.hh is None or self.hh.n <= 0:
            return

        reinvest_rate = float(self.params.get("reinvest_rate_of_retained", 0.0))
        if reinvest_rate <= 0.0:
            return

        # Respect explicit initial lagged retained settings if provided by scenario config.
        if any(
            abs(float(self.nodes[node_id].memo.get("retained_prev", 0.0))) > 1e-12
            for node_id in ("FA", "FH", "BANK")
        ):
            return

        seed_sol = self.solve_within_tick_population()
        if seed_sol is None:
            return

        scale = float(self.params.get("startup_bootstrap_retained_scale", 1.0))
        scale = max(0.0, scale)

        self.nodes["FA"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_fa", 0.0)))
        self.nodes["FH"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_fh", 0.0)))
        self.nodes["BANK"].memo["retained_prev"] = scale * max(0.0, float(seed_sol.get("retained_bk", 0.0)))

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
            vat = max(0.0, float(self.params.get("vat_rate", 0.0)))
            return max(1e-9, p * (1.0 + vat))
        return max(1e-9, p)

    def _mort_income_series_value(self, wages_total: float, div_house_total: float, ubi_per_h: float) -> float:
        n_hh = float(self.hh.n) if (self.hh is not None and self.hh.n > 0) else 1.0
        wages = max(0.0, float(wages_total))
        div_hh = max(0.0, float(div_house_total))
        ubi_total = max(0.0, float(ubi_per_h)) * n_hh

        income_series = str(self.params.get("mort_index_income_series", "NominalHHIncome")).strip()
        if income_series == "NominalWages":
            y = wages
        elif income_series == "NominalMarketIncome":
            y = wages + div_hh
        else:
            # NominalHHIncome (default): wages + household dividends + UBI.
            y = wages + div_hh + ubi_total
        return max(1e-9, float(y))

    def _ensure_mortgage_index_anchors(self, p_series_now: float, y_series_now: float, rL: float) -> None:
        if self.hh is None or self.hh.n <= 0:
            return
        hh = self.hh
        hh.ensure_memos()

        mort = _as_np(hh.mortgage_loans, dtype=float)
        active = mort > 1e-12
        inactive = ~active
        new_mask = active & (hh.mort_t0 < 0)

        if np.any(new_mask):
            mort_pay_rate = max(0.0, min(1.0, float(self.params.get("mortgage_principal_pay_rate_q", 0.0))))
            ctr_at_anchor = mort[new_mask] * (max(0.0, float(rL)) + mort_pay_rate)
            hh.mort_P0[new_mask] = float(max(1e-9, p_series_now))
            hh.mort_Y0[new_mask] = float(max(1e-9, y_series_now))
            hh.mort_t0[new_mask] = int(self.state.get("t", 0))
            hh.mort_pay_base[new_mask] = ctr_at_anchor
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
        ubi_per_h: float,
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

        mort_pay_rate = max(0.0, min(1.0, float(self.params.get("mortgage_principal_pay_rate_q", 0.0))))
        mort_interest_due_i = np.maximum(0.0, mort_vec) * max(0.0, float(rL))
        mort_principal_ctr_i = np.maximum(0.0, mort_vec) * mort_pay_rate
        mort_pay_ctr_i = mort_interest_due_i + mort_principal_ctr_i

        p_series_now = self._mort_price_series_value(float(self.state.get("price_level", 1.0)))
        y_series_now = self._mort_income_series_value(wages_total, div_house_total, ubi_per_h)
        p_series_prev = float(self.state.get("mort_price_series_prev", p_series_now))
        y_series_prev = float(self.state.get("mort_income_series_prev", y_series_now))
        p_series_prev = max(1e-9, p_series_prev)
        y_series_prev = max(1e-9, y_series_prev)

        enabled = bool(self.params.get("mort_index_enable", False))
        active = (mort_vec > 1e-12) & (hh.mort_t0 >= 0)
        max_pay_i = mort_interest_due_i + np.maximum(0.0, mort_vec)

        if enabled:
            if not bool(self.params.get("mort_corridor_apply_in_logspace", True)):
                raise ValueError("mort_corridor_apply_in_logspace must be True for mortgage index module.")
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
            # Caveat: this model uses stylized proportional paydown mortgages (no amortization object).
            # Indexing the current contractual flow preserves "deflation relief" semantics.
            mort_pay_req_i = mort_pay_ctr_i * i_curr

        mort_pay_req_i = np.minimum(np.maximum(0.0, mort_pay_req_i), max_pay_i)
        mort_interest_paid_i = np.minimum(mort_interest_due_i, mort_pay_req_i)
        mort_interest_paid_i = np.maximum(0.0, mort_interest_paid_i)
        mort_principal_paid_i = np.maximum(0.0, mort_pay_req_i - mort_interest_paid_i)
        mort_principal_paid_i = np.minimum(mort_principal_paid_i, np.maximum(0.0, mort_vec))

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
            "mort_gap_i": mort_gap_i,
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

    def _apply_mortgage_gap_neutralization(
        self,
        *,
        gap_i: np.ndarray,
        mort_interest_due_total: float,
        mort_pay_ctr_total: float,
    ) -> Dict[str, float]:
        gap_total_raw = float(np.sum(np.maximum(0.0, gap_i)))
        if gap_total_raw <= 0.0:
            return {"gap_total": 0.0, "paid_gov": 0.0, "paid_fund": 0.0, "paid_issuance": 0.0, "paid_total": 0.0}

        if not bool(self.params.get("mort_bank_neutralize_enable", True)):
            return {"gap_total": gap_total_raw, "paid_gov": 0.0, "paid_fund": 0.0, "paid_issuance": 0.0, "paid_total": 0.0}
        if not self._neutralize_stress_active():
            return {"gap_total": gap_total_raw, "paid_gov": 0.0, "paid_fund": 0.0, "paid_issuance": 0.0, "paid_total": 0.0}

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
            return {"gap_total": gap_total, "paid_gov": 0.0, "paid_fund": 0.0, "paid_issuance": 0.0, "paid_total": 0.0}

        stack_raw = self.params.get("mort_neutralize_funding_stack", ["GOV", "FUND", "ISSUANCE"])
        if isinstance(stack_raw, (list, tuple)):
            stack = [str(x).strip().upper() for x in stack_raw]
        else:
            stack = ["GOV", "FUND", "ISSUANCE"]

        remaining = float(gap_total)
        paid_gov = 0.0
        paid_fund = 0.0
        paid_iss = 0.0

        for src in stack:
            if remaining <= 0.0:
                break
            if src == "GOV":
                paid = self._pay_bank_income_from_payer("GOV", remaining)
                paid_gov += paid
                remaining -= paid
            elif src == "FUND":
                allow_if_debt = bool(self.params.get("mort_neutralize_fund_allowed_if_debt_outstanding", False))
                fund_debt = float(self.nodes["FUND"].get("loans", 0.0))
                if (fund_debt <= 1e-12) or allow_if_debt:
                    paid = self._pay_bank_income_from_payer("FUND", remaining)
                    paid_fund += paid
                    remaining -= paid
            elif src == "ISSUANCE":
                pay = max(0.0, remaining)
                if pay > 0.0:
                    self.nodes["BANK"].add("deposit_liab", +pay)
                    self.nodes["BANK"].add("reserves", +pay)
                    self.nodes["GOV"].add("deposits", +pay)
                    self.nodes["GOV"].add("money_issued", +pay)
                    paid = self._pay_bank_income_from_payer("GOV", pay)
                    paid_iss += paid
                    remaining -= paid

        paid_total = float(paid_gov + paid_fund + paid_iss)
        return {
            "gap_total": float(gap_total),
            "paid_gov": float(paid_gov),
            "paid_fund": float(paid_fund),
            "paid_issuance": float(paid_iss),
            "paid_total": float(paid_total),
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
        if self.state["trust_active"]:
            return

        # Allow one baseline quarter before any trigger logic (needed for baseline targeting)
        if int(self.state["t"]) == 0:
            return

        # Trigger metric: use last-quarter debt-service stress.
        # We take the maximum of:
        #   - pop_dti_w_p90: interest / wage among employed debtors (wage-only)
        #   - pop_dti_p90:   interest / (wage + UBI) among debtors (inclusive)
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

        # Optional leveraged launch: BANK lends to FUND and FUND buys 10% equity from aggregate HH.
        # If launch_loan is 0, we still activate the Trust regime (dilution can still grow ownership),
        # but we skip the leveraged purchase to avoid an unpaid equity transfer.
        if launch_loan > 0:
            # (1) Loan creates deposits
            self._create_loan("FUND", launch_loan, memo_tag="launch_loan_created")

            # (2) Purchase 10% equity from HH with those deposits.
            seller = "HH"

            cash_per = launch_loan / 3.0
            issuers = [("FA", "shares_FA"), ("FH", "shares_FH"), ("BANK", "shares_BANK")]

            for issuer, key in issuers:
                shares_out = self.nodes[issuer].get("shares_outstanding", 0.0)
                equity_jump = 0.10 * shares_out

                self.nodes[seller].add(key, -equity_jump)
                self.nodes["FUND"].add(key, +equity_jump)
                self._xfer_deposits_to_households("FUND", cash_per)

            for _, key in issuers:
                if self.nodes[seller].get(key, 0.0) < -1e-9:
                    raise ValueError(f"{seller} has negative {key} after launch. Check initial ownership.")

        self._assert_sfc_ok(context=f"after_trigger_t{t}")

    # ---------------------------------------------------------
    # Phase B: Dilution (2% annual) to FUND until policy cap
    # ---------------------------------------------------------

    def issue_social_shares(self) -> None:
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

    def solve_within_tick_population(self, max_iter: int = 200, tol: float = 1e-8) -> Optional[Dict[str, Any]]:
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
        rL = float(self.state.get("policy_rate_q", self.params["loan_rate_per_quarter"]))

        ws_fh = float(self.params["wage_share_of_revenue"]["FH"])
        ws_fa_base = float(self.params["wage_share_of_revenue"]["FA"])
        ws_fa = ws_fa_base * (1.0 - auto_eff)

        # Price level (nominal $ per unit real consumption). Consumption rule is applied in real terms.
        P = float(self.state.get("price_level", 1.0))
        if P <= 0:
            P = 1e-9
        # VAT wedge: consumer price includes VAT as a markup
        vat_rate = float(self.params.get("vat_rate", 0.0))
        if vat_rate < 0:
            vat_rate = 0.0
        P_cons = P * (1.0 + vat_rate)  # tax-exclusive VAT treated as higher consumer price

        # Baseline wage weights used to distribute wages and (temporarily) dividends.
        w0 = _as_np(hh.wages0_q, dtype=float)
        w0_sum = float(w0.sum())
        if w0_sum <= 0:
            return None
        w_weights = w0 / w0_sum

        # Static household vectors
        base_real = _as_np(hh.base_real_cons_q, dtype=float)
        mpc = _as_np(hh.mpc_q, dtype=float)
        mort = _as_np(hh.mortgage_loans, dtype=float)
        rev = _as_np(hh.revolving_loans, dtype=float)

        # Beginning-of-tick deposits (nominal). Use a copy so the solver is not affected by post-tick mutations.
        dep0 = _as_np(hh.deposits, dtype=float).copy()

        # Initial income guess: previous-quarter income memo.
        y_guess = _as_np(hh.prev_income, dtype=float)
        if y_guess.shape[0] != hh.n:
            y_guess = w0.copy()

        # Trust ownership fractions (same as legacy solver)
        def own_frac(issuer: str, key: str) -> float:
            so = self.nodes[issuer].get("shares_outstanding", 1.0)
            return self.nodes["FUND"].get(key, 0.0) / so if so > 0 else 0.0

        f_fa = own_frac("FA", "shares_FA")
        f_fh = own_frac("FH", "shares_FH")
        f_bk = own_frac("BANK", "shares_BANK")

        fund_loan = float(self.nodes["FUND"].get("loans", 0.0))
        fa_loan = float(self.nodes["FA"].get("loans", 0.0))
        fh_loan = float(self.nodes["FH"].get("loans", 0.0))

        fa_interest = fa_loan * rL
        fh_interest = fh_loan * rL

        # Income-target pool (population mode):
        # Store the pool in REAL terms at baseline, then index back to nominal each tick via current P.
        # This prevents deflation from mechanically increasing UBI issuance.
        target_pool_real_pop = self.state.get("income_target_pool_real_pop", None)
        reinvest_rate = float(self.params.get("reinvest_rate_of_retained", 0.0))
        reinvest_rate = max(0.0, min(1.0, reinvest_rate))
        # Lagged CAPEX: this quarter's investment is funded from LAST quarter's retained earnings
        # (breaks same-quarter circularity)
        retained_fa_prev = float(self.nodes["FA"].memo.get("retained_prev", 0.0))
        retained_fh_prev = float(self.nodes["FH"].memo.get("retained_prev", 0.0))

        capex_fa_nom = reinvest_rate * max(0.0, retained_fa_prev)
        capex_fh_nom = reinvest_rate * max(0.0, retained_fh_prev)
        capex_total_nom = capex_fa_nom + capex_fh_nom
        div_house_total_est = 0.0

        for _ in range(max_iter):
            # 1) Household consumption (nominal), vectorized
            # Consumption decision uses the consumer price (includes VAT wedge)
            y_real = y_guess / P_cons

            # Desired real consumption
            c_real_des = np.maximum(0.0, base_real + mpc * y_real)
            c_hh_nom_des = P_cons * c_real_des

            # Cash-in-advance constraint (no new borrowing for consumption inside the solver):
            # available = beginning deposits + current-quarter disposable income guess.
            # If disposable income is negative, available is floored at 0.
            avail_nom = np.maximum(0.0, dep0 + y_guess)
            c_hh_nom = np.minimum(c_hh_nom_des, avail_nom)

            # Convert back to real and tax-exclusive firm base
            c_real = c_hh_nom / P_cons
            c_firm_nom = P * c_real
            c_total = float(c_firm_nom.sum())

            # 2b) Firm revenues and wages, given fixed (lagged) CAPEX demand
            # Split consumption demand by automation share, but allow CAPEX demand to flow primarily to a supplier sector.
            # Default: CAPEX is produced by the physical sector (FH), so capex_supply_share_fa defaults to 0.0.
            capex_supply_share_fa = float(self.params.get("capex_supply_share_fa", 0.0))
            capex_supply_share_fa = max(0.0, min(1.0, capex_supply_share_fa))
            capex_supply_share_fh = 1.0 - capex_supply_share_fa

            rev_fa = (auto_eff * c_total) + (capex_supply_share_fa * capex_total_nom)
            rev_fh = ((1.0 - auto_eff) * c_total) + (capex_supply_share_fh * capex_total_nom)

            w_fa = rev_fa * ws_fa
            w_fh = rev_fh * ws_fh
            w_total = float(w_fa + w_fh)

            # profits pre-tax (capex is not expensed; it's a cash outflow later)
            p_fa_pre_tax = max(0.0, rev_fa - w_fa - fa_interest)
            p_fh_pre_tax = max(0.0, rev_fh - w_fh - fh_interest)

            mort_interest_due = mort * rL
            rev_interest = rev * rL
            interest_hh = mort_interest_due + rev_interest
            trust_interest = fund_loan * rL
            bank_interest_ex_mort = float(rev_interest.sum() + trust_interest + fa_interest + fh_interest)

            # Corporate income tax policy:
            # - Distributed dividends are NOT taxed at the corporate level.
            # - Retained earnings ARE taxed at the corporate level.
            # Household recipients are still taxed via income_tax_i; FUND is untaxed.
            # Optional policy: raise tax rate as wages fall relative to baseline.
            corp_tax_rate = float(self.params.get("corporate_tax_rate", 0.0))
            corp_tax_rate = max(0.0, min(1.0, corp_tax_rate))

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

            # Dividend payout policy: allow retained earnings for reinvestment.
            # Dividends are set off pre-tax profits; retained profits bear corporate tax.
            payout_firms = float(self.params.get("dividend_payout_rate_firms", 1.0))
            payout_bank = float(self.params.get("dividend_payout_rate_bank", 1.0))
            payout_firms = max(0.0, min(1.0, payout_firms))
            payout_bank = max(0.0, min(1.0, payout_bank))

            div_fa_total = payout_firms * p_fa_pre_tax
            div_fh_total = payout_firms * p_fh_pre_tax
            div_house_firms = (div_fa_total * (1.0 - f_fa)) + (div_fh_total * (1.0 - f_fh))
            div_fund_firms = (div_fa_total * f_fa) + (div_fh_total * f_fh)

            # 4) UBI policy (population total pool)
            # Index baseline REAL pool back to nominal using the current tax-exclusive price level P.
            if target_pool_real_pop is None:
                ubi = 0.0
            else:
                target_pool_nom = float(target_pool_real_pop) * float(P)
                ubi = max(0.0, (target_pool_nom - w_total) / float(hh.n))
            # Optional monotonic policy floor: do not allow policy UBI to decline.
            if bool(self.params.get("ubi_monotonic_floor", True)):
                ubi = max(float(hh.prev_ubi), float(ubi))

            # Mortgage index module: compute indexed required payment per household mortgage.
            mort_index_enable = bool(self.params.get("mort_index_enable", False))
            p_series_now = self._mort_price_series_value(P)
            y_series_now = self._mort_income_series_value(w_total, float(div_house_total_est), float(ubi))
            self._ensure_mortgage_index_anchors(p_series_now, y_series_now, rL)
            mort_terms = self._compute_mortgage_index_terms(
                mort=mort,
                rL=rL,
                wages_total=w_total,
                div_house_total=float(div_house_total_est),
                ubi_per_h=float(ubi),
                commit_state=False,
            )
            mort_pay_req_i = _as_np(mort_terms.get("mort_pay_req_i", np.zeros(hh.n, dtype=float)), dtype=float)
            mort_interest_paid_i = _as_np(mort_terms.get("mort_interest_paid_i", np.zeros(hh.n, dtype=float)), dtype=float)
            bank_profit_pre_tax = float(bank_interest_ex_mort + np.sum(np.maximum(0.0, mort_interest_paid_i)))

            div_bk_total = payout_bank * max(0.0, bank_profit_pre_tax)
            div_fund = div_fund_firms + (div_bk_total * f_bk)
            div_house_total = div_house_firms + (div_bk_total * (1.0 - f_bk))
            div_house_total_est = float(div_house_total)

            retained_fa_pre_tax = max(0.0, p_fa_pre_tax - div_fa_total)
            retained_fh_pre_tax = max(0.0, p_fh_pre_tax - div_fh_total)
            retained_bk_pre_tax = max(0.0, bank_profit_pre_tax - div_bk_total)

            corp_tax_fa = corp_tax_rate * retained_fa_pre_tax
            corp_tax_fh = corp_tax_rate * retained_fh_pre_tax
            corp_tax_bk = corp_tax_rate * retained_bk_pre_tax

            retained_fa = max(0.0, retained_fa_pre_tax - corp_tax_fa)
            retained_fh = max(0.0, retained_fh_pre_tax - corp_tax_fh)
            retained_bk = max(0.0, retained_bk_pre_tax - corp_tax_bk)

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
            taxable_income = wages_i + div_i  # excludes UBI by policy

            # Income tax: 15% marginal above a percentile threshold (nearest-rank)
            it_rate = float(self.params.get("income_tax_rate", 0.0))
            it_pct = float(self.params.get("income_tax_cutoff_pct", 100.0))
            if it_rate < 0:
                it_rate = 0.0
            if it_pct < 0:
                it_pct = 0.0
            if it_pct > 100:
                it_pct = 100.0
            if taxable_income.size > 0:
                k_it = int(math.ceil((it_pct / 100.0) * taxable_income.size)) - 1
                k_it = max(0, min(taxable_income.size - 1, k_it))
                it_thr = float(np.partition(taxable_income, k_it)[k_it])
            else:
                it_thr = 0.0
            income_tax_i = it_rate * np.maximum(0.0, taxable_income - it_thr)

            # VAT credit ("prebate"): vat_rate * poverty-line consumption, with a linear
            # phaseout over the configured eligibility-income percentile band.
            vc_start_pct = float(self.params.get("vat_credit_phaseout_start_pct", 25.0))
            vc_end_pct = float(self.params.get("vat_credit_phaseout_end_pct", 45.0))
            vc_start_pct = max(0.0, min(100.0, vc_start_pct))
            vc_end_pct = max(vc_start_pct, min(100.0, vc_end_pct))

            elig_income = taxable_income + float(ubi)  # user policy: eligibility uses taxable income + UBI
            if elig_income.size > 0:
                k_vc_start = int(math.ceil((vc_start_pct / 100.0) * elig_income.size)) - 1
                k_vc_start = max(0, min(elig_income.size - 1, k_vc_start))
                k_vc_end = int(math.ceil((vc_end_pct / 100.0) * elig_income.size)) - 1
                k_vc_end = max(0, min(elig_income.size - 1, k_vc_end))

                vc_thr_start = float(np.partition(elig_income, k_vc_start)[k_vc_start])
                vc_thr_end = float(np.partition(elig_income, k_vc_end)[k_vc_end])
            else:
                vc_thr_start = 0.0
                vc_thr_end = 0.0

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
            # When mortgage indexing is enabled, households service mortgage-required payment
            # plus revolving interest (contractual); otherwise retain legacy interest-only burden.
            if mort_index_enable:
                y_new = wages_i + float(ubi) + div_i + vat_credit_i - rev_interest - mort_pay_req_i - income_tax_i
            else:
                y_new = wages_i + float(ubi) + div_i + vat_credit_i - interest_hh - income_tax_i
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
                    "f_fa": f_fa,
                    "f_fh": f_fh,
                    "f_bk": f_bk,
                    "div_fund": div_fund,
                    "div_house_total": float(div_house_total),
                    "ubi": float(ubi),
                    "wages_i": wages_i,
                    "div_i": div_i,
                    "y": y_new,
                    "mort_index_enable": bool(mort_index_enable),
                    "mort_pay_req_i": mort_pay_req_i,
                    "mort_pay_ctr_i": _as_np(mort_terms.get("mort_pay_ctr_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_interest_due_i": _as_np(mort_terms.get("mort_interest_due_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_interest_paid_i": _as_np(mort_terms.get("mort_interest_paid_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_principal_paid_i": _as_np(mort_terms.get("mort_principal_paid_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_gap_i": _as_np(mort_terms.get("mort_gap_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_gap_total": float(mort_terms.get("mort_gap_total", 0.0)),
                    "mort_pay_req_total": float(mort_terms.get("mort_pay_req_total", 0.0)),
                    "mort_pay_ctr_total": float(mort_terms.get("mort_pay_ctr_total", 0.0)),
                    "mort_index_mean": float(mort_terms.get("mort_index_mean", 1.0)),
                    "mort_index_min": float(mort_terms.get("mort_index_min", 1.0)),
                    "mort_index_max": float(mort_terms.get("mort_index_max", 1.0)),
                    "mort_index_i": _as_np(mort_terms.get("mort_index_i", np.ones(hh.n, dtype=float)), dtype=float),
                    "mort_dln_i": _as_np(mort_terms.get("mort_dln_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "mort_dln_sm_i": _as_np(mort_terms.get("mort_dln_sm_i", np.zeros(hh.n, dtype=float)), dtype=float),
                    "p_series_now": float(mort_terms.get("p_series_now", P)),
                    "y_series_now": float(mort_terms.get("y_series_now", max(1e-9, w_total + div_house_total + float(ubi) * float(hh.n)))),
                    "rev_interest_i": rev_interest,
                    "taxable_income": taxable_income,
                    "income_tax_i": income_tax_i,
                    "vat_credit_i": vat_credit_i,
                    "it_threshold": float(it_thr),
                    "vc_phaseout_start_threshold": float(vc_thr_start),
                    "vc_phaseout_end_threshold": float(vc_thr_end),
                }

            y_guess = y_new

        return None


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
        mort_pay_ctr_i = _as_np(sol.get("mort_pay_ctr_i", []), dtype=float)
        mort_interest_due_i = _as_np(sol.get("mort_interest_due_i", []), dtype=float)
        mort_interest_paid_i = _as_np(sol.get("mort_interest_paid_i", []), dtype=float)
        mort_principal_paid_i = _as_np(sol.get("mort_principal_paid_i", []), dtype=float)
        mort_gap_i = _as_np(sol.get("mort_gap_i", []), dtype=float)
        mort_index_i = _as_np(sol.get("mort_index_i", []), dtype=float)
        mort_dln_i = _as_np(sol.get("mort_dln_i", []), dtype=float)
        mort_dln_sm_i = _as_np(sol.get("mort_dln_sm_i", []), dtype=float)
        mort_index_enable = bool(sol.get("mort_index_enable", False))
        y_vec = _as_np(sol.get("y", []), dtype=float)

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
        if mort_pay_ctr_i.shape[0] != n:
            mort_pay_ctr_i = np.zeros(n, dtype=float)
        if mort_interest_due_i.shape[0] != n:
            mort_interest_due_i = np.zeros(n, dtype=float)
        if mort_interest_paid_i.shape[0] != n:
            mort_interest_paid_i = np.zeros(n, dtype=float)
        if mort_principal_paid_i.shape[0] != n:
            mort_principal_paid_i = np.zeros(n, dtype=float)
        if mort_gap_i.shape[0] != n:
            mort_gap_i = np.zeros(n, dtype=float)
        if mort_index_i.shape[0] != n:
            mort_index_i = np.ones(n, dtype=float)
        if mort_dln_i.shape[0] != n:
            mort_dln_i = np.zeros(n, dtype=float)
        if mort_dln_sm_i.shape[0] != n:
            mort_dln_sm_i = np.zeros(n, dtype=float)

        c_total = float(sol.get("c_total", 0.0))

        # -------------------------------------------------
        # 1) Consumption: households -> firms, VAT remitted to GOV
        # -------------------------------------------------
        vat_rate = float(self.params.get("vat_rate", 0.0))
        if vat_rate < 0:
            vat_rate = 0.0

        deposits[:] = deposits - c_hh_nom

        # Firms receive the tax-exclusive base (this is what drives revenues in the solver)
        self.nodes["FA"].add("deposits", float(auto_eff) * c_total)
        self.nodes["FH"].add("deposits", (1.0 - float(auto_eff)) * c_total)

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

        # Settle CAPEX cash flows: investors pay; suppliers receive (split by capex_supply_share_fa).
        # This credits firms with the investment-demand revenue base that the solver used.
        if capex_total_nom > 0:
            capex_supply_share_fa = float(self.params.get("capex_supply_share_fa", 0.0))
            capex_supply_share_fa = max(0.0, min(1.0, capex_supply_share_fa))
            capex_to_fa = capex_supply_share_fa * capex_total_nom
            capex_to_fh = (1.0 - capex_supply_share_fa) * capex_total_nom

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

        # -------------------------------------------------
        # 2) Wages: firms -> households
        # -------------------------------------------------
        w_fa = float(sol.get("w_fa", 0.0))
        w_fh = float(sol.get("w_fh", 0.0))
        self.nodes["FA"].add("deposits", -w_fa)
        self.nodes["FH"].add("deposits", -w_fh)
        deposits[:] = deposits + wages_i

        # -------------------------------------------------
        # 2a) Firm interest: firms -> BANK
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
        # 2b) Corporate tax: corporations -> GOV (before dividends / reinvestment)
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
        self.state["mort_index_mean"] = 1.0
        self.state["mort_index_min"] = 1.0
        self.state["mort_index_max"] = 1.0
        self.state["mort_overdraft_due_to_payment_total"] = 0.0
        self.state["mort_overdraft_due_to_payment_count"] = 0.0

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

            # Revolving interest remains contractual and is always paid (fallback via overdraft later).
            deposits[:] = deposits - rev_interest_i
            rev_int_total = float(np.sum(np.maximum(0.0, rev_interest_i)))
            if rev_int_total > 0.0:
                bank.add("deposit_liab", -rev_int_total)
                bank.add("equity", +rev_int_total)

            # Mortgage required payment (indexed).
            dep_before_mort = deposits.copy()
            deposits[:] = deposits - mort_pay_req_i
            mort_overdraft_need = np.maximum(0.0, mort_pay_req_i - np.maximum(0.0, dep_before_mort))
            self.state["mort_overdraft_due_to_payment_total"] = float(np.sum(mort_overdraft_need))
            self.state["mort_overdraft_due_to_payment_count"] = float(np.sum(mort_overdraft_need > 1e-12))

            mort_int_paid_total = float(np.sum(np.maximum(0.0, mort_interest_paid_i)))
            mort_prin_paid_total = float(np.sum(np.maximum(0.0, mort_principal_paid_i)))

            if mort_int_paid_total > 0.0:
                bank.add("deposit_liab", -mort_int_paid_total)
                bank.add("equity", +mort_int_paid_total)
            if mort_prin_paid_total > 0.0:
                mort[:] = np.maximum(0.0, mort - mort_principal_paid_i)
                bank.add("loan_assets", -mort_prin_paid_total)
                bank.add("deposit_liab", -mort_prin_paid_total)

            # Optional bank neutralization transfer for reduced mortgage cashflow.
            neutral = self._apply_mortgage_gap_neutralization(
                gap_i=np.maximum(0.0, mort_gap_i),
                mort_interest_due_total=float(np.sum(np.maximum(0.0, mort_interest_due_i))),
                mort_pay_ctr_total=float(np.sum(np.maximum(0.0, mort_pay_ctr_i))),
            )
            self.state["mort_gap_total"] = float(neutral["gap_total"])
            self.state["mort_gap_paid_by_gov"] = float(neutral["paid_gov"])
            self.state["mort_gap_paid_by_fund"] = float(neutral["paid_fund"])
            self.state["mort_gap_paid_by_issuance"] = float(neutral["paid_issuance"])
            self.state["bank_mort_neutralize_inflow"] = float(neutral["paid_total"])
        else:
            deposits[:] = deposits - interest_hh
            tot_int_hh = float(interest_hh.sum())
            if tot_int_hh > 0.0:
                bank.add("deposit_liab", -tot_int_hh)
                bank.add("equity", +tot_int_hh)

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
        # 5) Taxes + VAT credit (before UBI)
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
                # Distribute in proportion to the computed credits
                denom_credit = float(np.sum(vat_credit_i))
                if denom_credit > 0:
                    deposits[:] = deposits + (vat_credit_i * (pay_gov / denom_credit))
                vat_credit_total -= pay_gov

            if vat_credit_total > 0:
                # Issuance to complete the credit
                vat_credit_issued += vat_credit_total
                denom_credit = float(np.sum(vat_credit_i))
                if denom_credit > 0:
                    deposits[:] = deposits + (vat_credit_i * (vat_credit_total / denom_credit))
                self.nodes["BANK"].add("deposit_liab", vat_credit_total)
                self.nodes["BANK"].add("reserves", vat_credit_total)
                self.nodes["GOV"].add("money_issued", vat_credit_total)

        # Diagnostics: store per-tick VAT credit totals
        self.state["vat_credit_paid_total"] = float(max(0.0, vat_credit_paid_from_gov))
        self.state["vat_credit_issued_total"] = float(max(0.0, vat_credit_issued))
        self.state["vat_credit_total"] = float(max(0.0, vat_credit_total_initial))

        # -------------------------------------------------
        # 5) Trust amortization (BEFORE UBI)
        # -------------------------------------------------
        fund_loan = float(self.nodes["FUND"].get("loans", 0.0))
        if fund_loan > 0:
            fund_dep = float(self.nodes["FUND"].get("deposits", 0.0))
            repay_amt = min(fund_dep, fund_loan)
            if repay_amt > 0:
                self._repay_loan("FUND", repay_amt)

        if self.params.get("send_fund_residual_to_gov", False):
            residual = max(0.0, self.nodes["FUND"].get("deposits"))
            if residual > 0:
                self._xfer_deposits("FUND", "GOV", residual)

        # -------------------------------------------------
        # 6) UBI payments: issuance share -> FUND dep -> GOV dep -> extra issuance
        # -------------------------------------------------
        ubi = float(sol.get("ubi", 0.0))
        fund_paid_from_dep_total = 0.0
        gov_paid_from_dep_total = 0.0
        issued_total = 0.0

        if ubi > 0:
            ubi_total = ubi * float(n)
            issue_share = float(self.params.get("ubi_issuance_share", 0.0))
            issue_share = max(0.0, min(1.0, issue_share))

            # Policy issuance tranche (e.g., 5% of total UBI each tick).
            issue_target = ubi_total * issue_share
            if issue_target > 0:
                issued_total += issue_target
                deposits[:] = deposits + (issue_target / float(n))
                self.nodes["BANK"].add("deposit_liab", issue_target)
                self.nodes["BANK"].add("reserves", issue_target)
                self.nodes["GOV"].add("money_issued", issue_target)

            fund_needed = max(0.0, ubi_total - issue_target)

            # 1) FUND deposits
            pay_fund_total = min(fund_needed, max(0.0, self.nodes["FUND"].get("deposits")))
            if pay_fund_total > 0:
                self.nodes["FUND"].add("deposits", -pay_fund_total)
                fund_paid_from_dep_total += pay_fund_total
                deposits[:] = deposits + (pay_fund_total / float(n))
                fund_needed -= pay_fund_total

            # 2) GOV deposits
            if fund_needed > 0:
                pay_gov_total = min(fund_needed, max(0.0, self.nodes["GOV"].get("deposits")))
                if pay_gov_total > 0:
                    self.nodes["GOV"].add("deposits", -pay_gov_total)
                    gov_paid_from_dep_total += pay_gov_total
                    deposits[:] = deposits + (pay_gov_total / float(n))
                    fund_needed -= pay_gov_total

            # 3) Additional issuance if deposits are insufficient
            if fund_needed > 0:
                issued_total += fund_needed
                deposits[:] = deposits + (fund_needed / float(n))

                # Bank liability expands by issuance
                self.nodes["BANK"].add("deposit_liab", fund_needed)
                self.nodes["BANK"].add("reserves", fund_needed)
                self.nodes["GOV"].add("money_issued", fund_needed)

        # Store diagnostics for this tick (totals across all households)
        self.state["ubi_from_fund_dep_total"] = float(fund_paid_from_dep_total)
        self.state["ubi_from_gov_dep_total"] = float(gov_paid_from_dep_total)
        self.state["ubi_issued_total"] = float(issued_total)

        # -------------------------------------------------
        # 6a) Optional tax refund: rebate a share of remaining GOV deposits
        # in proportion to each household's tax paid (VAT + income tax).
        # -------------------------------------------------
        tax_rebate_total = 0.0
        rebate_rate = float(self.params.get("gov_tax_rebate_rate", 0.0))
        rebate_rate = max(0.0, min(1.0, rebate_rate))
        if rebate_rate > 0.0:
            gov_dep_avail = max(0.0, self.nodes["GOV"].get("deposits", 0.0))
            tax_rebate_total = rebate_rate * gov_dep_avail
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

        # -------------------------------------------------
        # 6b) Household principal repayment (revolving first, then mortgage)
        # -------------------------------------------------
        rev_pay_rate = float(self.params.get("revolving_principal_pay_rate_q", 0.0))
        mort_pay_rate = float(self.params.get("mortgage_principal_pay_rate_q", 0.0))
        rev_pay_rate = max(0.0, min(1.0, rev_pay_rate))
        mort_pay_rate = max(0.0, min(1.0, mort_pay_rate))

        principal_paid_total = 0.0

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

        if (not mort_index_enable) and mort_pay_rate > 0.0:
            desired_mort_pay = mort * mort_pay_rate
            mort_pay = np.minimum(desired_mort_pay, deposits)
            mort_pay = np.maximum(0.0, mort_pay)

            pay_sum = float(mort_pay.sum())
            if pay_sum > 0.0:
                deposits[:] = deposits - mort_pay
                mort[:] = mort - mort_pay
                principal_paid_total += pay_sum

        if principal_paid_total > 0.0:
            # Principal repayment destroys deposits and loans (double-entry)
            self.nodes["BANK"].add("loan_assets", -principal_paid_total)
            self.nodes["BANK"].add("deposit_liab", -principal_paid_total)

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
        self.nodes["HH"].set("loans", hh_total_loan)

        if y_vec.shape[0] == n:
            hh.prev_income = y_vec.astype(float, copy=True)
        hh.prev_ubi = float(ubi)
        hh.prev_wages_total = float(sol.get("w_total", 0.0))

        # 8b) Store last-quarter retained earnings for lagged CAPEX decision
        self.nodes["FA"].memo["retained_prev"] = float(sol.get("retained_fa", 0.0))
        self.nodes["FH"].memo["retained_prev"] = float(sol.get("retained_fh", 0.0))
        self.nodes["BANK"].memo["retained_prev"] = float(sol.get("retained_bk", 0.0))

        for firm_id in ("FA", "FH"):
            dep = float(self.nodes[firm_id].get("deposits", 0.0))
            if dep < 0.0:
                # Convert overdraft into an explicit bank loan so deposits cannot remain negative.
                self._create_loan(firm_id, -dep, memo_tag="firm_overdraft")
                # _create_loan adds deposits by the same amount, so firm deposits should now be ~0.

        self._assert_sfc_ok(context=f"post_tick_population_t{self.state['t']}")

    # ---------------------------------------------------------
    # Phase E: One tick execution
    # ---------------------------------------------------------

    def step(self) -> None:
        # Set this quarter's policy rate from lagged inflation/DTI observables.
        self._update_policy_rate()

        # Automation path (levels + per-quarter flow for visualization)
        t = int(self.state["t"])
        path = str(self.params.get("automation_path", "two_hump")).lower()

        if path == "linear":
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

        # Productivity multiplier (>=1). This is the channel through which reinvestment raises output capacity.
        prod_mult = 1.0 + (kappa * (K_per_h / K_scale))
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
            solp = self.solve_within_tick_population()
            if solp is not None:
                self.post_tick_population(solp)

                # Establish baseline income target pool (population) after first successful baseline solve.
                # Store in REAL terms and index to P each tick.
                if self.state.get("income_target_pool_real_pop", None) is None:
                    P_base = float(self.state.get("price_level", 1.0))
                    if P_base <= 0:
                        P_base = 1e-9

                    target_pool_nom_base = float(solp["w_total"]) + float(self.hh.n) * float(solp["ubi"])  # ubi should be 0 here
                    self.state["income_target_pool_real_pop"] = float(target_pool_nom_base) / float(P_base)
                    self.state["baseline_price_level_pop"] = float(P_base)
                    self.state["baseline_wages_total_pop"] = float(solp["w_total"])

                # Population inequality diagnostics (vectorized)
                y_vec = _as_np(solp.get("y", []), dtype=float)             # disposable income
                wages_i = _as_np(solp.get("wages_i", []), dtype=float)     # market component
                div_i = _as_np(solp.get("div_i", []), dtype=float)         # market component

                market_inc = wages_i + div_i if (wages_i.size and div_i.size and wages_i.size == div_i.size) else np.asarray([], dtype=float)

                gini_market = calculate_gini_np(market_inc) if market_inc.size else 0.0
                gini_disp = calculate_gini_np(y_vec) if y_vec.size else 0.0

                # Net-wealth Gini proxy:
                #   wealth_i = deposits_i + allocated_hh_equity_i - loans_i
                # Household equity claims are allocated by baseline wage weights because ownership is tracked at HH aggregate.
                dep_i = _as_np(self.hh.deposits, dtype=float)
                loan_i = _as_np(self.hh.mortgage_loans, dtype=float) + _as_np(self.hh.revolving_loans, dtype=float)

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

                    def hh_share_frac(issuer: str, key: str) -> float:
                        so = float(self.nodes[issuer].get("shares_outstanding", 0.0))
                        if so <= 0.0:
                            return 0.0
                        frac_hh = float(self.nodes["HH"].get(key, 0.0)) / so
                        return max(0.0, min(1.0, frac_hh))

                    fa_equity_proxy = max(0.0, float(self.nodes["FA"].get("deposits", 0.0)) + float(self.nodes["FA"].get("K", 0.0)) * P_wealth - float(self.nodes["FA"].get("loans", 0.0)))
                    fh_equity_proxy = max(0.0, float(self.nodes["FH"].get("deposits", 0.0)) + float(self.nodes["FH"].get("K", 0.0)) * P_wealth - float(self.nodes["FH"].get("loans", 0.0)))
                    bank_equity_proxy = max(0.0, float(self.nodes["BANK"].get("equity", 0.0)))

                    hh_equity_total = (
                        hh_share_frac("FA", "shares_FA") * fa_equity_proxy
                        + hh_share_frac("FH", "shares_FH") * fh_equity_proxy
                        + hh_share_frac("BANK", "shares_BANK") * bank_equity_proxy
                    )
                    private_equity_total = float(max(0.0, hh_equity_total))
                    equity_i = wealth_weights * hh_equity_total
                    wealth_i = dep_i + equity_i - loan_i
                    gini_wealth = calculate_gini_np(wealth_i)
                else:
                    gini_wealth = 0.0

                # Private-capital diagnostics (all nominal, population totals unless noted).
                priv_payout_total = float(max(0.0, float(solp.get("div_house_total", 0.0))))
                prev_priv_eq_total = float(self.state.get("private_equity_prev_total", 0.0))
                private_roe_q = (priv_payout_total / prev_priv_eq_total) if prev_priv_eq_total > 1e-9 else 0.0

                retained_fa = float(solp.get("retained_fa", 0.0))
                retained_fh = float(solp.get("retained_fh", 0.0))
                retained_bk = float(solp.get("retained_bk", 0.0))
                f_fa = float(solp.get("f_fa", 0.0))
                f_fh = float(solp.get("f_fh", 0.0))
                f_bk = float(solp.get("f_bk", 0.0))
                private_retained_total = (
                    retained_fa * (1.0 - f_fa)
                    + retained_fh * (1.0 - f_fh)
                    + retained_bk * (1.0 - f_bk)
                )

                capex_total_nom = float(solp.get("capex_total_nom", 0.0))
                private_inv_cov = (private_retained_total / capex_total_nom) if capex_total_nom > 1e-9 else 0.0
                self.state["private_equity_prev_total"] = float(max(0.0, private_equity_total))

                def frac(issuer: str, key: str) -> float:
                    so = self.nodes[issuer].get("shares_outstanding", 1.0)
                    return self.nodes["FUND"].get(key, 0.0) / so if so > 0 else 0.0

                own_avg = (frac("FA", "shares_FA") + frac("FH", "shares_FH") + frac("BANK", "shares_BANK")) / 3.0

                # Population DTI percentiles directly from solver components (vectorized)
                interest_hh = _as_np(solp.get("interest_hh", []), dtype=float)
                ubi = float(solp.get("ubi", 0.0))

                if interest_hh.size and wages_i.size and interest_hh.size == wages_i.size:
                    gross = wages_i + ubi

                    # Income percentiles: everyone with gross income
                    incs = (gross - interest_hh)[gross > 0]

                    # IMPORTANT: apply the mask *before* dividing to avoid divide-by-zero / invalid warnings.
                    dti_mask = (interest_hh > 0) & (gross > 0)
                    dtis = (interest_hh[dti_mask] / gross[dti_mask]) if np.any(dti_mask) else np.asarray([], dtype=float)

                    wage_dti_mask = (interest_hh > 0) & (wages_i > 0)
                    dtis_w = (interest_hh[wage_dti_mask] / wages_i[wage_dti_mask]) if np.any(wage_dti_mask) else np.asarray([], dtype=float)

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

                wages_total = float(solp["w_total"])
                c_total = float(solp["c_total"])

                self.history.append(TickResult(
                    t=self.state["t"],
                    automation=float(self.state["automation"]),
                    automation_flow=float(self.state.get("automation_flow", 0.0)),
                    automation_info=float(self.state.get("automation_info", 0.0)),
                    automation_info_flow=float(self.state.get("automation_info_flow", 0.0)),
                    automation_phys=float(self.state.get("automation_phys", 0.0)),
                    automation_phys_flow=float(self.state.get("automation_phys_flow", 0.0)),
                    price_level=float(self.state.get("price_level", 1.0)),
                    inflation=float(self.state.get("inflation", 0.0)),
                    gini=float(gini_disp),
                    gini_market=float(gini_market),
                    gini_disp=float(gini_disp),
                    gini_wealth=float(gini_wealth),
                    private_eq_per_h=float(private_equity_total) / float(self.hh.n),
                    private_roe_q=float(private_roe_q),
                    private_inv_cov=float(private_inv_cov),
                    # --- Fiscal / funding diagnostics (per household) ---
                    vat_per_h=float(self.state.get("vat_receipts_total", 0.0)) / float(self.hh.n),
                    inc_tax_per_h=float(self.state.get("income_tax_total", 0.0)) / float(self.hh.n),
                    corp_tax_per_h=float(self.state.get("corp_tax_total", 0.0)) / float(self.hh.n),
                    vat_credit_per_h=float(self.state.get("vat_credit_total", 0.0)) / float(self.hh.n),
                    gov_dep_per_h=float(self.nodes["GOV"].get("deposits", 0.0)) / float(self.hh.n),
                    fund_dep_per_h=float(self.nodes["FUND"].get("deposits", 0.0)) / float(self.hh.n),
                    capex_per_h=float(self.state.get("capex_total", 0.0)) / float(self.hh.n),
                    ubi_per_h=float(ubi),
                    ubi_from_fund_dep_per_h=float(self.state.get("ubi_from_fund_dep_total", 0.0)) / float(self.hh.n),
                    ubi_from_gov_dep_per_h=float(self.state.get("ubi_from_gov_dep_total", 0.0)) / float(self.hh.n),
                    ubi_issued_per_h=float(self.state.get("ubi_issued_total", 0.0)) / float(self.hh.n),
                    trust_equity_pct=float(own_avg),
                    trust_debt=float(self.nodes["FUND"].get("loans", 0.0)),
                    wages_total=wages_total,
                    total_consumption=c_total,

                    real_avg_income=float(((wages_total / float(self.hh.n)) + float(ubi)) / P_now),
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

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    def write_csv(self, filename: str = "run_history.csv") -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TickResult.__dataclass_fields__.keys())
            writer.writeheader()
            for r in self.history:
                writer.writerow(r.__dict__)
