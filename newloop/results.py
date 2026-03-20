# Author: Roger Ison   roger@miximum.info
"""Reusable simulation-output helpers for CLI, plotting, and Streamlit layers."""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import config as default_config
from engine import NewLoop
from newloop_types import TickResult


@dataclass
class SimulationRun:
    """Container for one simulation run and its row-oriented outputs."""

    sim: NewLoop
    rows: List[Dict[str, Any]]
    population_distributions: Dict[str, Dict[str, Any]] | None = None
    startup_diagnostics: Dict[str, Any] | None = None
    burn_in_quarters: int = 0

    @property
    def history(self) -> List[TickResult]:
        return self.sim.history


def history_to_rows(history: Sequence[TickResult]) -> List[Dict[str, Any]]:
    """Convert TickResult history to plain dictionaries."""
    return [asdict(tick) for tick in history]


def _visible_rows(history: Sequence[TickResult]) -> List[Dict[str, Any]]:
    """Convert visible history rows and rebase t so visible Q0 starts at zero."""
    rows = history_to_rows(history)
    if not rows:
        return rows
    t0 = int(rows[0].get("t", 0))
    for row in rows:
        row["t"] = int(row.get("t", 0)) - t0
    return rows


def _population_distribution_snapshot(sim: NewLoop) -> Dict[str, Any] | None:
    """Capture household income/wealth vectors for before/after distribution plotting."""
    if sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    n = int(hh.n)

    dep_i = np.asarray(hh.deposits, dtype=float)
    loan_i = np.asarray(hh.mortgage_loans, dtype=float) + np.asarray(hh.revolving_loans, dtype=float)
    income_i = np.asarray(hh.prev_income, dtype=float)
    if income_i.shape[0] != n:
        income_i = np.asarray(hh.wages0_q, dtype=float)

    w0 = np.asarray(hh.wages0_q, dtype=float)
    w0_sum = float(w0.sum()) if w0.shape[0] == n else 0.0
    if w0_sum > 0.0:
        weights = w0 / w0_sum
    else:
        weights = np.full(n, 1.0 / float(n), dtype=float)

    p_now = float(sim.state.get("price_level", 1.0))
    if p_now <= 0.0:
        p_now = 1e-9

    def _hh_share_frac(issuer: str, key: str) -> float:
        shares_out = float(sim.nodes[issuer].get("shares_outstanding", 0.0))
        if shares_out <= 0.0:
            return 0.0
        frac = float(sim.nodes["HH"].get(key, 0.0)) / shares_out
        return max(0.0, min(1.0, frac))

    fa_eq = max(
        0.0,
        float(sim.nodes["FA"].get("deposits", 0.0))
        + float(sim.nodes["FA"].get("K", 0.0)) * p_now
        - float(sim.nodes["FA"].get("loans", 0.0)),
    )
    fh_eq = max(
        0.0,
        float(sim.nodes["FH"].get("deposits", 0.0))
        + float(sim.nodes["FH"].get("K", 0.0)) * p_now
        - float(sim.nodes["FH"].get("loans", 0.0)),
    )
    bk_eq = max(0.0, float(sim.nodes["BANK"].get("equity", 0.0)))

    hh_equity_total = (
        _hh_share_frac("FA", "shares_FA") * fa_eq
        + _hh_share_frac("FH", "shares_FH") * fh_eq
        + _hh_share_frac("BANK", "shares_BANK") * bk_eq
    )
    equity_i = weights * hh_equity_total
    wealth_i = dep_i + equity_i - loan_i

    return {
        "price_level": float(p_now),
        "income": income_i.astype(float).tolist(),
        "wealth": wealth_i.astype(float).tolist(),
    }


def _startup_diagnostics(sim: NewLoop) -> Dict[str, Any] | None:
    """Summarize quarter-0 household buffer consistency and debt stress."""
    if sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    hh.ensure_memos()

    wages_i = np.asarray(hh.wages0_q, dtype=float)
    deposits_i = np.asarray(hh.deposits, dtype=float)
    mort_i = np.asarray(hh.mortgage_loans, dtype=float)
    rev_i = np.asarray(hh.revolving_loans, dtype=float)
    base_real_i = np.asarray(hh.base_real_cons_q, dtype=float)
    mpc_i = np.asarray(hh.mpc_q, dtype=float)
    target_months_i = np.asarray(hh.liquid_buffer_months_target, dtype=float)
    prev_income_i = np.asarray(hh.prev_income, dtype=float)
    if prev_income_i.shape != wages_i.shape:
        prev_income_i = wages_i.copy()

    p_now = float(sim.state.get("price_level", 1.0))
    if p_now <= 0.0:
        p_now = 1e-9
    vat_rate = 0.0 if bool(sim.params.get("disable_vat", False)) else max(0.0, float(sim.params.get("vat_rate", 0.0)))
    p_cons = p_now * (1.0 + vat_rate)

    # Mirror the quarter-0 solver target before taxes/interest feedback updates y_guess.
    y_real_i = prev_income_i / p_cons
    c_real_core_i = np.maximum(0.0, base_real_i + mpc_i * y_real_i)
    target_buffer_i = (target_months_i / 3.0) * (p_cons * c_real_core_i)
    buffer_gap_i = deposits_i - target_buffer_i
    deposit_to_target_i = np.divide(
        deposits_i,
        np.maximum(target_buffer_i, 1e-9),
        out=np.zeros_like(deposits_i, dtype=float),
        where=np.isfinite(target_buffer_i),
    )

    rate_q = float(sim.state.get("policy_rate_q", sim.params.get("loan_rate_per_quarter", 0.0)))
    rate_q = max(0.0, rate_q)
    interest_i = np.maximum(0.0, mort_i * rate_q) + np.maximum(0.0, rev_i * rate_q)
    wage_income_i = np.maximum(0.0, wages_i)
    dti_mask = (interest_i > 0.0) & (wage_income_i > 0.0)
    wage_dti_i = (interest_i[dti_mask] / wage_income_i[dti_mask]) if np.any(dti_mask) else np.asarray([], dtype=float)

    def _summary_row(label: str, values: np.ndarray, *, pct: bool = False) -> Dict[str, Any]:
        xs = np.asarray(values, dtype=float)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return {"Metric": label, "Mean": 0.0, "P10": 0.0, "P50": 0.0, "P90": 0.0, "percent": pct}
        return {
            "Metric": label,
            "Mean": float(np.mean(xs)),
            "P10": float(np.percentile(xs, 10.0)),
            "P50": float(np.percentile(xs, 50.0)),
            "P90": float(np.percentile(xs, 90.0)),
            "percent": pct,
        }

    if not bool(sim.params.get("disable_income_support", False)):
        sim.income_support_policy.warm_start_anchor_if_needed(
            state=sim.state,
            baseline_wages_i=hh.wages0_q,
            price_level=float(sim.state.get("price_level", 1.0)),
        )
    sol = sim.solve_within_tick_population()

    circular_flow: Dict[str, float] = {}
    decile_rows: List[Dict[str, Any]] = []
    if sol is not None:
        c_hh_nom_i = np.asarray(sol.get("c_hh_nom", []), dtype=float)
        c_firm_nom_i = np.asarray(sol.get("c_firm_nom", []), dtype=float)
        income_tax_i = np.asarray(sol.get("income_tax_i", []), dtype=float)
        vat_credit_i = np.asarray(sol.get("vat_credit_i", []), dtype=float)
        y_i = np.asarray(sol.get("y", []), dtype=float)
        rev_interest_i = np.asarray(sol.get("rev_interest_i", []), dtype=float)
        mort_pay_req_i = np.asarray(sol.get("mort_pay_req_i", []), dtype=float)
        interest_hh_i = np.asarray(sol.get("interest_hh", []), dtype=float)
        mort_index_enable = bool(sol.get("mort_index_enable", False))

        if mort_index_enable and rev_interest_i.shape == wages_i.shape and mort_pay_req_i.shape == wages_i.shape:
            debt_service_i = np.maximum(0.0, rev_interest_i) + np.maximum(0.0, mort_pay_req_i)
        else:
            debt_service_i = np.maximum(0.0, interest_hh_i) if interest_hh_i.shape == wages_i.shape else interest_i

        f_fa = float(sol.get("f_fa", 0.0))
        f_fh = float(sol.get("f_fh", 0.0))
        f_bk = float(sol.get("f_bk", 0.0))
        private_retained_total = (
            float(sol.get("retained_fa", 0.0)) * (1.0 - f_fa)
            + float(sol.get("retained_fh", 0.0)) * (1.0 - f_fh)
            + float(sol.get("retained_bk", 0.0)) * (1.0 - f_bk)
        )
        circular_flow = {
            "hh_consumption_nom": float(np.sum(np.maximum(0.0, c_hh_nom_i))),
            "firm_revenue_nom": float(sol.get("c_total", 0.0)),
            "wages_total": float(sol.get("w_total", 0.0)),
            "household_dividends_total": float(sol.get("div_house_total", 0.0)),
            "hh_disposable_income_total": float(np.sum(y_i)) if y_i.shape == wages_i.shape else 0.0,
            "income_tax_total": float(np.sum(np.maximum(0.0, income_tax_i))) if income_tax_i.shape == wages_i.shape else 0.0,
            "corporate_tax_total": float(sol.get("corp_tax_fa", 0.0) + sol.get("corp_tax_fh", 0.0) + sol.get("corp_tax_bk", 0.0)),
            "vat_receipts_total": float(np.sum(np.maximum(0.0, c_hh_nom_i - c_firm_nom_i))) if (c_hh_nom_i.shape == wages_i.shape and c_firm_nom_i.shape == wages_i.shape) else 0.0,
            "vat_credit_total": float(np.sum(np.maximum(0.0, vat_credit_i))) if vat_credit_i.shape == wages_i.shape else 0.0,
            "income_support_total": float(sol.get("uis", 0.0)) * float(hh.n),
            "debt_service_total": float(np.sum(np.maximum(0.0, debt_service_i))) if debt_service_i.shape == wages_i.shape else 0.0,
            "private_retained_total": float(private_retained_total),
            "capex_total_nom": float(sol.get("capex_total_nom", 0.0)),
            "buffer_shortfall_total": float(np.sum(np.maximum(0.0, -buffer_gap_i))),
        }

        order = np.argsort(wage_income_i, kind="stable")
        for decile_idx, idx in enumerate(np.array_split(order, 10), start=1):
            if idx.size == 0:
                continue
            dec_gap = buffer_gap_i[idx]
            dec_target = target_buffer_i[idx]
            dec_dep = deposits_i[idx]
            dec_interest = debt_service_i[idx] if debt_service_i.shape == wages_i.shape else interest_i[idx]
            dec_wages = wage_income_i[idx]
            dec_mask = (dec_interest > 0.0) & (dec_wages > 0.0)
            dec_dti = (dec_interest[dec_mask] / dec_wages[dec_mask]) if np.any(dec_mask) else np.asarray([], dtype=float)
            decile_rows.append(
                {
                    "Decile": f"D{decile_idx}",
                    "Mean Wage": float(np.mean(dec_wages)),
                    "Mean Deposits": float(np.mean(dec_dep)),
                    "Mean Target": float(np.mean(dec_target)),
                    "Mean Gap": float(np.mean(dec_gap)),
                    "Below Buffer": float(np.mean(dec_gap < 0.0)),
                    "DTI P90": float(np.percentile(dec_dti, 90.0)) if dec_dti.size else 0.0,
                }
            )

    return {
        "household_count": int(hh.n),
        "stabilization_enabled": bool(sim.params.get("startup_stabilization_enabled", False)),
        "stabilization_quarters": int(max(0, sim.params.get("startup_stabilization_quarters", 0))),
        "share_below_runtime_buffer": float(np.mean(buffer_gap_i < 0.0)),
        "mean_deposit_to_target_ratio": float(np.mean(deposit_to_target_i)),
        "median_buffer_gap": float(np.median(buffer_gap_i)),
        "mean_buffer_gap": float(np.mean(buffer_gap_i)),
        "buffer_shortfall_total": float(np.sum(np.maximum(0.0, -buffer_gap_i))),
        "startup_dti_w_p90": float(np.percentile(wage_dti_i, 90.0)) if wage_dti_i.size else 0.0,
        "circular_flow": circular_flow,
        "decile_rows": decile_rows,
        "table_rows": [
            _summary_row("Deposits", deposits_i),
            _summary_row("Runtime Buffer Target", target_buffer_i),
            _summary_row("Buffer Gap", buffer_gap_i),
            _summary_row("Wage Income", wage_income_i),
            _summary_row("Interest Burden", interest_i),
            _summary_row("Wage-only DTI", wage_dti_i, pct=True),
        ],
    }


def _apply_startup_stabilization(sim: NewLoop) -> int:
    """Run hidden stabilization quarters before visible Q0."""
    if not bool(sim.params.get("startup_stabilization_enabled", False)):
        return 0

    n_quarters = int(max(0, sim.params.get("startup_stabilization_quarters", 0)))
    if n_quarters <= 0:
        return 0

    orig_automation_disabled = bool(sim.params.get("automation_disabled", False))
    sim.params["automation_disabled"] = True
    try:
        for _ in range(n_quarters):
            sim.step()
    finally:
        sim.params["automation_disabled"] = orig_automation_disabled

    # Visible Q0 should start from the stabilized state, not from a later automation-clock quarter.
    # Keep stocks/memos from burn-in, but reset time/history so visible automation and policy timing
    # begin at quarter 0 on top of the stabilized balance-sheet state.
    sim.state["t"] = 0
    sim.history.clear()
    sim.bs_history.clear()
    sim.inv_history.clear()
    sim.uis_debug_history.clear()
    sim.gov_obligation_history.clear()
    return n_quarters


def run_simulation(n_quarters: int = 80, cfg: Dict[str, Any] | None = None) -> SimulationRun:
    """Run NewLoop for n_quarters and return structured outputs."""
    effective_cfg = copy.deepcopy(default_config if cfg is None else cfg)
    startup_diag_sim = NewLoop(copy.deepcopy(effective_cfg))
    burn_in_quarters = _apply_startup_stabilization(startup_diag_sim)
    startup_diag = _startup_diagnostics(startup_diag_sim)
    sim = NewLoop(effective_cfg)
    burn_in_quarters = _apply_startup_stabilization(sim)
    before = _population_distribution_snapshot(sim)
    for _ in range(int(n_quarters)):
        sim.step()
    after = _population_distribution_snapshot(sim)
    pop_dist = {"before": before, "after": after} if (before is not None and after is not None) else None
    return SimulationRun(
        sim=sim,
        rows=_visible_rows(sim.history),
        population_distributions=pop_dist,
        startup_diagnostics=startup_diag,
        burn_in_quarters=burn_in_quarters,
    )


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return compact summary metrics from the final row and simple deltas."""
    if not rows:
        return {}

    first = rows[0]
    last = rows[-1]

    def _get(row: Dict[str, Any], key: str) -> float:
        return float(row.get(key, 0.0))

    return {
        "quarters": float(len(rows)),
        "automation_end": _get(last, "automation"),
        "price_end": _get(last, "price_level"),
        "real_consumption_end": _get(last, "real_consumption"),
        "real_consumption_delta": _get(last, "real_consumption") - _get(first, "real_consumption"),
        "gini_disp_end": _get(last, "gini_disp"),
        "trust_equity_end": _get(last, "trust_equity_pct"),
        "income_support_end": _get(last, "uis_per_h"),
        "income_support_issued_end": _get(last, "uis_issued_per_h"),
    }
