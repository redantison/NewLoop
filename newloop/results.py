# Author: Roger Ison   roger@miximum.info
"""Reusable simulation-output helpers for simulation and Streamlit layers."""

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

from config import get_default_config
from engine import NewLoop
from newloop_types import TickResult


@dataclass
class SimulationRun:
    """Container for one simulation run and its row-oriented outputs."""

    sim: NewLoop
    rows: List[Dict[str, Any]]
    population_distributions: Dict[str, Dict[str, Any]] | None = None
    startup_diagnostics: Dict[str, Any] | None = None
    baseline_calibration: Dict[str, Any] | None = None

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

    snapshot = _startup_solver_snapshot(sim)
    sol = None if snapshot is None else snapshot.get("sol")
    if snapshot is None:
        return None

    p_cons = float(snapshot["p_cons"])
    target_buffer_i = np.asarray(snapshot["target_buffer_i"], dtype=float)
    disposable_income_i = np.asarray(snapshot["disposable_income_i"], dtype=float)
    debt_service_i = np.asarray(snapshot["debt_service_i"], dtype=float)
    base_cons_gap_i = np.asarray(snapshot["base_consumption_gap_i"], dtype=float)
    buffer_gap_i = deposits_i - target_buffer_i
    deposit_to_target_i = np.divide(
        deposits_i,
        np.maximum(target_buffer_i, 1e-9),
        out=np.zeros_like(deposits_i, dtype=float),
        where=np.isfinite(target_buffer_i),
    )
    wage_income_i = np.maximum(0.0, wages_i)
    dti_mask = (debt_service_i > 0.0) & (wage_income_i > 0.0)
    wage_dti_i = (debt_service_i[dti_mask] / wage_income_i[dti_mask]) if np.any(dti_mask) else np.asarray([], dtype=float)

    circular_flow: Dict[str, float] = {}
    decile_rows: List[Dict[str, Any]] = []
    if sol is not None:
        c_hh_nom_i = np.asarray(sol.get("c_hh_nom", []), dtype=float)
        c_firm_nom_i = np.asarray(sol.get("c_firm_nom", []), dtype=float)
        income_tax_i = np.asarray(sol.get("income_tax_i", []), dtype=float)
        vat_credit_i = np.asarray(sol.get("vat_credit_i", []), dtype=float)
        y_i = disposable_income_i

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
            dec_interest = debt_service_i[idx]
            dec_wages = wage_income_i[idx]
            dec_net = disposable_income_i[idx]
            dec_base_gap = base_cons_gap_i[idx]
            dec_mask = (dec_interest > 0.0) & (dec_wages > 0.0)
            dec_dti = (dec_interest[dec_mask] / dec_wages[dec_mask]) if np.any(dec_mask) else np.asarray([], dtype=float)
            decile_rows.append(
                {
                    "Decile": f"D{decile_idx}",
                    "Mean Wage": float(np.mean(dec_wages)),
                    "Mean Net Disp": float(np.mean(dec_net)),
                    "Mean Deposits": float(np.mean(dec_dep)),
                    "Mean Target": float(np.mean(dec_target)),
                    "Mean Gap": float(np.mean(dec_gap)),
                    "Mean Base Gap": float(np.mean(dec_base_gap)),
                    "Below Buffer": float(np.mean(dec_gap < 0.0)),
                    "Base Uncovered": float(np.mean(dec_base_gap < 0.0)),
                    "DTI P90": float(np.percentile(dec_dti, 90.0)) if dec_dti.size else 0.0,
                }
            )

    return {
        "household_count": int(hh.n),
        "share_below_runtime_buffer": float(np.mean(buffer_gap_i < 0.0)),
        "share_base_consumption_uncovered": float(np.mean(base_cons_gap_i < 0.0)),
        "mean_deposit_to_target_ratio": float(np.mean(deposit_to_target_i)),
        "median_buffer_gap": float(np.median(buffer_gap_i)),
        "mean_buffer_gap": float(np.mean(buffer_gap_i)),
        "mean_base_consumption_gap": float(np.mean(base_cons_gap_i)),
        "buffer_shortfall_total": float(np.sum(np.maximum(0.0, -buffer_gap_i))),
        "startup_dti_w_p90": float(np.percentile(wage_dti_i, 90.0)) if wage_dti_i.size else 0.0,
        "circular_flow": circular_flow,
        "decile_rows": decile_rows,
        "table_rows": [
            _summary_row("Deposits", deposits_i),
            _summary_row("Runtime Buffer Target", target_buffer_i),
            _summary_row("Buffer Gap", buffer_gap_i),
            _summary_row("Wage Income", wage_income_i),
            _summary_row("Net Disposable Income", disposable_income_i),
            _summary_row("Debt Service Burden", debt_service_i),
            _summary_row("Base Consumption Gap", base_cons_gap_i),
            _summary_row("Wage-only DTI", wage_dti_i, pct=True),
        ],
    }


def _quarter_state_diagnostics(sim: NewLoop) -> Dict[str, Any] | None:
    if sim.hh is None or sim.hh.n <= 0 or not sim.history:
        return None

    hh = sim.hh
    hh.ensure_memos()
    snapshot = _startup_solver_snapshot(sim)
    if snapshot is None:
        return None

    row = asdict(sim.history[-1])
    wages_i = np.asarray(hh.wages0_q, dtype=float)
    deposits_i = np.asarray(hh.deposits, dtype=float)
    target_buffer_i = np.asarray(snapshot["target_buffer_i"], dtype=float)
    disposable_income_i = np.asarray(snapshot["disposable_income_i"], dtype=float)
    debt_service_i = np.asarray(snapshot["debt_service_i"], dtype=float)
    base_cons_gap_i = np.asarray(snapshot["base_consumption_gap_i"], dtype=float)
    p_cons = max(float(snapshot["p_cons"]), 1e-9)
    base_real_i = np.asarray(hh.base_real_cons_q, dtype=float)
    mpc_i = np.asarray(hh.mpc_q, dtype=float)
    disp_income_real_i = disposable_income_i / p_cons
    mpc_income_real_i = mpc_i * disp_income_real_i
    core_real_i = np.maximum(0.0, base_real_i + mpc_income_real_i)

    buffer_gap_i = deposits_i - target_buffer_i
    deposit_to_target_i = np.divide(
        deposits_i,
        np.maximum(target_buffer_i, 1e-9),
        out=np.zeros_like(deposits_i, dtype=float),
        where=np.isfinite(target_buffer_i),
    )
    wage_income_i = np.maximum(0.0, wages_i)
    dti_mask = (debt_service_i > 0.0) & (wage_income_i > 0.0)
    wage_dti_i = (debt_service_i[dti_mask] / wage_income_i[dti_mask]) if np.any(dti_mask) else np.asarray([], dtype=float)
    sol = snapshot.get("sol") or {}

    return {
        "t": int(row.get("t", 0)),
        "real_consumption": float(row.get("real_consumption", 0.0)),
        "real_avg_income": float(row.get("real_avg_income", 0.0)),
        "wages_total": float(row.get("wages_total", 0.0)),
        "capex_per_h": float(row.get("capex_per_h", 0.0)),
        "pop_dti_p90": float(row.get("pop_dti_p90", 0.0)),
        "mean_deposits": float(np.mean(deposits_i)),
        "mean_runtime_buffer_target": float(np.mean(target_buffer_i)),
        "mean_buffer_gap": float(np.mean(buffer_gap_i)),
        "share_below_runtime_buffer": float(np.mean(buffer_gap_i < 0.0)),
        "mean_deposit_to_target_ratio": float(np.mean(deposit_to_target_i)),
        "mean_base_real_consumption": float(np.mean(base_real_i)),
        "mean_mpc_income_real": float(np.mean(mpc_income_real_i)),
        "mean_core_real_consumption": float(np.mean(core_real_i)),
        "mean_net_disposable_income": float(np.mean(disposable_income_i)),
        "mean_debt_service": float(np.mean(debt_service_i)),
        "mean_base_consumption_gap": float(np.mean(base_cons_gap_i)),
        "share_base_consumption_uncovered": float(np.mean(base_cons_gap_i < 0.0)),
        "wage_dti_p90": float(np.percentile(wage_dti_i, 90.0)) if wage_dti_i.size else 0.0,
        "household_dividends_total": float(sol.get("div_house_total", 0.0)),
    }


def _quarter_comparison(q0_diag: Dict[str, Any] | None, qn_diag: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not q0_diag or not qn_diag:
        return None

    metrics = [
        ("Real Consumption", "real_consumption", False),
        ("Real Avg Income", "real_avg_income", False),
        ("Wages Total", "wages_total", False),
        ("Household Dividends", "household_dividends_total", False),
        ("Mean Deposits", "mean_deposits", False),
        ("Mean Runtime Buffer Target", "mean_runtime_buffer_target", False),
        ("Mean Buffer Gap", "mean_buffer_gap", False),
        ("HH Below Buffer", "share_below_runtime_buffer", True),
        ("Deposit / Target", "mean_deposit_to_target_ratio", False),
        ("Mean Base Real Cons", "mean_base_real_consumption", False),
        ("Mean MPC x Income", "mean_mpc_income_real", False),
        ("Mean Core Real Cons", "mean_core_real_consumption", False),
        ("Mean Net Disp Income", "mean_net_disposable_income", False),
        ("Mean Debt Service", "mean_debt_service", False),
        ("Mean Base Gap", "mean_base_consumption_gap", False),
        ("Base Uncovered", "share_base_consumption_uncovered", True),
        ("Wage DTI P90", "wage_dti_p90", True),
        ("CAPEX per H", "capex_per_h", False),
    ]

    rows: List[Dict[str, Any]] = []
    for label, key, is_pct in metrics:
        q0_val = float(q0_diag.get(key, 0.0))
        qn_val = float(qn_diag.get(key, 0.0))
        rows.append(
            {
                "Metric": label,
                "Q0": q0_val,
                "Q10": qn_val,
                "Delta": qn_val - q0_val,
                "percent": is_pct,
            }
        )

    return {
        "q0_t": int(q0_diag.get("t", 0)),
        "q10_t": int(qn_diag.get("t", 0)),
        "rows": rows,
    }


def _quintile_boundaries(n_quintiles: int) -> List[float]:
    n = max(1, int(n_quintiles))
    step = 100.0 / float(n)
    return [step * float(i + 1) for i in range(n)]


def _bucket_means(values: np.ndarray, rank_source: np.ndarray, n_buckets: int) -> np.ndarray:
    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=float)

    order = np.argsort(np.asarray(rank_source, dtype=float), kind="stable")
    buckets = np.array_split(order, max(1, int(n_buckets)))
    means: List[float] = []
    vals = np.asarray(values, dtype=float)
    for idx in buckets:
        if idx.size <= 0:
            means.append(0.0)
        else:
            means.append(float(np.mean(vals[idx])))
    return np.asarray(means, dtype=float)


def _startup_solver_snapshot(sim: NewLoop) -> Dict[str, Any] | None:
    if sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    hh.ensure_memos()
    wages_i = np.asarray(hh.wages0_q, dtype=float)
    base_real_i = np.asarray(hh.base_real_cons_q, dtype=float)
    mpc_i = np.asarray(hh.mpc_q, dtype=float)
    target_months_i = np.asarray(hh.liquid_buffer_months_target, dtype=float)

    p_now = max(float(sim.state.get("price_level", 1.0)), 1e-9)
    p_cons = p_now * (1.0 + float(sim._effective_vat_rate()))

    if not bool(sim.params.get("disable_income_support", False)):
        sim.income_support_policy.warm_start_anchor_if_needed(
            state=sim.state,
            baseline_wages_i=hh.wages0_q,
            price_level=float(sim.state.get("price_level", 1.0)),
        )

    sol = sim.solve_within_tick_population()
    if sol is None:
        return None

    disposable_income_i = np.asarray(sol.get("y", []), dtype=float)
    if disposable_income_i.shape != wages_i.shape:
        disposable_income_i = np.asarray(hh.prev_income, dtype=float)
        if disposable_income_i.shape != wages_i.shape:
            disposable_income_i = wages_i.copy()

    mort_index_enable = bool(sol.get("mort_index_enable", False))
    rev_interest_i = np.asarray(sol.get("rev_interest_i", []), dtype=float)
    mort_pay_req_i = np.asarray(sol.get("mort_pay_req_i", []), dtype=float)
    interest_hh_i = np.asarray(sol.get("interest_hh", []), dtype=float)
    mort_i = np.asarray(hh.mortgage_loans, dtype=float)
    rev_i = np.asarray(hh.revolving_loans, dtype=float)
    rate_q = max(0.0, float(sim.state.get("policy_rate_q", sim.params.get("loan_rate_per_quarter", 0.0))))
    raw_interest_i = np.maximum(0.0, mort_i * rate_q) + np.maximum(0.0, rev_i * rate_q)

    if mort_index_enable and rev_interest_i.shape == wages_i.shape and mort_pay_req_i.shape == wages_i.shape:
        debt_service_i = np.maximum(0.0, rev_interest_i) + np.maximum(0.0, mort_pay_req_i)
    else:
        debt_service_i = np.maximum(0.0, interest_hh_i) if interest_hh_i.shape == wages_i.shape else raw_interest_i

    y_real_i = disposable_income_i / p_cons
    c_real_core_i = np.maximum(0.0, base_real_i + (mpc_i * y_real_i))
    target_buffer_i = (target_months_i / 3.0) * (p_cons * c_real_core_i)
    base_consumption_nom_i = p_cons * np.maximum(0.0, base_real_i)
    base_consumption_gap_i = disposable_income_i - base_consumption_nom_i

    return {
        "sol": sol,
        "p_cons": float(p_cons),
        "disposable_income_i": disposable_income_i.astype(float, copy=True),
        "debt_service_i": debt_service_i.astype(float, copy=True),
        "target_buffer_i": target_buffer_i.astype(float, copy=True),
        "base_consumption_gap_i": base_consumption_gap_i.astype(float, copy=True),
        "base_consumption_nom_i": base_consumption_nom_i.astype(float, copy=True),
    }


def _sync_startup_household_state(sim: NewLoop) -> None:
    if sim.hh is None or sim.hh.n <= 0:
        return

    hh = sim.hh
    if sim.population is not None:
        sim.population.deposits = hh.deposits.astype(float).tolist()
        sim.population.mpc_q = np.asarray(hh.mpc_q, dtype=float).astype(float).tolist()

    sim.nodes["HH"].set("deposits", hh.sum_deposits())
    bank = sim.nodes["BANK"]
    dep_liab = float(sim._sum_deposits_all())
    bank.set("deposit_liab", dep_liab)
    bank.set(
        "reserves",
        dep_liab + float(bank.get("equity", 0.0)) - float(bank.get("loan_assets", 0.0)),
    )
    sim._assert_sfc_ok(context="startup_state_sync")


def _apply_startup_income_buffer_reset(
    sim: NewLoop,
    max_iter: int = 8,
    reset_deposits: bool = True,
) -> Dict[str, Any] | None:
    if sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    hh.ensure_memos()

    try:
        import population as pop_mod
    except Exception:
        pop_mod = None

    mpc_schedule = ()
    if sim.population_cfg is not None:
        mpc_schedule = tuple(
            (float(pct), float(val))
            for pct, val in getattr(sim.population_cfg, "mpc_by_wealth_pct", ())
        )

    prev_deposits = np.asarray(hh.deposits, dtype=float).copy()
    prev_income = np.asarray(hh.prev_income, dtype=float).copy()
    last_snapshot: Dict[str, Any] | None = None
    for _ in range(max(1, int(max_iter))):
        snapshot = _startup_solver_snapshot(sim)
        if snapshot is None:
            break
        last_snapshot = snapshot

        hh.prev_income = np.asarray(snapshot["disposable_income_i"], dtype=float).astype(float, copy=True)
        if reset_deposits:
            target_i = np.maximum(0.0, np.asarray(snapshot["target_buffer_i"], dtype=float))
            hh.deposits = target_i.astype(float, copy=True)

        if pop_mod is None or not mpc_schedule:
            _sync_startup_household_state(sim)
            break

        new_mpc = np.asarray(pop_mod._assign_mpc_from_deposits(hh.deposits.tolist(), mpc_schedule), dtype=float)
        mpc_delta = float(np.max(np.abs(new_mpc - np.asarray(hh.mpc_q, dtype=float)))) if new_mpc.size else 0.0
        dep_delta = float(np.max(np.abs(hh.deposits - prev_deposits))) if hh.deposits.size else 0.0
        income_delta = float(np.max(np.abs(hh.prev_income - prev_income))) if hh.prev_income.size else 0.0
        hh.mpc_q = new_mpc.astype(float, copy=True)
        prev_deposits = hh.deposits.copy()
        prev_income = hh.prev_income.copy()
        _sync_startup_household_state(sim)
        if max(mpc_delta, dep_delta, income_delta) <= 1e-8:
            break

    _sync_startup_household_state(sim)
    if last_snapshot is None:
        last_snapshot = _startup_solver_snapshot(sim)
    if last_snapshot is None:
        return None

    target_i = np.asarray(last_snapshot["target_buffer_i"], dtype=float)
    gap_i = np.asarray(hh.deposits, dtype=float) - target_i
    ratio_i = np.divide(
        np.asarray(hh.deposits, dtype=float),
        np.maximum(target_i, 1e-9),
        out=np.zeros_like(target_i, dtype=float),
        where=np.isfinite(target_i),
    )
    return {
        "share_below_runtime_buffer": float(np.mean(gap_i < 0.0)) if gap_i.size else 0.0,
        "mean_deposit_to_target_ratio": float(np.mean(ratio_i)) if ratio_i.size else 0.0,
        "buffer_shortfall_total": float(np.sum(np.maximum(0.0, -gap_i))) if gap_i.size else 0.0,
        "deposit_total": float(np.sum(np.asarray(hh.deposits, dtype=float))),
        "share_base_consumption_uncovered": float(np.mean(np.asarray(last_snapshot["base_consumption_gap_i"], dtype=float) < 0.0)),
    }


def _baseline_calibration_regime_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    regime_cfg = copy.deepcopy(cfg)
    params = regime_cfg.setdefault("parameters", {})
    params["baseline_calibration_enabled"] = False
    params["automation_disabled"] = True
    params["disable_trust"] = True
    params["disable_mortgage_policy"] = True
    params["disable_income_support"] = True
    return regime_cfg


def _run_baseline_calibration(cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    effective_cfg = copy.deepcopy(cfg)
    params = effective_cfg.get("parameters", {})
    if not bool(params.get("baseline_calibration_enabled", False)):
        return effective_cfg, None
    if not bool(params.get("use_population", False)):
        return effective_cfg, {"enabled": True, "skipped_reason": "population_disabled", "iterations_completed": 0}

    candidate_cfg = copy.deepcopy(effective_cfg)
    candidate_params = candidate_cfg.setdefault("parameters", {})
    pop_cfg = candidate_params.setdefault("population_config", {})
    if not isinstance(pop_cfg, dict):
        raise TypeError("population_config must be a dict for baseline calibration.")

    max_iters = max(1, int(candidate_params.get("baseline_calibration_max_iters", 8)))
    calib_quarters = max(1, int(candidate_params.get("baseline_calibration_quarters", 8)))
    n_quintiles = max(1, int(candidate_params.get("baseline_calibration_quintiles", 5)))
    alpha = max(0.0, float(candidate_params.get("baseline_calibration_alpha", 0.92)))
    damping = max(0.0, min(1.0, float(candidate_params.get("baseline_calibration_damping", 0.30))))
    tol_pct = max(0.0, float(candidate_params.get("baseline_calibration_tol_pct", 0.02)))
    reset_deposits = bool(candidate_params.get("baseline_calibration_reset_deposits_to_runtime_target", True))

    report: Dict[str, Any] = {
        "enabled": True,
        "iterations": [],
        "iterations_completed": 0,
        "quintile_boundaries_pct": _quintile_boundaries(n_quintiles),
        "alpha": alpha,
        "damping": damping,
        "calibration_quarters": calib_quarters,
        "max_target_change_pct": 0.0,
    }

    for iter_idx in range(max_iters):
        regime_cfg = _baseline_calibration_regime_cfg(candidate_cfg)
        sim = NewLoop(copy.deepcopy(regime_cfg))
        reset_stats = _apply_startup_income_buffer_reset(sim, reset_deposits=reset_deposits)
        if sim.hh is None or sim.hh.n <= 0:
            report["skipped_reason"] = "no_households"
            break

        startup_snapshot = _startup_solver_snapshot(sim)
        if startup_snapshot is None:
            report["skipped_reason"] = "startup_snapshot_failed"
            break

        hh = sim.hh
        hh.ensure_memos()
        rank_source = np.asarray(hh.wages0_q, dtype=float)
        current_base_targets = _bucket_means(np.asarray(hh.base_real_cons_q, dtype=float), rank_source, n_quintiles)
        p_cons = max(float(startup_snapshot["p_cons"]), 1e-9)
        sustainable_disp_real = np.maximum(0.0, np.asarray(startup_snapshot["disposable_income_i"], dtype=float) / p_cons)
        target_base_targets = alpha * _bucket_means(sustainable_disp_real, rank_source, n_quintiles)
        updated_targets = np.maximum(
            0.0,
            ((1.0 - damping) * current_base_targets) + (damping * target_base_targets),
        )

        denom = np.maximum(np.abs(current_base_targets), 1e-9)
        max_change_pct = float(np.max(np.abs(updated_targets - current_base_targets) / denom)) if updated_targets.size else 0.0

        for _ in range(calib_quarters):
            sim.step()

        rows = _visible_rows(sim.history)
        if rows:
            cons_first = max(abs(float(rows[0].get("real_consumption", 0.0))), 1e-9)
            wage_first = max(abs(float(rows[0].get("wages_total", 0.0))), 1e-9)
            cons_drift_pct = (
                (float(rows[-1].get("real_consumption", 0.0)) - float(rows[0].get("real_consumption", 0.0)))
                / cons_first
            )
            wage_drift_pct = (
                (float(rows[-1].get("wages_total", 0.0)) - float(rows[0].get("wages_total", 0.0)))
                / wage_first
            )
        else:
            cons_drift_pct = 0.0
            wage_drift_pct = 0.0

        report["iterations"].append(
            {
                "iteration": int(iter_idx + 1),
                "current_targets": current_base_targets.astype(float).tolist(),
                "target_targets": target_base_targets.astype(float).tolist(),
                "updated_targets": updated_targets.astype(float).tolist(),
                "max_target_change_pct": float(max_change_pct),
                "real_consumption_drift_pct": float(cons_drift_pct),
                "wage_base_drift_pct": float(wage_drift_pct),
                "startup_dti_w_p90": float(np.percentile(
                    np.divide(
                        np.asarray(startup_snapshot["debt_service_i"], dtype=float),
                        np.maximum(np.asarray(hh.wages0_q, dtype=float), 1e-9),
                    ),
                    90.0,
                )),
                "share_base_consumption_uncovered": float(np.mean(np.asarray(startup_snapshot["base_consumption_gap_i"], dtype=float) < 0.0)),
                "reset_stats": dict(reset_stats or {}),
            }
        )
        report["iterations_completed"] = int(iter_idx + 1)
        report["max_target_change_pct"] = float(max_change_pct)
        report["real_consumption_drift_pct"] = float(cons_drift_pct)
        report["wage_base_drift_pct"] = float(wage_drift_pct)
        report["final_targets"] = updated_targets.astype(float).tolist()

        pop_cfg["base_real_cons_by_wealth_pct"] = tuple(
            (float(boundary), float(target))
            for boundary, target in zip(_quintile_boundaries(n_quintiles), updated_targets.astype(float).tolist())
        )
        if updated_targets.size > 0:
            pop_cfg["base_real_cons_q"] = float(np.mean(updated_targets))

        if max_change_pct <= tol_pct:
            report["converged"] = True
            break
    else:
        report["converged"] = False

    return candidate_cfg, report


def _prepare_startup_sim(sim: NewLoop) -> Dict[str, Any] | None:
    if not (
        bool(sim.params.get("baseline_calibration_enabled", False))
        or bool(sim.params.get("startup_buffer_alignment_enabled", False))
    ):
        return None
    return _apply_startup_income_buffer_reset(
        sim,
        max_iter=int(sim.params.get("startup_buffer_alignment_max_iters", 8)),
        reset_deposits=bool(sim.params.get("baseline_calibration_reset_deposits_to_runtime_target", True)),
    )


def run_simulation(n_quarters: int = 80, cfg: Dict[str, Any] | None = None) -> SimulationRun:
    """Run NewLoop for n_quarters and return structured outputs."""
    base_cfg = copy.deepcopy(get_default_config() if cfg is None else cfg)
    effective_cfg, baseline_calibration = _run_baseline_calibration(base_cfg)

    startup_diag_sim = NewLoop(copy.deepcopy(effective_cfg))
    _prepare_startup_sim(startup_diag_sim)
    startup_diag = _startup_diagnostics(startup_diag_sim)

    sim = NewLoop(effective_cfg)
    _prepare_startup_sim(sim)
    before = _population_distribution_snapshot(sim)
    quarter_diag_q0: Dict[str, Any] | None = None
    quarter_diag_q10: Dict[str, Any] | None = None
    for _ in range(int(n_quarters)):
        sim.step()
        visible_t = len(sim.history) - 1
        if visible_t == 0:
            quarter_diag_q0 = _quarter_state_diagnostics(sim)
        elif visible_t == 10:
            quarter_diag_q10 = _quarter_state_diagnostics(sim)
    after = _population_distribution_snapshot(sim)
    pop_dist = {"before": before, "after": after} if (before is not None and after is not None) else None
    startup_diag_out = dict(startup_diag or {})
    quarter_compare = _quarter_comparison(quarter_diag_q0, quarter_diag_q10)
    if quarter_compare is not None:
        startup_diag_out["quarter_comparison"] = quarter_compare
    return SimulationRun(
        sim=sim,
        rows=_visible_rows(sim.history),
        population_distributions=pop_dist,
        startup_diagnostics=startup_diag_out,
        baseline_calibration=baseline_calibration,
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
