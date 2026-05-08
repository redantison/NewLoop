# Author: Roger Ison   roger@miximum.info
"""Reusable simulation-output helpers for simulation and Streamlit layers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import apply_economic_regime_overrides, get_default_config, normalize_economic_regime_name
from .engine import NewLoop
from .housing_affordability import compute_affordable_housing_profile
from .income_support import make_income_support_policy
from .mortgage import (
    annuity_factor,
    balance_from_orig_principal,
    payment_from_orig_principal,
    remaining_term,
    scheduled_payment_components,
)
from .newloop_types import TickResult
from .tax_policy import make_tax_policy

COMPREHENSIVE_WEALTH_DISTRIBUTION = True


def _startup_reset_deposits_enabled(sim: NewLoop) -> bool:
    """Return whether startup alignment should reseed household deposits."""
    regime = str(sim.params.get("economic_regime", "NewLoop")).strip()
    if regime == "OldLoop" and bool(sim.params.get("old_loop_startup_preserve_deposits", True)):
        return False
    return bool(sim.params.get("baseline_calibration_reset_deposits_to_runtime_target", True))


def _startup_deposit_blend(sim: NewLoop) -> float:
    """Return the startup deposit blend share after regime-specific overrides."""
    regime = str(sim.params.get("economic_regime", "NewLoop")).strip()
    if regime == "OldLoop" and bool(sim.params.get("old_loop_startup_preserve_deposits", True)):
        return 0.0
    return max(0.0, min(1.0, float(sim.params.get("startup_buffer_alignment_deposit_blend", 0.0))))


def _old_to_new_transition_quarters(cfg: Dict[str, Any] | Dict[str, float] | None) -> int:
    """Return the visible-quarter handoff point for OldToNew runs."""
    if not isinstance(cfg, dict):
        return 0
    params = cfg.get("parameters", cfg)
    if not isinstance(params, dict):
        return 0
    return max(0, int(params.get("old_to_new_transition_quarters", 0)))


def _old_to_new_transition_mode(cfg: Dict[str, Any] | None) -> str:
    """Return the configured OldToNew experiment after the OldLoop settling period."""
    if not isinstance(cfg, dict):
        return "NewLoopPolicies"
    params = cfg.get("parameters", cfg)
    if not isinstance(params, dict):
        return "NewLoopPolicies"
    raw = params.get("old_to_new_transition_mode", None)
    if raw is None:
        return (
            "NewLoopPolicies"
            if bool(params.get("old_to_new_launch_newloop_policies", True))
            else "AutomationOnly"
        )
    normalized = str(raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"stayoldloop", "oldloop", "oldlooponly", "none", "baseline"}:
        return "StayOldLoop"
    if normalized in {"automationonly", "automation", "oldloopautomation", "automationoldloop"}:
        return "AutomationOnly"
    return "NewLoopPolicies"


def _old_to_new_launch_newloop_policies(cfg: Dict[str, Any] | None) -> bool:
    """Return whether OldToNew should switch into the NewLoop policy stack."""
    return _old_to_new_transition_mode(cfg) == "NewLoopPolicies"


def _old_to_new_old_phase_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the OldLoop phase config used before the visible handoff."""
    phase_cfg = copy.deepcopy(cfg)
    params = phase_cfg.setdefault("parameters", {})
    params["economic_regime"] = "OldLoop"
    params["tax_policy_mode"] = "auto"
    params["neutral_warmup_quarters"] = 0
    return apply_economic_regime_overrides(phase_cfg)


def _old_to_new_new_phase_cfg(cfg: Dict[str, Any], *, switch_t: int | None = None) -> Dict[str, Any]:
    """Build the NewLoop phase config used after the visible handoff."""
    phase_cfg = copy.deepcopy(cfg)
    params = phase_cfg.setdefault("parameters", {})
    base_automation_start = max(0, int(params.get("automation_start_quarter", 0)))
    params["economic_regime"] = "NewLoop"
    params["neutral_warmup_quarters"] = 0
    if switch_t is not None:
        params["automation_start_quarter"] = int(max(0, switch_t) + base_automation_start)
    return apply_economic_regime_overrides(phase_cfg)


def _old_to_new_oldloop_decay_phase_cfg(cfg: Dict[str, Any], *, switch_t: int | None = None) -> Dict[str, Any]:
    """Build the OldLoop-with-automation phase config used for unmanaged OldLoop decay."""
    phase_cfg = _old_to_new_old_phase_cfg(cfg)
    params = phase_cfg.setdefault("parameters", {})
    src_params = cfg.get("parameters", {}) if isinstance(cfg.get("parameters", {}), dict) else {}
    base_automation_start = max(0, int(src_params.get("automation_start_quarter", 0)))
    params["automation_disabled"] = bool(src_params.get("automation_disabled", False))
    if switch_t is not None:
        params["automation_start_quarter"] = int(max(0, switch_t) + base_automation_start)
    return phase_cfg


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


def _visible_rows(history: Sequence[TickResult], start_idx: int = 0) -> List[Dict[str, Any]]:
    """Convert visible history rows and rebase t so visible Q0 starts at zero."""
    rows = history_to_rows(history[start_idx:])
    if not rows:
        return rows
    t0 = int(rows[0].get("t", 0))
    for row in rows:
        row["t"] = int(row.get("t", 0)) - t0
    return rows


def _household_wealth_snapshot(sim: NewLoop, *, comprehensive: bool = COMPREHENSIVE_WEALTH_DISTRIBUTION) -> Dict[str, np.ndarray]:
    """Build household wealth vectors using either the narrow or comprehensive definition."""
    if sim.hh is None or sim.hh.n <= 0:
        return {
            "wealth": np.asarray([], dtype=float),
            "deposits": np.asarray([], dtype=float),
            "housing": np.asarray([], dtype=float),
            "private_equity": np.asarray([], dtype=float),
            "trust_value": np.asarray([], dtype=float),
            "loans": np.asarray([], dtype=float),
        }

    hh = sim.hh
    deposits_i = np.asarray(hh.deposits, dtype=float)
    housing_i = np.asarray(hh.housing_escrow, dtype=float)
    mort_i = np.asarray(hh.mortgage_loans, dtype=float)
    rev_i = np.asarray(hh.revolving_loans, dtype=float)
    loan_i = mort_i + rev_i
    equity_i = np.zeros_like(deposits_i, dtype=float)
    trust_i = np.zeros_like(deposits_i, dtype=float)

    if comprehensive and deposits_i.size and (loan_i.size == deposits_i.size):
        wages0_i = np.asarray(hh.wages0_q, dtype=float)
        wages0_sum = float(wages0_i.sum()) if wages0_i.shape[0] == deposits_i.shape[0] else 0.0
        if wages0_sum > 0.0:
            wealth_weights = wages0_i / wages0_sum
        else:
            wealth_weights = np.full(deposits_i.shape[0], 1.0 / float(deposits_i.shape[0]), dtype=float)
        equity_weights = sim._household_equity_weights(fallback_weights=wealth_weights)

        price_level = float(sim.state.get("price_level", 1.0))
        if price_level <= 0.0:
            price_level = 1e-9

        def node_share_frac(holder: str, issuer: str, key: str) -> float:
            shares_out = float(sim.nodes[issuer].get("shares_outstanding", 0.0))
            if shares_out <= 0.0:
                return 0.0
            frac = float(sim.nodes[holder].get(key, 0.0)) / shares_out
            return max(0.0, min(1.0, frac))

        fa_equity_proxy = sim._firm_balance_sheet_equity_proxy("IS", price_level)
        fh_equity_proxy = sim._firm_balance_sheet_equity_proxy("PS", price_level)
        bank_equity_proxy = sim._firm_balance_sheet_equity_proxy("BANK", price_level)

        hh_equity_total = (
            node_share_frac("HH", "IS", "shares_IS") * fa_equity_proxy
            + node_share_frac("HH", "PS", "shares_PS") * fh_equity_proxy
            + node_share_frac("HH", "BANK", "shares_BANK") * bank_equity_proxy
        )
        trust_equity_total = (
            node_share_frac("FUND", "IS", "shares_IS") * fa_equity_proxy
            + node_share_frac("FUND", "PS", "shares_PS") * fh_equity_proxy
            + node_share_frac("FUND", "BANK", "shares_BANK") * bank_equity_proxy
        )
        trust_value_total = (
            float(sim.nodes["FUND"].get("deposits", 0.0))
            + trust_equity_total
            - float(sim.nodes["FUND"].get("loans", 0.0))
        )
        equity_i = equity_weights * max(0.0, float(hh_equity_total))
        trust_i = np.full(deposits_i.shape[0], trust_value_total / float(deposits_i.shape[0]), dtype=float)

    wealth_i = deposits_i + housing_i + equity_i + trust_i - loan_i
    return {
        "wealth": wealth_i,
        "deposits": deposits_i,
        "housing": housing_i,
        "private_equity": equity_i,
        "trust_value": trust_i,
        "loans": loan_i,
    }


def _population_distribution_snapshot(
    sim: NewLoop,
    *,
    sol: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Capture household income and net-worth vectors for before/after plotting."""
    if sim.hh is None or sim.hh.n <= 0:
        return None
    if sol is None:
        raise ValueError("_population_distribution_snapshot requires a precomputed solver result.")

    hh = sim.hh
    n = int(hh.n)

    wealth_snapshot = _household_wealth_snapshot(sim)
    dep_i = np.asarray(hh.deposits, dtype=float)
    housing_i = np.asarray(hh.housing_escrow, dtype=float)
    mort_i = np.asarray(hh.mortgage_loans, dtype=float)
    rev_i = np.asarray(hh.revolving_loans, dtype=float)
    loan_i = mort_i + rev_i
    income_i = np.asarray(sol.get("y", []), dtype=float)
    if income_i.shape[0] != n:
        income_i = np.asarray(hh.prev_income, dtype=float)
    if income_i.shape[0] != n:
        income_i = np.asarray(hh.wages0_q, dtype=float)

    wealth_i = np.asarray(wealth_snapshot.get("wealth", np.asarray([], dtype=float)), dtype=float)
    renters = (mort_i <= 1e-12) & (housing_i <= 1e-12)
    mortgagors = mort_i > 1e-12
    outright_owners = (mort_i <= 1e-12) & (housing_i > 1e-12)
    initial_tenure_code = np.asarray(getattr(hh, "initial_tenure_code", np.asarray([], dtype=int)), dtype=int)
    if initial_tenure_code.shape[0] != n:
        initial_tenure_code = np.ones(n, dtype=int)
    initial_outright_owner = initial_tenure_code == 2
    initial_other = ~initial_outright_owner
    income_groups_policy: Dict[str, List[float]] = {}
    if sol is not None:
        wages_i = np.asarray(sol.get("wages_i", []), dtype=float)
        income_tax_i = np.asarray(sol.get("income_tax_i", []), dtype=float)
        vat_credit_i = np.asarray(sol.get("vat_credit_i", []), dtype=float)
        if wages_i.shape[0] == n and income_tax_i.shape[0] == n and vat_credit_i.shape[0] == n:
            has_vat_credit = vat_credit_i > 1e-9
            pays_income_tax = income_tax_i > 1e-9
            income_groups_policy = {
                "vat_credit_no_income_tax": income_i[has_vat_credit & (~pays_income_tax)].astype(float).tolist(),
                "vat_credit_and_income_tax": income_i[has_vat_credit & pays_income_tax].astype(float).tolist(),
                "no_vat_credit_no_income_tax": income_i[(~has_vat_credit) & (~pays_income_tax)].astype(float).tolist(),
                "no_vat_credit_and_income_tax": income_i[(~has_vat_credit) & pays_income_tax].astype(float).tolist(),
            }

    return {
        "price_level": float(sim.state.get("price_level", 1.0)),
        "income": income_i.astype(float).tolist(),
        "income_groups": {
            "renters": income_i[renters].astype(float).tolist(),
            "mortgagors": income_i[mortgagors].astype(float).tolist(),
            "outright_owners": income_i[outright_owners].astype(float).tolist(),
        },
        "income_groups_initial_owner": {
            "initial_outright_owner": income_i[initial_outright_owner].astype(float).tolist(),
            "initial_other_households": income_i[initial_other].astype(float).tolist(),
        },
        "income_groups_policy": income_groups_policy,
        "wealth_definition": "comprehensive" if COMPREHENSIVE_WEALTH_DISTRIBUTION else "narrow",
        "wealth": wealth_i.astype(float).tolist(),
        "wealth_components": {
            key: np.asarray(val, dtype=float).tolist()
            for key, val in wealth_snapshot.items()
            if key != "wealth"
        },
        "wealth_groups": {
            "renters": wealth_i[renters].astype(float).tolist(),
            "mortgagors": wealth_i[mortgagors].astype(float).tolist(),
            "outright_owners": wealth_i[outright_owners].astype(float).tolist(),
        },
    }


def _startup_diagnostics(
    sim: NewLoop,
    *,
    snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Summarize quarter-0 household buffer consistency and debt stress."""
    if sim.hh is None or sim.hh.n <= 0:
        return None
    if snapshot is None:
        raise ValueError("_startup_diagnostics requires a precomputed startup snapshot.")

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
    op_margin_info = 0.0
    op_margin_phys = 0.0
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
        rev_fa = float(sol.get("rev_fa", 0.0))
        rev_fh = float(sol.get("rev_fh", 0.0))
        w_fa = float(sol.get("w_fa", 0.0))
        w_fh = float(sol.get("w_fh", 0.0))
        overhead_fa = float(sol.get("overhead_fa", 0.0))
        overhead_fh = float(sol.get("overhead_fh", 0.0))
        input_cost_fa = float(sol.get("input_cost_fa", 0.0))
        input_cost_fh = float(sol.get("input_cost_fh", 0.0))
        if rev_fa > 1e-9:
            op_margin_info = (rev_fa - w_fa - overhead_fa - input_cost_fa) / rev_fa
        if rev_fh > 1e-9:
            op_margin_phys = (rev_fh - w_fh - overhead_fh - input_cost_fh) / rev_fh
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
            "input_cost_info_total": float(input_cost_fa),
            "input_cost_phys_total": float(input_cost_fh),
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
        "startup_op_margin_info": float(op_margin_info),
        "startup_op_margin_phys": float(op_margin_phys),
        "startup_ums_deposits": float(sim.nodes["UMS"].get("deposits", 0.0)),
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


def _quarter_state_diagnostics(
    sim: NewLoop,
    *,
    snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if sim.hh is None or sim.hh.n <= 0 or not sim.history:
        return None
    if snapshot is None:
        raise ValueError("_quarter_state_diagnostics requires a precomputed startup snapshot.")

    hh = sim.hh
    hh.ensure_memos()
    if snapshot is None:
        return None

    row = asdict(sim.history[-1])
    wages_i = np.asarray(hh.wages0_q, dtype=float)
    deposits_i = np.asarray(hh.deposits, dtype=float)
    target_buffer_i = np.asarray(snapshot["target_buffer_i"], dtype=float)
    disposable_income_i = np.asarray(snapshot["disposable_income_i"], dtype=float)
    debt_service_i = np.asarray(snapshot["debt_service_i"], dtype=float)
    base_cons_gap_i = np.asarray(snapshot["base_consumption_gap_i"], dtype=float)
    base_cons_nom_i = np.asarray(snapshot.get("base_consumption_nom_i", []), dtype=float)
    p_cons = max(float(snapshot["p_cons"]), 1e-9)
    base_real_i = np.asarray(hh.base_real_cons_q, dtype=float)
    mpc_i = np.asarray(hh.mpc_q, dtype=float)
    disp_income_real_i = disposable_income_i / p_cons
    mpc_income_real_i = mpc_i * disp_income_real_i
    if base_cons_nom_i.shape == base_real_i.shape:
        core_real_i = np.maximum(0.0, base_cons_nom_i / p_cons)
    else:
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


def _startup_solver_snapshot(
    sim: NewLoop,
    *,
    sol: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
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

    if sol is None:
        sim.tax_policy.warm_start_anchor_if_needed(
            state=sim.state,
            baseline_wages_i=hh.wages0_q,
            price_level=float(sim.state.get("price_level", 1.0)),
        )

    if sol is None and not bool(sim.params.get("disable_income_support", False)):
        sim.income_support_policy.warm_start_anchor_if_needed(
            state=sim.state,
            baseline_wages_i=hh.wages0_q,
            price_level=float(sim.state.get("price_level", 1.0)),
        )

    if sol is None:
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
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

    if rev_interest_i.shape == wages_i.shape and mort_pay_req_i.shape == wages_i.shape:
        debt_service_i = np.maximum(0.0, rev_interest_i) + np.maximum(0.0, mort_pay_req_i)
    else:
        debt_service_i = np.maximum(0.0, interest_hh_i) if interest_hh_i.shape == wages_i.shape else raw_interest_i

    renter_rent_q = np.maximum(0.0, np.asarray(hh.renter_rent_q, dtype=float))
    if renter_rent_q.shape != wages_i.shape:
        renter_rent_q = np.zeros_like(wages_i, dtype=float)
    mort_payment_sched_q = np.maximum(0.0, np.asarray(hh.mort_payment_sched_q, dtype=float))
    if mort_payment_sched_q.shape != wages_i.shape:
        mort_payment_sched_q = np.zeros_like(wages_i, dtype=float)

    consumption_targets = sim._household_consumption_targets(
        y_guess=disposable_income_i,
        dep0=np.asarray(hh.deposits, dtype=float),
        base_real=base_real_i,
        mpc=mpc_i,
        liquid_buffer_months_target=target_months_i,
        baseline_wages_i=wages_i,
        p_cons=p_cons,
        rev_interest_nom=np.maximum(0.0, rev_interest_i),
        rev_balance_nom=np.maximum(0.0, rev_i),
        mort_payment_nom=np.maximum(0.0, mort_pay_req_i if mort_pay_req_i.shape == wages_i.shape else mort_payment_sched_q),
        renter_rent_q=renter_rent_q,
    )
    c_real_core_i = np.asarray(consumption_targets["c_real_core"], dtype=float)
    target_buffer_i = np.asarray(consumption_targets["target_buffer_nom"], dtype=float)
    base_consumption_nom_i = np.asarray(consumption_targets["c_hh_nom_income"], dtype=float)
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
    sim.nodes["HOUSING"].set(
        "deposits",
        float(sim.state.get("housing_financing_deposits_total", 0.0)),
    )
    sim.nodes["HH"].set("loans", hh.sum_loans())
    bank = sim.nodes["BANK"]
    dep_liab = float(sim._sum_deposits_all())
    loan_assets = float(sim._sum_loans_borrowers())
    bank.set("deposit_liab", dep_liab)
    bank.set("loan_assets", loan_assets)
    bank.set(
        "reserves",
        dep_liab + float(bank.get("equity", 0.0)) - loan_assets,
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
        from . import population as pop_mod
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
    deposit_blend = _startup_deposit_blend(sim)
    for _ in range(max(1, int(max_iter))):
        snapshot = _startup_solver_snapshot(sim)
        if snapshot is None:
            break
        last_snapshot = snapshot

        hh.prev_income = np.asarray(snapshot["disposable_income_i"], dtype=float).astype(float, copy=True)
        if hasattr(hh, "prev_perm_income"):
            prev_perm_income = (
                np.asarray(hh.prev_perm_income, dtype=float)
                if np.asarray(hh.prev_perm_income, dtype=float).shape == hh.prev_income.shape
                else np.maximum(0.0, hh.prev_income)
            )
            lambda_q = sim._old_loop_perm_income_update_rate_q()
            hh.prev_perm_income = (
                ((1.0 - lambda_q) * np.maximum(0.0, prev_perm_income))
                + (lambda_q * np.maximum(0.0, hh.prev_income))
            ).astype(float, copy=True)
        if reset_deposits:
            target_i = np.maximum(0.0, np.asarray(snapshot["target_buffer_i"], dtype=float))
            hh.deposits = target_i.astype(float, copy=True)
        elif deposit_blend > 0.0:
            target_i = np.maximum(0.0, np.asarray(snapshot["target_buffer_i"], dtype=float))
            hh.deposits = (((1.0 - deposit_blend) * np.asarray(hh.deposits, dtype=float)) + (deposit_blend * target_i)).astype(float, copy=True)

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
    if normalize_economic_regime_name(params.get("economic_regime", "NewLoop")) == "OldToNew":
        params["economic_regime"] = "OldLoop"
        params["neutral_warmup_quarters"] = 0
    params["baseline_calibration_enabled"] = False
    params["automation_disabled"] = True
    params["disable_trust"] = True
    # Hidden startup-consistency regime:
    # preserve the configured household/fiscal structure, but suppress
    # payout/recycling behavior that can obscure the sustainable startup
    # consumption ladder we are trying to fit.
    params["dividend_payout_rate_firms"] = 0.0
    params["dividend_payout_rate_firms_mature_max"] = 0.0
    params["dividend_payout_rate_bank"] = 0.0
    params["sector_surplus_distribution_share"] = 0.0
    params["gov_tax_rebate_rate"] = 0.0
    params["sector_capex_share_min"] = 0.0
    params["sector_capex_share_max"] = 0.0
    params["sector_capex_gap_close_rate"] = 0.0
    params["sector_capex_growth_cap_rate_q"] = 0.0
    params["sector_install_rate_q"] = 0.0
    return regime_cfg


def _run_baseline_calibration(cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    effective_cfg = copy.deepcopy(cfg)
    uncalibrated_cfg = copy.deepcopy(effective_cfg)
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
        "mode": "startup_consistency",
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
            report["converged"] = False
            return uncalibrated_cfg, report

        startup_snapshot = _startup_solver_snapshot(sim)
        if startup_snapshot is None:
            report["skipped_reason"] = "startup_snapshot_failed"
            report["converged"] = False
            return uncalibrated_cfg, report

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

        try:
            for _ in range(calib_quarters):
                sim.step()
        except Exception as exc:
            report["skipped_reason"] = "infeasible_hidden_baseline_regime"
            report["error"] = str(exc)
            report["failed_iteration"] = int(iter_idx + 1)
            report["converged"] = False
            return uncalibrated_cfg, report

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
    reset_stats = _apply_startup_income_buffer_reset(
        sim,
        max_iter=int(sim.params.get("startup_buffer_alignment_max_iters", 8)),
        reset_deposits=_startup_reset_deposits_enabled(sim),
    )
    sim._bootstrap_startup_lagged_retained()
    return reset_stats


def _startup_mortgage_money_seed_estimate(sim: NewLoop) -> Dict[str, Any] | None:
    """Estimate the missing historical mortgage-money stock from inherited runoff."""
    regime = normalize_economic_regime_name(sim.params.get("economic_regime", "NewLoop"))
    if regime != "OldLoop" or sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    horizon_q = max(
        0,
        int(sim.params.get("old_loop_startup_mortgage_money_seed_horizon_q", 20)),
    )
    seed_fraction = max(
        0.0,
        float(sim.params.get("old_loop_startup_mortgage_money_seed_fraction", 0.44)),
    )

    balance = np.maximum(0.0, np.asarray(hh.mortgage_loans, dtype=float)).copy()
    rate_q = np.maximum(0.0, np.asarray(hh.mort_rate_q, dtype=float)).copy()
    payment_q = np.maximum(0.0, np.asarray(hh.mort_payment_sched_q, dtype=float)).copy()
    term_q = np.maximum(0.0, np.asarray(hh.mort_term_q, dtype=float))
    age_q = np.maximum(0.0, np.asarray(hh.mort_age_q, dtype=float))
    if not (
        balance.shape == rate_q.shape == payment_q.shape == term_q.shape == age_q.shape
    ):
        return None

    active = balance > 1e-9
    remaining_q = remaining_term(term_q, age_q)
    n = float(max(1, hh.n))
    scheduled_payment_total = 0.0
    scheduled_interest_total = 0.0
    scheduled_principal_total = 0.0
    scheduled_final_principal_total = 0.0
    runoff_rows: List[Dict[str, Any]] = []

    for q in range(horizon_q + 1):
        due_i, interest_i, principal_i = scheduled_payment_components(
            balance,
            rate_q,
            payment_q,
            remaining_q,
        )
        active_q = balance > 1e-9
        final_q = active_q & (remaining_q <= 1.0 + 1e-12)
        payment_total = float(np.sum(np.maximum(0.0, due_i)))
        interest_total = float(np.sum(np.maximum(0.0, interest_i)))
        principal_total = float(np.sum(np.maximum(0.0, principal_i)))
        final_principal_total = float(np.sum(np.maximum(0.0, principal_i[final_q])))
        scheduled_payment_total += payment_total
        scheduled_interest_total += interest_total
        scheduled_principal_total += principal_total
        scheduled_final_principal_total += final_principal_total
        if q in {0, 1, 2, 5, 10, 15, 20, 40, 60, 79, horizon_q}:
            runoff_rows.append(
                {
                    "q": int(q),
                    "active_count": int(np.sum(active_q)),
                    "payment_total": payment_total,
                    "interest_total": interest_total,
                    "principal_total": principal_total,
                    "payment_per_h": payment_total / n,
                    "interest_per_h": interest_total / n,
                    "principal_per_h": principal_total / n,
                    "final_period_count": int(np.sum(final_q)),
                    "final_principal_total": final_principal_total,
                }
            )
        balance = np.maximum(0.0, balance - principal_i)
        remaining_q = np.maximum(0.0, remaining_q - 1.0)

    initial_balance_total = float(np.sum(np.maximum(0.0, np.asarray(hh.mortgage_loans, dtype=float))))
    initial_sched_payment_total = float(np.sum(np.maximum(0.0, payment_q[active])))
    if np.any(active):
        rem_active = np.maximum(0.0, remaining_term(term_q, age_q)[active])
        remaining_stats = {
            "remaining_q_min": float(np.min(rem_active)),
            "remaining_q_p10": float(np.percentile(rem_active, 10.0)),
            "remaining_q_median": float(np.median(rem_active)),
            "remaining_q_p90": float(np.percentile(rem_active, 90.0)),
            "remaining_q_max": float(np.max(rem_active)),
        }
    else:
        remaining_stats = {
            "remaining_q_min": 0.0,
            "remaining_q_p10": 0.0,
            "remaining_q_median": 0.0,
            "remaining_q_p90": 0.0,
            "remaining_q_max": 0.0,
        }

    seed_total = seed_fraction * scheduled_principal_total
    out: Dict[str, Any] = {
        "horizon_q": int(horizon_q),
        "included_quarter_count": int(horizon_q + 1),
        "seed_fraction": float(seed_fraction),
        "active_count": int(np.sum(active)),
        "initial_balance_total": initial_balance_total,
        "initial_balance_per_h": initial_balance_total / n,
        "initial_sched_payment_total": initial_sched_payment_total,
        "initial_sched_payment_per_h": initial_sched_payment_total / n,
        "scheduled_payment_total": float(scheduled_payment_total),
        "scheduled_interest_total": float(scheduled_interest_total),
        "scheduled_principal_total": float(scheduled_principal_total),
        "scheduled_final_principal_total": float(scheduled_final_principal_total),
        "scheduled_payment_per_h": float(scheduled_payment_total / n),
        "scheduled_interest_per_h": float(scheduled_interest_total / n),
        "scheduled_principal_per_h": float(scheduled_principal_total / n),
        "estimated_seed_total": float(seed_total),
        "estimated_seed_per_h": float(seed_total / n),
        "remaining_balance_after_horizon_total": float(np.sum(balance)),
        "remaining_balance_after_horizon_per_h": float(np.sum(balance) / n),
        "runoff_rows": runoff_rows,
    }
    out.update(remaining_stats)
    return out


def _distribute_startup_mortgage_money_seed(sim: NewLoop) -> Dict[str, Any] | None:
    """Create estimated historical mortgage-money deposits at OldLoop startup."""
    regime = normalize_economic_regime_name(sim.params.get("economic_regime", "NewLoop"))
    if regime != "OldLoop" or sim.hh is None or sim.hh.n <= 0:
        return None
    if not bool(sim.params.get("old_loop_startup_mortgage_money_seed_enabled", True)):
        return None

    estimate = _startup_mortgage_money_seed_estimate(sim)
    if estimate is None:
        return None
    seed_total = max(0.0, float(estimate.get("estimated_seed_total", 0.0)))
    if seed_total <= 1e-12:
        return None

    hh = sim.hh
    hh.ensure_memos()
    n = int(hh.n)
    deposits_before = np.asarray(hh.deposits, dtype=float).copy()
    mort = np.maximum(0.0, np.asarray(hh.mortgage_loans, dtype=float))
    housing = np.maximum(0.0, np.asarray(hh.housing_escrow, dtype=float))
    mpc = np.maximum(0.0, np.asarray(hh.mpc_q, dtype=float))
    if mort.shape[0] != n or housing.shape[0] != n or mpc.shape[0] != n:
        return None

    raw_universal = max(0.0, float(sim.params.get("old_loop_startup_mortgage_money_seed_universal_share", 0.50)))
    raw_liquidity = max(0.0, float(sim.params.get("old_loop_startup_mortgage_money_seed_liquidity_share", 0.35)))
    raw_seller = max(0.0, float(sim.params.get("old_loop_startup_mortgage_money_seed_seller_share", 0.15)))
    share_sum = raw_universal + raw_liquidity + raw_seller
    if share_sum <= 1e-12:
        return None
    universal_share = raw_universal / share_sum
    liquidity_share = raw_liquidity / share_sum
    seller_share = raw_seller / share_sum

    allocation = np.zeros(n, dtype=float)
    universal_total = seed_total * universal_share
    liquidity_total = seed_total * liquidity_share
    seller_total = seed_total * seller_share

    if universal_total > 0.0:
        allocation += universal_total / float(n)

    snapshot = _startup_solver_snapshot(sim)
    target_buffer = (
        np.asarray(snapshot.get("target_buffer_i", np.zeros(n, dtype=float)), dtype=float)
        if isinstance(snapshot, dict)
        else np.zeros(n, dtype=float)
    )
    if target_buffer.shape[0] != n:
        target_buffer = np.zeros(n, dtype=float)
    liquidity_gap = np.maximum(0.0, target_buffer - deposits_before)
    liquidity_weight = liquidity_gap * np.maximum(0.0, mpc)
    if liquidity_total > 0.0:
        weight_sum = float(np.sum(liquidity_weight))
        if weight_sum <= 1e-12:
            liquidity_weight = np.ones(n, dtype=float)
            weight_sum = float(n)
        allocation += liquidity_total * (liquidity_weight / weight_sum)

    active_mort = mort > 1e-9
    nonmort = ~active_mort
    seller_weight = np.zeros(n, dtype=float)
    owner_nonmort = nonmort & (housing > 1e-9)
    renter_nonmort = nonmort & (~owner_nonmort)
    seller_weight[owner_nonmort] = housing[owner_nonmort]
    if np.any(owner_nonmort) and np.any(renter_nonmort):
        renter_base = 0.25 * float(np.median(housing[owner_nonmort]))
        seller_weight[renter_nonmort] = max(0.0, renter_base)
    elif np.any(renter_nonmort):
        seller_weight[renter_nonmort] = 1.0
    if seller_total > 0.0:
        seller_weight_sum = float(np.sum(seller_weight))
        if seller_weight_sum <= 1e-12 and np.any(nonmort):
            seller_weight[nonmort] = 1.0
            seller_weight_sum = float(np.sum(seller_weight))
        if seller_weight_sum > 1e-12:
            allocation += seller_total * (seller_weight / seller_weight_sum)

    actual_total = float(np.sum(allocation))
    if actual_total <= 1e-12:
        return None
    hh.deposits = (deposits_before + allocation).astype(float, copy=True)
    _sync_startup_household_state(sim)

    mortgagor_seed_total = float(np.sum(allocation[active_mort]))
    nonmort_seed_total = float(np.sum(allocation[nonmort]))
    liquidity_recipient_count = int(np.sum(liquidity_weight > 1e-12))
    seller_recipient_count = int(np.sum(seller_weight > 1e-12))
    out: Dict[str, Any] = {
        "enabled": True,
        "seed_total": actual_total,
        "seed_per_h": actual_total / float(n),
        "universal_total": float(universal_total),
        "liquidity_total": float(liquidity_total),
        "seller_total": float(seller_total),
        "universal_share": float(universal_share),
        "liquidity_share": float(liquidity_share),
        "seller_share": float(seller_share),
        "mortgagor_seed_total": mortgagor_seed_total,
        "mortgagor_seed_per_active": mortgagor_seed_total / max(1.0, float(np.sum(active_mort))),
        "nonmortgagor_seed_total": nonmort_seed_total,
        "nonmortgagor_seed_per_active": nonmort_seed_total / max(1.0, float(np.sum(nonmort))),
        "liquidity_recipient_count": liquidity_recipient_count,
        "seller_recipient_count": seller_recipient_count,
        "mean_deposit_before": float(np.mean(deposits_before)),
        "mean_deposit_after": float(np.mean(np.asarray(hh.deposits, dtype=float))),
        "mortgagor_mean_deposit_after": float(np.mean(np.asarray(hh.deposits, dtype=float)[active_mort])) if np.any(active_mort) else 0.0,
        "nonmortgagor_mean_deposit_after": float(np.mean(np.asarray(hh.deposits, dtype=float)[nonmort])) if np.any(nonmort) else 0.0,
        "estimate": estimate,
    }
    return out


def _reunderwrite_old_loop_startup_housing(sim: NewLoop) -> Dict[str, Any] | None:
    """Reset OldLoop startup housing burdens to the model's own settled income path."""
    regime = str(sim.params.get("economic_regime", "NewLoop")).strip()
    if regime != "OldLoop" or sim.hh is None or sim.hh.n <= 0:
        return None

    hh = sim.hh
    hh.ensure_memos()
    snapshot = _startup_solver_snapshot(sim)
    if snapshot is None:
        return None
    sol = snapshot.get("sol")
    if not isinstance(sol, dict):
        return None

    n = int(hh.n)
    wages_i = np.asarray(sol.get("wages_i", hh.wages0_q), dtype=float)
    if wages_i.shape[0] != n:
        wages_i = np.asarray(hh.wages0_q, dtype=float)
    div_i = np.asarray(sol.get("div_i", np.zeros(n, dtype=float)), dtype=float)
    if div_i.shape[0] != n:
        div_i = np.zeros(n, dtype=float)
    rev_interest_i = np.asarray(sol.get("rev_interest_i", np.zeros(n, dtype=float)), dtype=float)
    if rev_interest_i.shape[0] != n:
        rev_interest_i = np.zeros(n, dtype=float)
    uis = max(0.0, float(sol.get("uis", 0.0)))

    gross_income_q = np.maximum(0.0, wages_i) + np.maximum(0.0, div_i) + uis
    affordability = compute_affordable_housing_profile(
        gross_income_q,
        np.maximum(0.0, np.asarray(hh.wages0_q, dtype=float)),
        sim.params,
        existing_fixed_obligations_q=np.maximum(0.0, rev_interest_i),
    )
    housing_payment_target_q = np.maximum(
        0.0,
        np.asarray(affordability["housing_payment_target_q"], dtype=float),
    )

    mortgage_loans = np.asarray(hh.mortgage_loans, dtype=float)
    mort_rate_q = np.asarray(hh.mort_rate_q, dtype=float)
    mort_term_q = np.asarray(hh.mort_term_q, dtype=float)
    mort_age_q = np.asarray(hh.mort_age_q, dtype=float)
    active_mort = mortgage_loans > 1e-9

    if np.any(active_mort):
        active_rate_q = np.maximum(0.0, mort_rate_q.copy())
        default_rate_q = float(sim.params.get("mortgage_fixed_rate_q", 0.0))
        active_rate_q = np.where(active_rate_q > 1e-12, active_rate_q, default_rate_q)

        active_term_q = np.maximum(1.0, mort_term_q.copy())
        default_term_q = float(sim.params.get("mortgage_term_quarters", 60.0))
        active_term_q = np.where(active_term_q > 1e-12, active_term_q, default_term_q)
        supportable_orig_principal = np.zeros(n, dtype=float)
        supportable_orig_principal[active_mort] = (
            housing_payment_target_q[active_mort]
            * annuity_factor(active_rate_q[active_mort], active_term_q[active_mort])
        )
        supportable_balance = np.zeros(n, dtype=float)
        supportable_balance[active_mort] = balance_from_orig_principal(
            supportable_orig_principal[active_mort],
            active_rate_q[active_mort],
            active_term_q[active_mort],
            mort_age_q[active_mort],
        )

        new_orig_principal = np.asarray(hh.mort_orig_principal, dtype=float).copy()
        new_payment_sched_q = np.asarray(hh.mort_payment_sched_q, dtype=float).copy()
        new_balance = mortgage_loans.copy()

        new_orig_principal[active_mort] = np.minimum(
            np.maximum(0.0, np.asarray(hh.mort_orig_principal, dtype=float)[active_mort]),
            np.maximum(0.0, supportable_orig_principal[active_mort]),
        )
        new_payment_sched_q[active_mort] = payment_from_orig_principal(
            new_orig_principal[active_mort],
            active_rate_q[active_mort],
            active_term_q[active_mort],
        )
        new_balance[active_mort] = np.minimum(
            np.maximum(0.0, mortgage_loans[active_mort]),
            np.maximum(0.0, supportable_balance[active_mort]),
        )
        hh.mortgage_loans = new_balance.astype(float, copy=True)
        hh.mort_orig_principal = new_orig_principal.astype(float, copy=True)
        hh.mort_payment_sched_q = new_payment_sched_q.astype(float, copy=True)

    renter_rent_q = np.asarray(hh.renter_rent_q, dtype=float)
    hh.renter_rent_q = np.minimum(
        np.maximum(0.0, renter_rent_q),
        housing_payment_target_q,
    ).astype(float, copy=True)

    _sync_startup_household_state(sim)
    return {
        "active_mortgages": float(np.sum(active_mort)),
        "mean_target_housing_payment_q": float(np.mean(housing_payment_target_q)) if housing_payment_target_q.size else 0.0,
        "mean_mortgage_payment_q": float(np.mean(np.asarray(hh.mort_payment_sched_q, dtype=float)[active_mort])) if np.any(active_mort) else 0.0,
        "mean_rent_q": float(np.mean(np.asarray(hh.renter_rent_q, dtype=float))) if hh.renter_rent_q.size else 0.0,
    }


def _reseed_visible_start_capacity(sim: NewLoop) -> Dict[str, Any] | None:
    """Re-anchor sector capacity after neutral warmup using visible-regime demand."""
    snapshot = _startup_solver_snapshot(sim)
    if snapshot is None:
        return None

    sol = snapshot.get("sol")
    if not isinstance(sol, dict):
        return None

    p_now = max(float(sim.state.get("price_level", 1.0)), 1e-9)
    hh_demand_fa_real = float(sol.get("hh_demand_fa_real", 0.0))
    hh_demand_fh_real = float(sol.get("hh_demand_fh_real", 0.0))
    supplier_fa_real = float(sol.get("supplier_sales_fa_real", 0.0))
    supplier_fh_real = float(sol.get("supplier_sales_fh_real", 0.0))
    ums_fa_real = float(sol.get("ums_recycle_fa_nom", 0.0)) / p_now
    ums_fh_real = float(sol.get("ums_recycle_fh_nom", 0.0)) / p_now

    sim.state.pop("sector_base_capacity_info_real", None)
    sim.state.pop("sector_base_capacity_phys_real", None)
    sim._ensure_sector_capacity_anchors(
        hh_demand_fa_real,
        hh_demand_fh_real,
        supplier_fa_real=supplier_fa_real,
        supplier_fh_real=supplier_fh_real,
        ums_fa_real=ums_fa_real,
        ums_fh_real=ums_fh_real,
    )
    sim.state["sector_capacity_info_real_prev"] = float(sim._sector_capacity_real("IS"))
    sim.state["sector_capacity_phys_real_prev"] = float(sim._sector_capacity_real("PS"))
    return {
        "hh_demand_fa_real": float(hh_demand_fa_real),
        "hh_demand_fh_real": float(hh_demand_fh_real),
        "supplier_fa_real": float(supplier_fa_real),
        "supplier_fh_real": float(supplier_fh_real),
        "ums_fa_real": float(ums_fa_real),
        "ums_fh_real": float(ums_fh_real),
        "capacity_info_real": float(sim._sector_capacity_real("IS")),
        "capacity_phys_real": float(sim._sector_capacity_real("PS")),
    }


def _seed_old_loop_startup_retained_cash(sim: NewLoop) -> Dict[str, Any] | None:
    """Seed OldLoop visible-start firms with retained cash for initial maintenance CAPEX."""
    regime = normalize_economic_regime_name(sim.params.get("economic_regime", "NewLoop"))
    if regime != "OldLoop" or not bool(sim.params.get("old_loop_startup_seed_retained_cash", True)):
        return None

    quarters = max(0.0, float(sim.params.get("old_loop_startup_retained_cash_quarters", 0.0)))
    if quarters <= 0.0:
        return None

    p_now = max(float(sim.state.get("price_level", 1.0)), 1e-9)
    total_added = 0.0
    out: Dict[str, Any] = {"quarters": float(quarters)}
    for firm_id, free_cash_key in (
        ("IS", "sector_free_cash_info_prev"),
        ("PS", "sector_free_cash_phys_prev"),
    ):
        maintenance_nom = quarters * max(0.0, float(sim._sector_maintenance_capex_nom(firm_id, p_now)))
        existing_cash = max(0.0, float(sim._firm_discretionary_deposits_nom(firm_id)))
        add_cash = max(0.0, maintenance_nom - existing_cash)
        if add_cash > 0.0:
            sim.nodes[firm_id].add("deposits", add_cash)
            total_added += add_cash
            existing_cash += add_cash
        sim.state[free_cash_key] = max(
            max(0.0, float(sim.state.get(free_cash_key, 0.0))),
            existing_cash,
        )
        suffix = "info" if firm_id == "IS" else "phys"
        out[f"{suffix}_maintenance_target_nom"] = float(maintenance_nom)
        out[f"{suffix}_retained_cash_added_nom"] = float(add_cash)
        out[f"{suffix}_free_cash_prev_nom"] = float(sim.state[free_cash_key])

    if total_added > 0.0:
        _sync_startup_household_state(sim)
    out["total_retained_cash_added_nom"] = float(total_added)
    return out


def _extract_sector_planner_seed(sim: NewLoop) -> Dict[str, float]:
    """Capture the lagged CAPEX planner state that should survive a policy switch."""
    keys = (
        "sector_capacity_info_real_prev",
        "sector_capacity_phys_real_prev",
        "sector_free_cash_info_prev",
        "sector_free_cash_phys_prev",
    )
    return {
        key: float(max(0.0, sim.state.get(key, 0.0)))
        for key in keys
    }


def _apply_sector_planner_seed(sim: NewLoop, planner_seed: Dict[str, Any] | None) -> None:
    """Restore legacy lagged planner state onto the visible-start simulation."""
    if not planner_seed:
        return
    for key, value in planner_seed.items():
        sim.state[key] = float(max(0.0, value))


def _build_legacy_sector_planner_seed(cfg: Dict[str, Any]) -> Dict[str, float] | None:
    """Simulate one hidden no-policy quarter to seed visible-start CAPEX planner state."""
    seed_sim = NewLoop(_neutral_warmup_regime_cfg(cfg))
    _prepare_startup_sim(seed_sim)
    try:
        seed_sim.step()
    except Exception:
        return None
    return _extract_sector_planner_seed(seed_sim)


def _neutral_warmup_regime_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of cfg with policy actions disabled for neutral warm-up."""
    warm_cfg = copy.deepcopy(cfg)
    params = warm_cfg["parameters"]
    params["automation_disabled"] = True
    params["disable_income_support"] = True
    params["disable_trust"] = True
    params["disable_mortgage_relief"] = True
    # Keep hidden startup warm-up from pre-building visible-quarter capacity.
    params["ums_recycle_rate_q"] = 0.0
    params["sector_capex_share_min"] = 0.0
    params["sector_capex_share_max"] = 0.0
    params["sector_capex_gap_close_rate"] = 0.0
    params["sector_capex_growth_cap_rate_q"] = 0.0
    params["sector_install_rate_q"] = 0.0
    params["mortgage_turnover_enabled"] = False
    params["gov_tax_rebate_rate"] = 0.0
    params["send_fund_residual_to_gov"] = False
    params["fund_residual_to_gov_share"] = 0.0
    return warm_cfg


def _reset_post_warmup_sector_planner_state(sim: NewLoop) -> None:
    """Clear hidden warm-up expansion pressure before visible Q0."""
    for key in (
        "sector_capex_queue_info_nom",
        "sector_capex_queue_phys_nom",
        "sector_unmet_info_real_prev",
        "sector_unmet_phys_real_prev",
        "sector_unmet_info_real_sm_prev",
        "sector_unmet_phys_real_sm_prev",
        "sector_load_gap_info_real_prev",
        "sector_load_gap_phys_real_prev",
        "sector_load_gap_info_real_sm_prev",
        "sector_load_gap_phys_real_sm_prev",
    ):
        sim.state[key] = 0.0
    sim.nodes["UMS"].set("deposits", 0.0)


def _activate_old_to_new_transition(sim: NewLoop, cfg: Dict[str, Any], visible_quarter: int) -> Dict[str, Any]:
    """Switch a live simulation into the selected OldToNew transition experiment."""
    current_t = int(sim.state.get("t", 0))
    transition_mode = _old_to_new_transition_mode(cfg)
    if transition_mode == "StayOldLoop":
        sim.state["old_to_new_transition_applied"] = True
        sim.state["old_to_new_transition_visible_quarter"] = int(visible_quarter)
        sim.state["old_to_new_transition_internal_t"] = int(current_t)
        sim.state["old_to_new_configured_regime"] = "OldToNew"
        return {
            "transition_applied": True,
            "visible_quarter": int(visible_quarter),
            "internal_t": int(current_t),
            "transition_mode": transition_mode,
            "launch_newloop_policies": False,
            "post_transition_regime": str(sim.params.get("economic_regime", "OldLoop")),
            "post_transition_tax_policy_mode": str(sim.params.get("tax_policy_mode", "")),
            "post_transition_automation_disabled": bool(sim.params.get("automation_disabled", True)),
            "post_transition_automation_start_quarter": int(sim.params.get("automation_start_quarter", 0)),
        }

    next_cfg = (
        _old_to_new_new_phase_cfg(cfg, switch_t=current_t)
        if transition_mode == "NewLoopPolicies" else
        _old_to_new_oldloop_decay_phase_cfg(cfg, switch_t=current_t)
    )
    sim.params = copy.deepcopy(next_cfg.get("parameters", {}))
    sim.income_support_policy = make_income_support_policy(sim.params)
    sim.tax_policy = make_tax_policy(sim.params)
    sim.state["old_to_new_transition_applied"] = True
    sim.state["old_to_new_transition_visible_quarter"] = int(visible_quarter)
    sim.state["old_to_new_transition_internal_t"] = int(current_t)
    sim.state["old_to_new_configured_regime"] = "OldToNew"
    return {
        "transition_applied": True,
        "visible_quarter": int(visible_quarter),
        "internal_t": int(current_t),
        "transition_mode": transition_mode,
        "launch_newloop_policies": bool(transition_mode == "NewLoopPolicies"),
        "post_transition_regime": str(sim.params.get("economic_regime", "NewLoop")),
        "post_transition_tax_policy_mode": str(sim.params.get("tax_policy_mode", "")),
        "post_transition_automation_disabled": bool(sim.params.get("automation_disabled", False)),
        "post_transition_automation_start_quarter": int(sim.params.get("automation_start_quarter", 0)),
    }


def _build_old_to_new_startup_sim(cfg: Dict[str, Any]) -> tuple[NewLoop, int, Dict[str, Any]]:
    """Create the visible-start sim for OldToNew without hidden NewLoop warm-start quarters."""
    old_phase_cfg = _old_to_new_old_phase_cfg(cfg)
    sim = NewLoop(old_phase_cfg)
    _prepare_startup_sim(sim)
    mortgage_money_seed = _distribute_startup_mortgage_money_seed(sim)
    retained_cash_seed = _seed_old_loop_startup_retained_cash(sim)
    report = {
        "requested_quarters": 0,
        "completed_quarters": 0,
        "completed_fully": True,
        "error": "",
        "old_to_new_transition_quarters": _old_to_new_transition_quarters(cfg),
        "old_to_new_transition_mode": _old_to_new_transition_mode(cfg),
        "old_to_new_launch_newloop_policies": _old_to_new_launch_newloop_policies(cfg),
        "startup_mode": "old_to_new_visible_old_loop",
    }
    if mortgage_money_seed is not None:
        report["visible_start_mortgage_money_seed"] = dict(mortgage_money_seed)
    if retained_cash_seed is not None:
        report["visible_start_retained_cash_seed"] = dict(retained_cash_seed)
    return sim, len(sim.history), report


def _build_startup_sim(cfg: Dict[str, Any]) -> tuple[NewLoop, int, Dict[str, Any]]:
    """Create a startup sim, optionally run hidden neutral warm-up quarters, and return the visible start index plus warm-up diagnostics."""
    effective_cfg = apply_economic_regime_overrides(cfg)
    if normalize_economic_regime_name(effective_cfg.get("parameters", {}).get("economic_regime", "NewLoop")) == "OldToNew":
        return _build_old_to_new_startup_sim(effective_cfg)
    warmup_quarters = max(0, int(effective_cfg.get("parameters", {}).get("neutral_warmup_quarters", 0)))
    startup_cfg = _neutral_warmup_regime_cfg(effective_cfg) if warmup_quarters > 0 else copy.deepcopy(effective_cfg)
    sim = NewLoop(startup_cfg)
    _prepare_startup_sim(sim)
    legacy_planner_seed: Dict[str, float] | None = None
    warmup_report: Dict[str, Any] = {
        "requested_quarters": int(warmup_quarters),
        "completed_quarters": 0,
        "completed_fully": True,
        "error": "",
    }

    if warmup_quarters > 0:
        for idx in range(warmup_quarters):
            try:
                sim.step()
            except Exception as exc:
                warmup_report["completed_fully"] = False
                warmup_report["error"] = f"{type(exc).__name__}: {exc}"
                break
            warmup_report["completed_quarters"] = int(idx + 1)
        _prepare_startup_sim(sim)
        legacy_planner_seed = _extract_sector_planner_seed(sim)
        sim.params = copy.deepcopy(effective_cfg["parameters"])
        sim.income_support_policy = make_income_support_policy(sim.params)
        sim.tax_policy = make_tax_policy(sim.params)
        _reset_post_warmup_sector_planner_state(sim)
        _sync_startup_household_state(sim)
        _apply_sector_planner_seed(sim, legacy_planner_seed)
        mortgage_money_seed = _distribute_startup_mortgage_money_seed(sim)
        if mortgage_money_seed is not None:
            warmup_report["visible_start_mortgage_money_seed"] = dict(mortgage_money_seed)
        reseed_stats = _reseed_visible_start_capacity(sim)
        if reseed_stats is not None:
            warmup_report["visible_start_capacity_reseed"] = dict(reseed_stats)
            warmup_report["visible_start_capex_seed"] = dict(legacy_planner_seed or {})
        retained_cash_seed = _seed_old_loop_startup_retained_cash(sim)
        if retained_cash_seed is not None:
            warmup_report["visible_start_retained_cash_seed"] = dict(retained_cash_seed)
    else:
        legacy_planner_seed = _build_legacy_sector_planner_seed(cfg)
        _apply_sector_planner_seed(sim, legacy_planner_seed)
        mortgage_money_seed = _distribute_startup_mortgage_money_seed(sim)
        if mortgage_money_seed is not None:
            warmup_report["visible_start_mortgage_money_seed"] = dict(mortgage_money_seed)
        reseed_stats = _reseed_visible_start_capacity(sim)
        if reseed_stats is not None:
            warmup_report["visible_start_capacity_reseed"] = dict(reseed_stats)
        if legacy_planner_seed is not None:
            warmup_report["visible_start_capex_seed"] = dict(legacy_planner_seed)
        retained_cash_seed = _seed_old_loop_startup_retained_cash(sim)
        if retained_cash_seed is not None:
            warmup_report["visible_start_retained_cash_seed"] = dict(retained_cash_seed)

    return sim, len(sim.history), warmup_report


def run_simulation(
    n_quarters: int = 80,
    cfg: Dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> SimulationRun:
    """Run NewLoop for n_quarters and return structured outputs."""
    total_quarters = max(0, int(n_quarters))

    def _notify_progress(stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(str(stage), int(completed), int(total_quarters))

    base_cfg = apply_economic_regime_overrides(copy.deepcopy(get_default_config() if cfg is None else cfg))
    effective_cfg, baseline_calibration = _run_baseline_calibration(base_cfg)
    effective_regime = normalize_economic_regime_name(
        effective_cfg.get("parameters", {}).get("economic_regime", "NewLoop")
    )
    old_to_new_transition_q = (
        _old_to_new_transition_quarters(effective_cfg)
        if effective_regime == "OldToNew" else None
    )

    _notify_progress("Preparing startup...", 0)
    startup_diag_sim, _, warmup_report = _build_startup_sim(effective_cfg)
    startup_snapshot = _startup_solver_snapshot(startup_diag_sim)
    startup_diag = _startup_diagnostics(startup_diag_sim, snapshot=startup_snapshot)

    sim, visible_history_start, _ = _build_startup_sim(effective_cfg)
    before_sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
    before = _population_distribution_snapshot(sim, sol=before_sol) if before_sol is not None else None
    quarter_diag_q0: Dict[str, Any] | None = None
    quarter_diag_q10: Dict[str, Any] | None = None
    old_to_new_transition_report: Dict[str, Any] | None = None
    _notify_progress("Running visible quarters...", 0)
    for step_idx in range(total_quarters):
        if old_to_new_transition_q is not None and old_to_new_transition_report is None and step_idx == old_to_new_transition_q:
            old_to_new_transition_report = _activate_old_to_new_transition(sim, effective_cfg, step_idx)
        sim.step()
        visible_t = len(sim.history) - visible_history_start - 1
        if visible_t == 0:
            quarter_diag_q0_snapshot = _startup_solver_snapshot(sim)
            quarter_diag_q0 = (
                _quarter_state_diagnostics(sim, snapshot=quarter_diag_q0_snapshot)
                if quarter_diag_q0_snapshot is not None else None
            )
        elif visible_t == 10:
            quarter_diag_q10_snapshot = _startup_solver_snapshot(sim)
            quarter_diag_q10 = (
                _quarter_state_diagnostics(sim, snapshot=quarter_diag_q10_snapshot)
                if quarter_diag_q10_snapshot is not None else None
            )
        _notify_progress("Running visible quarters...", step_idx + 1)
    after_sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
    after = _population_distribution_snapshot(sim, sol=after_sol) if after_sol is not None else None
    pop_dist = {"before": before, "after": after} if (before is not None and after is not None) else None
    startup_diag_out = dict(startup_diag or {})
    startup_diag_out["neutral_warmup_quarters"] = int(warmup_report.get("requested_quarters", 0))
    startup_diag_out["neutral_warmup_quarters_completed"] = int(warmup_report.get("completed_quarters", 0))
    startup_diag_out["neutral_warmup_completed_fully"] = bool(warmup_report.get("completed_fully", True))
    startup_diag_out["neutral_warmup_error"] = str(warmup_report.get("error", ""))
    mortgage_money_seed_estimate = _startup_mortgage_money_seed_estimate(startup_diag_sim)
    if mortgage_money_seed_estimate is not None:
        startup_diag_out["startup_mortgage_money_seed_estimate"] = mortgage_money_seed_estimate
    if "visible_start_mortgage_money_seed" in warmup_report:
        startup_diag_out["startup_mortgage_money_seed_distribution"] = dict(
            warmup_report["visible_start_mortgage_money_seed"]
        )
    if old_to_new_transition_q is not None:
        startup_diag_out["old_to_new_transition"] = (
            dict(old_to_new_transition_report)
            if old_to_new_transition_report is not None else {
                "transition_applied": False,
                "visible_quarter": int(old_to_new_transition_q),
                "transition_mode": _old_to_new_transition_mode(effective_cfg),
                "launch_newloop_policies": bool(_old_to_new_launch_newloop_policies(effective_cfg)),
            }
        )
    quarter_compare = _quarter_comparison(quarter_diag_q0, quarter_diag_q10)
    if quarter_compare is not None:
        startup_diag_out["quarter_comparison"] = quarter_compare
    _notify_progress("Completed.", total_quarters)
    return SimulationRun(
        sim=sim,
        rows=_visible_rows(sim.history, start_idx=visible_history_start),
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
