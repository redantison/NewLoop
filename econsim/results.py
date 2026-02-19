# Author: Roger Ison   roger@miximum.info
"""Reusable simulation-output helpers for CLI, plotting, and Streamlit layers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import config as default_config
from .engine import EconomySim
from .types import TickResult


@dataclass
class SimulationRun:
    """Container for one simulation run and its row-oriented outputs."""

    sim: EconomySim
    rows: List[Dict[str, Any]]
    population_distributions: Dict[str, Dict[str, Any]] | None = None

    @property
    def history(self) -> List[TickResult]:
        return self.sim.history


def history_to_rows(history: Sequence[TickResult]) -> List[Dict[str, Any]]:
    """Convert TickResult history to plain dictionaries."""
    return [asdict(tick) for tick in history]


def _population_distribution_snapshot(sim: EconomySim) -> Dict[str, Any] | None:
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


def run_simulation(n_quarters: int = 80, cfg: Dict[str, Any] | None = None) -> SimulationRun:
    """Run EconomySim for n_quarters and return structured outputs."""
    effective_cfg = copy.deepcopy(default_config if cfg is None else cfg)
    sim = EconomySim(effective_cfg)
    before = _population_distribution_snapshot(sim)
    for _ in range(int(n_quarters)):
        sim.step()
    after = _population_distribution_snapshot(sim)
    pop_dist = {"before": before, "after": after} if (before is not None and after is not None) else None
    return SimulationRun(sim=sim, rows=history_to_rows(sim.history), population_distributions=pop_dist)


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
        "ubi_end": _get(last, "ubi_per_h"),
        "ubi_issued_end": _get(last, "ubi_issued_per_h"),
    }
