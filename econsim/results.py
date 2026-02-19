# Author: Roger Ison   roger@miximum.info
"""Reusable simulation-output helpers for CLI, plotting, and Streamlit layers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from .config import config as default_config
from .engine import EconomySim
from .types import TickResult


@dataclass
class SimulationRun:
    """Container for one simulation run and its row-oriented outputs."""

    sim: EconomySim
    rows: List[Dict[str, Any]]

    @property
    def history(self) -> List[TickResult]:
        return self.sim.history


def history_to_rows(history: Sequence[TickResult]) -> List[Dict[str, Any]]:
    """Convert TickResult history to plain dictionaries."""
    return [asdict(tick) for tick in history]


def run_simulation(n_quarters: int = 80, cfg: Dict[str, Any] | None = None) -> SimulationRun:
    """Run EconomySim for n_quarters and return structured outputs."""
    effective_cfg = copy.deepcopy(default_config if cfg is None else cfg)
    sim = EconomySim(effective_cfg)
    for _ in range(int(n_quarters)):
        sim.step()
    return SimulationRun(sim=sim, rows=history_to_rows(sim.history))


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
