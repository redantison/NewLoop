# Author: Roger Ison   roger@miximum.info
"""NewLoop package exports."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import config, get_default_config
from engine import NewLoop
from results import SimulationRun, history_to_rows, run_simulation, summarize_rows
from table_of_data import TableOfData
from newloop_types import HouseholdState, Node, TickResult

__all__ = [
    "NewLoop",
    "Node",
    "HouseholdState",
    "TickResult",
    "SimulationRun",
    "run_simulation",
    "history_to_rows",
    "summarize_rows",
    "Population",
    "PopulationConfig",
    "generate_population",
    "baseline_report",
    "TableOfData",
    "config",
    "get_default_config",
]


def __getattr__(name: str):
    if name in {"Population", "PopulationConfig", "generate_population", "baseline_report"}:
        from population import Population, PopulationConfig, baseline_report, generate_population

        exports = {
            "Population": Population,
            "PopulationConfig": PopulationConfig,
            "generate_population": generate_population,
            "baseline_report": baseline_report,
        }
        return exports[name]
    raise AttributeError(f"module 'newloop' has no attribute {name!r}")
