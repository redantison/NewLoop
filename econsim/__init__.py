"""EconomySim package exports."""

from __future__ import annotations

from .config import config, get_default_config
from .engine import EconomySim
from .table_of_data import TableOfData
from .types import HouseholdState, Node, TickResult

__all__ = [
    "EconomySim",
    "Node",
    "HouseholdState",
    "TickResult",
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
        from .population import Population, PopulationConfig, baseline_report, generate_population

        exports = {
            "Population": Population,
            "PopulationConfig": PopulationConfig,
            "generate_population": generate_population,
            "baseline_report": baseline_report,
        }
        return exports[name]
    raise AttributeError(f"module 'econsim' has no attribute {name!r}")
