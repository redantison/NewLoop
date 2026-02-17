"""Package-native EconomySim entrypoint and public exports."""

from __future__ import annotations

# Support both execution modes:
# 1) module mode:   python -m econsim.economy_sim
# 2) script mode:   run this file directly in IDEs (no package context)
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from econsim.config import config, get_default_config
    from econsim.dashboard import run_cli
    from econsim.engine import EconomySim
    from econsim.types import HouseholdState, Node, TickResult
else:
    from .config import config, get_default_config
    from .dashboard import run_cli
    from .engine import EconomySim
    from .types import HouseholdState, Node, TickResult

__all__ = [
    "EconomySim",
    "Node",
    "HouseholdState",
    "TickResult",
    "config",
    "get_default_config",
    "main",
]


def main(n_quarters: int = 80) -> None:
    """Run the terminal dashboard simulation."""
    run_cli(config=config, n_quarters=int(n_quarters))


if __name__ == "__main__":
    main()
