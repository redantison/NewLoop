# Author: Roger Ison   roger@miximum.info
"""Package-native NewLoop entrypoint and public exports."""

from __future__ import annotations

# Support both execution modes:
# 1) module mode:   python -m newloop.newloop
# 2) script mode:   run this file directly in IDEs (no package context)
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from newloop.config import config, get_default_config
    from newloop.dashboard import run_cli
    from newloop.engine import NewLoop
    from newloop.types import HouseholdState, Node, TickResult
else:
    from .config import config, get_default_config
    from .dashboard import run_cli
    from .engine import NewLoop
    from .types import HouseholdState, Node, TickResult

__all__ = [
    "NewLoop",
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
