# Author: Roger Ison   roger@miximum.info
"""Package-native NewLoop entrypoint and public exports."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import config, get_default_config
from dashboard import run_cli
from engine import NewLoop
from newloop_types import HouseholdState, Node, TickResult

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
