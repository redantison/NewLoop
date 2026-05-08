# Author: Roger Ison   roger@miximum.info
"""Canonical sector identifiers and legacy aliases."""

from __future__ import annotations

INFO_SECTOR = "IS"
PHYSICAL_SECTOR = "PS"

LEGACY_INFO_SECTOR = "FA"
LEGACY_PHYSICAL_SECTOR = "FH"

SECTOR_ALIASES = {
    LEGACY_INFO_SECTOR: INFO_SECTOR,
    LEGACY_PHYSICAL_SECTOR: PHYSICAL_SECTOR,
}

SHARE_ALIASES = {
    "shares_FA": "shares_IS",
    "shares_FH": "shares_PS",
}
