"""Cash-limited household payments, without creating loans for unpaid bills."""
from __future__ import annotations

from typing import Mapping

import numpy as np


BILL_TYPES = ("revolving_interest", "rent", "owner_housing", "income_tax")
CURRENT_PRIORITY = ("revolving_interest", "mortgage", "rent", "owner_housing", "income_tax")
ARREARS_PRIORITY = ("mortgage_interest", "mortgage_principal") + BILL_TYPES
SHORTFALL_TOLERANCE_NOM = 1e-6


def allocate_household_payments(
    cash: np.ndarray,
    due: Mapping[str, np.ndarray],
    arrears: Mapping[str, np.ndarray],
) -> dict:
    """Pay current obligations, then old arrears, in the existing settlement order.

    Consumption has already been budgeted using income net of current obligations.
    This allocator does not change that budget, forgive debt, or compound arrears.
    Inputs are never mutated; each household can use only its own available cash.
    """
    available = np.maximum(0.0, np.asarray(cash, dtype=float)).copy()
    current_paid, arrears_paid = {}, {}
    for names, source, destination in (
        (CURRENT_PRIORITY, due, current_paid),
        (ARREARS_PRIORITY, arrears, arrears_paid),
    ):
        for name in names:
            required = np.maximum(0.0, np.asarray(source.get(name, np.zeros_like(available)), dtype=float))
            payment = np.minimum(available, required)
            destination[name] = payment
            available = np.maximum(0.0, available - payment)
    return {"current": current_paid, "arrears": arrears_paid, "remaining_cash": available}
