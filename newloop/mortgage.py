# Author: Roger Ison   roger@miximum.info
"""Shared fixed-rate mortgage schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _as_array(value: np.ndarray | float | int) -> np.ndarray:
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class FixedRateMortgageSchedule:
    """Canonical fixed-rate mortgage schedule for one product."""

    rate_q: float
    term_q: int
    contract_payment_factor: float
    balance_factor_by_age: np.ndarray
    total_due_factor_by_age: np.ndarray
    interest_factor_by_age: np.ndarray
    principal_factor_by_age: np.ndarray

    def age_index(self, age_q: np.ndarray | float | int) -> np.ndarray:
        age = np.floor(np.maximum(0.0, _as_array(age_q)) + 1e-12).astype(int, copy=False)
        return np.clip(age, 0, int(self.term_q))

    def remaining_term(self, age_q: np.ndarray | float | int) -> np.ndarray:
        return np.maximum(0.0, float(self.term_q) - _as_array(age_q))

    def balance_factor(self, age_q: np.ndarray | float | int) -> np.ndarray:
        return self.balance_factor_by_age[self.age_index(age_q)]

    def total_due_factor(self, age_q: np.ndarray | float | int) -> np.ndarray:
        return self.total_due_factor_by_age[self.age_index(age_q)]

    def interest_factor(self, age_q: np.ndarray | float | int) -> np.ndarray:
        return self.interest_factor_by_age[self.age_index(age_q)]

    def principal_factor(self, age_q: np.ndarray | float | int) -> np.ndarray:
        return self.principal_factor_by_age[self.age_index(age_q)]

    def payment_from_orig_principal(self, principal: np.ndarray | float | int) -> np.ndarray:
        return np.maximum(0.0, _as_array(principal)) * float(self.contract_payment_factor)

    def contract_payment_from_balance(
        self,
        balance: np.ndarray | float | int,
        age_q: np.ndarray | float | int,
    ) -> np.ndarray:
        orig_principal = self.orig_principal_from_balance(balance, age_q)
        return self.payment_from_orig_principal(orig_principal)

    def balance_from_orig_principal(
        self,
        principal: np.ndarray | float | int,
        age_q: np.ndarray | float | int,
    ) -> np.ndarray:
        return np.maximum(0.0, _as_array(principal)) * self.balance_factor(age_q)

    def orig_principal_from_balance(
        self,
        balance: np.ndarray | float | int,
        age_q: np.ndarray | float | int,
    ) -> np.ndarray:
        bal = np.maximum(0.0, _as_array(balance))
        factor = self.balance_factor(age_q)
        out = np.zeros(np.broadcast(bal, factor).shape, dtype=float)
        bal_b = np.broadcast_to(bal, out.shape)
        factor_b = np.broadcast_to(factor, out.shape)
        valid = factor_b > 1e-12
        out[valid] = bal_b[valid] / factor_b[valid]
        return out

    def canonical_components_from_orig_principal(
        self,
        principal: np.ndarray | float | int,
        age_q: np.ndarray | float | int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        principal_arr = np.maximum(0.0, _as_array(principal))
        return (
            principal_arr * self.total_due_factor(age_q),
            principal_arr * self.interest_factor(age_q),
            principal_arr * self.principal_factor(age_q),
        )

    def contractual_components(
        self,
        balance: np.ndarray | float | int,
        payment_q: np.ndarray | float | int,
        age_q: np.ndarray | float | int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bal = np.maximum(0.0, _as_array(balance))
        payment = np.maximum(0.0, _as_array(payment_q))
        age_idx = self.age_index(age_q)
        out_shape = np.broadcast(bal, payment, age_idx).shape
        bal_b = np.broadcast_to(bal, out_shape)
        payment_b = np.broadcast_to(payment, out_shape)
        age_b = np.broadcast_to(age_idx, out_shape)

        interest_due = np.zeros(out_shape, dtype=float)
        principal_due = np.zeros(out_shape, dtype=float)
        total_due = np.zeros(out_shape, dtype=float)

        active = (bal_b > 1e-12) & (age_b < int(self.term_q))
        if not np.any(active):
            return total_due, interest_due, principal_due

        interest_due[active] = bal_b[active] * float(self.rate_q)
        final_period = active & (age_b >= (int(self.term_q) - 1))
        regular_period = active & (~final_period)

        if np.any(regular_period):
            total_due[regular_period] = np.minimum(
                payment_b[regular_period],
                bal_b[regular_period] + interest_due[regular_period],
            )
        if np.any(final_period):
            total_due[final_period] = bal_b[final_period] + interest_due[final_period]

        principal_due[active] = np.minimum(
            bal_b[active],
            np.maximum(0.0, total_due[active] - interest_due[active]),
        )
        total_due[active] = interest_due[active] + principal_due[active]
        return total_due, interest_due, principal_due


@lru_cache(maxsize=32)
def get_fixed_rate_mortgage_schedule(rate_q: float, term_q: int) -> FixedRateMortgageSchedule:
    """Return a cached canonical schedule for one fixed-rate mortgage product."""
    rate = max(0.0, float(rate_q))
    term = max(1, int(term_q))
    payment_factor = float(payment_from_balance(np.asarray([1.0], dtype=float), rate, float(term))[0])

    balance_factor = np.zeros(term + 1, dtype=float)
    total_due_factor = np.zeros(term + 1, dtype=float)
    interest_factor = np.zeros(term + 1, dtype=float)
    principal_factor = np.zeros(term + 1, dtype=float)

    balance = 1.0
    for age in range(term):
        balance_factor[age] = balance
        total_due, interest_due, principal_due = scheduled_payment_components(
            np.asarray([balance], dtype=float),
            np.asarray([rate], dtype=float),
            np.asarray([payment_factor], dtype=float),
            np.asarray([float(term - age)], dtype=float),
        )
        total_due_factor[age] = float(total_due[0])
        interest_factor[age] = float(interest_due[0])
        principal_factor[age] = float(principal_due[0])
        balance = max(0.0, balance - principal_factor[age])

    balance_factor[term] = 0.0
    total_due_factor[term] = 0.0
    interest_factor[term] = 0.0
    principal_factor[term] = 0.0

    return FixedRateMortgageSchedule(
        rate_q=rate,
        term_q=term,
        contract_payment_factor=payment_factor,
        balance_factor_by_age=balance_factor,
        total_due_factor_by_age=total_due_factor,
        interest_factor_by_age=interest_factor,
        principal_factor_by_age=principal_factor,
    )


def remaining_term(term_q: np.ndarray | float | int, age_q: np.ndarray | float | int) -> np.ndarray:
    term = _as_array(term_q)
    age = _as_array(age_q)
    return np.maximum(0.0, term - age)


def annuity_factor(rate_q: np.ndarray | float | int, periods_q: np.ndarray | float | int) -> np.ndarray:
    rate = _as_array(rate_q)
    periods = np.maximum(0.0, _as_array(periods_q))
    out = np.zeros(np.broadcast(rate, periods).shape, dtype=float)
    valid = periods > 1e-12
    if not np.any(valid):
        return out

    rate_b = np.broadcast_to(rate, out.shape)
    periods_b = np.broadcast_to(periods, out.shape)
    near_zero = valid & (np.abs(rate_b) <= 1e-12)
    out[near_zero] = periods_b[near_zero]

    active = valid & (~near_zero)
    if np.any(active):
        out[active] = (1.0 - np.power(1.0 + rate_b[active], -periods_b[active])) / rate_b[active]
    return out


def payment_from_balance(
    balance: np.ndarray | float | int,
    rate_q: np.ndarray | float | int,
    remaining_term_q: np.ndarray | float | int,
) -> np.ndarray:
    bal = np.maximum(0.0, _as_array(balance))
    rem = np.maximum(0.0, _as_array(remaining_term_q))
    af = annuity_factor(rate_q, rem)
    out = np.zeros(np.broadcast(bal, af).shape, dtype=float)
    bal_b = np.broadcast_to(bal, out.shape)
    af_b = np.broadcast_to(af, out.shape)
    valid = (bal_b > 1e-12) & (af_b > 1e-12)
    out[valid] = bal_b[valid] / af_b[valid]
    return out


def payment_from_orig_principal(
    principal: np.ndarray | float | int,
    rate_q: np.ndarray | float | int,
    term_q: np.ndarray | float | int,
) -> np.ndarray:
    return payment_from_balance(principal, rate_q, term_q)


def balance_from_orig_principal(
    principal: np.ndarray | float | int,
    rate_q: np.ndarray | float | int,
    term_q: np.ndarray | float | int,
    age_q: np.ndarray | float | int,
) -> np.ndarray:
    payment = payment_from_orig_principal(principal, rate_q, term_q)
    rem = remaining_term(term_q, age_q)
    af_rem = annuity_factor(rate_q, rem)
    out = np.zeros(np.broadcast(payment, af_rem).shape, dtype=float)
    payment_b = np.broadcast_to(payment, out.shape)
    af_rem_b = np.broadcast_to(af_rem, out.shape)
    valid = (payment_b > 1e-12) & (af_rem_b > 1e-12)
    out[valid] = payment_b[valid] * af_rem_b[valid]
    return out


def orig_principal_from_balance(
    balance: np.ndarray | float | int,
    rate_q: np.ndarray | float | int,
    term_q: np.ndarray | float | int,
    age_q: np.ndarray | float | int,
) -> np.ndarray:
    rem = remaining_term(term_q, age_q)
    payment = payment_from_balance(balance, rate_q, rem)
    af_orig = annuity_factor(rate_q, term_q)
    out = np.zeros(np.broadcast(payment, af_orig).shape, dtype=float)
    payment_b = np.broadcast_to(payment, out.shape)
    af_orig_b = np.broadcast_to(af_orig, out.shape)
    valid = (payment_b > 1e-12) & (af_orig_b > 1e-12)
    out[valid] = payment_b[valid] * af_orig_b[valid]
    return out


def scheduled_payment_components(
    balance: np.ndarray | float | int,
    rate_q: np.ndarray | float | int,
    payment_q: np.ndarray | float | int,
    remaining_term_q: np.ndarray | float | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bal = np.maximum(0.0, _as_array(balance))
    rate = np.maximum(0.0, _as_array(rate_q))
    payment = np.maximum(0.0, _as_array(payment_q))
    rem = np.maximum(0.0, _as_array(remaining_term_q))

    out_shape = np.broadcast(bal, rate, payment, rem).shape
    bal_b = np.broadcast_to(bal, out_shape)
    rate_b = np.broadcast_to(rate, out_shape)
    payment_b = np.broadcast_to(payment, out_shape)
    rem_b = np.broadcast_to(rem, out_shape)

    interest_due = np.zeros(out_shape, dtype=float)
    principal_due = np.zeros(out_shape, dtype=float)
    total_due = np.zeros(out_shape, dtype=float)

    active = (bal_b > 1e-12) & (rem_b > 1e-12)
    if not np.any(active):
        return total_due, interest_due, principal_due

    interest_due[active] = bal_b[active] * rate_b[active]
    # On the final scheduled quarter, any remaining balance is due as a balloon payoff.
    final_period = active & (rem_b <= 1.0 + 1e-12)
    regular_period = active & (~final_period)

    if np.any(regular_period):
        total_due[regular_period] = np.minimum(
            payment_b[regular_period],
            bal_b[regular_period] + interest_due[regular_period],
        )
    if np.any(final_period):
        total_due[final_period] = bal_b[final_period] + interest_due[final_period]

    principal_due[active] = np.minimum(
        bal_b[active],
        np.maximum(0.0, total_due[active] - interest_due[active]),
    )
    total_due[active] = interest_due[active] + principal_due[active]
    return total_due, interest_due, principal_due
