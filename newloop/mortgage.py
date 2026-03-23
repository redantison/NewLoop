# Author: Roger Ison   roger@miximum.info
"""Shared fixed-rate mortgage schedule helpers."""

from __future__ import annotations

import numpy as np


def _as_array(value: np.ndarray | float | int) -> np.ndarray:
    return np.asarray(value, dtype=float)


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
