# Author: Roger Ison   roger@miximum.info
"""Numeric helpers used by EconomySim engine."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def _as_np(x, dtype=float) -> np.ndarray:
    """Convert list/array-like to 1D numpy array (copying only if needed)."""
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _pct(values: Any, p: float) -> float:
    """Nearest-rank percentile for p in [0,100]."""
    xs = _as_np(values, dtype=float)
    return _pct_np(xs, p)


def _pct_np(values: np.ndarray, p: float) -> float:
    """Nearest-rank percentile for numpy arrays (p in [0,100])."""
    xs = _as_np(values, dtype=float)
    if xs.size == 0:
        return 0.0
    xs = np.sort(xs)
    if p <= 0:
        return float(xs[0])
    if p >= 100:
        return float(xs[-1])
    k = int(math.ceil((p / 100.0) * xs.size)) - 1
    k = max(0, min(xs.size - 1, k))
    return float(xs[k])


def calculate_gini(incomes: Any) -> float:
    """Discrete Gini coefficient (negatives clamped to 0)."""
    xs = _as_np(incomes, dtype=float)
    return calculate_gini_np(xs)


def calculate_gini_np(incomes: np.ndarray) -> float:
    """Discrete Gini coefficient for a numpy vector (negatives clamped to 0)."""
    xs = _as_np(incomes, dtype=float)
    if xs.size == 0:
        return 0.0
    xs = np.maximum(0.0, xs)
    xs = np.sort(xs)
    s = float(xs.sum())
    if s <= 0:
        return 0.0
    n = xs.size
    i = np.arange(1, n + 1, dtype=float)
    num = float((i * xs).sum())
    return (2.0 * num) / (n * s) - (n + 1.0) / n


def _gompertz(t: float, k: float, t0: float, b: float) -> float:
    """Asymmetric S-curve with early gains and a long tail."""
    return math.exp(-b * math.exp(-k * (t - t0)))


def _logistic(t: float, k: float, t0: float) -> float:
    """Symmetric S-curve; derivative is bell-shaped."""
    return 1.0 / (1.0 + math.exp(-k * (t - t0)))


def automation_two_hump(
    t_qtr: int,
    w_info: float,
    ki: float,
    ti: float,
    bi: float,
    kp: float,
    tp: float,
    floor: float = 0.0,
    cap: float = 1.0,
    info_cap: float = 1.0,
    phys_cap: float = 1.0,
) -> Dict[str, float]:
    """Two-hump automation path with info + physical components."""
    A_info_raw = _gompertz(float(t_qtr), ki, ti, bi)
    A_phys_raw = _logistic(float(t_qtr), kp, tp)

    info_cap = max(floor, min(cap, float(info_cap)))
    phys_cap = max(floor, min(cap, float(phys_cap)))

    # Asymptotic cap mapping:
    # map raw S-curves in [0,1] into [floor, cap_i] by scaling, rather than hard clipping.
    # This avoids abrupt "hitting the cap" behavior and preserves smooth convergence.
    A_info_raw = max(0.0, min(1.0, A_info_raw))
    A_phys_raw = max(0.0, min(1.0, A_phys_raw))
    A_info = floor + (info_cap - floor) * A_info_raw
    A_phys = floor + (phys_cap - floor) * A_phys_raw

    A = w_info * A_info + (1.0 - w_info) * A_phys
    A = max(floor, min(cap, A))

    # Use t-1 for all quarters (including t=0) so flow is a true one-step difference,
    # avoiding an artificial startup jump from floor -> level at quarter 0.
    A_info_prev_raw = _gompertz(float(t_qtr - 1), ki, ti, bi)
    A_phys_prev_raw = _logistic(float(t_qtr - 1), kp, tp)

    A_info_prev_raw = max(0.0, min(1.0, A_info_prev_raw))
    A_phys_prev_raw = max(0.0, min(1.0, A_phys_prev_raw))
    A_info_prev = floor + (info_cap - floor) * A_info_prev_raw
    A_phys_prev = floor + (phys_cap - floor) * A_phys_prev_raw

    A_prev = w_info * A_info_prev + (1.0 - w_info) * A_phys_prev
    A_prev = max(floor, min(cap, A_prev))

    return {
        "level": A,
        "flow": A - A_prev,
        "info_level": A_info,
        "phys_level": A_phys,
        "info_flow": A_info - A_info_prev,
        "phys_flow": A_phys - A_phys_prev,
    }
