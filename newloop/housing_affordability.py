"""Shared OldLoop housing-affordability helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


_DEFAULT_CORE_NONHOUSING_KAPPA_BY_INCOME_PCT: tuple[tuple[float, float], ...] = (
    (20.0, 0.72),
    (50.0, 0.64),
    (80.0, 0.56),
    (95.0, 0.50),
    (100.0, 0.44),
)


def _coerce_schedule(
    raw: Sequence[Sequence[Any]] | None,
    fallback: Sequence[Sequence[Any]],
) -> tuple[tuple[float, float], ...]:
    if not raw:
        raw = fallback
    pts = [(float(pct), float(val)) for pct, val in raw]
    pts.sort(key=lambda item: item[0])
    if not pts:
        pts = [(100.0, 0.0)]
    if pts[0][0] > 0.0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 100.0:
        pts.append((100.0, pts[-1][1]))
    return tuple(pts)


def _assign_linear_by_percentile_rank(
    values: np.ndarray,
    schedule: tuple[tuple[float, float], ...],
) -> np.ndarray:
    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=float)
    order = np.argsort(values)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    rank_pct = 100.0 * (rank.astype(float) / float(max(1, n - 1)))
    pct_vals = np.array([pct for pct, _ in schedule], dtype=float)
    level_vals = np.array([val for _, val in schedule], dtype=float)
    return np.interp(rank_pct, pct_vals, level_vals).astype(float)


def estimate_old_loop_income_tax_q(
    permanent_income_q: np.ndarray,
    baseline_income_q: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Estimate quarterly OldLoop household income tax from gross permanent income."""
    income_q = np.maximum(0.0, np.asarray(permanent_income_q, dtype=float))
    base_q = np.maximum(0.0, np.asarray(baseline_income_q, dtype=float))
    if income_q.shape != base_q.shape:
        raise ValueError("permanent_income_q and baseline_income_q must have matching shapes.")
    if bool(params.get("disable_income_tax", False)):
        return np.zeros_like(income_q)

    annual_income = 4.0 * income_q
    baseline_annual = 4.0 * base_q
    pct_lo = max(0.0, min(100.0, float(params.get("old_loop_tax_threshold_lower_pct", 30.0))))
    pct_hi = max(pct_lo, min(100.0, float(params.get("old_loop_tax_threshold_upper_pct", 80.0))))
    thr_lo = float(np.percentile(baseline_annual, pct_lo)) if baseline_annual.size else 0.0
    thr_hi = float(np.percentile(baseline_annual, pct_hi)) if baseline_annual.size else thr_lo
    rate_lo = max(0.0, min(1.0, float(params.get("old_loop_tax_rate_lower", 0.15))))
    rate_hi = max(0.0, min(1.0, float(params.get("old_loop_tax_rate_upper", 0.28))))

    middle_band = np.clip(annual_income - thr_lo, 0.0, max(0.0, thr_hi - thr_lo))
    top_band = np.clip(annual_income - thr_hi, 0.0, None)
    tax_annual = (rate_lo * middle_band) + (rate_hi * top_band)
    return 0.25 * np.maximum(0.0, tax_annual)


def compute_affordable_housing_profile(
    permanent_income_q: np.ndarray,
    baseline_income_q: np.ndarray,
    params: Mapping[str, Any],
    *,
    existing_fixed_obligations_q: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return core-consumption, reserve, and supportable housing budgets."""
    perm_income_q = np.maximum(0.0, np.asarray(permanent_income_q, dtype=float))
    base_income_q = np.maximum(0.0, np.asarray(baseline_income_q, dtype=float))
    if perm_income_q.shape != base_income_q.shape:
        raise ValueError("permanent_income_q and baseline_income_q must have matching shapes.")
    fixed_obligations_q = (
        np.zeros_like(perm_income_q)
        if existing_fixed_obligations_q is None
        else np.maximum(0.0, np.asarray(existing_fixed_obligations_q, dtype=float))
    )
    if fixed_obligations_q.shape != perm_income_q.shape:
        raise ValueError("existing_fixed_obligations_q must match permanent_income_q shape.")

    tax_q = estimate_old_loop_income_tax_q(perm_income_q, base_income_q, params)
    disp_perm_q = np.maximum(0.0, perm_income_q - tax_q)
    kappa_schedule = _coerce_schedule(
        params.get("old_loop_core_nonhousing_kappa_by_income_pct"),
        _DEFAULT_CORE_NONHOUSING_KAPPA_BY_INCOME_PCT,
    )
    core_kappa = _assign_linear_by_percentile_rank(base_income_q, kappa_schedule)
    headroom_share = max(0.0, min(1.0, float(params.get("old_loop_housing_headroom_share", 0.08))))
    headroom_floor_q = max(0.0, float(params.get("old_loop_housing_headroom_floor_q", 15.0)))
    headroom_q = np.maximum(headroom_floor_q, headroom_share * disp_perm_q)
    core_floor_q = max(0.0, float(params.get("old_loop_core_nonhousing_floor_q", 150.0)))
    core_nonhousing_raw_q = np.maximum(core_floor_q, core_kappa * disp_perm_q)
    core_nonhousing_cap_q = np.maximum(0.0, disp_perm_q - headroom_q - fixed_obligations_q)
    core_nonhousing_q = np.minimum(core_nonhousing_raw_q, core_nonhousing_cap_q)

    housing_share_target = max(0.0, min(1.0, float(params.get("old_loop_housing_share_target", 0.20))))
    housing_share_cap = max(housing_share_target, min(1.0, float(params.get("old_loop_housing_share_cap", 0.25))))
    target_housing_q = housing_share_target * disp_perm_q
    hard_housing_cap_q = housing_share_cap * disp_perm_q
    residual_budget_q = np.maximum(0.0, disp_perm_q - core_nonhousing_q - headroom_q - fixed_obligations_q)
    supportable_housing_q = np.minimum(hard_housing_cap_q, residual_budget_q)
    housing_payment_target_q = np.minimum(target_housing_q, supportable_housing_q)

    return {
        "tax_q": tax_q.astype(float, copy=False),
        "disp_perm_q": disp_perm_q.astype(float, copy=False),
        "core_nonhousing_q": core_nonhousing_q.astype(float, copy=False),
        "headroom_q": headroom_q.astype(float, copy=False),
        "hard_housing_cap_q": hard_housing_cap_q.astype(float, copy=False),
        "supportable_housing_q": supportable_housing_q.astype(float, copy=False),
        "housing_payment_target_q": housing_payment_target_q.astype(float, copy=False),
    }
