# Author: Roger Ison   roger@miximum.info
"""NewLoop - Population Prototype (Phase 1)

Phase 1 goals:
- Build a synthetic population of "families" (consumer-workers) with heterogeneity in:
  * wage earning capacity
  * liquid deposits (a proxy for liquid wealth)
  * household debt (loan principal)
  * marginal propensity to consume (MPC)
- Print baseline moments (percentiles, Gini, aggregates) to sanity-check calibration.

This file is intentionally self-contained and does NOT implement the macro-sector simulation yet.
It produces a synthetic population that Phase 2 will plug into the NewLoop policy dynamics.

Notes / design choices:
- Wealth: modeled as a mixture of (1) lognormal body and (2) Pareto tail.
- Wages: modeled as lognormal.
- Debt: modeled as a mixture where a subset of families hold "mortgage-like" debt
        (scaled to income/wealth) and a subset hold smaller "revolving" debt.
- MPC: decreases with wealth percentile (simple piecewise schedule).

All dollar units are arbitrary "model dollars" per QUARTER unless otherwise stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import random
import statistics

from .mortgage import orig_principal_from_balance, payment_from_balance, remaining_term

# ----------------------------
# Utilities
# ----------------------------

def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


NP = _try_import_numpy()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rescale_probabilities_to_target(
    probs: "NP.ndarray",
    active_mask: "NP.ndarray",
    target_share: float,
) -> "NP.ndarray":
    """Shift probabilities in logit space so the active-group mean matches target_share."""
    np = NP
    if np is None:
        return probs
    out = np.asarray(probs, dtype=float).copy()
    if out.size == 0:
        return out
    mask = np.asarray(active_mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(out)
    target = clamp(float(target_share), 0.0, 1.0)
    if target <= 0.0:
        out[mask] = 0.0
        return out
    if target >= 1.0:
        out[mask] = 1.0
        return out
    eps = 1e-9
    base = np.clip(out[mask], eps, 1.0 - eps)
    logits = np.log(base / (1.0 - base))

    lo = -40.0
    hi = 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        shifted = 1.0 / (1.0 + np.exp(-(logits + mid)))
        mean_mid = float(np.mean(shifted))
        if mean_mid < target:
            lo = mid
        else:
            hi = mid

    shift = 0.5 * (lo + hi)
    out[mask] = 1.0 / (1.0 + np.exp(-(logits + shift)))
    out[~mask] = 0.0
    return out


def percentile(sorted_vals: List[float], p: float) -> float:
    """Percentile with linear interpolation. sorted_vals must be sorted ascending."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def gini(values: List[float]) -> float:
    """Gini coefficient for nonnegative values."""
    if not values:
        return 0.0
    vals = [max(0.0, v) for v in values]
    s = sum(vals)
    if s <= 0:
        return 0.0
    vals.sort()
    n = len(vals)
    # G = (2*sum_i i*x_i)/(n*sum x) - (n+1)/n
    cum = 0.0
    for i, x in enumerate(vals, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def summarize_distribution(name: str, values: List[float], money: bool = True) -> Dict[str, float]:
    vals = list(values)
    vals.sort()
    out: Dict[str, float] = {}
    out["n"] = float(len(vals))
    out["min"] = vals[0] if vals else 0.0
    out["p10"] = percentile(vals, 10)
    out["p25"] = percentile(vals, 25)
    out["p50"] = percentile(vals, 50)
    out["p75"] = percentile(vals, 75)
    out["p90"] = percentile(vals, 90)
    out["p99"] = percentile(vals, 99)
    out["max"] = vals[-1] if vals else 0.0
    out["mean"] = statistics.fmean(vals) if vals else 0.0
    out["gini"] = gini(vals)
    return out


def fmt_money(x: float) -> str:
    # compact integer-ish formatting for table readability
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.1f}"
    return f"{x:,.3f}"


# ----------------------------
# Population config + data
# ----------------------------

@dataclass(frozen=True)
class PopulationConfig:
    # Size / reproducibility
    n_families: int = 20000
    seed: int = 7919    # 1000th prime

    # Wages (quarterly) - lognormal
    # median_wage_q: median quarterly wage among working families
    # sigma_wage_ln: lognormal sigma (shape). Larger => fatter tail.
    median_wage_q: float = 450.0
    sigma_wage_ln: float = 0.60

    # Work attachment (share of families with positive wages)
    employment_rate: float = 0.94

    # Deposits / liquid wealth.
    # Default mode uses a wealth-conditioned liquid-buffer rule:
    # households target enough liquid savings for a piecewise number of months
    # of baseline consumption, depending on pre-liquidity wealth rank.
    deposit_generation_mode: str = "liquid_buffer_months"  # "liquid_buffer_months" | "legacy_mixture"
    liquid_buffer_months_by_wealth_pct: Tuple[Tuple[float, float], ...] = (
        (20.0, 1.5),
        (50.0, 3.0),
        (80.0, 5.0),
        (95.0, 8.0),
        (100.0, 12.0),
    )
    liquid_buffer_sigma_ln: float = 0.30

    # Legacy deposit-mixture parameters retained for compatibility.
    # body: lognormal with median_deposits_q and sigma
    # tail: Pareto with alpha and scale at a percentile boundary
    median_deposits_q: float = 1200.0
    sigma_deposits_ln: float = 1.05

    # Correlation between wage-earning capacity and deposits (liquid wealth proxy)
    # Implemented via a shared latent normal factor for lognormal draws.
    wage_deposit_corr: float = 0.40
    tail_share: float = 0.08  # fraction of families in Pareto tail
    pareto_alpha: float = 1.35

    # Debt (loan principal) mixture
    mortgage_share: float = 0.55      # fraction with mortgage-like debt
    revolving_share: float = 0.25     # fraction with revolving debt
    mortgage_income_mult_median: float = 3.25   # median mortgage principal as multiple of annual wage
    mortgage_income_mult_sigma: float = 0.55
    revolving_income_mult_median: float = 0.06  # revolving principal as multiple of annual wage
    revolving_income_mult_sigma: float = 0.80

    # Revolving balance caps (to prevent extreme tails in interest burden)
    revolving_cap_income_mult: float = 0.50   # cap revolving principal at this multiple of annual wage
    revolving_cap_deposits_mult: float = 2.0  # cap revolving principal at this multiple of deposits

    # Interest rates (effective rates on outstanding balances; baseline calibration)
    mortgage_rate_effective: float = 0.045
    revolving_rate_effective: float = 0.20
    mortgage_term_quarters: int = 60

    # MPC schedule by deposits percentile (piecewise)
    # (pct_upper_bound, mpc)
    # Interpreted as: if deposits percentile <= bound -> that mpc.
    mpc_by_wealth_pct: Tuple[Tuple[float, float], ...] = (
        (20.0, 0.55),
        (50.0, 0.45),
        (80.0, 0.35),
        (95.0, 0.25),
        (100.0, 0.18),
    )

    # Baseline real consumption per quarter (for Phase 2 integration).
    # Default: piecewise by pre-liquidity wealth rank; `base_real_cons_q` is retained as
    # a legacy/global fallback if the schedule is cleared.
    base_real_cons_q: float = 600.0
    base_real_cons_by_wealth_pct: Tuple[Tuple[float, float], ...] = (
        (20.0, 450.0),
        (50.0, 525.0),
        (80.0, 600.0),
        (95.0, 675.0),
        (100.0, 800.0),
    )


@dataclass
class Population:
    wages_q: List[float]
    deposits: List[float]
    loans: List[float]
    mortgage_loans: List[float]
    revolving_loans: List[float]
    mortgage_rate_q: List[float]
    mortgage_age_q: List[float]
    mortgage_term_q: List[float]
    mortgage_payment_sched_q: List[float]
    mortgage_orig_principal: List[float]
    mpc_q: List[float]
    base_real_cons_q: List[float]
    liquid_buffer_months_target: List[float]


# ----------------------------
# Generation logic
# ----------------------------


def _lognormal_samples(n: int, median: float, sigma_ln: float, rng: random.Random) -> List[float]:
    # If X ~ LogNormal(mu, sigma), then median = exp(mu)
    mu = math.log(max(median, 1e-12))
    out: List[float] = []
    for _ in range(n):
        z = rng.gauss(0.0, 1.0)
        out.append(math.exp(mu + sigma_ln * z))
    return out


def _pareto_samples(n: int, alpha: float, x_m: float, rng: random.Random) -> List[float]:
    """Pareto Type I with scale x_m and shape alpha: P(X>x)= (x_m/x)^alpha for x>=x_m."""
    alpha = max(alpha, 0.10)
    x_m = max(x_m, 1e-12)
    out: List[float] = []
    for _ in range(n):
        u = clamp(rng.random(), 1e-12, 1.0 - 1e-12)
        x = x_m / (u ** (1.0 / alpha))
        out.append(x)
    return out


def _assign_mpc_from_deposits(deposits: List[float], schedule: Tuple[Tuple[float, float], ...]) -> List[float]:
    """Assign per-family MPC based on deposits percentile rank.

    Semantics match the original implementation:
    - families are ranked by deposits ascending
    - the schedule is a list of (pct_upper_bound, mpc)
    - the cutoff index is round((pct/100)*(n-1))
    - ranks <= cutoff index get that mpc
    """
    n = len(deposits)
    if n == 0:
        return []

    if NP is None:
        # Fallback to the original pure-Python approach if numpy is unavailable
        order = sorted(range(n), key=lambda i: deposits[i])
        mpc = [0.35] * n

        cutoffs: List[Tuple[int, float]] = []
        for pct, m in schedule:
            idx = int(round((pct / 100.0) * (n - 1)))
            cutoffs.append((idx, m))

        next_cut = 0
        for rank, i in enumerate(order):
            while next_cut < len(cutoffs) and rank > cutoffs[next_cut][0]:
                next_cut += 1
            if next_cut >= len(cutoffs):
                mpc[i] = cutoffs[-1][1]
            else:
                mpc[i] = cutoffs[next_cut][1]
        return mpc

    np = NP
    dep = np.asarray(deposits, dtype=float)

    order = np.argsort(dep)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)

    # Build cutoff indices (ascending)
    cutoff_idx = np.array([int(round((pct / 100.0) * (n - 1))) for pct, _ in schedule], dtype=np.int32)
    mpc_vals = np.array([float(m) for _, m in schedule], dtype=float)

    # For each rank, choose the first cutoff index that is >= rank.
    # This matches the original loop which advanced when rank > cutoff.
    bin_idx = np.searchsorted(cutoff_idx, rank, side="left")
    mpc = mpc_vals[bin_idx]
    return mpc.astype(float).tolist()


def _assign_piecewise_by_percentile_rank(
    values: "NP.ndarray",
    schedule: Tuple[Tuple[float, float], ...],
) -> "NP.ndarray":
    """Assign piecewise-constant values from percentile-rank schedule."""
    np = NP
    if np is None:
        raise RuntimeError("NumPy is required for percentile-rank assignment.")

    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=float)

    order = np.argsort(values)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    cutoff_idx = np.array([int(round((pct / 100.0) * (n - 1))) for pct, _ in schedule], dtype=np.int32)
    piece_vals = np.array([float(v) for _, v in schedule], dtype=float)
    bin_idx = np.searchsorted(cutoff_idx, rank, side="left")
    return piece_vals[bin_idx]


def _assign_linear_by_percentile_rank(
    values: "NP.ndarray",
    schedule: Tuple[Tuple[float, float], ...],
) -> "NP.ndarray":
    """Assign linearly interpolated values from percentile-rank anchor schedule."""
    np = NP
    if np is None:
        raise RuntimeError("NumPy is required for percentile-rank assignment.")

    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=float)

    if len(schedule) == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(values)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    rank_pct = 100.0 * (rank.astype(float) / float(max(1, n - 1)))

    pct_vals = np.array([float(pct) for pct, _ in schedule], dtype=float)
    anchor_vals = np.array([float(v) for _, v in schedule], dtype=float)
    return np.interp(rank_pct, pct_vals, anchor_vals).astype(float)


def _with_linear_endpoint_anchors(
    schedule: Tuple[Tuple[float, float], ...],
    *,
    floor: float = 0.0,
) -> Tuple[Tuple[float, float], ...]:
    """Add endpoint anchors so linear interpolation does not create flat tail buckets."""
    if len(schedule) == 0:
        return tuple()

    pts = [(float(pct), float(val)) for pct, val in schedule]
    pts.sort(key=lambda item: item[0])

    if len(pts) >= 2 and pts[0][0] > 0.0:
        p0, v0 = pts[0]
        p1, v1 = pts[1]
        slope = (v1 - v0) / max(1e-9, (p1 - p0))
        extrap = max(float(floor), v0 - (slope * p0))
        pts.insert(0, (0.0, extrap))

    if len(pts) >= 2 and pts[-1][0] < 100.0:
        p0, v0 = pts[-2]
        p1, v1 = pts[-1]
        slope = (v1 - v0) / max(1e-9, (p1 - p0))
        extrap = max(float(floor), v1 + (slope * (100.0 - p1)))
        pts.append((100.0, extrap))

    return tuple(pts)


def generate_population(cfg: PopulationConfig) -> Population:
    # Vectorized implementation (NumPy). Keeps the same public interface and outputs.
    if NP is None:
        raise RuntimeError(
            "NumPy is required for the vectorized population generator. "
            "Install numpy or restore the non-vectorized generator."
        )

    np = NP
    # Pin the RNG algorithm for cross-version determinism.
    # Using PCG64 avoids potential future changes in np.random.default_rng().
    rng = np.random.Generator(np.random.PCG64(int(cfg.seed)))
    n = int(cfg.n_families)

    # Shared latent factor to induce correlation between wages and deposits
    rho = clamp(cfg.wage_deposit_corr, -0.95, 0.95)
    s = math.sqrt(max(0.0, 1.0 - rho * rho))
    z = rng.standard_normal(n)

    # --- Wages (correlated lognormal) ---
    mu_w = math.log(max(cfg.median_wage_q, 1e-12))
    e_w = rng.standard_normal(n)
    ln_w = mu_w + cfg.sigma_wage_ln * (rho * z + s * e_w)
    wages = np.exp(ln_w)

    wage_potential = wages.copy()

    # Apply employment rate by zeroing out some families
    if cfg.employment_rate < 1.0:
        employed = rng.random(n) < float(cfg.employment_rate)
        wages = wages * employed

    # --- Baseline consumption target (used in both startup calibration and in-run behavior) ---
    # Use interpolation between percentile anchors so households within a bucket can vary smoothly.
    wealth_signal = np.asarray(wage_potential, dtype=float)
    base_schedule = tuple((float(pct), float(val)) for pct, val in getattr(cfg, "base_real_cons_by_wealth_pct", ()))
    if len(base_schedule) > 0:
        base_real = _assign_linear_by_percentile_rank(wealth_signal, base_schedule)
    else:
        base_real = np.full(n, float(cfg.base_real_cons_q), dtype=float)

    # --- Liquid-buffer target (used both for initialization and in-run consumption behavior) ---
    # Interpolate between percentile anchors so the precautionary-buffer rule does not
    # create artificial household cohorts at a few discrete month targets.
    target_months = _assign_linear_by_percentile_rank(
        wealth_signal,
        _with_linear_endpoint_anchors(
            tuple((float(pct), float(months)) for pct, months in cfg.liquid_buffer_months_by_wealth_pct),
            floor=0.0,
        ),
    )

    # --- Deposits (liquid wealth proxy) ---
    deposit_mode = str(getattr(cfg, "deposit_generation_mode", "liquid_buffer_months")).strip().lower()
    if deposit_mode == "legacy_mixture":
        n_tail = int(round(float(cfg.tail_share) * n))
        n_tail = max(0, min(n, n_tail))
        n_body = n - n_tail

        idx_all = np.arange(n, dtype=np.int32)
        if n_tail > 0:
            idx_sorted_by_z = np.argsort(z)
            tail_idxs = idx_sorted_by_z[-n_tail:]
            body_mask = np.ones(n, dtype=bool)
            body_mask[tail_idxs] = False
            body_idxs = idx_all[body_mask]
        else:
            tail_idxs = np.empty(0, dtype=np.int32)
            body_idxs = idx_all

        mu_d = math.log(max(cfg.median_deposits_q, 1e-12))
        if n_body > 0:
            e_d = rng.standard_normal(n_body)
            ln_d = mu_d + cfg.sigma_deposits_ln * (rho * z[body_idxs] + s * e_d)
            body_deposits = np.exp(ln_d)
        else:
            body_deposits = np.empty(0, dtype=float)

        anchor_pct = 92.0
        if body_deposits.size > 0:
            anchor = float(np.percentile(body_deposits, anchor_pct))
        else:
            anchor = float(cfg.median_deposits_q)

        if n_tail > 0:
            alpha = max(float(cfg.pareto_alpha), 0.10)
            u = np.clip(rng.random(n_tail), 1e-12, 1.0 - 1e-12)
            tail_deposits = anchor / (u ** (1.0 / alpha))
        else:
            tail_deposits = np.empty(0, dtype=float)

        deposits = np.empty(n, dtype=float)
        if n_body > 0:
            deposits[body_idxs] = rng.permutation(body_deposits)
        if n_tail > 0:
            deposits[tail_idxs] = rng.permutation(tail_deposits)
    else:
        monthly_base_cons = np.maximum(0.0, base_real) / 3.0
        sigma_liq = max(0.0, float(cfg.liquid_buffer_sigma_ln))
        liq_noise = np.exp(sigma_liq * rng.standard_normal(n))
        deposits = np.maximum(0.0, monthly_base_cons * target_months * liq_noise)

    # --- MPC (decreasing with deposits percentile) ---
    mpc_q = _assign_mpc_from_deposits(deposits.tolist(), cfg.mpc_by_wealth_pct)

    # --- Debt (loan principal) ---
    wages_annual = 4.0 * wages

    # Compute deposits percentile rank for each family (0..100)
    order = np.argsort(deposits)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    denom = max(1, n - 1)
    pct_rank = 100.0 * (rank.astype(float) / float(denom))

    p = pct_rank

    # Mortgage probability schedule by deposits percentile. The config share
    # then rescales this schedule so the active-group mean matches the target.
    mort_p = np.zeros(n, dtype=float)
    mort_p = np.where((p > 10.0) & (p <= 70.0), 0.65 * (p - 10.0) / 60.0, mort_p)
    mort_p = np.where((p > 70.0) & (p <= 95.0), 0.65 + (0.45 - 0.65) * (p - 70.0) / 25.0, mort_p)
    mort_p = np.where(p > 95.0, 0.45 + (0.25 - 0.45) * (p - 95.0) / 5.0, mort_p)

    # Revolving probability schedule by deposits percentile. The config share
    # then rescales this schedule so the active-group mean matches the target.
    rev_p = np.empty(n, dtype=float)
    rev_p = np.where(p <= 20.0, 0.50, np.nan)
    rev_p = np.where((p > 20.0) & (p <= 50.0), 0.50 + (0.40 - 0.50) * (p - 20.0) / 30.0, rev_p)
    rev_p = np.where((p > 50.0) & (p <= 80.0), 0.40 + (0.25 - 0.40) * (p - 50.0) / 30.0, rev_p)
    rev_p = np.where((p > 80.0) & (p <= 95.0), 0.25 + (0.15 - 0.25) * (p - 80.0) / 15.0, rev_p)
    rev_p = np.where(p > 95.0, 0.15 + (0.10 - 0.15) * (p - 95.0) / 5.0, rev_p)

    has_wage = wages_annual > 0.0
    mort_p = _rescale_probabilities_to_target(mort_p, has_wage, float(cfg.mortgage_share))
    rev_p = _rescale_probabilities_to_target(rev_p, has_wage, float(cfg.revolving_share))

    mortgage_loans = np.zeros(n, dtype=float)
    revolving_loans = np.zeros(n, dtype=float)
    mortgage_rate_q = np.zeros(n, dtype=float)
    mortgage_age_q = np.zeros(n, dtype=float)
    mortgage_term_q = np.zeros(n, dtype=float)
    mortgage_payment_sched_q = np.zeros(n, dtype=float)
    mortgage_orig_principal = np.zeros(n, dtype=float)

    mort_mask = has_wage & (rng.random(n) < mort_p)
    if mort_mask.any():
        mort_mult = rng.lognormal(
            mean=math.log(max(cfg.mortgage_income_mult_median, 1e-12)),
            sigma=float(cfg.mortgage_income_mult_sigma),
            size=int(mort_mask.sum()),
        )
        mortgage_loans[mort_mask] = np.maximum(0.0, mort_mult * wages_annual[mort_mask])
        term_q = float(max(1, int(getattr(cfg, "mortgage_term_quarters", 60))))
        rate_q = float(max(0.0, float(cfg.mortgage_rate_effective) / 4.0))
        ages = rng.integers(0, int(term_q), size=int(mort_mask.sum()), endpoint=False).astype(float)
        mortgage_rate_q[mort_mask] = rate_q
        mortgage_age_q[mort_mask] = ages
        mortgage_term_q[mort_mask] = term_q
        rem_q = remaining_term(term_q, ages)
        mortgage_payment_sched_q[mort_mask] = payment_from_balance(mortgage_loans[mort_mask], rate_q, rem_q)
        mortgage_orig_principal[mort_mask] = orig_principal_from_balance(
            mortgage_loans[mort_mask],
            rate_q,
            term_q,
            ages,
        )

    rev_mask = has_wage & (rng.random(n) < rev_p)
    if rev_mask.any():
        rev_mult = rng.lognormal(
            mean=math.log(max(cfg.revolving_income_mult_median, 1e-12)),
            sigma=float(cfg.revolving_income_mult_sigma),
            size=int(rev_mask.sum()),
        )
        raw = np.maximum(0.0, rev_mult * wages_annual[rev_mask])
        cap_income = float(cfg.revolving_cap_income_mult) * wages_annual[rev_mask]
        cap_deposits = float(cfg.revolving_cap_deposits_mult) * deposits[rev_mask]
        revolving_loans[rev_mask] = np.maximum(0.0, np.minimum.reduce([raw, cap_income, cap_deposits]))

    loans = mortgage_loans + revolving_loans

    return Population(
        wages_q=wages.astype(float).tolist(),
        deposits=deposits.astype(float).tolist(),
        loans=loans.astype(float).tolist(),
        mortgage_loans=mortgage_loans.astype(float).tolist(),
        revolving_loans=revolving_loans.astype(float).tolist(),
        mortgage_rate_q=mortgage_rate_q.astype(float).tolist(),
        mortgage_age_q=mortgage_age_q.astype(float).tolist(),
        mortgage_term_q=mortgage_term_q.astype(float).tolist(),
        mortgage_payment_sched_q=mortgage_payment_sched_q.astype(float).tolist(),
        mortgage_orig_principal=mortgage_orig_principal.astype(float).tolist(),
        mpc_q=[float(x) for x in mpc_q],
        base_real_cons_q=base_real.astype(float).tolist(),
        liquid_buffer_months_target=target_months.astype(float).tolist(),
    )


# ----------------------------
# Baseline report
# ----------------------------


def baseline_report(pop: Population, cfg: PopulationConfig) -> None:
    wages = pop.wages_q
    deposits = pop.deposits
    loans = pop.loans

    # Debt service proxy: interest-only payment per quarter, using component-specific effective rates
    r_m_q = cfg.mortgage_rate_effective / 4.0
    r_r_q = cfg.revolving_rate_effective / 4.0
    mortgage_interest_q = [r_m_q * L for L in pop.mortgage_loans]
    revolving_interest_q = [r_r_q * L for L in pop.revolving_loans]
    interest_q = [mortgage_interest_q[i] + revolving_interest_q[i] for i in range(len(loans))]

    # Income proxy: wages only (Phase 1). Later add income support, taxes, dividends.
    income_q = wages

    # Avoid division by zero
    dti_interest = [interest_q[i] / max(1e-9, income_q[i]) if income_q[i] > 0 else 0.0 for i in range(len(wages))]

    # Summaries
    sw = summarize_distribution("Wages", wages)
    sd = summarize_distribution("Deposits", deposits)
    sl = summarize_distribution("Loans", loans)
    si = summarize_distribution("InterestQ", interest_q)
    sdt = summarize_distribution("DTI_interest", dti_interest, money=False)

    # Aggregates
    total_wages = sum(wages)
    total_deposits = sum(deposits)
    total_loans = sum(loans)
    total_interest = sum(interest_q)

    employed = sum(1 for w in wages if w > 0)

    print("\nSYNTHETIC POPULATION BASELINE (Phase 1)\n")
    print(f"Families: {len(wages):,d}   Seed: {cfg.seed}   Employment rate realized: {employed/len(wages):.3f}")
    print(f"Totals per quarter: Wages={fmt_money(total_wages)}  Deposits={fmt_money(total_deposits)}  Loans={fmt_money(total_loans)}  Interest={fmt_money(total_interest)}")
    print(f"Aggregate interest / aggregate wages (interest-only proxy): {total_interest / max(1e-9, total_wages):.3%}")

    # Incidence rates and conditional balance stats
    mort_idx = [i for i, L in enumerate(pop.mortgage_loans) if L > 0]
    rev_idx = [i for i, L in enumerate(pop.revolving_loans) if L > 0]
    any_idx = [i for i, L in enumerate(pop.loans) if L > 0]

    def _share(k: int) -> str:
        return f"{k / max(1, len(wages)):.1%}"

    print("\nDebt incidence (share of families):")
    print(f"  Mortgage debt:   {_share(len(mort_idx))}")
    print(f"  Revolving debt:  {_share(len(rev_idx))}")
    print(f"  Any debt:        {_share(len(any_idx))}")

    if mort_idx:
        mort_bal = [pop.mortgage_loans[i] for i in mort_idx]
        s = summarize_distribution("MortgageLoans", mort_bal)
        print("\nMortgage principal (conditional on having mortgage debt):")
        print(
            f"  median {fmt_money(s['p50'])} | p90 {fmt_money(s['p90'])} | p99 {fmt_money(s['p99'])} | mean {fmt_money(s['mean'])}"
        )

    if rev_idx:
        rev_bal = [pop.revolving_loans[i] for i in rev_idx]
        s = summarize_distribution("RevolvingLoans", rev_bal)
        print("\nRevolving principal (conditional on having revolving debt):")
        print(
            f"  median {fmt_money(s['p50'])} | p90 {fmt_money(s['p90'])} | p99 {fmt_money(s['p99'])} | mean {fmt_money(s['mean'])}"
        )

    # Interest burden conditional on having any debt (wage-attach only)
    debt_wage_idx = [i for i in any_idx if wages[i] > 0]
    if debt_wage_idx:
        dti_debt = [dti_interest[i] for i in debt_wage_idx]
        dti_debt.sort()
        print("\nInterest-to-wage ratio per quarter (conditional on having debt and wages > 0):")
        print(
            f"  median {percentile(dti_debt, 50):.3f} | p90 {percentile(dti_debt, 90):.3f} | p99 {percentile(dti_debt, 99):.3f}"
        )

        # Additional diagnostics
        avg_dti_debt = statistics.fmean(dti_interest[i] for i in debt_wage_idx)
        print(f"Average interest/wage among employed debtors: {avg_dti_debt:.3%}")

    tot_m_int = sum(mortgage_interest_q)
    tot_r_int = sum(revolving_interest_q)
    tot_int = tot_m_int + tot_r_int
    if tot_int > 0:
        print("\nInterest composition (share of total interest):")
        print(f"  Mortgage interest share:  {tot_m_int / tot_int:.1%}")
        print(f"  Revolving interest share: {tot_r_int / tot_int:.1%}")

    def show(name: str, s: Dict[str, float], as_money: bool = True):
        print(f"\n{name}:")

        def _fmt(x: float) -> str:
            return fmt_money(x) if as_money else f"{x:.3f}"

        row = (
            f"min {_fmt(float(s['min']))} | "
            f"p10 {_fmt(float(s['p10']))} | "
            f"p25 {_fmt(float(s['p25']))} | "
            f"median {_fmt(float(s['p50']))} | "
            f"p75 {_fmt(float(s['p75']))} | "
            f"p90 {_fmt(float(s['p90']))} | "
            f"p99 {_fmt(float(s['p99']))} | "
            f"max {_fmt(float(s['max']))} | "
            f"mean {_fmt(float(s['mean']))} | "
            f"gini {float(s['gini']):.3f}"
        )
        print(row)

    show("Wages (quarterly)", sw, as_money=True)
    show("Deposits (liquid wealth proxy)", sd, as_money=True)
    show("Loans (principal)", sl, as_money=True)
    show("Interest paid per quarter (interest-only proxy)", si, as_money=True)
    show("Interest-to-wage ratio per quarter (proxy)", sdt, as_money=False)

    # MPC sanity check by wealth quintile
    n = len(deposits)
    order = sorted(range(n), key=lambda i: deposits[i])
    quint = [order[int(q * n / 5.0): int((q + 1) * n / 5.0)] for q in range(5)]

    # Debt incidence by deposits quintile
    print("\nDebt incidence by deposits quintile (Q1=poorest):")
    for q, idxs in enumerate(quint, start=1):
        mort_share = sum(1 for i in idxs if pop.mortgage_loans[i] > 0) / max(1, len(idxs))
        rev_share = sum(1 for i in idxs if pop.revolving_loans[i] > 0) / max(1, len(idxs))
        any_share = sum(1 for i in idxs if pop.loans[i] > 0) / max(1, len(idxs))
        dep_p50 = percentile(sorted(deposits[i] for i in idxs), 50)
        mort_bal_q = sorted(pop.mortgage_loans[i] for i in idxs if pop.mortgage_loans[i] > 0)
        rev_bal_q = sorted(pop.revolving_loans[i] for i in idxs if pop.revolving_loans[i] > 0)
        dti_q = sorted(dti_interest[i] for i in idxs if pop.loans[i] > 0 and wages[i] > 0)
        dti_m_q = sorted((mortgage_interest_q[i] + 0.0) / max(1e-9, wages[i]) for i in idxs if pop.mortgage_loans[i] > 0 and wages[i] > 0)
        dti_r_q = sorted((revolving_interest_q[i] + 0.0) / max(1e-9, wages[i]) for i in idxs if pop.revolving_loans[i] > 0 and wages[i] > 0)

        mort_med = fmt_money(percentile(mort_bal_q, 50)) if mort_bal_q else "-"
        rev_med = fmt_money(percentile(rev_bal_q, 50)) if rev_bal_q else "-"
        dti_med = f"{percentile(dti_q, 50):.3f}" if dti_q else "-"
        dti_m_med = f"{percentile(dti_m_q, 50):.3f}" if dti_m_q else "-"
        dti_r_med = f"{percentile(dti_r_q, 50):.3f}" if dti_r_q else "-"

        print(
            f"  Q{q}: mortgage={mort_share:.1%}  revolving={rev_share:.1%}  any={any_share:.1%}  "
            f"medDep={fmt_money(dep_p50)}  mortMed={mort_med}  revMed={rev_med}  "
            f"dtiMed={dti_med}  mortDTI={dti_m_med}  revDTI={dti_r_med}"
        )

    print("\nMPC by deposits quintile (Q1=poorest):")
    for q, idxs in enumerate(quint, start=1):
        avg_mpc = statistics.fmean(pop.mpc_q[i] for i in idxs)
        dep_p50 = percentile(sorted(deposits[i] for i in idxs), 50)
        print(f"  Q{q}: avg MPC={avg_mpc:.3f}  median deposits={fmt_money(dep_p50)}")


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    cfg = PopulationConfig()
    pop = generate_population(cfg)
    baseline_report(pop, cfg)


if __name__ == "__main__":
    main()
