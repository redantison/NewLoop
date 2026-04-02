"""Modular household and corporate tax policies for NewLoop."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, MutableMapping, Protocol

import numpy as np

from .config import normalize_economic_regime_name, resolve_tax_policy_mode


@dataclass(frozen=True)
class HouseholdTaxResult:
    """Vectorized household-tax outputs for one solver iteration."""

    taxable_income_i: np.ndarray
    income_tax_i: np.ndarray
    vat_credit_i: np.ndarray
    taxable_income_before_deductions_i: np.ndarray
    mortgage_interest_deduction_i: np.ndarray
    threshold_lower: float
    threshold_upper: float


@dataclass(frozen=True)
class CorporateTaxResult:
    """Corporate-tax outputs for one solver iteration."""

    corp_tax_rate: float
    corp_tax_depr_rate_q: float
    corp_tax_depr_fa: float
    corp_tax_depr_fh: float
    corp_tax_base_fa: float
    corp_tax_base_fh: float
    corp_tax_fa: float
    corp_tax_fh: float
    corp_tax_bk: float


class TaxPolicy(Protocol):
    """Interface for tax-policy behavior inside the solver."""

    def mode_name(self) -> str:
        ...

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        ...

    def compute_household_taxes(
        self,
        *,
        wages_i: np.ndarray,
        div_i: np.ndarray,
        mort_interest_due_i: np.ndarray,
        support_per_h: float,
        price_level: float,
        state: MutableMapping[str, Any],
        base_real_avg: float | None = None,
        baseline_wages_i: np.ndarray | None = None,
        current_tax_anchor_wage: float = 0.0,
        current_vc_start_anchor_wage: float = 0.0,
        current_vc_end_anchor_wage: float = 0.0,
    ) -> HouseholdTaxResult:
        ...

    def compute_corporate_taxes(
        self,
        *,
        p_fa_pre_tax: float,
        p_fh_pre_tax: float,
        bank_profit_pre_tax: float,
        price_level: float,
        wages_total: float,
        state: MutableMapping[str, Any],
        fa_capital_real: float,
        fh_capital_real: float,
    ) -> CorporateTaxResult:
        ...


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _nearest_rank_percentile(values: Iterable[float], pct: float) -> float:
    data = [float(v) for v in values]
    if not data:
        return 0.0
    pct_clamped = max(0.0, min(100.0, float(pct)))
    data.sort()
    k = int(math.ceil((pct_clamped / 100.0) * len(data))) - 1
    k = max(0, min(len(data) - 1, k))
    return float(data[k])


class CurrentTaxPolicy:
    """Preserve the original NewLoop tax behavior."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def mode_name(self) -> str:
        return "CURRENT"

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        del state, baseline_wages_i, price_level

    def compute_household_taxes(
        self,
        *,
        wages_i: np.ndarray,
        div_i: np.ndarray,
        mort_interest_due_i: np.ndarray,
        support_per_h: float,
        price_level: float,
        state: MutableMapping[str, Any],
        base_real_avg: float | None = None,
        baseline_wages_i: np.ndarray | None = None,
        current_tax_anchor_wage: float = 0.0,
        current_vc_start_anchor_wage: float = 0.0,
        current_vc_end_anchor_wage: float = 0.0,
    ) -> HouseholdTaxResult:
        del mort_interest_due_i

        taxable_income = _as_array(wages_i) + _as_array(div_i)
        if bool(self.params.get("disable_income_tax", False)):
            it_rate = 0.0
        else:
            it_rate = max(0.0, float(self.params.get("income_tax_rate", 0.0)))

        income_total = float(np.sum(np.maximum(0.0, taxable_income)))
        baseline_sum = 0.0
        if baseline_wages_i is not None:
            baseline_sum = float(np.sum(np.maximum(0.0, _as_array(baseline_wages_i))))
        if baseline_sum <= 0.0:
            baseline_sum = max(1e-9, income_total)
        taxable_scale = income_total / max(1e-9, baseline_sum)

        it_thr = max(0.0, float(current_tax_anchor_wage)) * taxable_scale
        income_tax_i = it_rate * np.maximum(0.0, taxable_income - it_thr)

        if bool(self.params.get("disable_vat", False)):
            vat_rate = 0.0
        else:
            vat_rate = max(0.0, float(self.params.get("vat_rate", 0.0)))
        elig_income = taxable_income + float(support_per_h)
        vc_thr_start = (max(0.0, float(current_vc_start_anchor_wage)) * taxable_scale) + float(support_per_h)
        vc_thr_end = (max(0.0, float(current_vc_end_anchor_wage)) * taxable_scale) + float(support_per_h)

        if vc_thr_end <= (vc_thr_start + 1e-12):
            vat_credit_weight_i = (elig_income <= vc_thr_start).astype(float)
        else:
            vat_credit_weight_i = np.ones_like(elig_income, dtype=float)
            hi_mask = elig_income >= vc_thr_end
            mid_mask = (elig_income > vc_thr_start) & (~hi_mask)
            vat_credit_weight_i[hi_mask] = 0.0
            vat_credit_weight_i[mid_mask] = (
                (vc_thr_end - elig_income[mid_mask]) / (vc_thr_end - vc_thr_start)
            )

        p_now = max(1e-9, float(price_level))
        if base_real_avg is None:
            base_real_avg = float(state.get("baseline_real_cons_per_h", 0.0))
        if state.get("baseline_real_cons_per_h", None) is None:
            state["baseline_real_cons_per_h"] = float(max(0.0, base_real_avg or 0.0))
        pov_frac = max(0.0, float(self.params.get("vat_poverty_cons_frac", 0.0)))
        pov_nom = p_now * pov_frac * max(0.0, float(base_real_avg or 0.0))
        vat_credit_i = (vat_rate * pov_nom) * vat_credit_weight_i

        return HouseholdTaxResult(
            taxable_income_i=taxable_income.astype(float, copy=True),
            income_tax_i=income_tax_i.astype(float, copy=True),
            vat_credit_i=vat_credit_i.astype(float, copy=True),
            taxable_income_before_deductions_i=taxable_income.astype(float, copy=True),
            mortgage_interest_deduction_i=np.zeros_like(taxable_income, dtype=float),
            threshold_lower=float(it_thr),
            threshold_upper=float(vc_thr_end),
        )

    def compute_corporate_taxes(
        self,
        *,
        p_fa_pre_tax: float,
        p_fh_pre_tax: float,
        bank_profit_pre_tax: float,
        price_level: float,
        wages_total: float,
        state: MutableMapping[str, Any],
        fa_capital_real: float,
        fh_capital_real: float,
    ) -> CorporateTaxResult:
        corp_tax_rate = max(0.0, min(1.0, float(self.params.get("corporate_tax_rate", 0.0))))
        corp_tax_depr_rate_q = max(0.0, min(1.0, float(self.params.get("corporate_tax_depr_rate_q", 0.025))))

        if bool(self.params.get("corporate_tax_dynamic_with_wages", False)):
            wage_baseline = float(state.get("baseline_wages_total_pop", 0.0))
            wage_index = max(0.0, min(1.0, float(wages_total) / wage_baseline)) if wage_baseline > 0.0 else 1.0
            base_rate = float(self.params.get("corporate_tax_rate_base", corp_tax_rate))
            slope = float(self.params.get("corporate_tax_wage_sensitivity", 0.0))
            tax_min = max(0.0, min(1.0, float(self.params.get("corporate_tax_rate_min", 0.0))))
            tax_max = max(tax_min, min(1.0, float(self.params.get("corporate_tax_rate_max", 1.0))))
            corp_tax_rate = max(tax_min, min(tax_max, base_rate + (slope * (1.0 - wage_index))))

        p_now = max(1e-9, float(price_level))
        corp_tax_depr_fa = max(0.0, float(fa_capital_real)) * p_now * corp_tax_depr_rate_q
        corp_tax_depr_fh = max(0.0, float(fh_capital_real)) * p_now * corp_tax_depr_rate_q
        corp_tax_base_fa = max(0.0, float(p_fa_pre_tax) - corp_tax_depr_fa)
        corp_tax_base_fh = max(0.0, float(p_fh_pre_tax) - corp_tax_depr_fh)
        corp_tax_fa = corp_tax_rate * corp_tax_base_fa
        corp_tax_fh = corp_tax_rate * corp_tax_base_fh
        corp_tax_bk = corp_tax_rate * max(0.0, float(bank_profit_pre_tax))

        return CorporateTaxResult(
            corp_tax_rate=float(corp_tax_rate),
            corp_tax_depr_rate_q=float(corp_tax_depr_rate_q),
            corp_tax_depr_fa=float(corp_tax_depr_fa),
            corp_tax_depr_fh=float(corp_tax_depr_fh),
            corp_tax_base_fa=float(corp_tax_base_fa),
            corp_tax_base_fh=float(corp_tax_base_fh),
            corp_tax_fa=float(corp_tax_fa),
            corp_tax_fh=float(corp_tax_fh),
            corp_tax_bk=float(corp_tax_bk),
        )


class OldLoopTaxPolicy:
    """Conventional tax regime with progressive household tax and mortgage-interest deduction."""

    _LOWER_KEY = "old_loop_tax_threshold_lower_real"
    _UPPER_KEY = "old_loop_tax_threshold_upper_real"

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def mode_name(self) -> str:
        return "OLD_LOOP"

    def _index_level(self, price_level: float) -> float:
        p_now = float(price_level)
        if p_now <= 0.0:
            p_now = 1e-9
        return p_now

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        if state.get(self._LOWER_KEY, None) is not None and state.get(self._UPPER_KEY, None) is not None:
            return
        try:
            samples = [float(v) for v in baseline_wages_i]
        except Exception:
            samples = []
        if not samples:
            return
        self._initialize_threshold_anchors(
            state=state,
            market_income_i=np.asarray(samples, dtype=float),
            price_level=price_level,
        )

    def _initialize_threshold_anchors(
        self,
        *,
        state: MutableMapping[str, Any],
        market_income_i: np.ndarray,
        price_level: float,
    ) -> None:
        if state.get(self._LOWER_KEY, None) is not None and state.get(self._UPPER_KEY, None) is not None:
            return
        pct_lo = max(0.0, min(100.0, float(self.params.get("old_loop_tax_threshold_lower_pct", 30.0))))
        pct_hi = max(pct_lo, min(100.0, float(self.params.get("old_loop_tax_threshold_upper_pct", 80.0))))
        lo_nom = _nearest_rank_percentile(market_income_i.tolist(), pct_lo)
        hi_nom = _nearest_rank_percentile(market_income_i.tolist(), pct_hi)
        index_level = self._index_level(price_level)
        state[self._LOWER_KEY] = float(max(0.0, lo_nom) / index_level)
        state[self._UPPER_KEY] = float(max(lo_nom, hi_nom) / index_level)
        state["old_loop_tax_threshold_lower_pct_effective"] = float(pct_lo)
        state["old_loop_tax_threshold_upper_pct_effective"] = float(pct_hi)

    def compute_household_taxes(
        self,
        *,
        wages_i: np.ndarray,
        div_i: np.ndarray,
        mort_interest_due_i: np.ndarray,
        support_per_h: float,
        price_level: float,
        state: MutableMapping[str, Any],
        base_real_avg: float | None = None,
        baseline_wages_i: np.ndarray | None = None,
        current_tax_anchor_wage: float = 0.0,
        current_vc_start_anchor_wage: float = 0.0,
        current_vc_end_anchor_wage: float = 0.0,
    ) -> HouseholdTaxResult:
        del support_per_h, base_real_avg, baseline_wages_i, current_tax_anchor_wage, current_vc_start_anchor_wage, current_vc_end_anchor_wage

        taxable_before = _as_array(wages_i) + _as_array(div_i)
        if state.get(self._LOWER_KEY, None) is None or state.get(self._UPPER_KEY, None) is None:
            self._initialize_threshold_anchors(
                state=state,
                market_income_i=taxable_before,
                price_level=price_level,
            )

        deduction_enabled = bool(self.params.get("old_loop_mortgage_interest_deduction", True))
        mort_deduction_i = _as_array(mort_interest_due_i) if deduction_enabled else np.zeros_like(taxable_before, dtype=float)
        taxable_income = np.maximum(0.0, taxable_before - mort_deduction_i)

        index_level = self._index_level(price_level)
        thr_lo = max(0.0, float(state.get(self._LOWER_KEY, 0.0))) * index_level
        thr_hi = max(thr_lo, float(state.get(self._UPPER_KEY, 0.0)) * index_level)
        rate_lo = max(0.0, min(1.0, float(self.params.get("old_loop_tax_rate_lower", 0.15))))
        rate_hi = max(rate_lo, min(1.0, float(self.params.get("old_loop_tax_rate_upper", 0.28))))

        income_tax_i = np.zeros_like(taxable_income, dtype=float)
        if not bool(self.params.get("disable_income_tax", False)):
            middle_band = np.clip(taxable_income - thr_lo, 0.0, max(0.0, thr_hi - thr_lo))
            top_band = np.maximum(0.0, taxable_income - thr_hi)
            income_tax_i = (rate_lo * middle_band) + (rate_hi * top_band)

        return HouseholdTaxResult(
            taxable_income_i=taxable_income.astype(float, copy=True),
            income_tax_i=income_tax_i.astype(float, copy=True),
            vat_credit_i=np.zeros_like(taxable_income, dtype=float),
            taxable_income_before_deductions_i=taxable_before.astype(float, copy=True),
            mortgage_interest_deduction_i=mort_deduction_i.astype(float, copy=True),
            threshold_lower=float(thr_lo),
            threshold_upper=float(thr_hi),
        )

    def compute_corporate_taxes(
        self,
        *,
        p_fa_pre_tax: float,
        p_fh_pre_tax: float,
        bank_profit_pre_tax: float,
        price_level: float,
        wages_total: float,
        state: MutableMapping[str, Any],
        fa_capital_real: float,
        fh_capital_real: float,
    ) -> CorporateTaxResult:
        del wages_total, state
        corp_tax_rate = max(0.0, min(1.0, float(self.params.get("old_loop_corporate_tax_rate", 0.35))))
        corp_tax_depr_rate_q = max(0.0, min(1.0, float(self.params.get("corporate_tax_depr_rate_q", 0.025))))
        p_now = max(1e-9, float(price_level))
        corp_tax_depr_fa = max(0.0, float(fa_capital_real)) * p_now * corp_tax_depr_rate_q
        corp_tax_depr_fh = max(0.0, float(fh_capital_real)) * p_now * corp_tax_depr_rate_q
        corp_tax_base_fa = max(0.0, float(p_fa_pre_tax) - corp_tax_depr_fa)
        corp_tax_base_fh = max(0.0, float(p_fh_pre_tax) - corp_tax_depr_fh)
        corp_tax_fa = corp_tax_rate * corp_tax_base_fa
        corp_tax_fh = corp_tax_rate * corp_tax_base_fh
        corp_tax_bk = corp_tax_rate * max(0.0, float(bank_profit_pre_tax))

        return CorporateTaxResult(
            corp_tax_rate=float(corp_tax_rate),
            corp_tax_depr_rate_q=float(corp_tax_depr_rate_q),
            corp_tax_depr_fa=float(corp_tax_depr_fa),
            corp_tax_depr_fh=float(corp_tax_depr_fh),
            corp_tax_base_fa=float(corp_tax_base_fa),
            corp_tax_base_fh=float(corp_tax_base_fh),
            corp_tax_fa=float(corp_tax_fa),
            corp_tax_fh=float(corp_tax_fh),
            corp_tax_bk=float(corp_tax_bk),
        )


def make_tax_policy(params: Mapping[str, Any]) -> TaxPolicy:
    """Create the active tax policy after considering regime defaults."""
    normalized_params = dict(params)
    normalized_params["economic_regime"] = normalize_economic_regime_name(params.get("economic_regime", "NewLoop"))
    mode = resolve_tax_policy_mode(normalized_params)
    if mode == "old_loop":
        return OldLoopTaxPolicy(params=params)
    return CurrentTaxPolicy(params=params)
