# Author: Roger Ison   roger@miximum.info
"""Income-support policy and funding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, MutableMapping, Protocol


class NodeLike(Protocol):
    """Protocol for economy nodes used by income-support helpers."""

    def get(self, key: str, default: float = 0.0) -> float:
        ...

    def add(self, key: str, delta: float) -> None:
        ...


class IncomeSupportPolicy(Protocol):
    """Interface for policy-specific income-support behavior."""

    def mode_name(self) -> str:
        ...

    def label_short(self) -> str:
        ...

    def label_long(self) -> str:
        ...

    def compute_per_household(
        self,
        *,
        wages_total: float,
        div_house_total: float,
        price_level: float,
        n_households: int,
        previous_support_per_h: float,
        state: Mapping[str, Any],
    ) -> float:
        ...

    def initialize_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        wages_total: float,
        support_per_h: float,
        n_households: int,
        price_level: float,
        wages_i: Any = None,
        div_i: Any = None,
    ) -> None:
        ...

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        ...


@dataclass(frozen=True)
class IncomeSupportFundingBreakdown:
    """Totals for each income-support funding source in one tick."""

    from_fund_dep_total: float
    from_gov_dep_total: float
    issued_total: float


class UISPolicy:
    """Universal Income Stabilizer policy implementation."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def _monotonic_floor_enabled(self) -> bool:
        return bool(
            self.params.get(
                "income_support_monotonic_floor",
                self.params.get("uis_monotonic_floor", True),
            )
        )

    def mode_name(self) -> str:
        return "UIS"

    def label_short(self) -> str:
        return "UIS"

    def label_long(self) -> str:
        return "Universal Income Stabilizer (UIS)"

    def compute_per_household(
        self,
        *,
        wages_total: float,
        div_house_total: float,
        price_level: float,
        n_households: int,
        previous_support_per_h: float,
        state: Mapping[str, Any],
    ) -> float:
        del div_house_total  # reserved for future policy modes

        n = max(1, int(n_households))
        target_pool_real_pop = state.get("income_target_pool_real_pop", None)

        if target_pool_real_pop is None:
            support = 0.0
        else:
            p_now = float(price_level)
            if p_now <= 0.0:
                p_now = 1e-9
            target_pool_nom = float(target_pool_real_pop) * p_now
            support = max(0.0, (target_pool_nom - float(wages_total)) / float(n))

        if self._monotonic_floor_enabled():
            support = max(float(previous_support_per_h), float(support))

        return float(support)

    def initialize_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        wages_total: float,
        support_per_h: float,
        n_households: int,
        price_level: float,
        wages_i: Any = None,
        div_i: Any = None,
    ) -> None:
        del wages_i, div_i  # unused by UIS anchor logic

        if state.get("income_target_pool_real_pop", None) is not None:
            return

        n = max(1, int(n_households))
        p_base = float(price_level)
        if p_base <= 0.0:
            p_base = 1e-9

        target_pool_nom_base = float(wages_total) + float(n) * float(support_per_h)
        state["income_target_pool_real_pop"] = float(target_pool_nom_base) / float(p_base)
        state["baseline_price_level_pop"] = float(p_base)
        state["baseline_wages_total_pop"] = float(wages_total)

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        del state, baseline_wages_i, price_level  # UIS baseline comes from first solved tick


class UBIPolicy:
    """Fixed per-household UBI policy (price-indexed after baseline anchor initialization)."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def mode_name(self) -> str:
        return "UBI"

    def label_short(self) -> str:
        return "UBI"

    def label_long(self) -> str:
        return "Universal Basic Income (UBI)"

    def _index_level(self, price_level: float) -> float:
        p = float(price_level)
        if p <= 0.0:
            p = 1e-9
        series = str(self.params.get("ubi_index_series", "P_producer")).strip()
        if series == "C_consumer":
            if bool(self.params.get("disable_vat", False)):
                vat = 0.0
            else:
                vat = max(0.0, float(self.params.get("vat_rate", 0.0)))
            return max(1e-9, p * (1.0 + vat))
        return max(1e-9, p)

    def _monotonic_floor_enabled(self) -> bool:
        return bool(
            self.params.get(
                "income_support_monotonic_floor",
                self.params.get("uis_monotonic_floor", True),
            )
        )

    @staticmethod
    def _nearest_rank_percentile(values: Iterable[float], pct: float) -> float:
        data = [float(v) for v in values]
        if not data:
            return 0.0
        pct_clamped = max(0.0, min(100.0, float(pct)))
        data.sort()
        k = int(math.ceil((pct_clamped / 100.0) * len(data))) - 1
        k = max(0, min(len(data) - 1, k))
        return float(data[k])

    def compute_per_household(
        self,
        *,
        wages_total: float,
        div_house_total: float,
        price_level: float,
        n_households: int,
        previous_support_per_h: float,
        state: Mapping[str, Any],
    ) -> float:
        del wages_total, div_house_total, n_households  # UBI amount is anchor-based after initialization

        anchor_real_per_h = state.get("ubi_anchor_real_per_h", None)
        if anchor_real_per_h is None:
            support = 0.0
        else:
            support = max(0.0, float(anchor_real_per_h)) * self._index_level(float(price_level))

        if self._monotonic_floor_enabled():
            support = max(float(previous_support_per_h), float(support))

        return float(support)

    def initialize_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        wages_total: float,
        support_per_h: float,
        n_households: int,
        price_level: float,
        wages_i: Any = None,
        div_i: Any = None,
    ) -> None:
        del support_per_h

        # Dynamic corporate-tax logic uses baseline wages in both modes.
        # Ensure UBI mode seeds the same baseline wage anchor that UIS provides.
        if state.get("baseline_wages_total_pop", None) is None:
            state["baseline_wages_total_pop"] = float(wages_total)
        if state.get("baseline_price_level_pop", None) is None:
            p_base = float(price_level)
            if p_base <= 0.0:
                p_base = 1e-9
            state["baseline_price_level_pop"] = float(p_base)

        if state.get("ubi_anchor_real_per_h", None) is not None:
            return

        n = max(1, int(n_households))
        pct = float(self.params.get("ubi_target_percentile", 30.0))
        basis = str(self.params.get("ubi_anchor_income_basis", "market_income")).strip()

        samples: list[float] = []
        if basis == "wages_only":
            if wages_i is not None:
                samples = [float(v) for v in wages_i]
        else:
            if wages_i is not None and div_i is not None:
                try:
                    samples = [float(w) + float(d) for w, d in zip(wages_i, div_i)]
                except Exception:
                    samples = []

        if samples:
            anchor_nom_per_h = max(0.0, self._nearest_rank_percentile(samples, pct))
        else:
            anchor_nom_per_h = max(0.0, float(wages_total) / float(n))

        index_level = self._index_level(float(price_level))
        anchor_real_per_h = anchor_nom_per_h / max(1e-9, index_level)

        state["ubi_anchor_real_per_h"] = float(anchor_real_per_h)
        state["ubi_anchor_nominal_per_h_base"] = float(anchor_nom_per_h)
        state["ubi_anchor_percentile"] = float(max(0.0, min(100.0, pct)))
        state["ubi_anchor_basis"] = basis

    def warm_start_anchor_if_needed(
        self,
        *,
        state: MutableMapping[str, Any],
        baseline_wages_i: Any,
        price_level: float,
    ) -> None:
        if state.get("ubi_anchor_real_per_h", None) is not None:
            return
        if baseline_wages_i is None:
            return

        try:
            wages_seq = [float(v) for v in baseline_wages_i]
        except Exception:
            return
        if not wages_seq:
            return

        n = len(wages_seq)
        self.initialize_anchor_if_needed(
            state=state,
            wages_total=float(sum(wages_seq)),
            support_per_h=0.0,
            n_households=int(n),
            price_level=float(price_level),
            wages_i=wages_seq,
            div_i=[0.0] * n,
        )


def make_income_support_policy(params: Mapping[str, Any]) -> IncomeSupportPolicy:
    """Create the active income-support policy.

    Supported modes:
    - UIS: Universal Income Stabilizer
    - UBI: Universal Basic Income
    """

    mode = str(params.get("income_support_mode", "UIS")).strip().upper()
    if mode == "UBI":
        return UBIPolicy(params=params)
    return UISPolicy(params=params)


def apply_income_support_payment(
    *,
    support_per_household: float,
    n_households: int,
    issue_share: float,
    deposits: Any,
    nodes: Mapping[str, NodeLike],
) -> IncomeSupportFundingBreakdown:
    """Apply per-household income support using issuance -> FUND -> GOV -> issuance."""

    n = max(1, int(n_households))
    support_per_h = float(support_per_household)

    fund_paid_from_dep_total = 0.0
    gov_paid_from_dep_total = 0.0
    issued_total = 0.0

    if support_per_h <= 0.0:
        return IncomeSupportFundingBreakdown(
            from_fund_dep_total=0.0,
            from_gov_dep_total=0.0,
            issued_total=0.0,
        )

    support_total = support_per_h * float(n)
    issue_share_clamped = max(0.0, min(1.0, float(issue_share)))

    issue_target = support_total * issue_share_clamped
    if issue_target > 0.0:
        issued_total += issue_target
        deposits[:] = deposits + (issue_target / float(n))
        nodes["BANK"].add("deposit_liab", issue_target)
        nodes["BANK"].add("reserves", issue_target)
        nodes["GOV"].add("money_issued", issue_target)

    fund_needed = max(0.0, support_total - issue_target)

    pay_fund_total = min(fund_needed, max(0.0, nodes["FUND"].get("deposits")))
    if pay_fund_total > 0.0:
        nodes["FUND"].add("deposits", -pay_fund_total)
        fund_paid_from_dep_total += pay_fund_total
        deposits[:] = deposits + (pay_fund_total / float(n))
        fund_needed -= pay_fund_total

    if fund_needed > 0.0:
        pay_gov_total = min(fund_needed, max(0.0, nodes["GOV"].get("deposits")))
        if pay_gov_total > 0.0:
            nodes["GOV"].add("deposits", -pay_gov_total)
            gov_paid_from_dep_total += pay_gov_total
            deposits[:] = deposits + (pay_gov_total / float(n))
            fund_needed -= pay_gov_total

    if fund_needed > 0.0:
        issued_total += fund_needed
        deposits[:] = deposits + (fund_needed / float(n))
        nodes["BANK"].add("deposit_liab", fund_needed)
        nodes["BANK"].add("reserves", fund_needed)
        nodes["GOV"].add("money_issued", fund_needed)

    return IncomeSupportFundingBreakdown(
        from_fund_dep_total=float(fund_paid_from_dep_total),
        from_gov_dep_total=float(gov_paid_from_dep_total),
        issued_total=float(issued_total),
    )
