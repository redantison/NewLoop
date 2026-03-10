# Author: Roger Ison   roger@miximum.info
"""Income-support policy and funding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol


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

        if bool(self.params.get("uis_monotonic_floor", True)):
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
    ) -> None:
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


def make_income_support_policy(params: Mapping[str, Any]) -> IncomeSupportPolicy:
    """Create the active income-support policy.

    Phase 1 keeps UIS behavior only; additional policies can be added later.
    """

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
