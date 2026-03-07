# Author: Roger Ison   roger@miximum.info
"""Policy sweep runner for NewLoop (silent runs + compact summary table).

Default grid is 2D:
  - trust_equity_cap
  - gov_tax_rebate_rate

Optional extra grids (tax/payout) can be supplied when needed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
from typing import Dict, Iterable, List, Optional

import numpy as np

# Support both execution modes:
# 1) package mode: python -m newloop.sweep_experiments
# 2) script mode:  run this file directly in IDEs (no package context)
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from newloop.table_of_data import TableOfData
    from newloop.newloop_core import NewLoop, TickResult, config
else:
    from .table_of_data import TableOfData
    from .newloop_core import NewLoop, TickResult, config


def _parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def _slope_last(values: Iterable[float], window: int) -> float:
    ys = np.asarray(list(values), dtype=float)
    if ys.size < 2:
        return 0.0
    w = int(max(2, min(window, ys.size)))
    y = ys[-w:]
    x = np.arange(w, dtype=float)
    x_center = x - float(x.mean())
    denom = float(np.dot(x_center, x_center))
    if denom <= 0.0:
        return 0.0
    y_center = y - float(y.mean())
    return float(np.dot(x_center, y_center) / denom)


def _run_scenario(
    n_quarters: int,
    window: int,
    trust_cap: float,
    rebate_rate: float,
    tax_base: Optional[float],
    tax_sens: Optional[float],
    payout: Optional[float],
) -> Dict[str, float]:
    cfg = copy.deepcopy(config)
    p = cfg["parameters"]

    p["trust_equity_cap"] = float(max(0.0, min(1.0, trust_cap)))
    p["gov_tax_rebate_rate"] = float(max(0.0, min(1.0, rebate_rate)))

    if tax_base is not None:
        p["corporate_tax_rate"] = float(max(0.0, min(1.0, tax_base)))
        if bool(p.get("corporate_tax_dynamic_with_wages", False)):
            base = float(max(0.0, min(1.0, tax_base)))
            p["corporate_tax_rate_base"] = base

    if tax_sens is not None and bool(p.get("corporate_tax_dynamic_with_wages", False)):
        p["corporate_tax_wage_sensitivity"] = float(max(0.0, tax_sens))

    if payout is not None:
        p["dividend_payout_rate_firms"] = float(max(0.0, min(1.0, payout)))

    sim = NewLoop(cfg)
    for _ in range(int(n_quarters)):
        sim.step()

    hist: List[TickResult] = sim.history
    if not hist:
        raise RuntimeError("Scenario produced no history rows.")

    denom = float(sim.hh.n) if sim.hh is not None else 1.0

    rcons = [(r.total_consumption / denom) / max(r.price_level, 1e-9) for r in hist]
    ubi = [float(r.ubi_per_h) for r in hist]
    ubi_f = [float(r.ubi_from_fund_dep_per_h) for r in hist]
    ubi_g = [float(r.ubi_from_gov_dep_per_h) for r in hist]
    ubi_i = [float(r.ubi_issued_per_h) for r in hist]

    ubi_total = float(sum(ubi))
    if ubi_total > 1e-12:
        ubi_f_share = float(sum(ubi_f) / ubi_total)
        ubi_g_share = float(sum(ubi_g) / ubi_total)
        ubi_i_share = float(sum(ubi_i) / ubi_total)
    else:
        ubi_f_share = 0.0
        ubi_g_share = 0.0
        ubi_i_share = 0.0

    last = hist[-1]
    return {
        "trust_cap": float(p["trust_equity_cap"]),
        "rebate_rate": float(p["gov_tax_rebate_rate"]),
        "tax_base": float(p["corporate_tax_rate"]),
        "tax_sens": float(p.get("corporate_tax_wage_sensitivity", 0.0)),
        "payout": float(p["dividend_payout_rate_firms"]),
        "rcons_end": float(rcons[-1]),
        "rcons_slope": _slope_last(rcons, window),
        "ubi_i_max": float(max(ubi_i)),
        "ubi_f_share": float(ubi_f_share),
        "ubi_g_share": float(ubi_g_share),
        "ubi_i_share": float(ubi_i_share),
        "eqcap_end": float(last.trust_equity_pct),
        "p_roe_end": float(last.private_roe_q),
        "p_eq_end": float(last.private_eq_per_h),
        "trust_active": float(1.0 if any(r.trust_active for r in hist) else 0.0),
    }


def _norm(value: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 1e-12:
        return 1.0
    return (value - lo) / span


def _add_rank_scores(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    rcons_vals = [float(r["rcons_end"]) for r in rows]
    slope_vals = [float(r["rcons_slope"]) for r in rows]
    ubi_i_vals = [float(r["ubi_i_max"]) for r in rows]
    gov_share_vals = [float(r["ubi_g_share"]) for r in rows]

    rcons_lo, rcons_hi = min(rcons_vals), max(rcons_vals)
    slope_lo, slope_hi = min(slope_vals), max(slope_vals)
    ubi_i_lo, ubi_i_hi = min(ubi_i_vals), max(ubi_i_vals)
    gov_lo, gov_hi = min(gov_share_vals), max(gov_share_vals)

    for r in rows:
        rcons_n = _norm(float(r["rcons_end"]), rcons_lo, rcons_hi)               # higher better
        slope_n = _norm(float(r["rcons_slope"]), slope_lo, slope_hi)             # higher better
        ubi_i_n = _norm(float(r["ubi_i_max"]), ubi_i_lo, ubi_i_hi)               # lower better
        gov_n = _norm(float(r["ubi_g_share"]), gov_lo, gov_hi)                   # lower better

        score = 100.0 * (
            (0.35 * rcons_n)
            + (0.25 * slope_n)
            + (0.20 * (1.0 - ubi_i_n))
            + (0.20 * (1.0 - gov_n))
        )
        r["score"] = float(score)

    sorted_scores = sorted((float(r["score"]) for r in rows), reverse=True)
    unique_scores: List[float] = []
    for s in sorted_scores:
        if not unique_scores or abs(s - unique_scores[-1]) > 1e-12:
            unique_scores.append(s)
    score_to_rank = {s: i + 1 for i, s in enumerate(unique_scores)}
    for r in rows:
        r["rank"] = float(score_to_rank[float(r["score"])])


def _build_table(rows: List[Dict[str, float]]) -> TableOfData:
    table = TableOfData(
        "id|Rr",
        "Rk|Rr",
        "Score|Rr|.1f",
        "cap|Rr|.2f",
        "reb|Rr|.2f",
        "RConsEnd|Rr|.2f",
        "dRCons20|Rr|.3f",
        "UBI_Imax|Rr|.2f",
        "Fsh|Rr|.1%",
        "Gsh|Rr|.1%",
        "Ish|Rr|.1%",
        "EqCap|Rr|.1%",
        "pROE|Rr|.2%",
        "pEq$|Rr|.2f",
        "Trust|Cc",
    )

    for i, r in enumerate(rows, start=1):
        row = table.AddRow()
        row[0] = int(i)
        row[1] = int(float(r.get("rank", 0.0)))
        row[2] = float(r.get("score", 0.0))
        row[3] = float(r["trust_cap"])
        row[4] = float(r["rebate_rate"])
        row[5] = float(r["rcons_end"])
        row[6] = float(r["rcons_slope"])
        row[7] = float(r["ubi_i_max"])
        row[8] = float(r["ubi_f_share"])
        row[9] = float(r["ubi_g_share"])
        row[10] = float(r["ubi_i_share"])
        row[11] = float(r["eqcap_end"])
        row[12] = float(r["p_roe_end"])
        row[13] = float(r["p_eq_end"])
        row[14] = "Y" if r["trust_active"] > 0.5 else "N"

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run silent NewLoop policy sweeps and print a compact summary table.")
    parser.add_argument("--quarters", type=int, default=80, help="Simulation length per scenario.")
    parser.add_argument("--window", type=int, default=20, help="Trailing window used for slope metrics.")
    parser.add_argument("--trust-cap", default="0.25,0.33,0.40,0.49", help="Comma-separated trust equity caps.")
    parser.add_argument("--rebate", default="0.80,0.90,1.00", help="Comma-separated gov tax rebate rates.")
    parser.add_argument("--tax-base", default="", help="Optional comma-separated base corporate tax rates.")
    parser.add_argument("--tax-sens", default="", help="Optional comma-separated wage-sensitivity values.")
    parser.add_argument("--payout", default="", help="Optional comma-separated firm payout rates.")
    parser.add_argument("--sort", choices=["none", "score", "rcons", "stability"], default="score", help="Sort rows by metric.")
    parser.add_argument("--top", type=int, default=0, help="Keep top N rows after sorting (0 = all).")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    trust_caps = _parse_float_list(args.trust_cap)
    rebates = _parse_float_list(args.rebate)

    tax_bases = _parse_float_list(args.tax_base) if args.tax_base.strip() else [None]
    tax_senses = _parse_float_list(args.tax_sens) if args.tax_sens.strip() else [None]
    payouts = _parse_float_list(args.payout) if args.payout.strip() else [None]

    rows: List[Dict[str, float]] = []
    for trust_cap, rebate, tax_base, tax_sens, payout in itertools.product(
        trust_caps, rebates, tax_bases, tax_senses, payouts
    ):
        row = _run_scenario(
            n_quarters=args.quarters,
            window=args.window,
            trust_cap=float(trust_cap),
            rebate_rate=float(rebate),
            tax_base=None if tax_base is None else float(tax_base),
            tax_sens=None if tax_sens is None else float(tax_sens),
            payout=None if payout is None else float(payout),
        )
        rows.append(row)

    _add_rank_scores(rows)

    if args.sort == "score":
        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    elif args.sort == "rcons":
        rows.sort(key=lambda r: float(r.get("rcons_end", 0.0)), reverse=True)
    elif args.sort == "stability":
        rows.sort(key=lambda r: abs(float(r.get("rcons_slope", 0.0))))

    if args.top > 0:
        rows = rows[: args.top]

    print(_build_table(rows))

    if args.csv:
        fieldnames = [
            "trust_cap",
            "rebate_rate",
            "tax_base",
            "tax_sens",
            "payout",
            "rcons_end",
            "rcons_slope",
            "ubi_i_max",
            "ubi_f_share",
            "ubi_g_share",
            "ubi_i_share",
            "eqcap_end",
            "p_roe_end",
            "p_eq_end",
            "trust_active",
            "score",
            "rank",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fieldnames})


if __name__ == "__main__":
    main()
