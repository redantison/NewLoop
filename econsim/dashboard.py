# Author: Roger Ison   roger@miximum.info
"""Terminal dashboard runner for EconomySim."""

from __future__ import annotations

import sys
from typing import Any, Dict

import numpy as np

from .mathutils import calculate_gini_np
from .table_of_data import TableOfData

from .engine import EconomySim


def run_cli(config: Dict[str, Any], n_quarters: int = 80) -> None:
    def terminal_reset() -> None:
        """Send terminal reset sequence when stdout is a TTY."""
        if sys.stdout.isatty():
            # RIS (Reset to Initial State)
            print("\033c", end="", flush=True)

    def terminal_clear_all() -> None:
        """Clear screen + scrollback in xterm-like terminals (incl. VS Code terminal) when stdout is a TTY."""
        if sys.stdout.isatty():
            # 3J: clear scrollback, 2J: clear screen, H: cursor home
            print("\033[3J\033[2J\033[H", end="", flush=True)

    def _fmt_compact(v: float, width: int = 7, decimals: int = 2) -> str:
        """Compact human-readable number formatting that stays within a fixed width."""
        x = float(v)
        ax = abs(x)
        d = max(0, int(decimals))

        if ax >= 1e12:
            s = f"{x/1e12:.{d}f}T"
        elif ax >= 1e9:
            s = f"{x/1e9:.{d}f}B"
        elif ax >= 1e6:
            s = f"{x/1e6:.{d}f}M"
        elif ax >= 1e3:
            s = f"{x/1e3:.{d}f}k"
        else:
            s = f"{x:.0f}"

        if len(s) > width:
            s = f"{x:.2e}"

        return s.rjust(width)

    def _dist_stats(values: np.ndarray) -> Dict[str, float]:
        xs = np.asarray(values, dtype=float)
        if xs.size == 0:
            return {
                "n": 0.0,
                "min": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "gini": 0.0,
            }
        ys = np.sort(xs)
        return {
            "n": float(ys.size),
            "min": float(ys[0]),
            "p10": float(np.percentile(ys, 10.0)),
            "p25": float(np.percentile(ys, 25.0)),
            "p50": float(np.percentile(ys, 50.0)),
            "p75": float(np.percentile(ys, 75.0)),
            "p90": float(np.percentile(ys, 90.0)),
            "p99": float(np.percentile(ys, 99.0)),
            "max": float(ys[-1]),
            "mean": float(np.mean(ys)),
            "gini": float(calculate_gini_np(ys)),
        }

    def _household_vectors_snapshot() -> Dict[str, np.ndarray] | None:
        if sim.hh is None or sim.hh.n <= 0:
            return None

        hh = sim.hh
        dep_i = np.asarray(hh.deposits, dtype=float)
        loan_i = np.asarray(hh.mortgage_loans, dtype=float) + np.asarray(hh.revolving_loans, dtype=float)

        # Disposable-income proxy: solver stores last-quarter disposable income in prev_income.
        # At t=0 this defaults to wages0_q.
        income_i = np.asarray(hh.prev_income, dtype=float)
        if income_i.shape[0] != hh.n:
            income_i = np.asarray(hh.wages0_q, dtype=float)

        # Align wealth proxy with engine's gini_wealth construction.
        w0 = np.asarray(hh.wages0_q, dtype=float)
        w0_sum = float(w0.sum()) if w0.shape[0] == hh.n else 0.0
        if w0_sum > 0.0:
            weights = w0 / w0_sum
        else:
            weights = np.full(hh.n, 1.0 / float(hh.n), dtype=float)

        p_now = float(sim.state.get("price_level", 1.0))
        if p_now <= 0.0:
            p_now = 1e-9

        def _hh_share_frac(issuer: str, key: str) -> float:
            so = float(sim.nodes[issuer].get("shares_outstanding", 0.0))
            if so <= 0.0:
                return 0.0
            frac_hh = float(sim.nodes["HH"].get(key, 0.0)) / so
            return max(0.0, min(1.0, frac_hh))

        fa_eq = max(
            0.0,
            float(sim.nodes["FA"].get("deposits", 0.0))
            + float(sim.nodes["FA"].get("K", 0.0)) * p_now
            - float(sim.nodes["FA"].get("loans", 0.0)),
        )
        fh_eq = max(
            0.0,
            float(sim.nodes["FH"].get("deposits", 0.0))
            + float(sim.nodes["FH"].get("K", 0.0)) * p_now
            - float(sim.nodes["FH"].get("loans", 0.0)),
        )
        bk_eq = max(0.0, float(sim.nodes["BANK"].get("equity", 0.0)))

        hh_equity_total = (
            _hh_share_frac("FA", "shares_FA") * fa_eq
            + _hh_share_frac("FH", "shares_FH") * fh_eq
            + _hh_share_frac("BANK", "shares_BANK") * bk_eq
        )
        equity_i = weights * hh_equity_total
        wealth_i = dep_i + equity_i - loan_i

        if show_price_normalized:
            scale = p0_display / p_now
            income_i = income_i * scale
            wealth_i = wealth_i * scale

        return {
            "income": income_i,
            "wealth": wealth_i,
        }

    def _print_before_after_distributions(before: Dict[str, np.ndarray] | None, after: Dict[str, np.ndarray] | None) -> None:
        if before is None or after is None:
            print("Population distribution summary unavailable (no household population state).")
            print()
            return

        metrics = ["min", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean", "gini"]
        labels = {
            "min": "min",
            "p10": "p10",
            "p25": "p25",
            "p50": "p50",
            "p75": "p75",
            "p90": "p90",
            "p99": "p99",
            "max": "max",
            "mean": "mean",
            "gini": "gini",
        }

        inc0 = _dist_stats(before["income"])
        incT = _dist_stats(after["income"])
        w0 = _dist_stats(before["wealth"])
        wT = _dist_stats(after["wealth"])

        print("Population Distribution Summary (Before vs After)")
        unit_label = "base-period dollars" if show_price_normalized else "nominal dollars"
        print(f"Income metric: household disposable-income proxy (prev_income), in {unit_label}.")
        print("Wealth metric: deposits + allocated HH equity claims - loans, in the same units.")

        t = TableOfData("Metric|Ll", "Inc_Before|Rr", "Inc_After|Rr", "Wealth_Before|Rr", "Wealth_After|Rr")
        for k in metrics:
            row = t.AddRow()
            row[0] = labels[k]
            if k == "gini":
                row[1] = f"{inc0[k]:.3f}"
                row[2] = f"{incT[k]:.3f}"
                row[3] = f"{w0[k]:.3f}"
                row[4] = f"{wT[k]:.3f}"
            else:
                row[1] = _fmt_compact(inc0[k], 8).strip()
                row[2] = _fmt_compact(incT[k], 8).strip()
                row[3] = _fmt_compact(w0[k], 8).strip()
                row[4] = _fmt_compact(wT[k], 8).strip()
        print(t)

        inc_before = np.asarray(before["income"], dtype=float)
        inc_after = np.asarray(after["income"], dtype=float)
        wealth_before = np.asarray(before["wealth"], dtype=float)
        wealth_after = np.asarray(after["wealth"], dtype=float)
        n_cmp = int(min(inc_before.size, inc_after.size, wealth_before.size, wealth_after.size))

        if n_cmp > 0:
            d_inc = inc_after[:n_cmp] - inc_before[:n_cmp]
            d_w = wealth_after[:n_cmp] - wealth_before[:n_cmp]

            share_wealth_up = float(np.mean(d_w > 0.0))
            share_wealth_down = float(np.mean(d_w < 0.0))

            d_w_p05 = float(np.percentile(d_w, 5.0))
            d_w_p50 = float(np.percentile(d_w, 50.0))
            d_w_p95 = float(np.percentile(d_w, 95.0))

            d_inc_p05 = float(np.percentile(d_inc, 5.0))
            d_inc_p50 = float(np.percentile(d_inc, 50.0))
            d_inc_p95 = float(np.percentile(d_inc, 95.0))

            print(
                "Change diagnostics: "
                f"Wealth up={share_wealth_up:.1%}, down={share_wealth_down:.1%}; "
                f"Wealth Δ (p05/p50/p95)=({_fmt_compact(d_w_p05, 8).strip()}, {_fmt_compact(d_w_p50, 8).strip()}, {_fmt_compact(d_w_p95, 8).strip()}); "
                f"Income Δ (p05/p50/p95)=({_fmt_compact(d_inc_p05, 8).strip()}, {_fmt_compact(d_inc_p50, 8).strip()}, {_fmt_compact(d_inc_p95, 8).strip()})."
            )

        print()

    # ---- run ----
    sim = EconomySim(config)
    value_mode = str(sim.params.get("dashboard_value_mode", "nominal")).strip().lower()
    price_normalized_modes = {"price_normalized", "price-normalized", "normalized", "real"}
    show_price_normalized = value_mode in price_normalized_modes
    p0_display = float(sim.params.get("price_level_initial", 1.0))
    if p0_display <= 0.0:
        p0_display = 1e-9

    def _disp_money(v: float, p_now: float) -> float:
        x = float(v)
        if not show_price_normalized:
            return x
        p = float(p_now) if float(p_now) > 0.0 else 1e-9
        return x * (p0_display / p)

    # Reset terminal state before printing anything
    terminal_reset()
    # Clear BEFORE printing anything else
    terminal_clear_all()

    dist_before = _household_vectors_snapshot()

    n_quarters = int(n_quarters)

    dashboard = TableOfData(
        "Q|Rr",
        "Auto|Rr|.2f",
        "Info|Rr|.2f",
        "Phys|Rr|.2f",
        "P|Rr|.3f",
        "Gm|Rr|.2f",
        "Gi|Rr|.2f",
        "Gw|Rr|.2f",
        "pEq$|Rr",
        "pROE|Rr|.1%",
        "pInv|Rr|.2f",
        "VAT|Rr",
        "IncTx|Rr",
        "CoTx|Rr",
        "CoR%|Rr",
        "VATCr|Rr",
        "GOVdep|Rr",
        "FUNDdep|Rr",
        "TrClm|Rr",
        "TrBal$|Rr",
        "UBI,F,G,I|Cc",
        "Wages|Rr",
        "CAPEX|Rr",
        "AvgInc|Rr",
        "RAvgInc|Rr",
        "Cons|Rr",
        "RCons|Rr",
        "HHLoan|Rr",
        "EqCap|Rr|.1%",
        "Resv|Rr",
        "DepLb|Rr",
        "LoanA|Rr",
    )

    print()
    print(f"Simulating {n_quarters} quarters of AI Transition and Structural Reform (SFC + diagnostics)...")
    if show_price_normalized:
        print(f"Value view: price_normalized (base-period dollars, P0={p0_display:.3f}).")
    else:
        print("Value view: nominal dollars.")
    zero_tol = 1e-6
    max_abs_trdebt = 0.0
    max_abs_dep_identity_gap = 0.0
    max_abs_loan_identity_gap = 0.0
    max_abs_bank_balance_gap = 0.0
    trust_ever_activated = False

    for _ in range(n_quarters):
        sim.step()
        r = sim.history[-1]
        trust_ever_activated = trust_ever_activated or bool(r.trust_active)

        denom = float(sim.hh.n) if (sim.hh is not None and bool(sim.params.get("use_population", False))) else 1.0
        avg_inc_nom = (float(r.wages_total) / denom) + float(r.ubi_per_h)
        P_now = float(r.price_level) if float(r.price_level) > 0 else 1e-9

        row = dashboard.AddRow()
        row[0] = f"{int(r.t):02d}"
        row[1] = float(r.automation)
        row[2] = float(r.automation_info)
        row[3] = float(r.automation_phys)
        row[4] = float(r.price_level)
        row[5] = float(r.gini_market)
        row[6] = float(r.gini_disp)
        row[7] = float(r.gini_wealth)
        row[8] = _fmt_compact(_disp_money(r.private_eq_per_h, P_now), 8).strip()
        row[9] = float(r.private_roe_q)
        row[10] = float(r.private_inv_cov)
        row[11] = _fmt_compact(_disp_money(sim.state.get("vat_receipts_total", 0.0) / denom, P_now), 5).strip()
        row[12] = _fmt_compact(_disp_money(sim.state.get("income_tax_total", 0.0) / denom, P_now), 5).strip()
        row[13] = _fmt_compact(_disp_money(sim.state.get("corp_tax_total", 0.0) / denom, P_now), 5).strip()
        corp_tax_rate_pct = int(round(100.0 * float(sim.state.get("corp_tax_rate_eff", sim.params.get("corporate_tax_rate", 0.0)))))
        row[14] = f"{corp_tax_rate_pct:02d}%"
        row[15] = _fmt_compact(_disp_money(sim.state.get("vat_credit_total", 0.0) / denom, P_now), 5).strip()
        row[16] = _fmt_compact(_disp_money(sim.nodes["GOV"].get("deposits", 0.0) / denom, P_now), 8).strip()
        row[17] = _fmt_compact(_disp_money(sim.nodes["FUND"].get("deposits", 0.0) / denom, P_now), 8).strip()

        # Trust balance diagnostics:
        # trust_balance_total = FUND deposits + market value of FUND equity claims - FUND loans.
        def _fund_share_frac(issuer: str, key: str) -> float:
            shares_out = float(sim.nodes[issuer].get("shares_outstanding", 0.0))
            if shares_out <= 0.0:
                return 0.0
            frac = float(sim.nodes["FUND"].get(key, 0.0)) / shares_out
            return max(0.0, min(1.0, frac))

        fa_equity_proxy = max(
            0.0,
            float(sim.nodes["FA"].get("deposits", 0.0))
            + float(sim.nodes["FA"].get("K", 0.0)) * P_now
            - float(sim.nodes["FA"].get("loans", 0.0)),
        )
        fh_equity_proxy = max(
            0.0,
            float(sim.nodes["FH"].get("deposits", 0.0))
            + float(sim.nodes["FH"].get("K", 0.0)) * P_now
            - float(sim.nodes["FH"].get("loans", 0.0)),
        )
        bank_equity_proxy = max(0.0, float(sim.nodes["BANK"].get("equity", 0.0)))

        fund_equity_claim_total = (
            _fund_share_frac("FA", "shares_FA") * fa_equity_proxy
            + _fund_share_frac("FH", "shares_FH") * fh_equity_proxy
            + _fund_share_frac("BANK", "shares_BANK") * bank_equity_proxy
        )

        trust_balance_total = (
            float(sim.nodes["FUND"].get("deposits", 0.0))
            + float(fund_equity_claim_total)
            - float(sim.nodes["FUND"].get("loans", 0.0))
        )
        trust_claim_per_h = trust_balance_total / denom

        row[18] = _fmt_compact(_disp_money(trust_claim_per_h, P_now), 8).strip()
        row[19] = _fmt_compact(_disp_money(trust_balance_total, P_now), 8).strip()
        row[20] = " ".join([
            _fmt_compact(_disp_money(r.ubi_per_h, P_now), 5).strip(),
            _fmt_compact(_disp_money(r.ubi_from_fund_dep_per_h, P_now), 5).strip(),
            _fmt_compact(_disp_money(r.ubi_from_gov_dep_per_h, P_now), 5).strip(),
            _fmt_compact(_disp_money(r.ubi_issued_per_h, P_now), 5).strip(),
        ])
        row[21] = _fmt_compact(_disp_money(r.wages_total, P_now), 8, 1).strip()
        row[22] = _fmt_compact(_disp_money(sim.state.get("capex_total", 0.0) / denom, P_now), 8).strip()
        row[23] = _fmt_compact(_disp_money(avg_inc_nom, P_now), 8).strip()
        row[24] = _fmt_compact(r.real_avg_income, 8).strip()
        # Consumption (per household)
        cons_per_h = float(r.total_consumption) / denom
        rcons_per_h = cons_per_h / P_now

        row[25] = _fmt_compact(_disp_money(cons_per_h, P_now), 8, 1).strip()
        row[26] = _fmt_compact(rcons_per_h, 8, 1).strip()

        # Household loans (per household)
        hh_loan_per_h = float(sim.nodes["HH"].get("loans", 0.0)) / denom
        row[27] = _fmt_compact(_disp_money(hh_loan_per_h, P_now), 8).strip()

        # Equity capture
        row[28] = float(r.trust_equity_pct)

        # --- BANK state (per household) ---
        bank_dep_liab_per_h = float(sim.nodes["BANK"].get("deposit_liab", 0.0)) / denom
        bank_loan_assets_per_h = float(sim.nodes["BANK"].get("loan_assets", 0.0)) / denom
        bank_reserves_per_h = float(sim.nodes["BANK"].get("reserves", 0.0)) / denom
        bank_equity_per_h = float(sim.nodes["BANK"].get("equity", 0.0)) / denom

        # SFC identity checks (totals, not per-household scaling)
        dep_identity_gap = float(sim.nodes["BANK"].get("deposit_liab", 0.0)) - float(sim._sum_deposits_all())
        loan_identity_gap = float(sim.nodes["BANK"].get("loan_assets", 0.0)) - float(sim._sum_loans_borrowers())
        bank_balance_gap = (
            float(sim.nodes["BANK"].get("loan_assets", 0.0))
            + float(sim.nodes["BANK"].get("reserves", 0.0))
            - float(sim.nodes["BANK"].get("deposit_liab", 0.0))
            - float(sim.nodes["BANK"].get("equity", 0.0))
        )
        max_abs_trdebt = max(max_abs_trdebt, abs(float(r.trust_debt)))
        max_abs_dep_identity_gap = max(max_abs_dep_identity_gap, abs(dep_identity_gap))
        max_abs_loan_identity_gap = max(max_abs_loan_identity_gap, abs(loan_identity_gap))
        max_abs_bank_balance_gap = max(max_abs_bank_balance_gap, abs(bank_balance_gap))

        row[29] = _fmt_compact(_disp_money(bank_reserves_per_h, P_now), 8).strip()
        row[30] = _fmt_compact(_disp_money(bank_dep_liab_per_h, P_now), 8).strip()
        row[31] = _fmt_compact(_disp_money(bank_loan_assets_per_h, P_now), 8).strip()

    print(dashboard)
    print()
    if not trust_ever_activated:
        print("ERROR: Trust was never activated during this run.")
        print()
    assert max_abs_trdebt <= zero_tol, (
        f"TrDebt non-zero beyond tolerance: max_abs={max_abs_trdebt:.6g}, tol={zero_tol:.1e}"
    )
    assert max_abs_dep_identity_gap <= zero_tol, (
        f"DepIdentityGap non-zero beyond tolerance: max_abs={max_abs_dep_identity_gap:.6g}, tol={zero_tol:.1e}"
    )
    assert max_abs_loan_identity_gap <= zero_tol, (
        f"LoanIdentityGap non-zero beyond tolerance: max_abs={max_abs_loan_identity_gap:.6g}, tol={zero_tol:.1e}"
    )
    assert max_abs_bank_balance_gap <= zero_tol, (
        f"BankBalanceGap non-zero beyond tolerance: max_abs={max_abs_bank_balance_gap:.6g}, tol={zero_tol:.1e}"
    )
    print(
        "Balance checks passed: "
        f"max |TrDebt|={max_abs_trdebt:.3g}, "
        f"max |DepIdentityGap|={max_abs_dep_identity_gap:.3g}, "
        f"max |LoanIdentityGap|={max_abs_loan_identity_gap:.3g}, "
        f"max |BankBalanceGap|={max_abs_bank_balance_gap:.3g} "
        f"(tol={zero_tol:.1e})"
    )
    print()

    # -------------------------------------------------
    # Dashboard legend: one line per column
    # -------------------------------------------------
    legend = [
        ("Q",     "Quarter index (two-digit display)") ,
        ("Auto",  "Overall automation share A(t) in [0,1]") ,
        ("Info",  "Information automation component A_info (capped)") ,
        ("Phys",  "Physical automation component A_phys (capped)") ,
        ("P",     "Price level index P(t)") ,
        ("Gm",   "Market-income Gini (wages + household dividends, pre-tax/pre-transfer)") ,
        ("Gi",   "Disposable-income Gini (after tax/transfers/interest)") ,
        ("Gw",   "Net-wealth Gini proxy (deposits + estimated equity claims - loans)") ,
        ("pEq$",  "Private-equity proxy per household (HH-held claims on FA/FH/BANK; display-currency)") ,
        ("pROE",  "Quarterly private payout yield proxy = private payouts / lagged private equity") ,
        ("pInv",  "Private retained earnings coverage of CAPEX = private_retained / CAPEX") ,
        ("VAT",   "VAT receipts per household this quarter (display-currency flow)") ,
        ("IncTx", "Income tax receipts per household this quarter (display-currency flow)") ,
        ("CoTx",  "Corporate tax receipts per household this quarter (display-currency flow; firms + bank)") ,
        ("CoR%",  "Effective corporate tax rate this quarter (percent, two digits)") ,
        ("VATCr", "VAT credit granted per household this quarter (display-currency transfer; gross)") ,
        ("GOVdep","Government deposits per household (display-currency stock, end of quarter)") ,
        ("FUNDdep","Trust/Fund deposits per household (display-currency stock, end of quarter)") ,
        ("TrClm", "Per-household beneficial claim on trust balance (display-currency stock)") ,
        ("TrBal$", "Total trust balance (FUND deposits + FUND equity claims - FUND loans; display-currency)") ,
        ("UBI,F,G,I",  "UBI bundle (per-household): UBI UBI_F UBI_G UBI_I, one-space separated") ,
        ("Wages", "Total wages paid economy-wide this quarter (display-currency total)") ,
        ("CAPEX", "Firm capital expenditures per household this quarter (display-currency flow; lagged-from-retained)") ,
        ("AvgInc","Average income proxy per household = wages/HH + UBI (display-currency; excludes dividends)") ,
        ("RAvgInc","Real AvgInc proxy = AvgInc / P") ,
        ("Cons",  "Consumption per household at producer prices (display-currency; tax-exclusive)") ,
        ("RCons", "Real consumption per household = Cons / P") ,
        ("HHLoan", "Household loans per household (mortgage + revolving; display-currency stock)") ,
        ("EqCap",  "Trust equity ownership fraction (avg across FA/FH/BANK)") ,
        ("Resv",   "BANK reserves per household (display-currency asset created by issuance)") ,
        ("DepLb",  "Bank deposit liabilities per household (display-currency stock)") ,
        ("LoanA",  "Bank loan assets per household (display-currency stock)") ,
    ]

    print("Dashboard legend:")
    print("  Note   Monetary columns follow parameters['dashboard_value_mode'] "
          "('nominal' or 'price_normalized' as base-period dollars).")
    for k, desc in legend:
        print(f"  {k:<6} {desc}")

    print()
    dist_after = _household_vectors_snapshot()
    _print_before_after_distributions(dist_before, dist_after)

    print()
