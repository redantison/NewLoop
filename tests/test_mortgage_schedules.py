import copy
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.mortgage import balance_from_orig_principal, payment_from_orig_principal, remaining_term, scheduled_payment_components
from newloop.results import (
    _baseline_calibration_regime_cfg,
    _prepare_startup_sim,
    _run_baseline_calibration,
    _startup_solver_snapshot,
    run_simulation,
)


def make_cfg():
    cfg = copy.deepcopy(get_default_config())
    cfg["parameters"]["hard_assert_sfc"] = True
    return cfg


class MortgageScheduleTests(unittest.TestCase):
    def test_fixed_rate_amortization_components_match_balance_and_payment(self):
        principal = np.asarray([1200.0], dtype=float)
        rate_q = np.asarray([0.01], dtype=float)
        term_q = np.asarray([60.0], dtype=float)
        age_q = np.asarray([0.0], dtype=float)

        payment_q = payment_from_orig_principal(principal, rate_q, term_q)
        due_q, interest_q, principal_q = scheduled_payment_components(
            principal,
            rate_q,
            payment_q,
            remaining_term(term_q, age_q),
        )

        self.assertAlmostEqual(float(due_q[0]), float(payment_q[0]), places=9)
        self.assertAlmostEqual(float(interest_q[0]), 12.0, places=9)
        self.assertAlmostEqual(float(principal_q[0]), float(payment_q[0]) - 12.0, places=9)

    def test_fixed_rate_amortization_clears_balance_by_term(self):
        balance = np.asarray([1200.0], dtype=float)
        rate_q = np.asarray([0.01], dtype=float)
        term_q = np.asarray([60.0], dtype=float)
        payment_q = payment_from_orig_principal(balance, rate_q, term_q)

        for step in range(60):
            rem_q = np.asarray([60.0 - float(step)], dtype=float)
            due_q, interest_q, principal_q = scheduled_payment_components(
                balance,
                rate_q,
                payment_q,
                rem_q,
            )
            self.assertGreaterEqual(float(due_q[0]), float(interest_q[0]))
            balance = np.maximum(0.0, balance - principal_q)

        self.assertLess(float(balance[0]), 1e-6)

    def test_final_period_balloon_clears_residual_balance(self):
        balance = np.asarray([500.0], dtype=float)
        rate_q = np.asarray([0.01], dtype=float)
        payment_q = np.asarray([0.5], dtype=float)
        rem_q = np.asarray([1.0], dtype=float)

        due_q, interest_q, principal_q = scheduled_payment_components(
            balance,
            rate_q,
            payment_q,
            rem_q,
        )

        self.assertAlmostEqual(float(interest_q[0]), 5.0, places=9)
        self.assertAlmostEqual(float(principal_q[0]), 500.0, places=9)
        self.assertAlmostEqual(float(due_q[0]), 505.0, places=9)

    def test_population_initializes_real_mortgage_schedule_state(self):
        sim = NewLoop(make_cfg())
        hh = sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None

        active = np.asarray(hh.mortgage_loans, dtype=float) > 1e-12
        self.assertTrue(bool(np.any(active)))
        self.assertTrue(np.all(np.asarray(hh.mort_rate_q, dtype=float)[active] > 0.0))
        self.assertTrue(np.all(np.asarray(hh.mort_term_q, dtype=float)[active] == 60.0))
        self.assertTrue(np.all(np.asarray(hh.mort_payment_sched_q, dtype=float)[active] > 0.0))
        self.assertTrue(
            np.allclose(
                np.asarray(hh.mortgage_loans, dtype=float)[active],
                balance_from_orig_principal(
                    np.asarray(hh.mort_orig_principal, dtype=float)[active],
                    np.asarray(hh.mort_rate_q, dtype=float)[active],
                    np.asarray(hh.mort_term_q, dtype=float)[active],
                    np.asarray(hh.mort_age_q, dtype=float)[active],
                ),
                rtol=1e-7,
                atol=1e-9,
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(hh.mort_payment_sched_q, dtype=float)[active],
                payment_from_orig_principal(
                    np.asarray(hh.mort_orig_principal, dtype=float)[active],
                    np.asarray(hh.mort_rate_q, dtype=float)[active],
                    np.asarray(hh.mort_term_q, dtype=float)[active],
                ),
                rtol=1e-7,
                atol=1e-9,
            )
        )

    def test_startup_disposable_income_tail_is_not_pathological(self):
        cfg = make_cfg()
        eff, _ = _run_baseline_calibration(cfg)
        sim = NewLoop(copy.deepcopy(eff))
        _prepare_startup_sim(sim)
        snap = _startup_solver_snapshot(sim)
        self.assertIsNotNone(snap)
        assert snap is not None

        y = np.asarray(snap["disposable_income_i"], dtype=float)
        self.assertGreater(float(np.min(y)), -10000.0)

    def test_baseline_calibration_enabled_does_not_crash_run(self):
        cfg = make_cfg()
        cfg["parameters"]["baseline_calibration_enabled"] = True

        run = run_simulation(12, cfg)

        self.assertTrue(bool(run.rows))
        self.assertIsNotNone(run.baseline_calibration)
        self.assertFalse(bool((run.baseline_calibration or {}).get("skipped_reason", "")))

    def test_baseline_calibration_regime_preserves_income_support_and_mortgage_relief(self):
        cfg = make_cfg()
        cfg["parameters"]["disable_income_support"] = False
        cfg["parameters"]["disable_mortgage_relief"] = False
        cfg["parameters"]["dividend_payout_rate_firms"] = 0.75
        cfg["parameters"]["gov_tax_rebate_rate"] = 0.25

        regime = _baseline_calibration_regime_cfg(cfg)
        params = regime["parameters"]

        self.assertFalse(bool(params.get("disable_income_support", True)))
        self.assertFalse(bool(params.get("disable_mortgage_relief", True)))
        self.assertTrue(bool(params.get("automation_disabled", False)))
        self.assertTrue(bool(params.get("disable_trust", False)))
        self.assertEqual(float(params.get("dividend_payout_rate_firms", -1.0)), 0.0)
        self.assertEqual(float(params.get("gov_tax_rebate_rate", -1.0)), 0.0)

    def test_baseline_calibration_skip_reverts_to_uncalibrated_config(self):
        cfg = make_cfg()
        cfg["parameters"]["baseline_calibration_enabled"] = True
        original = copy.deepcopy(cfg)

        with mock.patch("newloop.results.NewLoop.step", side_effect=ValueError("boom")):
            eff, report = _run_baseline_calibration(cfg)

        self.assertEqual(eff, original)
        self.assertEqual(str((report or {}).get("skipped_reason", "")), "infeasible_hidden_baseline_regime")

    def test_turnover_originates_new_age_zero_fixed_rate_mortgages(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["mortgage_turnover_enabled"] = True
        params["mortgage_turnover_target_payment_floor_share"] = 1.0
        params["mortgage_turnover_dti_cap"] = 5.0
        params["mortgage_turnover_income_mult_cap"] = 20.0
        params["mortgage_turnover_min_wage_q"] = 0.0

        sim = NewLoop(cfg)
        hh = sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None

        had_no_mortgage = np.asarray(hh.mortgage_loans, dtype=float) <= 1e-12
        sim.step()
        sim.step()

        new_mask = (np.asarray(hh.mortgage_loans, dtype=float) > 1e-12) & (np.asarray(hh.mort_t0, dtype=int) < 0)
        self.assertGreater(float(sim.state.get("mort_turnover_total", 0.0)), 0.0)
        self.assertTrue(bool(np.any(new_mask)))
        self.assertTrue(np.all(np.asarray(hh.mort_age_q, dtype=float)[new_mask] == 0.0))
        self.assertTrue(np.all(np.asarray(hh.mort_term_q, dtype=float)[new_mask] == 60.0))
        self.assertTrue(
            np.allclose(
                np.asarray(hh.mort_rate_q, dtype=float)[new_mask],
                float(cfg["parameters"]["mortgage_fixed_rate_q"]),
            )
        )
        self.assertTrue(np.all(np.asarray(hh.mort_payment_sched_q, dtype=float)[new_mask] > 0.0))
        self.assertTrue(
            np.allclose(
                np.asarray(hh.mort_payment_sched_q, dtype=float)[new_mask],
                payment_from_orig_principal(
                    np.asarray(hh.mortgage_loans, dtype=float)[new_mask],
                    float(cfg["parameters"]["mortgage_fixed_rate_q"]),
                    60.0,
                ),
                rtol=1e-7,
                atol=1e-9,
            )
        )

    def test_newly_issued_mortgages_start_index_at_issue_date(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["mortgage_turnover_enabled"] = True
        params["mortgage_turnover_target_payment_floor_share"] = 1.0
        params["mortgage_turnover_dti_cap"] = 5.0
        params["mortgage_turnover_income_mult_cap"] = 20.0
        params["mortgage_turnover_min_wage_q"] = 0.0
        params["mort_index_weight_w"] = 1.0
        params["mort_index_ewma_lambda"] = 0.5

        sim = NewLoop(cfg)
        hh = sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None

        pending_new = np.zeros(hh.n, dtype=bool)
        for _ in range(3):
            sim.step()
            pending_new = (np.asarray(hh.mortgage_loans, dtype=float) > 1e-12) & (np.asarray(hh.mort_t0, dtype=int) < 0)
            if np.any(pending_new):
                break

        self.assertTrue(bool(np.any(pending_new)))

        sim.state["price_level"] = 2.0
        sim.state["mort_price_series_prev"] = 1.0
        sim.state["mort_income_series_prev"] = 1.0
        rL = float(sim.state.get("policy_rate_q", sim.params.get("loan_rate_per_quarter", 0.0)))

        sim._ensure_mortgage_index_anchors(2.0, 1.0, rL)
        terms = sim._compute_mortgage_index_terms(
            mort=np.asarray(hh.mortgage_loans, dtype=float),
            rL=rL,
            wages_total=1.0,
            div_house_total=0.0,
            uis_per_h=0.0,
            commit_state=False,
        )

        mort_index_i = np.asarray(terms["mort_index_i"], dtype=float)
        mort_pay_req_i = np.asarray(terms["mort_pay_req_i"], dtype=float)
        mort_pay_ctr_i = np.asarray(terms["mort_pay_ctr_i"], dtype=float)
        mort_dln_sm_i = np.asarray(terms["mort_dln_sm_i"], dtype=float)

        self.assertTrue(np.allclose(mort_index_i[pending_new], 1.0, rtol=0.0, atol=1e-9))
        self.assertTrue(np.allclose(mort_pay_req_i[pending_new], mort_pay_ctr_i[pending_new], rtol=0.0, atol=1e-9))
        self.assertTrue(np.allclose(mort_dln_sm_i[pending_new], 0.0, rtol=0.0, atol=1e-9))

    def test_matured_mortgages_are_cleared_from_active_pool(self):
        sim = NewLoop(make_cfg())
        hh = sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None

        idx = int(np.argmax(np.asarray(hh.mortgage_loans, dtype=float)))
        hh.mortgage_loans[idx] = 25.0
        hh.mort_term_q[idx] = 60.0
        hh.mort_age_q[idx] = 60.0
        hh.mort_rate_q[idx] = 0.01
        hh.mort_payment_sched_q[idx] = 25.25
        hh.mort_orig_principal[idx] = 1000.0

        sim._refresh_mortgage_contract_state()

        self.assertAlmostEqual(float(hh.mortgage_loans[idx]), 0.0, places=9)
        self.assertAlmostEqual(float(hh.mort_age_q[idx]), 0.0, places=9)
        self.assertAlmostEqual(float(hh.mort_term_q[idx]), 0.0, places=9)
        self.assertAlmostEqual(float(hh.mort_payment_sched_q[idx]), 0.0, places=9)

    def test_revolving_credit_is_capped(self):
        cfg = make_cfg()
        run = run_simulation(120, cfg)
        hh = run.sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None

        rev = np.asarray(hh.revolving_loans, dtype=float)
        wages_q = np.asarray(hh.wages0_q, dtype=float)
        cap_mult = float(cfg["parameters"]["population_config"]["revolving_cap_income_mult"])
        rev_cap = cap_mult * np.maximum(0.0, 4.0 * wages_q)
        self.assertTrue(np.all(rev <= (rev_cap + 1e-6)))


if __name__ == "__main__":
    unittest.main()
