import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.mortgage import balance_from_orig_principal, payment_from_orig_principal, remaining_term, scheduled_payment_components
from newloop.results import _prepare_startup_sim, _run_baseline_calibration, _startup_solver_snapshot, run_simulation


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

    def test_turnover_originates_new_age_zero_fixed_rate_mortgages(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["mortgage_turnover_enabled"] = True
        params["mortgage_turnover_replace_share"] = 1.0
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
        self.assertTrue(bool(np.any(new_mask & had_no_mortgage)))
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
