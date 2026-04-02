import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.config import get_default_config
from newloop.tax_policy import make_tax_policy


def make_params():
    return copy.deepcopy(get_default_config()["parameters"])


class TaxPolicyTests(unittest.TestCase):
    def test_default_tax_policy_mode_tracks_newloop_regime(self):
        params = make_params()
        policy = make_tax_policy(params)
        self.assertEqual(policy.mode_name(), "CURRENT")

    def test_old_loop_tax_policy_mode_tracks_oldloop_regime(self):
        params = make_params()
        params["economic_regime"] = "OldLoop"
        policy = make_tax_policy(params)
        self.assertEqual(policy.mode_name(), "OLD_LOOP")

    def test_old_loop_household_tax_applies_full_mortgage_interest_deduction(self):
        params = make_params()
        params["economic_regime"] = "OldLoop"
        params["old_loop_tax_rate_lower"] = 0.15
        params["old_loop_tax_rate_upper"] = 0.28
        policy = make_tax_policy(params)

        state = {
            "old_loop_tax_threshold_lower_real": 100.0,
            "old_loop_tax_threshold_upper_real": 200.0,
        }
        result = policy.compute_household_taxes(
            wages_i=np.asarray([150.0, 250.0], dtype=float),
            div_i=np.asarray([0.0, 0.0], dtype=float),
            mort_interest_due_i=np.asarray([25.0, 25.0], dtype=float),
            support_per_h=0.0,
            price_level=1.0,
            state=state,
        )

        self.assertTrue(
            np.allclose(
                np.asarray(result.mortgage_interest_deduction_i, dtype=float),
                np.asarray([25.0, 25.0], dtype=float),
                rtol=1e-9,
                atol=1e-9,
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(result.taxable_income_i, dtype=float),
                np.asarray([125.0, 225.0], dtype=float),
                rtol=1e-9,
                atol=1e-9,
            )
        )
        self.assertAlmostEqual(float(result.income_tax_i[0]), 3.75, places=9)
        self.assertAlmostEqual(float(result.income_tax_i[1]), 22.0, places=9)

    def test_old_loop_corporate_tax_uses_fixed_rate(self):
        params = make_params()
        params["economic_regime"] = "OldLoop"
        params["old_loop_corporate_tax_rate"] = 0.35
        params["corporate_tax_depr_rate_q"] = 0.0
        policy = make_tax_policy(params)

        result = policy.compute_corporate_taxes(
            p_fa_pre_tax=100.0,
            p_fh_pre_tax=60.0,
            bank_profit_pre_tax=40.0,
            price_level=1.0,
            wages_total=0.0,
            state={},
            fa_capital_real=0.0,
            fh_capital_real=0.0,
        )

        self.assertAlmostEqual(float(result.corp_tax_rate), 0.35, places=9)
        self.assertAlmostEqual(float(result.corp_tax_fa), 35.0, places=9)
        self.assertAlmostEqual(float(result.corp_tax_fh), 21.0, places=9)
        self.assertAlmostEqual(float(result.corp_tax_bk), 14.0, places=9)


if __name__ == "__main__":
    unittest.main()
