import statistics
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.population import PopulationConfig, generate_population


class PopulationGenerationTests(unittest.TestCase):
    def test_liquid_buffer_rule_increases_deposits_with_wages(self):
        cfg = PopulationConfig(
            n_families=5000,
            seed=7919,
            employment_rate=1.0,
            deposit_generation_mode="liquid_buffer_months",
        )
        pop = generate_population(cfg)

        pairs = sorted(
            zip(pop.wages_q, pop.deposits, pop.base_real_cons_q, pop.liquid_buffer_months_target),
            key=lambda item: item[0],
        )
        n = len(pairs)
        bins = [pairs[int(i * n / 5): int((i + 1) * n / 5)] for i in range(5)]
        deposit_medians = [statistics.median(dep for _, dep, _, _ in bucket) for bucket in bins]
        base_real_medians = [statistics.median(base for _, _, base, _ in bucket) for bucket in bins]
        month_medians = [statistics.median(months for _, _, _, months in bucket) for bucket in bins]
        realized_buffer_months = [
            statistics.median((dep / max(base / 3.0, 1e-9)) for _, dep, base, _ in bucket)
            for bucket in bins
        ]

        for left, right in zip(deposit_medians, deposit_medians[1:]):
            self.assertLess(left, right)
        for left, right in zip(base_real_medians, base_real_medians[1:]):
            self.assertLessEqual(left, right)
        for left, right in zip(month_medians, month_medians[1:]):
            self.assertLessEqual(left, right)

        self.assertGreaterEqual(base_real_medians[0], 400.0)
        self.assertLessEqual(base_real_medians[0], 500.0)
        self.assertGreaterEqual(base_real_medians[-1], 625.0)
        self.assertLessEqual(base_real_medians[-1], 725.0)
        self.assertGreaterEqual(realized_buffer_months[0], 0.6)
        self.assertLessEqual(realized_buffer_months[0], 3.0)
        self.assertGreaterEqual(realized_buffer_months[-1], 7.0)
        self.assertLessEqual(realized_buffer_months[-1], 14.0)
        self.assertGreaterEqual(month_medians[0], 0.9)
        self.assertLessEqual(month_medians[0], 3.0)
        self.assertGreaterEqual(month_medians[-1], 7.0)
        self.assertLessEqual(month_medians[-1], 14.0)

    def test_liquid_buffer_targets_are_smoothed_between_percentile_anchors(self):
        cfg = PopulationConfig(
            n_families=5000,
            seed=7919,
            employment_rate=1.0,
            deposit_generation_mode="liquid_buffer_months",
        )
        pop = generate_population(cfg)

        rounded_targets = {round(float(x), 3) for x in pop.liquid_buffer_months_target}
        _, counts = np.unique(np.round(pop.liquid_buffer_months_target, 3), return_counts=True)
        self.assertGreater(len(rounded_targets), 100)
        self.assertIn(1.5, rounded_targets)
        self.assertIn(12.0, rounded_targets)
        self.assertLess(int(counts.max()), 100)

    def test_renters_receive_explicit_rent_burden(self):
        cfg = PopulationConfig(
            n_families=5000,
            seed=7919,
            employment_rate=1.0,
            deposit_generation_mode="liquid_buffer_months",
        )
        pop = generate_population(cfg)

        mort = np.asarray(pop.mortgage_loans, dtype=float)
        housing = np.asarray(pop.housing_values, dtype=float)
        rent = np.asarray(pop.renter_rent_q, dtype=float)
        pay = np.asarray(pop.mortgage_payment_sched_q, dtype=float)
        wages = np.asarray(pop.wages_q, dtype=float)

        renters = (mort <= 1e-12) & (housing <= 1e-12)
        mortgagors = mort > 1e-12
        self.assertTrue(bool(np.any(renters)))
        self.assertGreater(float(np.median(rent[renters])), 0.0)
        self.assertTrue(bool(np.all(rent[~renters] <= 1e-12)))
        low_end_cut = float(np.percentile(wages[mortgagors], 40.0))
        low_end_mort_pay = pay[mortgagors & (wages <= low_end_cut)]
        self.assertGreater(low_end_mort_pay.size, 0)
        renter_med = float(np.median(rent[renters]))
        low_end_med = float(np.median(low_end_mort_pay))
        self.assertGreater(renter_med, 0.5 * low_end_med)
        self.assertLess(renter_med, 1.5 * low_end_med)


if __name__ == "__main__":
    unittest.main()
