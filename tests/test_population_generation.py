import statistics
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newloop"))

from population import PopulationConfig, generate_population


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
            zip(pop.wages_q, pop.deposits, pop.liquid_buffer_months_target),
            key=lambda item: item[0],
        )
        n = len(pairs)
        bins = [pairs[int(i * n / 5): int((i + 1) * n / 5)] for i in range(5)]
        medians = [statistics.median(dep for _, dep, _ in bucket) for bucket in bins]
        month_medians = [statistics.median(months for _, _, months in bucket) for bucket in bins]

        for left, right in zip(medians, medians[1:]):
            self.assertLess(left, right)
        for left, right in zip(month_medians, month_medians[1:]):
            self.assertLessEqual(left, right)

        monthly_base = cfg.base_real_cons_q / 3.0
        bottom_months = medians[0] / monthly_base
        top_months = medians[-1] / monthly_base
        self.assertGreaterEqual(bottom_months, 1.0)
        self.assertLessEqual(bottom_months, 3.0)
        self.assertGreaterEqual(top_months, 7.0)
        self.assertLessEqual(top_months, 14.0)
        self.assertGreaterEqual(month_medians[0], 1.0)
        self.assertLessEqual(month_medians[0], 3.0)
        self.assertGreaterEqual(month_medians[-1], 7.0)
        self.assertLessEqual(month_medians[-1], 14.0)


if __name__ == "__main__":
    unittest.main()
