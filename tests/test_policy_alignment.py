import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newloop"))

from config import get_default_config
from engine import NewLoop


def make_cfg():
    cfg = copy.deepcopy(get_default_config())
    cfg["parameters"]["hard_assert_sfc"] = True
    return cfg


class PolicyAlignmentTests(unittest.TestCase):
    def test_default_run_stays_stock_flow_consistent(self):
        sim = NewLoop(make_cfg())
        for _ in range(12):
            sim.step()

        max_deposit_gap = max(abs(float(r["deposit_gap"])) for r in sim.inv_history)
        max_loan_gap = max(abs(float(r["loan_gap"])) for r in sim.inv_history)
        self.assertLessEqual(max_deposit_gap, 1e-6)
        self.assertLessEqual(max_loan_gap, 1e-6)

    def test_trust_launch_loan_changes_initial_equity_block(self):
        cfg_low = make_cfg()
        cfg_high = make_cfg()

        for cfg, loan in ((cfg_low, 6000.0), (cfg_high, 24000.0)):
            params = cfg["parameters"]
            params["trust_trigger_dti"] = 0.0
            params["trust_launch_loan"] = loan
            params["gov_tax_rebate_rate"] = 0.0

        sim_low = NewLoop(cfg_low)
        sim_high = NewLoop(cfg_high)
        for _ in range(3):
            sim_low.step()
            sim_high.step()

        self.assertTrue(sim_low.state["trust_active"])
        self.assertTrue(sim_high.state["trust_active"])

        low_shares = sum(sim_low.nodes["FUND"].get(key, 0.0) for key in ("shares_FA", "shares_FH", "shares_BANK"))
        high_shares = sum(sim_high.nodes["FUND"].get(key, 0.0) for key in ("shares_FA", "shares_FH", "shares_BANK"))
        self.assertGreater(high_shares, low_shares)

    def test_corporate_tax_depreciation_allowance_reduces_tax(self):
        cfg_no_depr = make_cfg()
        cfg_with_depr = make_cfg()

        for cfg, depr_rate in ((cfg_no_depr, 0.0), (cfg_with_depr, 0.025)):
            params = cfg["parameters"]
            params["trust_trigger_dti"] = 1.0
            params["gov_tax_rebate_rate"] = 0.0
            params["corporate_tax_depr_rate_q"] = depr_rate
            cfg["nodes"]["FA"]["stocks"]["K"] = 2000.0
            cfg["nodes"]["FH"]["stocks"]["K"] = 2000.0

        sim_no_depr = NewLoop(cfg_no_depr)
        sim_with_depr = NewLoop(cfg_with_depr)
        sim_no_depr.step()
        sim_with_depr.step()

        tax_no_depr = float(sim_no_depr.state.get("corp_tax_total", 0.0))
        tax_with_depr = float(sim_with_depr.state.get("corp_tax_total", 0.0))
        self.assertGreater(tax_no_depr, 0.0)
        self.assertLess(tax_with_depr, tax_no_depr)

    def test_fund_residual_share_supports_partial_sweep(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 1.0
        params["loan_rate_per_quarter"] = 0.0
        params["vat_rate"] = 0.0
        params["income_tax_rate"] = 0.0
        params["corporate_tax_rate"] = 0.0
        params["corporate_tax_dynamic_with_wages"] = False
        params["gov_tax_rebate_rate"] = 0.0
        params["fund_residual_to_gov_share"] = 0.25

        sim = NewLoop(cfg)
        sim.nodes["FUND"].set("deposits", 100.0)
        sim.nodes["BANK"].add("deposit_liab", 100.0)
        sim.nodes["BANK"].add("reserves", 100.0)
        sim.step()

        self.assertAlmostEqual(sim.nodes["GOV"].get("deposits", 0.0), 25.0, places=6)
        self.assertAlmostEqual(sim.nodes["FUND"].get("deposits", 0.0), 75.0, places=6)

    def test_gov_rebate_keeps_one_year_buffer_of_trailing_obligations(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 1.0
        params["loan_rate_per_quarter"] = 0.0
        params["vat_rate"] = 0.0
        params["income_tax_rate"] = 0.0
        params["corporate_tax_rate"] = 0.0
        params["corporate_tax_dynamic_with_wages"] = False
        params["gov_tax_rebate_rate"] = 1.0
        params["gov_rebate_buffer_quarters"] = 4
        params["gov_rebate_start_delay_quarters"] = 0
        params["gov_rebate_ramp_quarters"] = 0

        sim = NewLoop(cfg)
        sim.gov_obligation_history = [100.0, 100.0, 100.0, 100.0]
        sim.nodes["GOV"].set("deposits", 500.0)
        sim.nodes["BANK"].add("deposit_liab", 500.0)
        sim.nodes["BANK"].add("reserves", 500.0)
        sim.step()

        self.assertAlmostEqual(sim.state.get("tax_rebate_total", 0.0), 100.0, places=6)
        self.assertAlmostEqual(sim.nodes["GOV"].get("deposits", 0.0), 400.0, places=6)

    def test_gov_rebate_waits_for_full_buffer_history(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 1.0
        params["loan_rate_per_quarter"] = 0.0
        params["vat_rate"] = 0.0
        params["income_tax_rate"] = 0.0
        params["corporate_tax_rate"] = 0.0
        params["corporate_tax_dynamic_with_wages"] = False
        params["gov_tax_rebate_rate"] = 1.0
        params["gov_rebate_buffer_quarters"] = 4
        params["gov_rebate_start_delay_quarters"] = 0
        params["gov_rebate_ramp_quarters"] = 0

        sim = NewLoop(cfg)
        sim.gov_obligation_history = [100.0, 100.0, 100.0]
        sim.nodes["GOV"].set("deposits", 500.0)
        sim.nodes["BANK"].add("deposit_liab", 500.0)
        sim.nodes["BANK"].add("reserves", 500.0)
        sim.step()

        self.assertAlmostEqual(sim.state.get("tax_rebate_total", 0.0), 0.0, places=6)

    def test_gov_rebate_waits_one_year_then_ramps(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 1.0
        params["loan_rate_per_quarter"] = 0.0
        params["vat_rate"] = 0.0
        params["income_tax_rate"] = 0.0
        params["corporate_tax_rate"] = 0.0
        params["corporate_tax_dynamic_with_wages"] = False
        params["gov_tax_rebate_rate"] = 1.0
        params["gov_rebate_buffer_quarters"] = 4
        params["gov_rebate_start_delay_quarters"] = 4
        params["gov_rebate_ramp_quarters"] = 20

        sim = NewLoop(cfg)
        sim.gov_obligation_history = [100.0, 100.0, 100.0, 100.0]
        sim.state["t"] = 4
        sim.nodes["GOV"].set("deposits", 500.0)
        sim.nodes["BANK"].add("deposit_liab", 500.0)
        sim.nodes["BANK"].add("reserves", 500.0)
        sim.step()

        self.assertAlmostEqual(sim.state.get("tax_rebate_total", 0.0), 5.0, places=6)

    def test_precautionary_buffer_rule_reduces_spending_when_below_target(self):
        cfg_loose = make_cfg()
        cfg_tight = make_cfg()

        for cfg, conserve_rate in ((cfg_loose, 0.0), (cfg_tight, 0.5)):
            params = cfg["parameters"]
            params["trust_trigger_dti"] = 1.0
            params["trust_launch_loan"] = 0.0
            params["gov_tax_rebate_rate"] = 0.0
            params["hh_buffer_spend_excess_rate_q"] = 0.0
            params["hh_buffer_shortfall_conserve_rate_q"] = conserve_rate
            params["population_config"]["seed"] = 7919

        sim_loose = NewLoop(cfg_loose)
        sim_tight = NewLoop(cfg_tight)
        sim_loose.step()
        sim_tight.step()

        self.assertGreater(float(sim_tight.state.get("hh_buffer_gap_shortfall_total", 0.0)), 0.0)
        self.assertLess(float(sim_tight.history[0].total_consumption), float(sim_loose.history[0].total_consumption))


if __name__ == "__main__":
    unittest.main()
