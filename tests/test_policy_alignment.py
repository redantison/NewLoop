import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.results import _population_distribution_snapshot


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

    def test_trust_launch_targets_fixed_initial_equity_block(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 0.0
        params["trust_launch_loan"] = 12000.0
        params["trust_launch_target_pct"] = 0.10
        params["gov_tax_rebate_rate"] = 0.0

        sim = NewLoop(cfg)
        sim.step()
        sim.step()

        self.assertTrue(sim.state["trust_active"])
        # First visible trust point includes both the 10% launch buy and the first scheduled issuance.
        phi_qtr = 1.0 - (0.98 ** 0.25)
        expected_pct = (0.10 + phi_qtr) / (1.0 + phi_qtr)
        self.assertAlmostEqual(float(sim.history[1].trust_equity_pct), expected_pct, places=3)

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
        params["disable_income_support"] = True
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
        params["disable_income_support"] = True
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
        params["disable_income_support"] = True
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
        params["disable_income_support"] = True
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

    def test_indexed_mortgage_payment_is_capped_at_origination_real_burden(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)
        hh = sim.hh
        self.assertIsNotNone(hh)
        assert hh is not None
        hh.ensure_memos()

        active = np.asarray(hh.mortgage_loans, dtype=float) > 1e-12
        self.assertTrue(bool(np.any(active)))

        p_now = float(sim._mort_price_series_value(float(sim.state.get("price_level", 1.0))))
        sim.state["mort_price_series_prev"] = p_now
        sim.state["mort_income_series_prev"] = 1.0

        wages_total = 1e9
        terms = sim._compute_mortgage_index_terms(
            mort=np.asarray(hh.mortgage_loans, dtype=float),
            rL=float(sim.state.get("policy_rate_q", sim.params.get("loan_rate_per_quarter", 0.0))),
            wages_total=wages_total,
            div_house_total=0.0,
            uis_per_h=0.0,
            commit_state=False,
        )

        mort_pay_req_i = np.asarray(terms["mort_pay_req_i"], dtype=float)
        base_vec = np.maximum(0.0, np.asarray(hh.mort_pay_base, dtype=float))
        p0_vec = np.maximum(1e-9, np.asarray(hh.mort_P0, dtype=float))
        cap_i = base_vec * (p_now / p0_vec)

        self.assertTrue(np.all(mort_pay_req_i[active] <= (cap_i[active] + 1e-9)))

    def test_mortgage_neutralization_splits_interest_from_principal(self):
        cfg = make_cfg()
        cfg["parameters"]["mort_neutralize_trigger_mode"] = "Always"
        sim = NewLoop(cfg)
        n = sim.hh.n if sim.hh is not None else 0
        self.assertGreater(n, 0)
        interest_gap_i = np.zeros(n, dtype=float)
        principal_gap_i = np.zeros(n, dtype=float)
        interest_gap_i[0] = 10.0
        principal_gap_i[0] = 20.0

        sim.nodes["GOV"].set("deposits", 30.0)
        sim.nodes["BANK"].add("deposit_liab", 30.0)
        sim.nodes["BANK"].add("reserves", 30.0)
        loan_assets_before = float(sim.nodes["BANK"].get("loan_assets", 0.0))
        equity_before = float(sim.nodes["BANK"].get("equity", 0.0))

        neutral = sim._apply_mortgage_gap_neutralization(
            interest_gap_i=interest_gap_i,
            principal_gap_i=principal_gap_i,
            mort_interest_due_total=10.0,
            mort_pay_ctr_total=30.0,
        )

        self.assertAlmostEqual(float(neutral["paid_interest_total"]), 10.0, places=6)
        self.assertAlmostEqual(float(neutral["paid_principal_total"]), 20.0, places=6)
        self.assertAlmostEqual(float(np.sum(neutral["paid_principal_i"])), 20.0, places=6)
        self.assertAlmostEqual(float(sim.nodes["BANK"].get("equity", 0.0)), equity_before + 10.0, places=6)
        self.assertAlmostEqual(float(sim.nodes["BANK"].get("loan_assets", 0.0)), loan_assets_before - 20.0, places=6)
        self.assertAlmostEqual(float(sim.nodes["GOV"].get("deposits", 0.0)), 0.0, places=6)

    def test_corporate_roe_split_metrics_recompose_to_totals(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.step()
        p_prev = float(sim.state.get("price_level", 1.0))
        bank_eq_prev = float(sim._firm_balance_sheet_equity_proxy("BANK", p_prev))
        fa_eq_prev = float(sim._firm_broad_equity_proxy("FA", p_prev))
        fh_eq_prev = float(sim._firm_broad_equity_proxy("FH", p_prev))

        sim.step()
        row = sim.history[-1]

        total_eq_prev = bank_eq_prev + fa_eq_prev + fh_eq_prev
        nonbank_eq_prev = fa_eq_prev + fh_eq_prev
        self.assertGreater(total_eq_prev, 0.0)
        self.assertGreater(nonbank_eq_prev, 0.0)

        total_recomposed = (
            (float(row.bank_broad_roe_q) * bank_eq_prev)
            + (float(row.corporate_info_broad_roe_q) * fa_eq_prev)
            + (float(row.corporate_physical_broad_roe_q) * fh_eq_prev)
        ) / total_eq_prev
        nonbank_recomposed = (
            (float(row.corporate_info_broad_roe_q) * fa_eq_prev)
            + (float(row.corporate_physical_broad_roe_q) * fh_eq_prev)
        ) / nonbank_eq_prev

        self.assertAlmostEqual(float(row.corporate_broad_roe_q), total_recomposed, places=9)
        self.assertAlmostEqual(float(row.corporate_nonbank_broad_roe_q), nonbank_recomposed, places=9)

    def test_startup_bootstrap_seeds_positive_broad_equity_denominator_by_default(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim._bootstrap_startup_lagged_retained()

        self.assertGreater(float(sim.state.get("corporate_info_equity_prev_total", 0.0)), 0.0)
        self.assertGreater(float(sim.state.get("corporate_physical_equity_prev_total", 0.0)), 0.0)
        self.assertAlmostEqual(float(sim.nodes["FA"].get("K", 0.0)), 0.0, places=9)
        self.assertAlmostEqual(float(sim.nodes["FH"].get("K", 0.0)), 0.0, places=9)

    def test_startup_bootstrap_respects_explicit_initial_firm_capital(self):
        cfg = make_cfg()
        cfg["nodes"]["FA"]["stocks"]["K"] = 123.0
        cfg["nodes"]["FH"]["stocks"]["K"] = 456.0
        sim = NewLoop(cfg)

        sim._bootstrap_startup_lagged_retained()

        self.assertAlmostEqual(float(sim.nodes["FA"].get("K", 0.0)), 123.0, places=9)
        self.assertAlmostEqual(float(sim.nodes["FH"].get("K", 0.0)), 456.0, places=9)

    def test_startup_bootstrap_can_opt_in_to_firm_capital_seed(self):
        cfg = make_cfg()
        cfg["parameters"]["startup_bootstrap_firm_capital"] = True
        sim = NewLoop(cfg)

        sim._bootstrap_startup_lagged_retained()

        self.assertGreater(float(sim.nodes["FA"].get("K", 0.0)), 0.0)
        self.assertGreater(float(sim.nodes["FH"].get("K", 0.0)), 0.0)

    def test_population_wealth_snapshot_splits_trust_value_equally(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)
        assert sim.hh is not None

        baseline = _population_distribution_snapshot(sim)
        self.assertIsNotNone(baseline)
        assert baseline is not None

        sim.nodes["FUND"].set("deposits", 200.0)
        sim.nodes["BANK"].add("deposit_liab", 200.0)
        sim.nodes["BANK"].add("reserves", 200.0)

        with_trust = _population_distribution_snapshot(sim)
        self.assertIsNotNone(with_trust)
        assert with_trust is not None

        delta = np.asarray(with_trust["wealth"], dtype=float) - np.asarray(baseline["wealth"], dtype=float)
        np.testing.assert_allclose(delta, np.full(delta.shape, 200.0 / float(sim.hh.n)), rtol=0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
