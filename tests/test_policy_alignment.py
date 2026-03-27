import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.results import _population_distribution_snapshot, _prepare_startup_sim, _startup_diagnostics, run_simulation


def make_cfg():
    cfg = copy.deepcopy(get_default_config())
    cfg["parameters"]["hard_assert_sfc"] = True
    return cfg


class PolicyAlignmentTests(unittest.TestCase):
    def test_default_support_mode_is_ubi(self):
        cfg = make_cfg()
        self.assertEqual(str(cfg["parameters"].get("income_support_mode", "")).upper(), "UBI")

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

    def test_fund_dividends_start_after_prior_quarter_ownership(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["trust_trigger_dti"] = 0.0
        params["trust_launch_loan"] = 15000.0
        params["trust_launch_target_pct"] = 0.10
        params["gov_tax_rebate_rate"] = 0.0
        params["disable_income_support"] = True
        params["loan_rate_per_quarter"] = 0.0
        params["vat_rate"] = 0.0
        params["income_tax_rate"] = 0.0
        params["corporate_tax_rate"] = 0.0
        params["corporate_tax_dynamic_with_wages"] = False
        params["firm_overhead_rate_info"] = 0.0
        params["firm_overhead_rate_phys"] = 0.0
        params["sector_input_cost_rate_info"] = 0.0
        params["sector_input_cost_rate_phys"] = 0.0

        sim = NewLoop(cfg)
        for _ in range(3):
            sim.step()

        self.assertAlmostEqual(float(sim.history[0].fund_dividend_inflow_per_h), 0.0, places=9)
        self.assertTrue(bool(sim.history[1].trust_active))
        self.assertAlmostEqual(float(sim.history[1].fund_dividend_inflow_per_h), 0.0, places=9)
        self.assertGreater(float(sim.history[2].fund_dividend_inflow_per_h), 0.0)

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
        params["firm_overhead_rate_info"] = 0.0
        params["firm_overhead_rate_phys"] = 0.0
        params["sector_input_cost_rate_info"] = 0.0
        params["sector_input_cost_rate_phys"] = 0.0

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
        params["firm_overhead_rate_info"] = 0.0
        params["firm_overhead_rate_phys"] = 0.0
        params["sector_input_cost_rate_info"] = 0.0
        params["sector_input_cost_rate_phys"] = 0.0

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

    def test_sector_capex_plan_ignores_internal_load_gap_when_household_unmet_is_zero(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.state["price_level"] = 1.0
        sim.state["sector_capacity_info_real_prev"] = 1000.0
        sim.state["sector_free_cash_info_prev"] = 1000.0
        sim.state["sector_unmet_info_real_prev"] = 0.0
        sim.state["sector_unmet_info_real_sm_prev"] = 0.0
        sim.state["sector_load_gap_info_real_prev"] = 0.0
        sim.state["sector_load_gap_info_real_sm_prev"] = 0.0

        capex_without_load_gap = sim._sector_capex_plan_nom("FA", 1.0)

        sim.state["sector_load_gap_info_real_prev"] = 400.0
        sim.state["sector_load_gap_info_real_sm_prev"] = 400.0
        capex_with_load_gap = sim._sector_capex_plan_nom("FA", 1.0)

        self.assertAlmostEqual(capex_with_load_gap, capex_without_load_gap, places=9)

    def test_sector_capex_plan_funds_maintenance_before_expansion(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.state["price_level"] = 1.0
        sim.nodes["FH"].set("K", 1000.0)
        sim.state["sector_base_capacity_phys_real"] = 1000.0
        sim.state["sector_capacity_phys_real_prev"] = sim._sector_capacity_real("FH")
        maintenance_nom = sim._sector_maintenance_capex_nom("FH", 1.0)
        reserve_share = float(cfg["parameters"].get("sector_maintenance_reserve_share", 1.0))
        sim.state["sector_free_cash_phys_prev"] = maintenance_nom + 100.0
        sim.state["sector_unmet_phys_real_prev"] = 0.0
        sim.state["sector_unmet_phys_real_sm_prev"] = 0.0
        sim.state["sector_load_gap_phys_real_prev"] = 0.0
        sim.state["sector_load_gap_phys_real_sm_prev"] = 0.0

        capex_plan_nom = sim._sector_capex_plan_nom("FH", 1.0)

        self.assertGreaterEqual(capex_plan_nom, (reserve_share * maintenance_nom) - 1e-9)

    def test_sector_dividend_commit_reserves_maintenance_profit_first(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.state["price_level"] = 1.0
        sim.nodes["FH"].set("K", 1000.0)
        sim.state["sector_base_capacity_phys_real"] = 1000.0

        maintenance_nom = sim._sector_maintenance_capex_nom("FH", 1.0)
        reserve_share = float(cfg["parameters"].get("sector_maintenance_reserve_share", 1.0))
        distributable = sim._sector_profit_distributable_nom("FH", maintenance_nom + 50.0, 1.0)

        self.assertAlmostEqual(distributable, ((1.0 - reserve_share) * maintenance_nom) + 50.0, places=6)

    def test_mortgage_gap_neutralization_funds_bank_when_gap_exists(self):
        cfg = make_cfg()
        params = cfg["parameters"]
        params["mort_neutralize_trigger_mode"] = "Always"
        params["mort_neutralize_funding_stack"] = ["ISSUANCE"]
        params["mort_neutralize_cap_mode"] = "None"
        params["gov_tax_rebate_rate"] = 0.0

        sim = NewLoop(cfg)
        sim.step()

        self.assertGreater(float(sim.state.get("mort_gap_total", 0.0)), 0.0)
        self.assertGreater(float(sim.state.get("bank_mort_neutralize_inflow", 0.0)), 0.0)
        self.assertGreater(float(sim.state.get("bank_mort_neutralize_principal_inflow", 0.0)), 0.0)
        self.assertAlmostEqual(
            float(sim.state.get("bank_mort_neutralize_inflow", 0.0)),
            float(sim.state.get("mort_gap_total", 0.0)),
            places=6,
        )
        self.assertAlmostEqual(
            float(sim.state.get("mort_gap_paid_by_issuance", 0.0)),
            float(sim.state.get("mort_gap_total", 0.0)),
            places=6,
        )

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

    def test_sector_target_payout_rate_rises_when_unmet_demand_is_low(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.state["sector_capacity_info_real_prev"] = 100.0
        sim.state["sector_unmet_info_real_prev"] = 0.0
        low_gap_rate = sim._sector_target_payout_rate("FA")

        sim.state["sector_unmet_info_real_prev"] = 10.0
        high_gap_rate = sim._sector_target_payout_rate("FA")

        self.assertGreater(low_gap_rate, high_gap_rate)
        self.assertAlmostEqual(
            low_gap_rate,
            float(cfg["parameters"]["dividend_payout_rate_firms_mature_max"]),
            places=9,
        )

    def test_sector_surplus_distribution_requires_cash_above_reserves(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        sim.state["sector_capacity_info_real_prev"] = 100.0
        sim.state["sector_unmet_info_real_prev"] = 0.0
        sim.state["sector_base_capacity_info_real"] = 50.0
        sim.nodes["FA"].memo["revenue_prev"] = 200.0
        sim.nodes["FA"].set("deposits", 300.0)

        dist_nom = sim._sector_surplus_distribution_nom("FA", 1.0)
        reserve_nom = sim._sector_maintenance_capex_nom("FA", 1.0) + (0.10 * 200.0)
        expected = 0.50 * max(0.0, 300.0 - reserve_nom)

        self.assertAlmostEqual(dist_nom, expected, places=6)

        sim.nodes["FA"].set("deposits", reserve_nom - 1.0)
        self.assertAlmostEqual(sim._sector_surplus_distribution_nom("FA", 1.0), 0.0, places=9)

    def test_startup_diagnostics_report_target_like_sector_operating_margins(self):
        cfg = make_cfg()
        sim = NewLoop(cfg)

        _prepare_startup_sim(sim)
        diag = _startup_diagnostics(sim)

        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertAlmostEqual(float(diag.get("startup_op_margin_info", 0.0)), 0.25, delta=0.03)
        self.assertAlmostEqual(float(diag.get("startup_op_margin_phys", 0.0)), 0.10, delta=0.03)

    def test_uis_starts_at_zero_and_anchors_from_q0_wages(self):
        cfg = make_cfg()
        cfg["parameters"]["income_support_mode"] = "UIS"

        sim = NewLoop(cfg)
        sim.step()

        q0 = sim.history[0]
        self.assertAlmostEqual(float(q0.uis_per_h), 0.0, places=9)

        p0 = max(1e-9, float(q0.price_level))
        expected_real_target = float(q0.wages_total) / p0
        self.assertAlmostEqual(
            float(sim.state.get("income_target_pool_real_pop", 0.0)),
            expected_real_target,
            places=6,
        )

    def test_neutral_warmup_makes_before_snapshot_common_across_support_modes(self):
        cfg_uis = make_cfg()
        cfg_ubi = make_cfg()
        cfg_uis["parameters"]["income_support_mode"] = "UIS"
        cfg_ubi["parameters"]["income_support_mode"] = "UBI"
        cfg_uis["parameters"]["neutral_warmup_quarters"] = 2
        cfg_ubi["parameters"]["neutral_warmup_quarters"] = 2
        cfg_uis["parameters"]["baseline_calibration_enabled"] = False
        cfg_ubi["parameters"]["baseline_calibration_enabled"] = False

        run_uis = run_simulation(n_quarters=1, cfg=cfg_uis)
        run_ubi = run_simulation(n_quarters=1, cfg=cfg_ubi)

        before_uis = run_uis.population_distributions["before"]
        before_ubi = run_ubi.population_distributions["before"]

        self.assertTrue(np.allclose(np.asarray(before_uis["income"], dtype=float), np.asarray(before_ubi["income"], dtype=float)))
        self.assertTrue(np.allclose(np.asarray(before_uis["wealth"], dtype=float), np.asarray(before_ubi["wealth"], dtype=float)))
        self.assertEqual(int(run_uis.startup_diagnostics.get("neutral_warmup_quarters", 0)), 2)
        self.assertEqual(int(run_uis.startup_diagnostics.get("neutral_warmup_quarters_completed", 0)), 2)
        self.assertEqual(int(run_ubi.startup_diagnostics.get("neutral_warmup_quarters", 0)), 2)
        self.assertEqual(int(run_ubi.startup_diagnostics.get("neutral_warmup_quarters_completed", 0)), 2)

    def test_neutral_warmup_preserves_visible_q0_label(self):
        cfg = make_cfg()
        cfg["parameters"]["neutral_warmup_quarters"] = 2
        run = run_simulation(n_quarters=2, cfg=cfg)

        self.assertEqual(int(run.rows[0].get("t", -1)), 0)
        self.assertEqual(int(run.rows[1].get("t", -1)), 1)

    def test_neutral_warmup_fails_soft_when_requested_length_is_infeasible(self):
        cfg = make_cfg()
        cfg["parameters"]["neutral_warmup_quarters"] = 3

        run = run_simulation(n_quarters=1, cfg=cfg)

        self.assertEqual(int(run.startup_diagnostics.get("neutral_warmup_quarters", 0)), 3)
        self.assertLess(int(run.startup_diagnostics.get("neutral_warmup_quarters_completed", 0)), 3)
        self.assertFalse(bool(run.startup_diagnostics.get("neutral_warmup_completed_fully", True)))
        self.assertTrue(str(run.startup_diagnostics.get("neutral_warmup_error", "")))

    def test_sector_input_costs_accumulate_in_ums_before_trust_runs(self):
        cfg = make_cfg()
        cfg["parameters"]["disable_trust"] = True
        sim = NewLoop(cfg)

        sim.step()

        self.assertGreater(float(sim.nodes["UMS"].get("deposits", 0.0)), 0.0)
        self.assertGreater(float(sim.state.get("sector_input_cost_total", 0.0)), 0.0)

    def test_ums_recycle_flows_back_through_sectors_with_lag(self):
        cfg = make_cfg()
        cfg["parameters"]["disable_income_support"] = True
        cfg["parameters"]["ums_recycle_rate_q"] = 1.0
        cfg["parameters"]["fund_residual_to_gov_share"] = 0.0
        cfg["parameters"]["send_fund_residual_to_gov"] = False
        cfg["parameters"]["vat_rate"] = 0.0
        cfg["parameters"]["income_tax_rate"] = 0.0
        cfg["parameters"]["corporate_tax_rate"] = 0.0
        cfg["parameters"]["corporate_tax_dynamic_with_wages"] = False
        cfg["parameters"]["firm_overhead_rate_info"] = 0.0
        cfg["parameters"]["firm_overhead_rate_phys"] = 0.0
        cfg["parameters"]["sector_input_cost_rate_info"] = 0.0
        cfg["parameters"]["sector_input_cost_rate_phys"] = 0.0
        sim = NewLoop(cfg)

        sim.nodes["UMS"].set("deposits", 100.0)
        sim.nodes["BANK"].add("deposit_liab", 100.0)
        sim.nodes["BANK"].add("reserves", 100.0)
        sim.nodes["FA"].memo["revenue_prev"] = 60.0
        sim.nodes["FH"].memo["revenue_prev"] = 40.0

        sim.step()

        self.assertAlmostEqual(float(sim.nodes["UMS"].get("deposits", 0.0)), 0.0, places=6)
        self.assertAlmostEqual(float(sim.state.get("ums_recycle_to_info_total", 0.0)), 60.0, places=6)
        self.assertAlmostEqual(float(sim.state.get("ums_recycle_to_phys_total", 0.0)), 40.0, places=6)
        self.assertAlmostEqual(float(sim.state.get("ums_recycle_total", 0.0)), 100.0, places=6)
        self.assertAlmostEqual(float(sim.state.get("ums_drain_to_fund_total", 0.0)), 0.0, places=6)
        self.assertAlmostEqual(float(sim.state.get("ums_drain_to_gov_total", 0.0)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
