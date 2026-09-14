"""Financing-policy regressions: funding demand, household cash, and dilution."""
from types import SimpleNamespace
import unittest

import numpy as np

from test_accounting_regressions import config, sync_fixture
from newloop.engine import NewLoop
from newloop.results import run_simulation
from newloop.streamlit_params import PARAMETER_CONTROLS, resolve_control_default
from newloop.slnewloop import _build_cfg_from_state


class EquityFinancingTests(unittest.TestCase):
    def make_sim(self):
        cfg = config()
        cfg['parameters']['hh_equity_issue_price_smoothing_q'] = 1.0
        sim = NewLoop(cfg)
        sim.solve_within_tick_population(allow_income_support_trigger=False)
        return sim

    def test_existing_corporate_cash_blocks_issuance_and_consumption_reserve(self):
        sim = self.make_sim()
        for issuer in ('IS', 'PS'):
            sim.nodes[issuer].set('deposits', 1e7)
        sim.hh.deposits[:] = 1000
        sync_fixture(sim)
        before_shares = {k: sim.nodes[k].get('shares_outstanding') for k in ('IS', 'PS')}
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        self.assertGreater(sol['equity_offered_total'], 0)
        self.assertEqual(float(sol['planned_equity_investment_nom_i'].sum()), 0)
        sim.params['hh_equity_investment_enabled'] = False
        without_purchases = sim.solve_within_tick_population(allow_income_support_trigger=False)
        np.testing.assert_allclose(sol['c_hh_nom'], without_purchases['c_hh_nom'])
        sim.params['hh_equity_investment_enabled'] = True
        sim.post_tick_population(sol)
        self.assertEqual(sim.state['hh_equity_investment_total'], 0)
        for issuer in ('IS', 'PS'):
            self.assertEqual(sim.nodes[issuer].get('shares_outstanding'), before_shares[issuer])
        sim._assert_sfc_ok('no unnecessary issuance')

    def test_sector_caps_ration_buyers_and_leave_unaccepted_cash_with_households(self):
        sim = self.make_sim()
        sim.nodes['IS'].set('deposits', 70)
        sim.nodes['IS'].set('capex_reserve', 20)  # Already part of the 70, not additional cash.
        sim.nodes['PS'].set('deposits', 500)
        sim.hh.deposits[:] = 0
        sim.hh.deposits[:2] = [80, 40]
        sync_fixture(sim)
        money = sim._sum_deposits_all()
        shares_before = {k: sim.hh.shares_by_issuer[k].copy() for k in ('IS', 'PS')}
        sim._apply_household_equity_investment(deposits=sim.hh.deposits,
            target_buffer_nom=np.zeros(sim.hh.n), price_level=sim.state['price_level'],
            planned_investment_nom=sim.hh.deposits.copy(), funding_targets={'IS': 100, 'PS': 200})
        np.testing.assert_allclose(sim.hh.deposits[:2], [60, 30])
        self.assertAlmostEqual(sim.nodes['IS'].get('deposits'), 100)
        self.assertAlmostEqual(sim.nodes['IS'].get('capex_reserve'), 50)
        self.assertAlmostEqual(sim.nodes['PS'].get('deposits'), 500)
        np.testing.assert_array_equal(sim.hh.shares_by_issuer['PS'], shares_before['PS'])
        increments = sim.hh.shares_by_issuer['IS'] - shares_before['IS']
        self.assertAlmostEqual(increments[0] / increments[1], 2)
        self.assertEqual(sim.state['hh_equity_unfilled_total'], 90)
        sim.nodes['HH'].set('deposits', sim.hh.sum_deposits())
        self.assertAlmostEqual(sim._sum_deposits_all(), money)
        sim._assert_sfc_ok('rationed equity purchase')

    def test_settlement_reduces_offer_when_operating_cash_has_filled_gap(self):
        sim = self.make_sim()
        sim.nodes['IS'].set('deposits', 95)
        sim.nodes['PS'].set('deposits', 500)
        sim.hh.deposits[:] = 0
        sim.hh.deposits[:2] = [80, 40]
        planned = np.zeros(sim.hh.n)
        planned[:2] = [20, 10]  # IS only after the opening sector caps.
        sync_fixture(sim)
        sim._apply_household_equity_investment(deposits=sim.hh.deposits,
            target_buffer_nom=np.zeros(sim.hh.n), price_level=sim.state['price_level'],
            planned_investment_nom=planned, planned_info_nom=planned,
            funding_targets={'IS': 100, 'PS': 200},
            opening_funding_limits={'IS': 30, 'PS': 0}, offered_total=120)
        np.testing.assert_allclose(sim.hh.deposits[:2], [80-10/3, 40-5/3])
        self.assertAlmostEqual(sim.state['hh_equity_investment_info_total'], 5)
        self.assertEqual(sim.state['hh_equity_investment_phys_total'], 0)
        self.assertEqual(sim.state['hh_equity_unfilled_total'], 115)

    def test_funding_target_respects_installation_limit_queue_and_existing_cash(self):
        sim = self.make_sim()
        sim.params.update(sector_capacity_per_k_info=1, capital_depr_rate_info_per_quarter=.1,
            sector_capex_gap_close_rate=1, sector_capex_growth_cap_rate_q=1, sector_install_rate_q=.5)
        sim.nodes['IS'].set('K', 100)
        sim.state.update(sector_base_capacity_info_real=0, sector_capacity_info_real_prev=100,
            sector_unmet_info_real_sm_prev=100, sector_capex_queue_info_nom=20)
        sim.nodes['IS'].set('deposits', 30)
        sim.nodes['IS'].set('capex_reserve', 30)
        targets = sim._equity_funding_targets_nom(1)
        self.assertAlmostEqual(targets['IS'], 50)  # Only 50 can be installed next quarter.
        self.assertAlmostEqual(sim._equity_funding_limits_nom(targets)['IS'], 20)
        sim.params['disable_capex_and_depreciation'] = True
        self.assertEqual(sim._equity_funding_targets_nom(1), {'IS': 0, 'PS': 0})

    def test_insufficient_household_cash_never_creates_credit_for_subscriptions(self):
        sim = self.make_sim()
        sim.nodes['IS'].set('deposits', 0)
        sim.nodes['PS'].set('deposits', 0)
        sim.hh.deposits[:] = 0
        sim.hh.deposits[0] = 3
        planned = np.zeros(sim.hh.n)
        planned[0] = 20
        loans = sim.hh.sum_loans()
        sim._apply_household_equity_investment(deposits=sim.hh.deposits,
            target_buffer_nom=np.zeros(sim.hh.n), price_level=1,
            planned_investment_nom=planned, planned_info_nom=planned,
            funding_targets={'IS': 100, 'PS': 100})
        self.assertAlmostEqual(sim.state['hh_equity_investment_total'], 3)
        self.assertTrue(np.all(sim.hh.deposits >= 0))
        self.assertEqual(sim.hh.sum_loans(), loans)

    def test_dividends_cannot_manufacture_capital_funding_gap(self):
        sim = self.make_sim()
        sim.params.update(sector_install_rate_q=1, sector_capex_growth_cap_rate_q=1,
                          old_loop_autonomous_growth_capex_rate_q=.2)
        for issuer in ('IS', 'PS'):
            sim.nodes[issuer].memo['dividend_commit_prev'] = 1e8
        sync_fixture(sim)
        opening = {k: sim.nodes[k].get('deposits') for k in ('IS', 'PS')}
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        gaps = {}
        for issuer, suffix in (('IS', 'fa'), ('PS', 'fh')):
            before_dividend = opening[issuer] + sol['cash_profit_'+suffix] - sol['capex_'+suffix+'_nom']
            target = sol['equity_funding_targets_nom'][issuer]
            buffer = sim._equity_operating_buffer_nom(issuer)
            self.assertLessEqual(sol['div_'+suffix+'_total'], max(0, before_dividend-target-buffer)+1e-7)
            gaps[issuer] = max(0, target-max(0,before_dividend-buffer))
        sim.post_tick_population(sol)
        self.assertLessEqual(sim.state['hh_equity_investment_info_total'], gaps['IS']+1e-7)
        self.assertLessEqual(sim.state['hh_equity_investment_phys_total'], gaps['PS']+1e-7)
        sim._assert_sfc_ok('capital before dividends')

    def test_off_restores_unrestricted_subscriptions(self):
        sim = self.make_sim()
        sim.params['equity_issuance_needs_only'] = False
        sim.hh.deposits[:] = 0
        sim.hh.deposits[0] = 100
        for k in ('IS', 'PS'):sim.nodes[k].set('deposits', 1e7)
        sim._apply_household_equity_investment(deposits=sim.hh.deposits,
            target_buffer_nom=np.zeros(sim.hh.n), price_level=1,
            planned_investment_nom=sim.hh.deposits.copy())
        self.assertAlmostEqual(sim.state['hh_equity_investment_info_total'], 30)
        self.assertAlmostEqual(sim.state['hh_equity_investment_phys_total'], 70)
        self.assertEqual(sim.hh.deposits[0], 0)

    def test_limit_operates_when_investment_is_after_consumption(self):
        cfg = config()
        cfg['parameters']['hh_equity_investment_pre_consumption'] = False
        run = run_simulation(12, cfg)
        for row in run.rows:
            self.assertLessEqual(row['hh_equity_investment_per_h'], row['equity_funding_gap_per_h']+1e-7)
            self.assertAlmostEqual(row['hh_equity_offered_per_h'], row['hh_equity_investment_per_h']+row['hh_equity_unfilled_per_h'])
        run.sim._assert_sfc_ok('after consumption funding')

    def test_ui_toggle_defaults_on_and_round_trips_off(self):
        cfg = config()
        control = next(c for c in PARAMETER_CONTROLS if c.path == ('equity_issuance_needs_only',))
        self.assertTrue(resolve_control_default(control, cfg['parameters']))
        rebuilt = _build_cfg_from_state(SimpleNamespace(session_state={'param__equity_issuance_needs_only': False}), cfg)
        self.assertFalse(rebuilt['parameters']['equity_issuance_needs_only'])


if __name__ == '__main__':
    unittest.main()
