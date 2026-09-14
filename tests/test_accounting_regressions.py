"""Transaction-level regressions for the September accounting review."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.results import _household_wealth_snapshot, run_simulation
from newloop.slnewloop import _apply_loop_mode_mortgage_defaults, _rows_for_value_mode, _population_dist_for_value_mode, _rows_csv
from newloop.plotting import _series, plot_household_shortfall_sources


def config(mortgages=False):
    cfg = get_default_config()
    cfg['parameters']['population_config']['n_families'] = 64
    cfg['parameters'].update(economic_regime='OldLoop', hard_assert_sfc=True,
                             old_loop_steady_state_warmup_enabled=False,
                             neutral_warmup_quarters=0)
    _apply_loop_mode_mortgage_defaults(cfg['parameters'], not mortgages)
    return cfg


def sync_fixture(sim):
    """Close the initial balance sheet after a test supplies replacement stocks."""
    sim.nodes['HH'].set('deposits', sim.hh.sum_deposits())
    sim.nodes['HH'].set('loans', sim.hh.sum_loans())
    arrears = float(sim.hh.mort_interest_arrears_q.sum())
    sim.nodes['HH'].set('interest_payable', arrears)
    bank = sim.nodes['BANK']
    bank.set('interest_receivable', arrears)
    bank.set('deposit_liab', sim._sum_deposits_all())
    bank.set('loan_assets', sim._sum_loans_borrowers())
    bank.set('equity', bank.get('loan_assets') + bank.get('reserves') + arrears - bank.get('deposit_liab'))


class AccountingRegressions(unittest.TestCase):
    def test_equity_buyer_receives_new_claim_and_issuer_ownership(self):
        cfg = config()
        cfg['parameters']['hh_equity_issue_price_smoothing_q'] = 1.0
        cfg['parameters']['equity_issuance_needs_only'] = False
        sim = NewLoop(cfg)
        sim.solve_within_tick_population(allow_income_support_trigger=False)
        i = int(np.flatnonzero(sim.hh.equity_weight_i == 0)[0])
        sim.hh.deposits[:] = 0
        sim.hh.deposits[i] = 1000
        sync_fixture(sim)
        before = copy.deepcopy(_household_wealth_snapshot(sim))
        bank_shares = sim.hh.shares_by_issuer['BANK'].copy()
        planned = np.zeros(sim.hh.n)
        planned[i] = 100
        sim._apply_household_equity_investment(deposits=sim.hh.deposits,
            target_buffer_nom=np.zeros(sim.hh.n), price_level=sim.state['price_level'],
            planned_investment_nom=planned)
        sim.nodes['HH'].set('deposits', sim.hh.sum_deposits())
        after = _household_wealth_snapshot(sim)
        self.assertAlmostEqual(after['private_equity'][i] - before['private_equity'][i], 100)
        self.assertAlmostEqual(after['wealth'][i], before['wealth'][i])
        np.testing.assert_allclose(after['private_equity'][np.arange(sim.hh.n) != i], before['private_equity'][np.arange(sim.hh.n) != i])
        np.testing.assert_array_equal(sim.hh.shares_by_issuer['BANK'], bank_shares)
        self.assertGreater(sim._household_issuer_weights('IS')[i], 0)
        sim._assert_sfc_ok('equity purchase')
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        self.assertAlmostEqual(sol['div_i'][i], 0)
        sim.post_tick_population(sol)
        sim.nodes['IS'].add('deposits', 1000)
        sim.nodes['BANK'].add('deposit_liab', 1000)
        sim.nodes['BANK'].add('reserves', 1000)
        sim.nodes['IS'].memo['dividend_commit_prev'] = 100
        following = sim.solve_within_tick_population(allow_income_support_trigger=False)
        self.assertGreater(following['div_i'][i], 0)

    def test_corporate_earnings_reconcile_to_cash_capital_and_dividends(self):
        sim = NewLoop(config())
        sim.solve_within_tick_population(allow_income_support_trigger=False)
        for issuer in ('IS', 'PS'):
            sim.nodes[issuer].add('deposits', 100000)
            sim.nodes[issuer].memo['dividend_commit_prev'] = 50000
        sync_fixture(sim)
        price = sim.state['price_level']
        before = {issuer: sim._firm_broad_equity_proxy(issuer, price) for issuer in ('IS', 'PS')}
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        self.assertTrue(sol['retained_fa'] < 0 or sol['retained_fh'] < 0)
        housing = float(sol['renter_rent_q'].sum() + sol['owner_housing_payment_i'].sum())
        self.assertAlmostEqual(sol['housing_revenue_fa'] + sol['housing_revenue_fh'], housing)
        sim.post_tick_population(sol)
        for issuer, suffix, investment in [('IS', 'fa', 'info'), ('PS', 'fh', 'phys')]:
            profit = sol['rev_' + suffix] - sol['w_' + suffix] - sol[suffix + '_interest'] - sol['overhead_' + suffix] - sol['input_cost_' + suffix] - sol['corp_tax_' + suffix] - sol['depreciation_' + suffix]
            self.assertAlmostEqual(sol['p_' + suffix], profit)
            self.assertAlmostEqual(profit, sol['div_' + suffix + '_total'] + sol['retained_' + suffix])
            change = sim._firm_broad_equity_proxy(issuer, price) - before[issuer]
            self.assertAlmostEqual(change, sol['retained_' + suffix] + sim.state['hh_equity_investment_' + investment + '_total'], places=6)
        sim._assert_sfc_ok('profits')

    def test_bank_equity_corruption_is_detected(self):
        sim = NewLoop(config())
        sim.nodes['BANK'].add('equity', 1000)
        with self.assertRaisesRegex(AssertionError, 'bank balance sheet'):
            sim._assert_sfc_ok('injected discrepancy')

    def test_mortgage_turnover_advances_full_principal_above_house_value(self):
        cfg = config(True)
        cfg['parameters'].update(housing_turnover_rate_mortgagor_q=1.0)
        cfg['parameters']['population_config']['mortgage_startup_ltv_max'] = 1.15
        sim = NewLoop(cfg)
        for _ in range(12):
            sim.step()
            self.assertAlmostEqual(sim.state['mort_turnover_total'] - sim.state['mort_turnover_old_payoff_total'], sim.state['mort_turnover_deposit_delta_total'], places=6)
            self.assertAlmostEqual(sim.inv_history[-1]['bank_balance_gap'], 0, places=6)

    def mortgage_shortfall_fixture(self):
        cfg = config(True)
        cfg['parameters'].update(mortgage_turnover_enabled=False, mortgage_maturity_roll_enabled=False,
            old_loop_auto_reissue_paid_off_mortgages=False, revolving_credit_limit_income_mult=1e-6)
        sim = NewLoop(cfg)
        hh = sim.hh
        i = int(np.flatnonzero(hh.equity_weight_i == 0)[0])
        hh.deposits[:] = 0
        hh.revolving_loans[:] = 0
        for field in ('mortgage_loans', 'mort_payment_sched_q', 'mort_orig_principal', 'mort_rate_q', 'mort_age_q', 'mort_term_q'):
            getattr(hh, field)[:] = 0
        hh.mortgage_loans[i] = hh.mort_orig_principal[i] = 10000
        hh.mort_payment_sched_q[i] = 200
        hh.mort_rate_q[i] = .01
        hh.mort_term_q[i] = 120
        hh.wages0_q[i] = 0
        sim.nodes['BANK'].memo['dividend_commit_prev'] = 0
        sim._invalidate_mortgage_contract_state()
        sim._refresh_mortgage_contract_state()
        sync_fixture(sim)
        return sim, i

    def test_unpaid_interest_accrues_matching_claims_without_cash_dividend(self):
        sim, i = self.mortgage_shortfall_fixture()
        before = sim.nodes['BANK'].get('equity')
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        sim.post_tick_population(sol)
        self.assertAlmostEqual(sim.state['bank_interest_collected_total'], 0)
        self.assertAlmostEqual(sim.hh.mort_interest_arrears_q[i], 100)
        self.assertAlmostEqual(sim.hh.mort_principal_arrears_q[i], 100)
        self.assertAlmostEqual(sim.nodes['BANK'].get('interest_receivable'), 100)
        self.assertAlmostEqual(sim.nodes['HH'].get('interest_payable'), 100)
        self.assertAlmostEqual(sim.nodes['BANK'].get('equity') - before, sol['bank_profit'])
        self.assertAlmostEqual(sim.nodes['BANK'].memo['dividend_commit_prev'], 0)
        self.assertAlmostEqual(sim._lagged_dividend_commit_nom('BANK'), 0)
        self.assertAlmostEqual(_household_wealth_snapshot(sim)['loans'][i], 10100)
        sim._assert_sfc_ok('unpaid interest')

        # Pay the arrears plus this quarter's interest. Only the new interest
        # is accrual income; the old claim is settled without earning it twice.
        sim.hh.deposits[i] = 1000000
        sim.hh.revolving_loans[:] = 0
        sim.nodes['BANK'].memo['dividend_commit_prev'] = 0
        sync_fixture(sim)
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        sim.post_tick_population(sol)
        self.assertAlmostEqual(sim.nodes['BANK'].get('interest_receivable'), 0, places=6)
        self.assertAlmostEqual(sim.state['bank_interest_collected_total'], 200, places=6)
        self.assertAlmostEqual(sol['bank_profit'], 65, places=6)
        self.assertAlmostEqual(sol['bank_cash_profit'], 165, places=6)
        sim._assert_sfc_ok('arrears collection')

    def test_interest_claim_survives_principal_retirement(self):
        sim, i = self.mortgage_shortfall_fixture()
        sim.hh.mortgage_loans[i] = 0
        sim.hh.mort_interest_arrears_q[i] = 75
        sync_fixture(sim)
        sim._invalidate_mortgage_contract_state()
        sim._refresh_mortgage_contract_state()
        self.assertAlmostEqual(sim.hh.mort_interest_arrears_q[i], 75)
        sim._assert_sfc_ok('principal retired')

    def test_bank_compensation_is_taxed_and_counted_in_roe_once(self):
        cfg = get_default_config()
        cfg['parameters']['population_config']['n_families'] = 64
        cfg['parameters'].update(mort_neutralize_trigger_mode='Always',
            mort_neutralize_funding_stack=['ISSUANCE'], mort_neutralize_cap_mode='None',
            gov_tax_rebate_rate=0.0)
        sim = NewLoop(cfg)
        sim.hh.mort_P0 *= 100  # An indexed payment ceiling below contractual interest.
        before = sim.nodes['BANK'].get('equity')
        posted = {}
        original = sim.post_tick_population
        def post(sol):
            result = original(sol)
            posted.update(sol)
            return result
        sim.post_tick_population = post
        sim.step()
        self.assertGreater(sim.state['bank_mort_neutralize_interest_inflow'], 0)
        self.assertAlmostEqual(sim.nodes['BANK'].get('equity') - before, posted['retained_bk'], places=6)
        self.assertAlmostEqual(sim.history[-1].bank_broad_roe_q * before, posted['bank_profit'], places=6)
        self.assertAlmostEqual(posted['bank_profit'], posted['div_bk_total'] + posted['retained_bk'], places=6)

    def test_policy_transition_launches_trust_without_mortgages(self):
        cfg = config()
        cfg['parameters'].update(economic_regime='OldToNew', old_to_new_transition_quarters=2,
                                 old_to_new_transition_mode='NewLoopPolicies')
        run = run_simulation(6, cfg)
        self.assertFalse(run.rows[1]['trust_active'])
        self.assertTrue(run.rows[2]['trust_active'])
        self.assertGreater(run.rows[3]['fund_dividend_inflow_per_h'], 0)
        run.sim._assert_sfc_ok('trust')

        cfg['parameters']['old_to_new_transition_quarters'] = 0
        immediate = run_simulation(2, cfg)
        self.assertTrue(immediate.rows[0]['trust_active'])
        immediate.sim._assert_sfc_ok('immediate trust launch')

    def test_real_wealth_components_and_distributions_use_same_unit(self):
        row = dict(price_level=2, hh_deposits_per_h=100, hh_housing_value_per_h=200,
            private_eq_per_h=300, trust_value_per_h=0, hh_debt_per_h=50,
            sector_capacity_info_per_h=100)
        real = _rows_for_value_mode([row], 'real')[0]
        self.assertEqual(real['hh_deposits_per_h'] + real['hh_housing_value_per_h'] + real['private_eq_per_h'] - real['hh_debt_per_h'], 275)
        self.assertEqual(real['sector_capacity_info_per_h'], 100)
        snap = dict(price_level=2, wealth=[550], income=[100])
        dist = _population_dist_for_value_mode(dict(before=snap, after=snap), 'real', 2)
        self.assertEqual(dist['after']['wealth'], [275])
        self.assertNotIn('_monetary_scale', _rows_csv([real]))

    def test_roe_visibility_is_independent_of_display_units(self):
        rows = [dict(price_level=2, corporate_eq_info_per_h=800, corporate_info_broad_roe_q=.05)] * 2
        self.assertEqual(_series(rows, 'corporate_info_broad_roe_q')[1], _series(_rows_for_value_mode(rows, 'real'), 'corporate_info_broad_roe_q')[1])

    def test_drawdowns_are_summed_before_aggregation(self):
        sim = NewLoop(config())
        opening = sim.hh.deposits.copy()
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        sim.post_tick_population(sol)
        self.assertAlmostEqual(sim.state['hh_deposit_drawdown_total'], np.maximum(0, opening - sim.hh.deposits).sum())
        # Aggregate income exceeds aggregate uses, but one household spent savings.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        rows = [dict(t=t, hh_cash_income_per_h=50, hh_realized_consumption_per_h=45,
                     hh_deposit_drawdown_per_h=5) for t in range(2)]
        plot_household_shortfall_sources(rows, axes=axes)
        self.assertGreater(len(axes[1].collections), 0)
        self.assertAlmostEqual(max(axes[1].collections[0].get_paths()[0].vertices[:, 1]), 5)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
