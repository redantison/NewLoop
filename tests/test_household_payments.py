"""Payment distress must accumulate without invented cash or pooled household funds."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

import test_accounting_regressions
from test_accounting_regressions import config, sync_fixture
from newloop.engine import NewLoop
from newloop.household_payments import allocate_household_payments
from newloop.results import run_simulation, _household_wealth_snapshot
from newloop.slnewloop import _build_cfg_from_state, _rows_for_value_mode


class HouseholdPaymentsTests(unittest.TestCase):
    def fixture(self):
        cfg = config()
        cfg['parameters']['hh_shortfall_financing_enabled'] = False
        sim = NewLoop(cfg)
        i = int(np.flatnonzero(sim.hh.equity_weight_i == 0)[0])
        sim.hh.wages0_q[i] = 0
        sim.hh.deposits[i] = 0
        sim.hh.revolving_loans[i] = 1000
        sim.hh.renter_rent_q[i] = 50
        sync_fixture(sim)
        return sim, i

    def test_cash_priority_never_pools_household_funds(self):
        cash = np.array([10., 100.])
        due = {'revolving_interest': np.array([15., 5.]), 'rent': np.array([20., 20.]),
               'income_tax': np.array([10., 10.])}
        arrears = {'rent': np.array([4., 4.])}
        originals = copy.deepcopy((cash, due, arrears))
        plan = allocate_household_payments(cash, due, arrears)
        np.testing.assert_array_equal(plan['current']['revolving_interest'], [10, 5])
        np.testing.assert_array_equal(plan['current']['rent'], [0, 20])
        np.testing.assert_array_equal(plan['current']['income_tax'], [0, 10])
        np.testing.assert_array_equal(plan['arrears']['rent'], [0, 4])
        np.testing.assert_array_equal(plan['remaining_cash'], [0, 61])
        np.testing.assert_array_equal(cash, originals[0])
        for k in due: np.testing.assert_array_equal(due[k], originals[1][k])
        np.testing.assert_array_equal(arrears['rent'], originals[2]['rent'])

    def test_failed_household_accumulates_bills_while_others_keep_operating(self):
        sim, i = self.fixture()
        for quarter in range(3):
            sim.step()
            self.assertAlmostEqual(sim.hh.payment_arrears['revolving_interest'][i], 12.5*(quarter+1))
            self.assertAlmostEqual(sim.hh.payment_arrears['rent'][i], 50*(quarter+1))
            self.assertAlmostEqual(sim.hh.revolving_loans[i], 1000)
            self.assertAlmostEqual(sim.hh.deposits[i], 0, places=7)
            self.assertEqual(sim.history[-1].hh_overdraft_to_revolving_per_h, 0)
            self.assertGreater(sim.history[-1].real_consumption, 0)
        self.assertEqual(len(sim.history), 3)
        self.assertTrue(sim.hh.ever_payment_shortfall[i])
        self.assertGreater(sim.history[-1].hh_in_arrears_share, 0)
        sim._assert_sfc_ok('accumulated unpaid bills')

    def test_unpaid_claims_do_not_create_deposits_or_net_creditor_equity(self):
        sim, i = self.fixture()
        cash = sim._sum_deposits_all()
        bank_equity = sim.nodes['BANK'].get('equity')
        sim.hh.payment_arrears['revolving_interest'][i] = 70
        sim.hh.payment_arrears['rent'][i] = 30
        sim.hh.payment_arrears['income_tax'][i] = 20
        before = _household_wealth_snapshot(sim)['wealth'][i]
        sim._sync_unpaid_bill_claims()
        self.assertAlmostEqual(sim._sum_deposits_all(), cash)
        self.assertAlmostEqual(sim.nodes['BANK'].get('equity'), bank_equity)
        self.assertEqual(sim.nodes['BANK'].get('household_bill_receivable'), 70)
        self.assertEqual(sim.nodes['BANK'].get('household_bill_allowance'), 70)
        self.assertEqual(sim.nodes['HH'].get('unpaid_bills'), 120)
        sim._assert_sfc_ok('gross claims with collection allowances')
        sim.hh.payment_arrears['income_tax'][i] += 10
        self.assertAlmostEqual(_household_wealth_snapshot(sim)['wealth'][i], before-10)

    def test_corrupted_creditor_claim_is_detected(self):
        sim, _ = self.fixture()
        sim.step()
        sim.nodes['IS'].add('household_bill_allowance', 1)
        with self.assertRaisesRegex(AssertionError, 'unpaid bill claim/allowance'):
            sim._assert_sfc_ok('corrupt allowance')

    def test_old_bills_can_be_paid_when_household_cash_recovers(self):
        sim, i = self.fixture()
        sim.step()
        self.assertAlmostEqual(sim.hh.unpaid_bills_i()[i], 62.5)
        sim.hh.deposits[i] = 1000
        sim.hh.base_real_cons_q[i] = 0
        sim.hh.mpc_q[i] = 0
        sim.hh.prev_perm_income[i] = 0
        sync_fixture(sim)
        sim.step()
        self.assertAlmostEqual(sim.hh.unpaid_bills_i()[i], 0, places=6)
        self.assertGreaterEqual(sim.state['hh_unpaid_bills_paid_total'], 62.5-1e-6)
        self.assertTrue(sim.hh.ever_payment_shortfall[i])
        self.assertAlmostEqual(sim.hh.revolving_loans[i], 1000)
        sim._assert_sfc_ok('arrears collection')

    def test_credit_financing_remains_available_for_comparison(self):
        sim, i = self.fixture()
        sim.params['hh_shortfall_financing_enabled'] = True
        sim.step()
        self.assertAlmostEqual(sim.hh.revolving_loans[i], 1062.5)
        self.assertAlmostEqual(sim.hh.unpaid_bills_i()[i], 0)
        self.assertTrue(sim.hh.ever_payment_shortfall[i])

    def test_restoring_financing_preserves_arrears_and_reconciles_later_collection(self):
        sim, i = self.fixture()
        sim.step()
        sim.params['hh_shortfall_financing_enabled'] = True
        sim.step()
        self.assertAlmostEqual(sim.hh.unpaid_bills_i()[i], 62.5)
        sim.hh.deposits[i] = 1000
        sim.hh.base_real_cons_q[i] = sim.hh.mpc_q[i] = sim.hh.prev_perm_income[i] = 0
        sync_fixture(sim)
        before_bank = sim.nodes['BANK'].get('equity')
        price = sim.state['price_level']
        before_firms = {name: sim._firm_broad_equity_proxy(name, price) for name in ('IS', 'PS')}
        sol = sim.solve_within_tick_population(allow_income_support_trigger=False)
        sim.post_tick_population(sol)
        self.assertAlmostEqual(sim.hh.unpaid_bills_i()[i], 0, places=6)
        self.assertAlmostEqual(sim.nodes['BANK'].get('equity') - before_bank, sol['retained_bk'], places=6)
        for name, suffix, investment in (('IS', 'fa', 'info'), ('PS', 'fh', 'phys')):
            self.assertAlmostEqual(sim._firm_broad_equity_proxy(name, price) - before_firms[name],
                sol['retained_' + suffix] + sim.state['hh_equity_investment_' + investment + '_total'], places=6)
        self.assertAlmostEqual(sol['housing_revenue_fa'] + sol['housing_revenue_fh'],
            sim.state['renter_rent_total'] + sim.state['owner_housing_payment_total']
            + sim.state['hh_arrears_housing_receipts_total'], places=6)
        sim._assert_sfc_ok('collection after restoring financing')

    def test_disabling_shortfall_credit_also_disables_mortgage_bridge(self):
        sim, i = test_accounting_regressions.AccountingRegressions().mortgage_shortfall_fixture()
        sim.params['hh_shortfall_financing_enabled'] = False
        sim.params['revolving_credit_limit_income_mult'] = 0  # Otherwise unlimited headroom.
        sim.step()
        self.assertEqual(sim.state['mort_revolving_bridge_total'], 0)
        self.assertEqual(sim.hh.revolving_loans[i], 0)
        self.assertGreater(sim.hh.mort_interest_arrears_q[i], 0)
        self.assertGreater(sim.hh.mort_principal_arrears_q[i], 0)
        sim._assert_sfc_ok('unfinanced mortgage payment')

    def test_unfunded_mortgage_counts_as_shortfall_even_when_financing_is_enabled(self):
        sim, i = test_accounting_regressions.AccountingRegressions().mortgage_shortfall_fixture()
        sim.hh.renter_rent_q[i] = 0
        sim.step()
        self.assertEqual(sim.hh.revolving_loans[i], 0)
        self.assertGreater(sim.hh.mort_interest_arrears_q[i], 0)
        self.assertTrue(sim.hh.ever_payment_shortfall[i])

    def test_automation_run_continues_beyond_first_failure_without_new_shortfall_credit(self):
        cfg = config()
        cfg['parameters'].update(economic_regime='OldToNew', old_to_new_transition_mode='AutomationOnly',
                                 hh_shortfall_financing_enabled=False)
        run = run_simulation(100, cfg)
        first = next(row['t'] for row in run.rows if row['hh_payment_shortfall_share'] > 0)
        self.assertLess(first, 99)
        self.assertEqual(len(run.rows), 100)
        self.assertGreater(run.rows[-1]['hh_unpaid_bills_per_h'], 0)
        self.assertTrue(all(row['hh_overdraft_to_revolving_per_h'] == 0 for row in run.rows))
        self.assertTrue(all(row['hh_mortgage_bridge_to_revolving_per_h'] == 0 for row in run.rows))
        self.assertGreater(run.rows[-1]['real_consumption'], 0)

    def test_new_cash_and_distress_diagnostics_use_correct_display_units(self):
        row = {'price_level': 2., 'hh_unpaid_bills_per_h': 100.,
               'hh_unpaid_bills_paid_per_h': 10., 'hh_in_arrears_share': .25}
        real = _rows_for_value_mode([row], 'real')[0]
        self.assertEqual(real['hh_unpaid_bills_per_h'], 50.)
        self.assertEqual(real['hh_unpaid_bills_paid_per_h'], 5.)
        self.assertEqual(real['hh_in_arrears_share'], .25)

    def test_financing_toggle_defaults_on_and_accepts_off(self):
        cfg = config()
        self.assertTrue(cfg['parameters']['hh_shortfall_financing_enabled'])
        actual = _build_cfg_from_state(SimpleNamespace(session_state={'param__hh_shortfall_financing_enabled': False}), cfg)
        self.assertFalse(actual['parameters']['hh_shortfall_financing_enabled'])


if __name__ == '__main__':
    unittest.main()
