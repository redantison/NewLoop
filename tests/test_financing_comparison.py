"""Paired dashboard runs must isolate the financing rule and retain useful results."""
import copy
import json
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_accounting_regressions import config
from newloop.plotting import plot_shortfall_financing_comparison
from newloop.slnewloop import _cfg_json, _run_dashboard_payload


class FinancingComparisonTests(unittest.TestCase):
    def automation_config(self, financed=True):
        cfg = config()
        cfg['parameters'].update(economic_regime='OldToNew', old_to_new_transition_mode='AutomationOnly',
                                 hh_shortfall_financing_enabled=financed)
        return cfg

    def test_financing_on_is_one_run_without_a_comparison(self):
        primary = {'rows': [{'t': 0}], 'error': ''}
        with patch('newloop.slnewloop._cached_run_payload', return_value=primary) as run:
            result = _run_dashboard_payload(120, _cfg_json(self.automation_config()))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result['rows'], [{'t': 0}])
        self.assertEqual(result['shortfall_comparison'], {})

    def test_comparison_runs_on_then_off_and_returns_all_off_outputs(self):
        cfg = self.automation_config(False)
        original = copy.deepcopy(cfg)
        calls, messages = [], []

        def run(quarters, cfg_json, progress_callback=None):
            actual = json.loads(cfg_json)
            financed = actual['parameters']['hh_shortfall_financing_enabled']
            calls.append((quarters, actual))
            progress_callback('Preparing startup...', 0, 100)
            return dict(rows=[{'selected': financed}], population_distributions={'financed': financed},
                        startup_diagnostics={'financed': financed}, baseline_calibration={'financed': financed},
                        support_debug={'household_count': 64, 'financed': financed}, error='')

        with patch('newloop.slnewloop._cached_run_payload', side_effect=run):
            result = _run_dashboard_payload(120, _cfg_json(cfg), lambda *args: messages.append(args))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(quarters == 120 for quarters, _ in calls))
        expected_off = json.loads(_cfg_json(original))
        expected_on = copy.deepcopy(expected_off)
        expected_on['parameters']['hh_shortfall_financing_enabled'] = True
        self.assertEqual(calls[0][1], expected_on)
        self.assertEqual(calls[1][1], expected_off)
        self.assertEqual(cfg, original)
        self.assertEqual(result['rows'], [{'selected': False}])
        for key in ('population_distributions', 'startup_diagnostics', 'baseline_calibration', 'support_debug'):
            self.assertFalse(result[key]['financed'])
        self.assertEqual(result['shortfall_comparison']['on']['rows'], [{'selected': True}])
        self.assertEqual(result['shortfall_comparison']['off']['rows'], [{'selected': False}])
        self.assertIn('Run 1 of 2 — financing on', messages[0][0])
        self.assertIn('Run 2 of 2 — financing off', messages[1][0])

    def test_other_modes_do_not_run_a_comparison(self):
        for regime, transition in (('OldToNew', 'StayOldLoop'), ('OldToNew', 'NewLoopPolicies'), ('OldLoop', 'AutomationOnly')):
            with self.subTest(regime=regime, transition=transition):
                cfg = self.automation_config(False)
                cfg['parameters'].update(economic_regime=regime, old_to_new_transition_mode=transition)
                with patch('newloop.slnewloop._cached_run_payload', return_value={'rows': [1], 'error': ''}) as run:
                    result = _run_dashboard_payload(120, _cfg_json(cfg))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(result['shortfall_comparison'], {})

    def test_primary_failure_does_not_launch_counterpart(self):
        with patch('newloop.slnewloop._cached_run_payload', return_value={'rows': [], 'error': 'failed'}) as run:
            result = _run_dashboard_payload(120, _cfg_json(self.automation_config()))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result['error'], 'failed')
        self.assertEqual(result['shortfall_comparison'], {})

    def test_baseline_failure_still_runs_off_case_and_exposes_comparison_error(self):
        off = {'rows': [{'t': 0}], 'error': '', 'support_debug': {'household_count': 64}}
        failed = {'rows': [], 'error': 'baseline failed'}
        with patch('newloop.slnewloop._cached_run_payload', side_effect=[failed, off]) as run:
            result = _run_dashboard_payload(120, _cfg_json(self.automation_config(False)))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(result['rows'], [{'t': 0}])
        self.assertEqual(result['error'], '')
        self.assertEqual(result['shortfall_comparison']['on']['error'], 'baseline failed')

    def test_off_case_failure_does_not_substitute_financed_dashboard(self):
        on = {'rows': [{'t': 0}], 'error': '', 'support_debug': {'household_count': 64}}
        failed = {'rows': [], 'error': 'off case failed'}
        with patch('newloop.slnewloop._cached_run_payload', side_effect=[on, failed]):
            result = _run_dashboard_payload(120, _cfg_json(self.automation_config(False)))
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['error'], 'off case failed')
        self.assertEqual(result['shortfall_comparison']['on']['rows'], [{'t': 0}])

    def test_actual_pair_runs_both_full_horizons_with_distinct_financing_outcomes(self):
        result = _run_dashboard_payload(100, _cfg_json(self.automation_config(False)))
        self.assertEqual(result['error'], '')
        cases = result['shortfall_comparison']
        on, off = cases['on']['rows'], cases['off']['rows']
        self.assertEqual(len(on), 100)
        self.assertEqual(len(off), 100)
        self.assertEqual(result['rows'], off)
        self.assertEqual(cases['on']['household_count'], cases['off']['household_count'])
        self.assertAlmostEqual(on[0]['real_consumption'], off[0]['real_consumption'], places=7)
        self.assertGreater(on[-1]['money_supply_total'], off[-1]['money_supply_total'])
        self.assertGreater(off[-1]['hh_unpaid_bills_per_h'], 0)
        self.assertEqual(on[-1]['hh_unpaid_bills_per_h'], 0)

    def plot_rows(self):
        return [dict(t=t, real_consumption=10+t, price_level=2,
                     money_supply_total=100+t, hh_zero_consumption_share=.1,
                     hh_in_arrears_share=.2, hh_payment_shortfall_share=.2 if t == 7 else 0,
                     hh_revolving_debt_per_h=3+t, hh_unpaid_bills_per_h=2+t) for t in (6, 7)]

    def test_plot_uses_real_consumption_nominal_aggregate_stocks_and_actual_shortfall_quarter(self):
        on, off = self.plot_rows(), self.plot_rows()
        original = copy.deepcopy((on, off))
        fig = plot_shortfall_financing_comparison(on, off, household_count=10)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 4)
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [16, 17])
        np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [90, 100])
        np.testing.assert_array_equal(fig.axes[2].lines[1].get_ydata(), [80, 90])
        np.testing.assert_array_equal(fig.axes[3].lines[0].get_ydata(), [106, 107])
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [.1, .1])
        self.assertIn('Q7', fig.axes[1].texts[0].get_text())
        self.assertEqual((on, off), original)

    def test_plot_rejects_mismatched_quarters_or_already_deflated_money(self):
        rows = self.plot_rows()
        with self.assertRaisesRegex(ValueError, 'matching visible quarters'):
            plot_shortfall_financing_comparison(rows, rows[:-1], household_count=10)
        real_rows = [dict(r, _monetary_scale=.5) for r in rows]
        with self.assertRaisesRegex(ValueError, 'raw nominal rows'):
            plot_shortfall_financing_comparison(real_rows, real_rows, household_count=10)


if __name__ == '__main__':
    unittest.main()
