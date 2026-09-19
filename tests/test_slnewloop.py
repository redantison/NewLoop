import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.slnewloop import (
    AUTOMATION_FINANCING_MODE_KEY,
    UBI_PERCENTILE_PARAM_KEY,
    UBI_PERCENTILE_UI_KEY,
    _sync_ubi_percentile_state,
)


class StreamlitStateTests(unittest.TestCase):
    def test_sync_ubi_percentile_prefers_visible_ui_value(self):
        state = {
            UBI_PERCENTILE_PARAM_KEY: 30.0,
            UBI_PERCENTILE_UI_KEY: 45.0,
        }

        chosen = _sync_ubi_percentile_state(state, fallback_default=30.0)

        self.assertAlmostEqual(chosen, 45.0, places=9)
        self.assertAlmostEqual(float(state[UBI_PERCENTILE_PARAM_KEY]), 45.0, places=9)
        self.assertAlmostEqual(float(state[UBI_PERCENTILE_UI_KEY]), 45.0, places=9)

    def test_sync_ubi_percentile_falls_back_to_param_or_default_when_ui_invalid(self):
        state = {
            UBI_PERCENTILE_PARAM_KEY: 55.0,
            UBI_PERCENTILE_UI_KEY: 0.0,
        }

        chosen = _sync_ubi_percentile_state(state, fallback_default=30.0)
        self.assertAlmostEqual(chosen, 55.0, places=9)

        state = {
            UBI_PERCENTILE_PARAM_KEY: 0.0,
            UBI_PERCENTILE_UI_KEY: 0.0,
        }

        chosen = _sync_ubi_percentile_state(state, fallback_default=30.0)
        self.assertAlmostEqual(chosen, 30.0, places=9)


class ShortfallFinancingUITests(unittest.TestCase):
    """Exercise Streamlit's widget lifecycle, not just a plain session-state dict."""

    def setUp(self):
        # Capture the actual run configuration without running the economy.
        runner = patch('newloop.slnewloop._run_dashboard_payload', return_value={'rows': [], 'error': ''})
        self.run_payload = runner.start()
        self.addCleanup(runner.stop)
        self.at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(self.at.exception)

    def select_mode(self, mode):
        self.at.selectbox(key='run__economic_regime_select').select(mode).run()
        self.assertFalse(self.at.exception)

    def financing_control(self):
        if self.at.session_state['run__economic_regime_select'] == 'AutomationOnly':
            return next(w for w in self.at.radio if w.label == 'Shortfall financing')
        return next(w for w in self.at.checkbox if w.label == 'Finance Household Shortfalls with New Debt')

    def run_config(self):
        next(b for b in self.at.button if b.label == 'Run Model').click().run()
        self.assertFalse(self.at.exception)
        return json.loads(self.at.session_state['last_run_cfg_json'])

    def test_financing_on_survives_radio_to_checkbox_switch(self):
        self.select_mode('AutomationOnly')
        self.assertEqual(self.financing_control().value, 'on')
        for mode in ('OldLoop', 'NewLoopPolicies', 'MortgagePolicy', 'AutomationOnly'):
            self.select_mode(mode)
            self.assertEqual(self.financing_control().value, 'on' if mode == 'AutomationOnly' else True, mode)
            self.assertTrue(self.run_config()['parameters']['hh_shortfall_financing_enabled'])

    def test_financing_off_survives_checkbox_to_radio_switch(self):
        self.select_mode('OldLoop')
        self.financing_control().set_value(False).run()
        for mode in ('AutomationOnly', 'OldLoop', 'NewLoopPolicies', 'AutomationOnly'):
            self.select_mode(mode)
            self.assertEqual(self.financing_control().value, 'off' if mode == 'AutomationOnly' else False, mode)
            self.assertFalse(self.run_config()['parameters']['hh_shortfall_financing_enabled'])

    def test_rerun_and_mode_round_trip_preserve_complete_run_config(self):
        self.select_mode('AutomationOnly')
        self.financing_control().set_value('off').run()
        self.at.slider(key='run__quarters').set_value(240)
        original = self.run_config()
        self.assertEqual(self.run_config(), original)
        self.select_mode('OldLoop')
        self.select_mode('AutomationOnly')
        self.assertEqual(self.run_config(), original)

    def test_three_choices_select_single_cases_or_explicit_comparison(self):
        self.select_mode('AutomationOnly')
        self.assertEqual(len(self.financing_control().options), 3)
        for choice in ('on', 'off', 'compare', 'off', 'on'):
            self.financing_control().set_value(choice).run()
            cfg = self.run_config()
            self.assertEqual(cfg['parameters']['hh_shortfall_financing_enabled'], choice == 'on')
            self.assertEqual(self.run_payload.call_args.kwargs['compare_financing'], choice == 'compare')
            self.assertEqual(self.at.session_state['last_run_compare_financing'], choice == 'compare')

    def test_comparison_survives_mode_switches_and_resets_to_single_financing_on(self):
        self.select_mode('AutomationOnly')
        self.financing_control().set_value('compare').run()
        for mode in ('OldLoop', 'NewLoopPolicies', 'MortgagePolicy'):
            self.select_mode(mode)
            self.assertFalse(self.financing_control().value)
            self.run_config()
            self.assertFalse(self.run_payload.call_args.kwargs['compare_financing'])
            self.select_mode('AutomationOnly')
            self.assertEqual(self.financing_control().value, 'compare')
        next(b for b in self.at.button if b.label == 'Reset').click().run()
        self.select_mode('AutomationOnly')
        self.assertEqual(self.financing_control().value, 'on')

    def test_legacy_two_run_selection_migrates_without_resetting_other_controls(self):
        self.select_mode('AutomationOnly')
        self.at.session_state['param__hh_shortfall_financing_enabled'] = False
        self.at.session_state['ui__hh_shortfall_financing_radio'] = False
        del self.at.session_state[AUTOMATION_FINANCING_MODE_KEY]
        self.at.slider(key='run__quarters').set_value(240).run()
        self.assertEqual(self.financing_control().value, 'compare')
        self.assertEqual(self.at.slider(key='run__quarters').value, 240)

    def test_reset_restores_financing_default_after_switching_modes(self):
        self.select_mode('AutomationOnly')
        self.financing_control().set_value('off').run()
        self.select_mode('OldLoop')
        next(b for b in self.at.button if b.label == 'Reset').click().run()
        self.select_mode('AutomationOnly')
        self.assertEqual(self.financing_control().value, 'on')
        self.assertTrue(self.run_config()['parameters']['hh_shortfall_financing_enabled'])


if __name__ == "__main__":
    unittest.main()
