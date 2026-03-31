import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newloop.slnewloop import (
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


if __name__ == "__main__":
    unittest.main()
