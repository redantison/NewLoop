"""Cash-source reporting must reconcile to paid uses and household deposit changes."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_accounting_regressions import config
from newloop.engine import NewLoop
from newloop.plotting import plot_household_cash_sources
from newloop.slnewloop import _rows_for_value_mode


CASH_USES = (
    'hh_realized_consumption_per_h', 'hh_actual_mortgage_payment_per_h',
    'hh_rev_interest_per_h', 'hh_rent_per_h', 'hh_owner_housing_payment_per_h',
    'hh_income_tax_cash_per_h', 'hh_other_debt_payments_per_h',
    'hh_unpaid_bills_paid_per_h', 'hh_equity_investment_per_h',
)


class HouseholdCashSourcesTests(unittest.TestCase):
    def test_cash_sources_reconcile_to_paid_uses_and_positive_deposit_changes(self):
        for financed, rollover, issuance in ((True, 0, 0), (False, 0, 0), (True, .75, .02), (False, .75, .02)):
            with self.subTest(financed=financed, rollover=rollover, issuance=issuance):
                cfg = config()
                cfg['parameters'].update(hh_shortfall_financing_enabled=financed,
                    revolving_rollover_share=rollover, revolving_principal_pay_rate_q=.1,
                    old_loop_household_money_issuance_rate_annual=issuance)
                sim = NewLoop(cfg)
                total_borrowing = 0
                for _ in range(32):
                    opening = sim.hh.deposits.copy()
                    opening_debt = sim.hh.revolving_loans.sum()
                    sim.step()
                    row = vars(sim.history[-1])
                    n = sim.hh.n
                    self.assertAlmostEqual(row['hh_wages_per_h'], row['wages_total'] / n)
                    self.assertAlmostEqual(row['hh_wages_per_h'] + row['hh_dividends_per_h'],
                                           row['hh_cash_income_per_h'])
                    self.assertGreaterEqual(row['hh_dividends_per_h'], 0)
                    borrowing = row['hh_new_revolving_borrowing_per_h']
                    self.assertAlmostEqual(borrowing, (sim.hh.revolving_loans.sum() - opening_debt) / n
                                           + row['household_credit_retired_per_h'])
                    sources = (row['hh_wages_per_h'] + row['hh_dividends_per_h'] + borrowing
                               + row['hh_deposit_drawdown_per_h'] + row['hh_money_issuance_per_h'])
                    saving = float(np.maximum(sim.hh.deposits - opening, 0).sum()) / n
                    self.assertAlmostEqual(sources, sum(row[key] for key in CASH_USES) + saving, places=7)
                    total_borrowing += borrowing
                if financed or rollover:
                    self.assertGreater(total_borrowing, 0)
                else:
                    self.assertEqual(total_borrowing, 0)

    def test_plot_uses_quarterly_flows_and_deflates_each_source_once(self):
        rows = [dict(t=t, price_level=2, hh_wages_per_h=100, hh_dividends_per_h=20,
                     hh_new_revolving_borrowing_per_h=10, hh_deposit_drawdown_per_h=6,
                     hh_revolving_debt_per_h=9999) for t in range(3)]
        for mode, expected in (('nominal', [100, 120, 130, 136]), ('real', [50, 60, 65, 68])):
            displayed = _rows_for_value_mode(rows, mode)
            fig = plot_household_cash_sources(displayed)
            ax = fig.axes[0]
            self.assertEqual([c.get_label() for c in ax.collections],
                             ['Wages', 'Dividends', 'New Revolving Borrowing', 'Deposit Drawdown'])
            np.testing.assert_allclose([max(c.get_paths()[0].vertices[:, 1]) for c in ax.collections], expected)
            self.assertIn('/ Household / Quarter', ax.get_ylabel())
            plt.close(fig)
        self.assertEqual(rows[0]['hh_wages_per_h'], 100)

    def test_existing_session_requires_new_data_instead_of_showing_zero_income(self):
        fig = plot_household_cash_sources([dict(t=0, hh_cash_income_per_h=100)])
        self.assertFalse(fig.axes[0].collections)
        self.assertIn('Run Model', fig.axes[0].texts[0].get_text())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
