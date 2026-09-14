"""Run the four default experiments and independently reconcile posted accounts.

Usage: python scripts/audit_accounting.py --output /path/to/results --quarters 120
The output includes raw nominal rows, invariant histories, and a summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newloop.config import get_default_config
from newloop.engine import NewLoop
from newloop.results import run_simulation
from newloop.slnewloop import _apply_loop_mode_mortgage_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quarters', type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_post = NewLoop.post_tick_population
    records = []

    def post(sim, sol):
        price = sim.state['price_level']
        before_cash = {issuer: sim.nodes[issuer].get('deposits') for issuer in ('IS', 'PS')}
        before_equity = {issuer: sim._firm_broad_equity_proxy(issuer, price) for issuer in ('IS', 'PS')}
        before_bank = sim.nodes['BANK'].get('equity')
        result = original_post(sim, sol)
        record = dict(internal_t=sim.state['t'])
        for issuer, suffix, investment in [('IS', 'fa', 'info'), ('PS', 'fh', 'phys')]:
            investment_cash = sim.state['hh_equity_investment_' + investment + '_total']
            operating_cash = (sol['rev_' + suffix] - sol['w_' + suffix] - sol[suffix + '_interest']
                              - sol['overhead_' + suffix] - sol['input_cost_' + suffix] - sol['corp_tax_' + suffix])
            record[issuer + '_cash_gap'] = sim.nodes[issuer].get('deposits') - before_cash[issuer] - (
                operating_cash - sol['div_' + suffix + '_total'] - sol['capex_' + suffix + '_nom'] + investment_cash)
            record[issuer + '_profit_gap'] = sol['p_' + suffix] - (operating_cash - sol['depreciation_' + suffix])
            record[issuer + '_equity_gap'] = sim._firm_broad_equity_proxy(issuer, price) - before_equity[issuer] - (sol['retained_' + suffix] + investment_cash)
        record['bank_earnings_gap'] = sim.nodes['BANK'].get('equity') - before_bank - sol['retained_bk']
        record['housing_revenue_gap'] = (sol['housing_revenue_fa'] + sol['housing_revenue_fh']
            - sim.state['renter_rent_total'] - sim.state['owner_housing_payment_total'])
        record['interest_claim_gap'] = sim.nodes['BANK'].get('interest_receivable') - float(sim.hh.mort_interest_arrears_q.sum())
        record['max_solver_delta'] = sol['solver_max_delta']
        records.append(record)
        sim._assert_sfc_ok('accounting audit')
        for key, gap in record.items():
            if key.endswith('_gap') and abs(gap) > 1e-5:
                raise AssertionError(f"{key}={gap} at t={sim.state['t']}")
        return result

    summary = {}
    NewLoop.post_tick_population = post
    try:
        for mode in ('OldLoop', 'AutomationOnly', 'NewLoopPolicies', 'MortgagePolicy'):
            print('START', mode, flush=True)
            records = []
            cfg = get_default_config()
            cfg['parameters'].update(economic_regime='OldToNew', hard_assert_sfc=True,
                old_to_new_transition_mode={'OldLoop': 'StayOldLoop', 'MortgagePolicy': 'StayOldLoop'}.get(mode, mode))
            _apply_loop_mode_mortgage_defaults(cfg['parameters'], mode != 'MortgagePolicy')
            run = run_simulation(args.quarters, cfg)
            invariants = run.sim.inv_history
            summary[mode] = {
                'visible_quarters': len(run.rows),
                'settlements_including_warmup': len(records),
                'max_gaps': {key: max(abs(row[key]) for row in records) for key in records[0] if key.endswith('_gap')},
                'max_invariant_gaps': {key: max(abs(row[key]) for row in invariants) for key in ('deposit_gap', 'loan_gap', 'bank_balance_gap', 'interest_claim_gap')},
                'first_trust_quarter': next((row['t'] for row in run.rows if row['trust_active']), None),
                'final_real_consumption': run.rows[-1]['real_consumption'],
                'final_price': run.rows[-1]['price_level'],
                'final_corporate_equity_nominal_per_h': run.rows[-1]['corporate_eq_total_per_h'],
                'final_corporate_equity_real_per_h': run.rows[-1]['corporate_eq_total_per_h'] / run.rows[-1]['price_level'],
                'max_unpaid_interest_per_h': max(row['hh_interest_arrears_per_h'] for row in run.rows),
            }
            (args.output / (mode + '.json')).write_text(json.dumps(dict(rows=run.rows,
                startup=run.startup_diagnostics, settlements=records, invariants=invariants,
                population_distributions=run.population_distributions), indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(mode, json.dumps(summary[mode]), flush=True)
    finally:
        NewLoop.post_tick_population = original_post


if __name__ == '__main__':
    main()
