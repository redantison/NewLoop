# NewLoop Streamlit App

NewLoop is a simulation app built with Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Developer Note

When running Python checks in the `perpetual` environment, prefer the repo-local wrapper so bytecode goes to `/tmp` instead of `__pycache__` directories in the source tree:

```bash
bash scripts/perpetual-python -B -m unittest
bash scripts/perpetual-python -B -m py_compile newloop/engine.py
```

## Accounting conventions

Household ownership is recorded separately for IS, PS, and BANK. New equity
purchases credit the paying households, and trust purchases pay the actual
sellers. Dividends use the previous quarter's ownership; new shares participate
starting in the following quarter.

Corporate revenue includes housing-service receipts. Nonhousing production
costs and capacity remain attached to nonhousing output. After-tax earnings
deduct actual depreciation of both installed capital and legacy capacity.
Retained earnings equal earnings minus dividends and may be negative. Cash
profit is recorded separately so dividend planning can reserve maintenance
spending without subtracting depreciation twice. Tax depreciation continues to
use the configured tax allowance, which can differ from economic depreciation.

Mortgage interest uses accrual accounting with matching BANK.interest_receivable
and HH.interest_payable stocks, backed by the individual household arrears.
Collection reduces those claims without recognizing income again. Required but
unpaid principal is marked overdue within the existing loan balance, not added
as another liability. Refinancing principal does not erase unpaid interest.
Interest claims are carried at face value; there is currently no default or
impairment policy. Contractual relief in NewLoop reduces the household's interest
obligation; any bank compensation is separate income when paid.

Bank corporate tax uses accrued income. The next dividend commitment uses
collected income after tax, subject to the existing equity limit. Cash income,
accrued mortgage interest, collected mortgage interest, and interest arrears are
exported separately. Bank balance-sheet checks include accrued receivables.

The NewLoopPolicies transition explicitly launches the trust. The mortgage
stress trigger remains available for direct NewLoop runs. OldLoop and
AutomationOnly continue to disable the trust and compensating policies.

All real monetary displays use nominal value divided by the current price level,
including population distributions. Physical quantities and dimensionless ratios
are not deflated. Deposit drawdown is the sum of each household's positive fall
in deposits during the quarter; another household's saving does not offset it.

Run all tests and the independent 120-quarter accounting reconciliations from
the repository root:

```bash
bash scripts/perpetual-python -B -m unittest discover -s tests -v
bash scripts/perpetual-python -B scripts/audit_accounting.py --quarters 120 --output /tmp/newloop-accounting-audit
```

The audit runs all four experiment modes with their default population and
startup settings. It checks cash movements, earnings, changes in equity, bank
balance sheets, and interest claims, and saves raw rows and diagnostics. Inspect
the startup convergence flags separately: accounting consistency does not imply
that the economy has reached a steady state.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set the main file path to `app.py`.

## Files

- `app.py`: Streamlit wrapper entrypoint
- `requirements.txt`: Python dependencies and editable package install
- `newloop/slnewloop.py`: Streamlit app module
- Core model modules: `newloop/engine.py`, `newloop/results.py`, and related files
