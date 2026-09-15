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

### Equity issuance policy

Under **Price & Capital**, **Limit Equity Issuance to Financing Needs** defaults
to on and applies during both the OldLoop pre-run and the visible experiment.
Turn it off to reproduce unrestricted household subscriptions and the previous
dividend policy.

With the limit on, each nonbank sector offers equity only for a one-quarter
capital funding gap. Its capital target is maintenance plus expansion under the
existing demand rule, including queued investment and capped by installation
capacity. Existing corporate deposits, including cash labelled as a CAPEX
reserve, cover that target first. The configured operating cash buffer is
protected; subscriptions cannot exceed the capital target. Dividends may only
use cash left after current CAPEX and this capital target, so payouts cannot
create a gap to be filled by replacement equity.

Targets use opening capital stocks and lagged demand. Household offers retain
the configured sector split (30% IS / 70% PS by default); each issuer rations
buyers proportionally, without redirecting rejected offers to the other sector.
Only accepted offers reserve household cash before consumption. Settlement
rechecks the gap after operating cash flows and may accept less. Rejected cash
is never debited; cash released at settlement becomes available next quarter.
Subscriptions arrive at quarter end and fund subsequent investment. Shares
continue to use the existing issue-price and ownership rules. This is a capital
financing rule, without a new profitability test or share-trading mechanism.
The separate social-share issuance to FUND remains part of NewLoopPolicies;
the toggle governs cash-financed subscriptions by households.

**Selected Metrics** and quarterly data include the household equity purchase
budget, unfilled purchase budget, accepted investment, and equity funding gap.
The gap is measured just before subscriptions and capped by the opening offer.
An unfilled purchase budget is a flow diagnostic, not an additional cash stock:
money rejected during planning can already support consumption. Monetary
diagnostics follow the Real/Nominal display setting.

Compare both policies with independent accounting checks:

```bash
bash scripts/perpetual-python -B scripts/audit_accounting.py --limit-equity-issuance --output /tmp/newloop-equity-limited
bash scripts/perpetual-python -B scripts/audit_accounting.py --no-limit-equity-issuance --output /tmp/newloop-equity-unrestricted
```

The audit saves the full warmup history and the original convergence result.
When the policies produce different pre-run durations, their visible runs also
start from different household distributions.

### Household shortfall financing and unpaid bills

Household shortfall financing defaults to on, preserving the existing automatic
revolving loans. Turn it off to run the economy
with cash-limited household payments. The simulation continues for the requested
duration: a household's inability to pay does not stop other households or firms.
This is separate from the existing revolving principal rollover setting.

Consumption retains the existing rule based on income after required current
payments and available deposits. Current bills keep their settlement priority:
revolving interest, mortgage payment, rent, owner housing costs, then income tax.
After current payments, remaining cash can cure mortgage interest and principal
arrears, then earlier unpaid revolving interest, rent, owner costs, and income
tax, before discretionary revolving principal repayment and equity investment.
Late tax rebates or money issuance become available for new bill collections in
the next quarter's payment plan. No household can spend another's cash.

With shortfall financing off, unpaid nonmortgage bills become separate household
liabilities and matching creditor receivables, with a full collection allowance.
The allowance makes their net creditor carrying value zero until collected:
unpaid bills create no cash, net earnings, taxes on those earnings, or dividend
funding. Later cash collection releases the allowance and recognizes the
creditor's income; the household pays down its existing liability. These new
arrears do not compound or disappear. Existing revolving principal remains
outstanding and continues to incur its contractual interest. Restoring financing
does not erase earlier arrears or automatically refinance them.

Mortgage interest retains its separate existing accrual treatment described
below; unpaid principal stays within the original loan balance. The toggle also
disables the automatic mortgage payment bridge, but does not disable mortgage
originations, refinancing, or other explicitly configured lending. Mortgage case
dynamics remain deferred as noted below.

**Household Payment Distress** shows the share with a current payment shortfall,
the share currently in arrears, the share that has ever had a shortfall (including
pre-run history), and the share consuming zero this quarter. Financed shortfalls
count as shortfalls even though the bills are paid. **Accumulated Unpaid Household
Bills** shows the four new arrears stocks. **Household Funding Gap Response**
includes new unpaid bills, and actual cash uses include collections of old bills.
Monetary amounts follow the Real/Nominal selection; household shares do not.
Wealth includes the new household liabilities. Disposable income continues to
deduct contractual current obligations, so a missed payment is not income.
There is no new bankruptcy, eviction, or household survival mechanism.

Compare the two settings with the same 240-quarter Automation experiment:

```bash
bash scripts/perpetual-python -B scripts/audit_accounting.py --modes AutomationOnly --quarters 240 --finance-household-shortfalls --output /tmp/newloop-credit-on
bash scripts/perpetual-python -B scripts/audit_accounting.py --modes AutomationOnly --quarters 240 --no-finance-household-shortfalls --output /tmp/newloop-credit-off
```

The audit also reconciles the household cash flow and unpaid bill rollforward,
and verifies that no automatic shortfall loans are created when disabled.

Under **AutomationOnly → Run Controls → Shortfall financing**, choose:

- **Financing on only (one run)**, the default: the main dashboard shows automatic
  overdraft-to-revolving financing and performs one simulation.
- **Financing on, then off (two runs)**: run the financed case first, then the
  case without shortfall financing. All main charts, distributions, quarterly
  data, and the run CSV show the second, financing-off case.

The displayed case is labeled above the charts and in **Household Funding Gap
Response**. A control change does not relabel previously generated results;
they retain their actual financing label and are marked stale until rerun.
The **Shortfall Financing: On vs Off** graphs conclude the two-run output,
comparing real consumption, affected households, revolving loan and unpaid bill
balances, and money supply. Balances and money are nominal totals in this
comparison regardless of the general Real/Nominal display selection.

Both cases use the same parameters and population seed, changing only shortfall
financing. Each performs its own pre-run; their pre-run durations and convergence
status appear below the comparison. Changing display controls redraws stored
results; other experiment modes run only once and retain their ordinary financing
checkbox in **Price & Capital**. If the financed baseline fails, the financing-off
case still runs and the comparison reports the failure. If the financing-off
case fails, an error is shown; the dashboard does not substitute the financed case.

### Cash, earnings, and ownership

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

### Deferred issue: mortgage pre-run convergence

The 2026-09-14 default MortgagePolicy audits reached the 720-quarter pre-run cap
with the equity financing limit both on and off. Maximum drift over 40 quarters
was 5.10% with the limit and 4.92% without it, against a 0.1% threshold. Accounting
checks passed in both cases. The cause of this continued drift remains unresolved
and mortgage-case dynamics are deferred for a separate investigation. Reproduce
with the two audit commands above; inspect the MortgagePolicy startup diagnostics.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set the main file path to `app.py`.

## Files

- `app.py`: Streamlit wrapper entrypoint
- `requirements.txt`: Python dependencies and editable package install
- `newloop/slnewloop.py`: Streamlit app module
- Core model modules: `newloop/engine.py`, `newloop/results.py`, and related files
