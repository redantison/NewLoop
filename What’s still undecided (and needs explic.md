What’s still undecided (and needs explicit resolution)

A) Tax policy (you’re right: it’s a whole discussion)
	1.	Whether there is any tax at all in the new regime, versus relying on:
	•	dilution (equity issuance to the FUND),
	•	dividend recycling via FUND→UBI,
	•	and (if needed) issuance for shortfalls.
	2.	Tax base (what is taxed):
	•	deposits (cash wealth / liquid wealth),
	•	income (wages + dividends + UBI?),
	•	consumption (value-added tax / sales tax),
	•	assets (equity, housing—if you ever model them),
	•	debt (penalty/fee, or none).
	3.	Incidence / who pays in population mode:
	•	proportional across all households,
	•	progressive schedule (needs brackets/parameters),
	•	only “top tail” (requires percentile cutoffs),
	•	firm-side taxes instead (profit tax, payroll tax).
	4.	Timing within the tick (order matters):
	•	pre-UBI vs post-UBI,
	•	pre-dividend vs post-dividend,
	•	before/after trust debt service,
	•	before/after overdraft conversion.
	5.	Destination of tax receipts:
	•	FUND (increases UBI capacity / accelerates debt retirement),
	•	GOV (and then GOV may pay UBI second),
	•	BANK capital / stabilization account (if you create one).
	6.	Purpose of tax in the model:
	•	finance UBI (but you already have dividend recycling + issuance),
	•	manage inequality (distributional objective),
	•	manage leverage / DTI (macroprudential objective),
	•	damp consumption / inflation (stabilization objective),
	•	retire trust debt faster (balance-sheet hygiene).

Until you decide the purpose, the “right” tax instrument is ambiguous.

⸻

B) Trust launch / debt mechanics
	1.	Kick-start loan magnitude and trigger conditions
	•	confirm trust_launch_loan is nonzero and actually executes once.
	2.	Should the loan ever be repaid?
	•	never (it’s effectively permanent base money / quasi-fiscal),
	•	repaid slowly (rule),
	•	repaid aggressively (rule),
	•	converted to equity-like claim (a different accounting treatment).
	3.	Repayment funding source
	•	taxes (if any),
	•	FUND dividends,
	•	dedicated haircut of UBI (politically hard),
	•	residual sweep (but you already turned off sweeping to GOV by default).
	4.	Debt service priority vs UBI
	•	you stated a preference: politically, pay UBI first.
	•	but you still need an explicit rule for debt service after UBI (cap? percent of inflow?).

Right now, the code shows trust interest is paid (with capitalization if short), but principal repayment rules are still more of a placeholder than a designed policy.

⸻

C) UBI policy definition
	1.	UBI sizing rule
	•	currently: income-target pool (wages_total + n*UBI constant after baseline).
	•	decide if this is the long-run rule, or just a temporary “stabilize demand” experiment.
	2.	UBI recipients
	•	per-family uniform payment (current),
	•	weighted by household size (not modeled),
	•	means-tested (needs income definition and schedule).
	3.	Fallback funding hierarchy
	•	you decided order (FUND → GOV → issuance),
	•	but you still need to decide whether GOV ever has deposits and why (tax? exogenous spending?).

⸻

D) Government sector role (GOV)
	1.	Is GOV purely a sink/source for bookkeeping (money_issued), or does it:
	•	collect taxes,
	•	spend on goods/services,
	•	hold stabilizing balances,
	•	run deficits with explicit rules?
	2.	If GOV is inactive, the “GOV deposits” step in UBI order is mostly dead code.

⸻

E) Population-mode equity ownership representation
Right now there’s a hybrid:
	•	you moved initial equity from H5 → HH in population mode (good),
	•	but you do not yet track equity ownership distribution across households—HH is aggregate.
Open decisions:

	1.	Do you want heterogeneous equity ownership (top-heavy wealth) in the population itself?
	2.	If yes: you need per-household equity holdings (arrays) or at least a parametric approximation.
	3.	If no: keeping equity aggregated in HH is fine, but then inequality dynamics are limited.

⸻

F) Trigger metric and calibration
	1.	Trigger uses p90 DTI (good), but:
	•	DTI definition (interest / (wages+UBI)) is a choice,
	•	whether to use p90 vs median vs p95 is a choice,
	•	whether to use previous quarter vs current quarter is a choice.
	2.	Calibration of threshold, loan rate, baseline wages/loans etc. is still open.

⸻

G) Printing / dashboard spec (minor but still decisions)
	1.	Column set and widths (you’re tuning this now).
	2.	Whether to print headers at top and bottom each run (you want bottom too).
	3.	Whether to print summary lines (totals, growth rates) at end.
