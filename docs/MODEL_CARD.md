# Model card: consumer loan probability of default (PD)

## Purpose

Estimate the probability that an applicant for a consumer cash or revolving loan
will have payment difficulties (Home Credit `TARGET = 1`), and support an
approve or decline decision with a reason for each outcome. It is a portfolio
project, not a production model.

## Data

- Home Credit Default Risk (Kaggle), `application_train.csv` (307,511 applications,
  about 8% positive). `bureau.csv` aggregates are joined if present.
- `application_test.csv` (unlabelled) is used as the "current" population for
  stability monitoring.
- Known limitations: one lender, one country, no application dates (so no
  out-of-time validation is possible), and a label defined by the lender
  ("payment difficulties") rather than a regulatory default definition.

## Model

- LightGBM, early-stopped on a stratified validation split (60/20/20
  train/valid/test). No class reweighting, so outputs stay calibrated PDs.
- Monotone constraints: PD can only fall as bureau scores (`EXT_SOURCE_*`) rise,
  and only rise as repayment-to-income and social-circle defaults rise. This
  keeps reason codes intuitive and is a common validation requirement.
- Benchmark: L2 logistic regression on the same features (a scorecard proxy).
  If the GBM does not clearly beat it, the simpler model should be preferred.
- Decision: decline if PD ≥ threshold. The threshold is set on the validation
  set to hit a target reject rate (default 15%), as a stand-in for a
  profit/loss-based cut-off.

## Explainability

- `shap.TreeExplainer` (exact, path-dependent) in log-odds space. The pipeline
  checks that `base value + Σ SHAP = logit(PD)` for every test applicant.
- **Global**: mean |SHAP| ranking, beeswarm, and dependence plots for the top features.
- **Local**: a waterfall per example applicant, plus `reason_codes_test.csv` with
  the top four risk-increasing factors and the main mitigating factor in plain
  English for every test applicant.
- Use: reason codes support the "principal reasons" an applicant is entitled to
  under UK GDPR Art. 22 and ICO/Turing *Explaining decisions made with AI*
  guidance, the FCA Consumer Duty's consumer-understanding outcome, and
  US-style adverse action notices (ECOA / Reg B).
- Caveat: SHAP explains the *model*, not the world. Correlated features (the
  three `EXT_SOURCE_*` scores and their mean/min) share credit, so a single
  feature's SHAP value should not be read as a causal effect. Reason codes
  should be grouped by concept before being sent to a customer.

## Fairness note

**What is controlled.** `CODE_GENDER` (sex is a protected characteristic under
the Equality Act 2010) is never a model input. It is held aside and used only
for the audit. Age *is* an input. The Equality Act's age exceptions for
financial services allow age-based risk assessment when it is backed by
relevant, accurate data. It is still audited because younger applicants are
systematically declined more often.

**What is measured** (test set, at the actual decision threshold, per group):

| Metric | Question it answers |
|---|---|
| Approval rate and adverse impact ratio | Do groups get credit at similar rates? A ratio < 0.8 (the "four-fifths" rule of thumb) is flagged. |
| Good-decline rate (FPR on repayers) | Of people who *would have repaid*, are some groups wrongly declined more? This is the most direct measure of harm. |
| Bad-capture rate (TPR on defaulters) | Is the model equally effective at catching risk in each group? |
| AUC per group | Is ranking quality similar across groups? |
| Calibration gap (mean PD − observed rate) | Does a PD of 10% mean 10% for everyone? |
| Proxy AUC | How well can the model's inputs reconstruct gender? About 0.5 means little proxy signal. Higher means excluding the attribute did not remove its influence. |
| Mean SHAP by group | Which features drive any gap in predicted risk. |

**How to read it.** These criteria conflict. When base rates differ, a
calibrated model cannot also equalise error rates (Kleinberg et al.,
Chouldechova). For credit, the defensible default is to *require* calibration
within groups (so prices and decisions reflect real risk), *monitor* approval
and good-decline gaps, and *investigate* any gap that is not explained by
legitimate risk factors. Mitigation options in order of preference: remove or
replace proxy features with no business justification; per-group threshold
review (legally sensitive, needs sign-off); constrained training (e.g.
`fairlearn` reductions). Each option should be documented, including its cost
in AUC.

**Not covered.** Intersectional groups (e.g. young women), other protected
characteristics absent from the data (ethnicity, disability), and the
Consumer Duty's focus on vulnerable customers. These need data the dataset
does not have.

## Monitoring plan

| What | How | Trigger |
|---|---|---|
| Score distribution | PSI of PD, reference = validation set | > 0.10 investigate, > 0.25 escalate / consider recalibration |
| Input drift | PSI per feature (CSI), categorical levels included, missing values as their own bin | Same bands. New category levels are mapped to missing and logged. |
| Decision rate | Decline rate at the fixed threshold vs target | ± 5 pp from target |
| Explanation drift | Change in each feature's share of total \|SHAP\| | Any feature moving > 5 pp |
| Performance | AUC/Gini, KS, and calibration by decile on matured cohorts | Gini drop > 5 points, or calibration-in-the-large off by > 20% relative |
| Fairness | Rerun the group table each cycle | Adverse impact ratio < 0.8, or good-decline gap widening |

Labels mature slowly: a 12-month default flag is not known until a year after
booking. So the leading indicators (PSI, decline rate, SHAP drift, early
delinquency such as 30+ DPD at 3 months) carry the monitoring until outcomes
arrive. Declined applicants never get labels, so performance on the approved
book is biased (reject inference). Note this in validation, and consider a
small randomised approve band if risk appetite allows.

These controls follow the PRA's SS1/23 *Model risk management principles*:
documented ownership, independent validation, and ongoing performance
monitoring with defined thresholds.
