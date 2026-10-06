# Explainable credit risk model

Predicts loan default on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset and explains every decision with SHAP. The pipeline also audits outcomes
by gender and age and monitors population drift. It aims to look like the work
a regulated lender would need to see, not just a leaderboard score.

**What it produces** (`reports/`):

- **Calibrated PDs** from a monotone-constrained LightGBM, benchmarked against a
  logistic-regression scorecard proxy. Metrics: AUC, Gini, KS, PR-AUC, Brier,
  and calibration by decile.
- **Reason codes for every applicant**: the top risk drivers in plain English,
  with SHAP waterfall plots and global beeswarm, bar and dependence plots.
- **Fairness audit**: approval rate, adverse impact ratio, good-decline rate,
  per-group AUC and calibration, a proxy-discrimination test, and SHAP by group.
- **Stability monitoring**: PSI for the score and each feature, decline-rate
  drift, and SHAP attribution drift.
- `REPORT.md` that collects all of the above, plus a saved model bundle that
  can score new applications.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the fairness note and monitoring plan.

## Quick start

```bash
pip install -e ".[dev]"

# 1. Get the data (needs a Kaggle account and API token)
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw   # application_train/test.csv, bureau.csv

# 2. Train, explain, audit, monitor
python -m credit_risk.pipeline --data-dir data/raw --out reports --reject-rate 0.15

# 3. Tests
pytest
```

**No Kaggle data?** If `data/raw/application_train.csv` is missing, the pipeline
generates a synthetic dataset with the same schema and quirks. These include
the `DAYS_EMPLOYED = 365243` pensioner sentinel, mostly-missing `EXT_SOURCE_1`
and a ~8% default rate. It also adds a deliberately drifted "current" cohort,
so the whole pipeline runs in about 15 seconds. The tests use this path.
**Numbers from synthetic data are illustrative only.** Run on the real data
before quoting any results.

### Scoring new applications

```python
import pandas as pd
from credit_risk.scoring import load_bundle, score_applications

bundle = load_bundle("reports/model.joblib")
apps = pd.read_csv("data/raw/application_test.csv").head(10)
print(score_applications(bundle, apps))   # SK_ID_CURR, pd, decision, reason_1..4, top_mitigating
```

Example reason code row:

| SK_ID_CURR | pd | decision | reason_1 | reason_2 | reason_3 |
|---|---|---|---|---|---|
| 110159 | 0.556 | decline | Average external credit bureau score = 0.264 (+0.72 log-odds) | Repayment amount relative to income = 0.512 (+0.68) | Loan amount relative to income = 8.2 (+0.60) |

## Project layout

```
src/credit_risk/
  data.py        load Home Credit (+ bureau aggregates) or generate schema-matched synthetic data
  features.py    feature engineering, protected-attribute policy, plain-English feature names
  model.py       LightGBM (monotone constraints, early stopping), logistic baseline, metrics, threshold policy
  explain.py     SHAP TreeExplainer, reason codes, waterfall/beeswarm/dependence plots
  fairness.py    group metrics, adverse impact ratio, proxy AUC, SHAP by group
  monitoring.py  PSI / CSI, score stability, SHAP attribution drift
  scoring.py     score + explain new applications from a saved bundle
  pipeline.py    end-to-end run that writes reports/REPORT.md
docs/MODEL_CARD.md   intended use, limitations, fairness note, monitoring plan
tests/               unit tests + an end-to-end pipeline test on synthetic data
```

## Design choices

- **`CODE_GENDER` is excluded from training and kept for auditing.** Dropping a
  protected attribute is not enough on its own, so the audit also checks how
  well the remaining features can predict it (the proxy AUC).
- **No class reweighting.** `is_unbalance` / `scale_pos_weight` would distort the
  probabilities. A lender needs calibrated PDs for pricing, IFRS 9 and capital,
  so imbalance is handled by choosing the threshold instead.
- **Monotone constraints** on bureau scores and affordability. They cost little
  AUC, stop the model from learning counter-intuitive local patterns, and make
  sure a customer never sees "your credit score was too high" as a decline reason.
- **Logistic baseline.** Regulated lenders still run scorecards. A GBM has to
  earn its extra complexity, and on small or near-linear data it often does not.
- **The threshold is fit on validation and the test set is used once**, so the
  reported decline rate and fairness numbers are honest out-of-sample figures.

## Talking points for interviews

- Why SHAP over LIME or permutation importance for credit decisions? It is exact
  for trees, additive in log-odds, and consistent, so the reasons add up to the
  score.
- Why can't calibration and equal error rates both hold when base rates
  differ? Which one would you prioritise for lending, and why?
- How do you monitor a model whose labels take 12 months to mature?
- What is the reject-inference problem, and how does it bias monitoring on the
  approved book?
- How does this map to the FCA Consumer Duty, UK GDPR Art. 22, and PRA SS1/23
  model risk management?
