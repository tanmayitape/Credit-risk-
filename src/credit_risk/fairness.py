"""Group fairness audit and proxy-discrimination check.

The protected attribute is not a model input, but that alone does not make a
model fair: other features can act as proxies. So we (1) measure outcomes and
error rates by group and (2) test how well the model's own inputs can recover
the protected attribute.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def group_metrics(y_true, pd_hat, decline, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "pd": np.asarray(pd_hat), "decline": np.asarray(decline), "g": groups.to_numpy()}
    )
    rows = []
    for g, d in df.groupby("g", observed=True):
        good, bad = d[d.y == 0], d[d.y == 1]
        rows.append(
            {
                "group": g,
                "n": len(d),
                "default_rate": d.y.mean(),
                "mean_pd": d.pd.mean(),
                "approval_rate": 1 - d.decline.mean(),
                # Of applicants who would have repaid, how many did we wrongly decline?
                "good_decline_rate": good.decline.mean() if len(good) else np.nan,
                # Of applicants who defaulted, how many did we catch?
                "bad_capture_rate": bad.decline.mean() if len(bad) else np.nan,
                "auc": roc_auc_score(d.y, d.pd) if d.y.nunique() == 2 else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("group")
    ref = out["approval_rate"].idxmax()
    # Adverse impact ratio vs the most-approved group; < 0.8 fails the
    # "four-fifths" rule of thumb borrowed from US employment law.
    out["adverse_impact_ratio"] = out["approval_rate"] / out.loc[ref, "approval_rate"]
    # Calibration-in-the-large per group: a fair PD should match the observed rate.
    out["calibration_gap"] = out["mean_pd"] - out["default_rate"]
    return out.round(4)


def summary(table: pd.DataFrame) -> dict:
    return {
        "min_adverse_impact_ratio": float(table["adverse_impact_ratio"].min()),
        "approval_rate_gap": float(table["approval_rate"].max() - table["approval_rate"].min()),
        "good_decline_rate_gap": float(table["good_decline_rate"].max() - table["good_decline_rate"].min()),
        "auc_gap": float(table["auc"].max() - table["auc"].min()),
        "max_abs_calibration_gap": float(table["calibration_gap"].abs().max()),
    }


def proxy_auc(X: pd.DataFrame, protected: pd.Series, seed: int = 0) -> float:
    """Cross-validated AUC for predicting a binary protected attribute from the
    model's inputs. ~0.5 means no proxy signal; well above that means the
    features encode the attribute and outcome gaps need closer scrutiny."""
    y = pd.Series(pd.factorize(protected)[0], index=protected.index)
    if y.nunique() != 2:
        top = protected.value_counts().index[0]
        y = (protected == top).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 4)


def shap_by_group(shap_values: np.ndarray, feature_names, groups: pd.Series, top: int = 8) -> pd.DataFrame:
    """Mean SHAP contribution per feature per group: shows *which* features
    drive any difference in predicted risk between groups."""
    df = pd.DataFrame(shap_values, columns=feature_names)
    df["g"] = groups.to_numpy()
    means = df.groupby("g", observed=True).mean().T
    if means.shape[1] >= 2:
        means["spread"] = means.max(axis=1) - means.min(axis=1)
        means = means.sort_values("spread", ascending=False)
    return means.head(top).round(4)
