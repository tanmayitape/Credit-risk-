"""Model training, evaluation and decision thresholds."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    n_estimators=3000,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    # Monotone constraints are set per feature in train_lgbm; no class weighting,
    # so predicted probabilities stay calibrated PDs.
    verbose=-1,
)

# Business-sense monotonicity: higher bureau scores must never increase risk and
# a heavier repayment burden must never decrease it. This makes SHAP reason codes
# consistent and is something model validators routinely ask for.
MONOTONE = {
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    "EXT_SOURCE_MEAN": -1,
    "EXT_SOURCE_MIN": -1,
    "ANNUITY_TO_INCOME": 1,
    "DEF_30_CNT_SOCIAL_CIRCLE": 1,
}


def train_lgbm(X_train, y_train, X_valid, y_valid, seed: int = 0) -> lgb.LGBMClassifier:
    constraints = [MONOTONE.get(c, 0) for c in X_train.columns]
    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, monotone_constraints=constraints)
    # LightGBM >= 4.6 renamed eval_set to eval_X / eval_y.
    if "eval_X" in inspect.signature(model.fit).parameters:
        eval_kwargs = dict(eval_X=(X_valid,), eval_y=(y_valid,))
    else:
        eval_kwargs = dict(eval_set=[(X_valid, y_valid)])
    model.fit(
        X_train,
        y_train,
        **eval_kwargs,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_logistic(X_train: pd.DataFrame, y_train) -> Pipeline:
    """Scorecard-style baseline: impute, scale / one-hot, L2 logistic regression."""
    cat_cols = [c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols),
            ("cat", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
            ), cat_cols),
        ]
    )
    X = X_train.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=2000))])
    pipe.fit(X, y_train)
    return pipe


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, Pipeline):
        X = X.copy()
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].astype(object)
    return model.predict_proba(X)[:, 1]


def ks_statistic(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def evaluate(y_true, pd_hat) -> dict:
    auc = roc_auc_score(y_true, pd_hat)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, pd_hat), 4),
        "pr_auc": round(float(average_precision_score(y_true, pd_hat)), 4),
        "brier": round(float(brier_score_loss(y_true, pd_hat)), 5),
        "mean_pd": round(float(np.mean(pd_hat)), 4),
        "default_rate": round(float(np.mean(y_true)), 4),
    }


def calibration_table(y_true, pd_hat, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": pd_hat})
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), bins, labels=False) + 1
    return df.groupby("decile").agg(
        n=("y", "size"), predicted_pd=("pd", "mean"), observed_rate=("y", "mean")
    ).round(4)


@dataclass
class DecisionPolicy:
    """Approve applicants whose predicted PD is below ``threshold``."""

    threshold: float

    @classmethod
    def from_reject_rate(cls, pd_hat, reject_rate: float) -> "DecisionPolicy":
        return cls(threshold=float(np.quantile(pd_hat, 1 - reject_rate)))

    def decide(self, pd_hat) -> np.ndarray:
        """1 = decline, 0 = approve."""
        return (np.asarray(pd_hat) >= self.threshold).astype(int)
