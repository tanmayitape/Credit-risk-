"""Score and explain new applications with a saved model bundle.

    from credit_risk.scoring import load_bundle, score_applications
    bundle = load_bundle("reports/model.joblib")
    decisions = score_applications(bundle, raw_application_rows)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import ID_COL
from .explain import Explainer, reason_codes
from .features import align_categories, engineer, split_xy
from .model import predict_pd


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def score_applications(bundle: dict, applications: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """Return PD, approve/decline decision and SHAP reason codes per application.

    ``applications`` uses the raw Home Credit column names (as in application_test.csv).
    """
    df = engineer(applications)
    X, _, _ = split_xy(df)
    X = align_categories(X, bundle["schema"])
    pd_hat = predict_pd(bundle["model"], X)
    decision = bundle["policy"].decide(pd_hat)
    sv = Explainer(bundle["model"]).explain(X)
    return reason_codes(sv, X, df[ID_COL], pd_hat, decision, top_k=top_k)
