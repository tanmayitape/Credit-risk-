"""Feature engineering and the protected-attribute policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DAYS_EMPLOYED_SENTINEL, ID_COL, TARGET

# Protected characteristics under the UK Equality Act 2010 that appear in the data.
# CODE_GENDER is never a model input; it is kept aside for the fairness audit.
# Age is used as an input (the Act's financial-services exception permits age in
# risk assessment where it is backed by relevant data) but is still audited.
PROTECTED_EXCLUDED = ["CODE_GENDER"]
AUDIT_ATTRIBUTES = ["CODE_GENDER", "AGE_BAND"]

AGE_BANDS = [0, 30, 45, 60, 200]
AGE_LABELS = ["<30", "30-44", "45-59", "60+"]

# Plain-English descriptions used to turn SHAP values into reason codes.
FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_MEAN": "Average external credit bureau score",
    "EXT_SOURCE_MIN": "Lowest external credit bureau score",
    "EXT_SOURCE_1": "External credit bureau score 1",
    "EXT_SOURCE_2": "External credit bureau score 2",
    "EXT_SOURCE_3": "External credit bureau score 3",
    "EXT_SOURCE_COUNT": "Number of external bureau scores available",
    "CREDIT_TO_INCOME": "Loan amount relative to income",
    "ANNUITY_TO_INCOME": "Repayment amount relative to income",
    "CREDIT_TERM": "Implied loan term (credit / annuity)",
    "CREDIT_TO_GOODS": "Loan amount relative to goods price",
    "INCOME_PER_PERSON": "Income per household member",
    "AGE_YEARS": "Applicant age",
    "YEARS_EMPLOYED": "Time in current employment",
    "IS_PENSIONER": "Pensioner (no current employment)",
    "EMPLOYED_TO_AGE": "Share of adult life in current job",
    "YEARS_ID_PUBLISH": "Years since ID document issued",
    "YEARS_REGISTRATION": "Years since registration change",
    "YEARS_LAST_PHONE_CHANGE": "Years since phone number change",
    "AMT_INCOME_TOTAL": "Total income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Repayment amount",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "NAME_CONTRACT_TYPE": "Product type",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
    "OWN_CAR_AGE": "Age of car",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Household size",
    "REGION_RATING_CLIENT": "Regional risk rating",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit enquiries in the last year",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults (30+ days) in social circle",
    "BUREAU_LOAN_COUNT": "Number of loans at other lenders",
    "BUREAU_ACTIVE_COUNT": "Active loans at other lenders",
    "BUREAU_DEBT_SUM": "Outstanding debt at other lenders",
    "BUREAU_MAX_OVERDUE": "Largest overdue amount at other lenders",
    "BUREAU_DAYS_CREDIT_MEAN": "Average age of other credit lines",
}

_DROP_RAW = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE"]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived features, raw day counts dropped and
    string columns cast to ``category`` (LightGBM handles them natively)."""
    out = df.copy()

    pensioner = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["IS_PENSIONER"] = pensioner.astype(int)
    days_employed = out["DAYS_EMPLOYED"].where(~pensioner)

    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["YEARS_EMPLOYED"] = -days_employed / 365.25
    out["EMPLOYED_TO_AGE"] = _safe_div(out["YEARS_EMPLOYED"], out["AGE_YEARS"])
    out["YEARS_ID_PUBLISH"] = -out["DAYS_ID_PUBLISH"] / 365.25
    out["YEARS_REGISTRATION"] = -out["DAYS_REGISTRATION"] / 365.25
    out["YEARS_LAST_PHONE_CHANGE"] = -out["DAYS_LAST_PHONE_CHANGE"] / 365.25

    out["CREDIT_TO_INCOME"] = _safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["ANNUITY_TO_INCOME"] = _safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_TERM"] = _safe_div(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    out["CREDIT_TO_GOODS"] = _safe_div(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    out["INCOME_PER_PERSON"] = _safe_div(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])

    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_COUNT"] = ext.notna().sum(axis=1)

    out = out.drop(columns=[c for c in _DROP_RAW if c in out.columns])
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype("category")
    return out


def age_band(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, bins=AGE_BANDS, labels=AGE_LABELS, right=False).astype(str)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Split an engineered frame into (model features, target, audit attributes).

    The audit frame carries protected attributes that must not reach the model.
    """
    audit = pd.DataFrame(index=df.index)
    audit["CODE_GENDER"] = df["CODE_GENDER"].astype(str)
    audit["AGE_BAND"] = age_band(df["AGE_YEARS"])

    y = df[TARGET].astype(int) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in [ID_COL, TARGET, *PROTECTED_EXCLUDED] if c in df.columns])
    return X, y, audit


def align_categories(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Give ``X`` the same columns and category levels as the training frame so
    LightGBM's categorical codes line up at scoring time."""
    X = X.reindex(columns=reference.columns).copy()
    for col in reference.columns:
        if isinstance(reference[col].dtype, pd.CategoricalDtype):
            cats = reference[col].cat.categories
            values = X[col].astype(object)
            # Levels never seen in training become missing, which LightGBM handles.
            X[col] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
    return X


def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").capitalize())
