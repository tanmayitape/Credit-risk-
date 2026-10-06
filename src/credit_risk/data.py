"""Data loading for the Home Credit Default Risk dataset.

Real data: download from https://www.kaggle.com/c/home-credit-default-risk/data
and place ``application_train.csv`` (and optionally ``application_test.csv`` and
``bureau.csv``) in ``data/raw/``.

If the real files are missing, :func:`make_synthetic` produces a dataset with the
same column names, dtypes and quirks (the ``DAYS_EMPLOYED == 365243`` pensioner
sentinel, missing ``EXT_SOURCE_*`` values, ~8% default rate) so the full pipeline
can be run and tested without a Kaggle account.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"

# Columns we use from application_train.csv. The full file has 122; this subset
# carries almost all of the signal and keeps the reason codes readable.
APPLICATION_COLS = [
    ID_COL,
    TARGET,
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DEF_30_CNT_SOCIAL_CIRCLE",
]

DAYS_EMPLOYED_SENTINEL = 365243


def load_home_credit(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """Load application data and, if present, join bureau aggregates."""
    data_dir = Path(data_dir)
    path = data_dir / f"application_{split}.csv"
    cols = [c for c in APPLICATION_COLS if not (split == "test" and c == TARGET)]
    df = pd.read_csv(path, usecols=cols)

    bureau_path = data_dir / "bureau.csv"
    if bureau_path.exists():
        df = df.merge(aggregate_bureau(pd.read_csv(bureau_path)), on=ID_COL, how="left")
    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant summary of credit history held at other lenders."""
    b = bureau.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    agg = b.groupby(ID_COL).agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    )
    return agg.reset_index()


def make_synthetic(n: int = 20_000, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    """Generate an application table that mimics Home Credit's schema.

    ``drift`` in [0, 1] shifts the population (younger, lower income, larger loans,
    weaker bureau scores) to simulate a later production cohort for monitoring.
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
    age_years = np.clip(rng.normal(43 - 6 * drift, 11.5, n), 21, 69)
    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant"],
        size=n,
        p=[0.52, 0.23, 0.18, 0.07],
    )
    income_type = np.where(age_years > 60, "Pensioner", income_type)
    pensioner = income_type == "Pensioner"

    years_employed = np.minimum(rng.gamma(1.6, 4.0, n), age_years - 18)
    days_employed = np.where(
        pensioner, DAYS_EMPLOYED_SENTINEL, -(years_employed * 365.25).round()
    ).astype(float)

    education = rng.choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
        ],
        size=n,
        p=[0.71, 0.24, 0.035, 0.015],
    )
    edu_effect = pd.Series(education).map(
        {
            "Higher education": -0.35,
            "Secondary / secondary special": 0.0,
            "Incomplete higher": 0.1,
            "Lower secondary": 0.35,
        }
    ).to_numpy()

    income = np.exp(rng.normal(11.9 - 0.25 * drift, 0.5, n)).round(-2)
    income = income * np.where(gender == "M", 1.15, 1.0)  # observed pay gap in data
    credit = (income * rng.lognormal(0.9 + 0.3 * drift, 0.55, n)).round(-2)
    goods = (credit * rng.uniform(0.8, 1.0, n)).round(-2)
    annuity = (credit * rng.uniform(0.03, 0.07, n)).round(1)

    # Latent creditworthiness drives bureau scores and default.
    latent = rng.normal(0, 1, n)
    ext = []
    for k, miss in zip(range(3), [0.56, 0.002, 0.2]):
        s = 1 / (1 + np.exp(-(1.1 * latent + 0.025 * (age_years - 43) + rng.normal(0, 0.9, n))))
        s = np.clip(s - 0.06 * drift, 0.0, 1.0)
        s[rng.random(n) < miss] = np.nan
        ext.append(s)

    own_car = rng.choice(["Y", "N"], size=n, p=[0.34, 0.66])
    own_car_age = np.where(own_car == "Y", rng.gamma(2.0, 6.0, n).round(), np.nan)
    children = rng.poisson(0.4, n)
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n,
        p=[0.64, 0.15, 0.1, 0.06, 0.05],
    )
    region_rating = rng.choice([1, 2, 3], size=n, p=[0.1, 0.74, 0.16])
    occupation = rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", None],
        size=n,
        p=[0.18, 0.1, 0.09, 0.07, 0.06, 0.04, 0.46],
    )
    def30 = rng.poisson(0.14 + 0.1 * (latent < -1), n).astype(float)
    req_bureau = rng.poisson(1.9, n).astype(float)
    req_bureau[rng.random(n) < 0.13] = np.nan
    days_phone = -rng.gamma(1.2, 800, n).round()

    payment_burden = annuity / income
    logit = (
        -2.75
        - 0.9 * latent
        - 0.025 * (age_years - 43)
        - 0.02 * np.where(pensioner, 0, years_employed)
        + 2.5 * (payment_burden - 0.18)
        + edu_effect
        + 0.25 * (region_rating - 2)
        # Non-linear effects of the kind seen in real credit data: risk spikes for
        # very short employment and for loans far above income.
        + 0.7 * (np.where(pensioner, 99, years_employed) < 1)
        + 0.6 * (credit / income > 6)
        + 0.3 * (own_car == "N")
        + 0.25 * def30
        + 0.0002 * (days_phone + 1000)
        + 0.6 * drift
    )
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            ID_COL: np.arange(100_000, 100_000 + n) + int(drift * 1_000_000),
            TARGET: target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1]),
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n, p=[0.69, 0.31]),
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"],
                size=n,
                p=[0.89, 0.05, 0.02, 0.04],
            ),
            "DAYS_BIRTH": -(age_years * 365.25).round(),
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": -rng.uniform(0, 365.25 * 30, n).round(),
            "DAYS_ID_PUBLISH": -rng.uniform(0, 365.25 * 18, n).round(),
            "OWN_CAR_AGE": own_car_age,
            "OCCUPATION_TYPE": occupation,
            "CNT_FAM_MEMBERS": children + np.where(np.isin(family_status, ["Married", "Civil marriage"]), 2, 1),
            "REGION_RATING_CLIENT": region_rating,
            "EXT_SOURCE_1": ext[0],
            "EXT_SOURCE_2": ext[1],
            "EXT_SOURCE_3": ext[2],
            "DAYS_LAST_PHONE_CHANGE": days_phone,
            "AMT_REQ_CREDIT_BUREAU_YEAR": req_bureau,
            "DEF_30_CNT_SOCIAL_CIRCLE": def30,
        }
    )
    return df


def load_or_synthesize(data_dir: str | Path, n_synthetic: int = 20_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Return (train, current_population_or_None, source_label).

    With real data, ``application_test.csv`` (unlabelled, later applicants) is used
    as the "current" population for drift monitoring.
    """
    data_dir = Path(data_dir)
    if (data_dir / "application_train.csv").exists():
        train = load_home_credit(data_dir, "train")
        current = load_home_credit(data_dir, "test") if (data_dir / "application_test.csv").exists() else None
        return train, current, "home-credit"
    train = make_synthetic(n_synthetic, seed=seed)
    current = make_synthetic(n_synthetic // 4, seed=seed + 1, drift=1.0).drop(columns=[TARGET])
    return train, current, "synthetic"
