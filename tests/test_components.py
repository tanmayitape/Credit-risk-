import numpy as np
import pandas as pd
import pytest

from credit_risk import fairness, monitoring
from credit_risk.data import DAYS_EMPLOYED_SENTINEL, TARGET, make_synthetic
from credit_risk.features import PROTECTED_EXCLUDED, align_categories, engineer, split_xy
from credit_risk.model import DecisionPolicy, evaluate


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(3000, seed=1)


def test_synthetic_matches_home_credit_quirks(raw):
    assert 0.04 < raw[TARGET].mean() < 0.15
    assert (raw["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()
    assert raw["EXT_SOURCE_1"].isna().mean() > 0.4
    assert (raw["DAYS_BIRTH"] < 0).all()


def test_engineer_handles_pensioner_sentinel(raw):
    df = engineer(raw)
    pensioners = df["IS_PENSIONER"] == 1
    assert df.loc[pensioners, "YEARS_EMPLOYED"].isna().all()
    assert (df.loc[~pensioners, "YEARS_EMPLOYED"] >= 0).all()
    assert df["AGE_YEARS"].between(18, 75).all()


def test_protected_attribute_never_reaches_model(raw):
    X, y, audit = split_xy(engineer(raw))
    for col in PROTECTED_EXCLUDED:
        assert col not in X.columns
    assert TARGET not in X.columns
    assert set(audit.columns) == {"CODE_GENDER", "AGE_BAND"}
    assert len(y) == len(X) == len(audit)


def test_align_categories_maps_unseen_levels_to_nan(raw):
    X, _, _ = split_xy(engineer(raw))
    new = X.head(5).copy()
    new["OCCUPATION_TYPE"] = new["OCCUPATION_TYPE"].cat.add_categories("Astronaut")
    new.loc[new.index[0], "OCCUPATION_TYPE"] = "Astronaut"
    aligned = align_categories(new, X.head(0))
    assert list(aligned["OCCUPATION_TYPE"].cat.categories) == list(X["OCCUPATION_TYPE"].cat.categories)
    assert pd.isna(aligned["OCCUPATION_TYPE"].iloc[0])


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert monitoring.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert monitoring.psi(a, rng.normal(0, 1, 5000)) < monitoring.PSI_AMBER
    assert monitoring.psi(a, rng.normal(1, 1, 5000)) > monitoring.PSI_RED


def test_psi_categorical_new_level():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 50 + ["c"] * 50)
    assert monitoring.psi_categorical(ref, cur) > monitoring.PSI_RED


def test_decision_policy_hits_reject_rate():
    p = np.linspace(0, 1, 1001)
    policy = DecisionPolicy.from_reject_rate(p, 0.2)
    assert policy.decide(p).mean() == pytest.approx(0.2, abs=0.01)


def test_group_metrics_adverse_impact():
    groups = pd.Series(["A"] * 100 + ["B"] * 100)
    decline = np.r_[np.zeros(90), np.ones(10), np.zeros(60), np.ones(40)]
    y = np.r_[np.zeros(95), np.ones(5), np.zeros(80), np.ones(20)]
    t = fairness.group_metrics(y, np.linspace(0, 1, 200), decline, groups)
    assert t.loc["A", "approval_rate"] == pytest.approx(0.9)
    assert t.loc["B", "adverse_impact_ratio"] == pytest.approx(0.6 / 0.9, abs=1e-3)


def test_evaluate_keys():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert m["auc"] == pytest.approx(0.75)
    assert m["gini"] == pytest.approx(0.5)
