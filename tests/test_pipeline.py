import json

import numpy as np
import pytest

from credit_risk.data import make_synthetic
from credit_risk.pipeline import run
from credit_risk.scoring import load_bundle, score_applications


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    # Empty data dir -> synthetic data path.
    data_dir = tmp_path_factory.mktemp("data")
    return run(data_dir, out, n_synthetic=6000, shap_sample=500, seed=0), out


def test_model_beats_random_and_is_calibrated(result):
    r, _ = result
    m = r["metrics"]["lightgbm"]
    assert m["auc"] > 0.65
    assert abs(m["mean_pd"] - m["default_rate"]) < 0.03


def test_shap_values_are_additive(result):
    r, _ = result
    assert r["shap_additivity_max_error"] < 1e-6


def test_artifacts_written(result):
    r, out = result
    for f in ["REPORT.md", "metrics.json", "reason_codes_test.csv", "fairness_code_gender.csv",
              "feature_psi.csv", "model.joblib", "figures/shap_beeswarm.png"]:
        assert (out / f).exists(), f
    assert json.loads((out / "metrics.json").read_text())["data_source"] == "synthetic"


def test_monitoring_detects_simulated_drift(result):
    r, _ = result
    assert r["monitoring"]["score"]["cur_mean_pd"] > r["monitoring"]["score"]["ref_mean_pd"]
    assert r["monitoring"]["features_red"] or r["monitoring"]["features_amber"]


def test_score_new_applications_with_reason_codes(result):
    _, out = result
    bundle = load_bundle(out / "model.joblib")
    apps = make_synthetic(50, seed=99).drop(columns=["TARGET"])
    scored = score_applications(bundle, apps)
    assert len(scored) == 50
    assert scored["pd"].between(0, 1).all()
    assert set(scored["decision"]) <= {"approve", "decline"}
    declined = scored[scored.decision == "decline"]
    if len(declined):
        assert declined["reason_1"].notna().all()
