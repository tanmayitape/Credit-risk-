"""Population stability monitoring.

PSI (Population Stability Index) compares the distribution of a variable in a
reference window (training/validation) with a current window. Industry
convention: < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant shift.
Applied to model inputs it is often called CSI (Characteristic Stability Index).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_AMBER = 0.10
PSI_RED = 0.25
_EPS = 1e-4


def _status(psi: float) -> str:
    return "red" if psi > PSI_RED else "amber" if psi > PSI_AMBER else "green"


def psi(reference, current, bins: int = 10) -> float:
    """PSI for a numeric variable, with quantile bins fitted on the reference.
    Missing values get their own bin."""
    ref = pd.Series(reference, dtype=float)
    cur = pd.Series(current, dtype=float)
    edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s: pd.Series) -> np.ndarray:
        counts = pd.cut(s.dropna(), edges, include_lowest=True).value_counts(sort=False).to_numpy()
        counts = np.append(counts, s.isna().sum())
        return counts / max(len(s), 1)

    return _psi_from_dists(dist(ref), dist(cur))


def psi_categorical(reference, current) -> float:
    ref = pd.Series(reference).astype(object).fillna("__missing__")
    cur = pd.Series(current).astype(object).fillna("__missing__")
    cats = sorted(set(ref) | set(cur), key=str)
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_dists(r, c)


def _psi_from_dists(r: np.ndarray, c: np.ndarray) -> float:
    r = np.clip(r, _EPS, None)
    c = np.clip(c, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def stability_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        if col not in current.columns:
            continue
        if isinstance(reference[col].dtype, pd.CategoricalDtype) or reference[col].dtype == object:
            value = psi_categorical(reference[col], current[col])
        else:
            value = psi(reference[col], current[col])
        rows.append({"feature": col, "psi": round(value, 4), "status": _status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def score_stability(ref_scores, cur_scores) -> dict:
    value = psi(ref_scores, cur_scores)
    return {
        "score_psi": round(value, 4),
        "status": _status(value),
        "ref_mean_pd": round(float(np.mean(ref_scores)), 4),
        "cur_mean_pd": round(float(np.mean(cur_scores)), 4),
    }


def attribution_drift(ref_shap: np.ndarray, cur_shap: np.ndarray, feature_names) -> pd.DataFrame:
    """Change in each feature's share of total |SHAP|. Catches cases where the
    *reasons* for decisions shift even if the score distribution looks stable."""
    ref = np.abs(ref_shap).mean(axis=0)
    cur = np.abs(cur_shap).mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "ref_share": ref / ref.sum(),
            "cur_share": cur / cur.sum(),
        }
    )
    df["change"] = df["cur_share"] - df["ref_share"]
    return df.reindex(df["change"].abs().sort_values(ascending=False).index).round(4).reset_index(drop=True)
