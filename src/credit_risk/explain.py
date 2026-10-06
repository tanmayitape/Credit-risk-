"""SHAP explanations: global importance, per-applicant waterfalls and reason codes.

SHAP values are computed in log-odds space by ``TreeExplainer``, so for every
applicant ``base_value + sum(shap_values) == logit(PD)`` exactly. That additivity
is what makes them usable as adverse-action reason codes ("principal reasons for
decline"), as expected by the FCA's Consumer Duty and by GDPR Art. 22 / ICO
guidance on explaining automated decisions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .features import describe


class Explainer:
    def __init__(self, model):
        # Path-dependent TreeExplainer needs no background data and is exact for
        # tree ensembles; LightGBM handles the categorical splits natively.
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        sv = self.explainer(X)
        # Older shap versions return one set of values per class for classifiers.
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    @staticmethod
    def global_importance(sv: shap.Explanation) -> pd.DataFrame:
        imp = pd.DataFrame(
            {
                "feature": sv.feature_names,
                "mean_abs_shap": np.abs(sv.values).mean(axis=0),
            }
        )
        imp["description"] = imp["feature"].map(describe)
        imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
        return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.0f}"
    return str(value)


def reason_codes(
    sv: shap.Explanation,
    X: pd.DataFrame,
    ids: pd.Series,
    pd_hat: np.ndarray,
    decision: np.ndarray,
    top_k: int = 4,
) -> pd.DataFrame:
    """Top ``top_k`` features pushing each applicant's risk *up*, in plain English.

    For declined applicants these are the principal reasons for decline. For
    approved applicants we also report the strongest mitigating factor.
    """
    values = sv.values
    names = np.array(sv.feature_names)
    rows = []
    for i in range(len(X)):
        order = np.argsort(-values[i])
        up = [j for j in order if values[i, j] > 0][:top_k]
        down = int(np.argmin(values[i]))
        row = {
            "SK_ID_CURR": ids.iloc[i],
            "pd": round(float(pd_hat[i]), 4),
            "decision": "decline" if decision[i] else "approve",
        }
        for r, j in enumerate(up, start=1):
            row[f"reason_{r}"] = f"{describe(names[j])} = {_fmt(X.iloc[i, j])} (+{values[i, j]:.2f} log-odds)"
        row["top_mitigating"] = f"{describe(names[down])} = {_fmt(X.iloc[i, down])} ({values[i, down]:.2f} log-odds)"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_global(sv: shap.Explanation, out_dir: Path, max_display: int = 15) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("bar", "beeswarm"):
        plt.figure()
        getattr(shap.plots, kind)(sv, max_display=max_display, show=False)
        p = out_dir / f"shap_{kind}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=130, bbox_inches="tight")
        plt.close("all")
        paths.append(p)
    return paths


def plot_waterfall(sv_row: shap.Explanation, path: Path, title: str, max_display: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.waterfall(sv_row, max_display=max_display, show=False)
    plt.title(title, fontsize=10)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return path


def plot_dependence(sv: shap.Explanation, feature: str, path: Path) -> Path:
    plt.figure()
    shap.plots.scatter(sv[:, feature], show=False)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return path
