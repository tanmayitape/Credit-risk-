"""End-to-end run: train, evaluate, explain, audit fairness, check stability.

    python -m credit_risk.pipeline --data-dir data/raw --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import explain, fairness, monitoring
from .data import ID_COL, TARGET, load_or_synthesize
from .features import AUDIT_ATTRIBUTES, align_categories, engineer, split_xy
from .model import DecisionPolicy, calibration_table, evaluate, predict_pd, train_lgbm, train_logistic


def run(
    data_dir: str | Path = "data/raw",
    out_dir: str | Path = "reports",
    reject_rate: float = 0.15,
    n_synthetic: int = 20_000,
    n_waterfalls: int = 3,
    shap_sample: int = 5_000,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for stale in fig_dir.glob("*.png"):
        stale.unlink()

    raw, current_raw, source = load_or_synthesize(data_dir, n_synthetic=n_synthetic, seed=seed)
    df = engineer(raw)

    # 60 / 20 / 20 stratified split. Validation is for early stopping and the
    # decision threshold; test is touched once for the reported numbers.
    train_df, temp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    valid_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp[TARGET], random_state=seed)

    X_tr, y_tr, _ = split_xy(train_df)
    X_va, y_va, _ = split_xy(valid_df)
    X_te, y_te, audit_te = split_xy(test_df)
    X_va = align_categories(X_va, X_tr)
    X_te = align_categories(X_te, X_tr)

    # --- Models -----------------------------------------------------------
    gbm = train_lgbm(X_tr, y_tr, X_va, y_va, seed=seed)
    logit = train_logistic(X_tr, y_tr)

    pd_va = predict_pd(gbm, X_va)
    pd_te = predict_pd(gbm, X_te)
    metrics = {
        "lightgbm": evaluate(y_te, pd_te),
        "logistic_baseline": evaluate(y_te, predict_pd(logit, X_te)),
        "best_iteration": int(gbm.best_iteration_ or gbm.n_estimators),
    }
    calib = calibration_table(y_te, pd_te)

    policy = DecisionPolicy.from_reject_rate(pd_va, reject_rate)
    decline_te = policy.decide(pd_te)

    # --- Explainability ---------------------------------------------------
    exp = explain.Explainer(gbm)
    sv_te = exp.explain(X_te)
    importance = explain.Explainer.global_importance(sv_te)
    shap_check = float(np.max(np.abs(sv_te.base_values + sv_te.values.sum(axis=1) - _logit(pd_te))))

    rng = np.random.default_rng(seed)
    idx_plot = rng.choice(len(X_te), size=min(shap_sample, len(X_te)), replace=False)
    explain.plot_global(sv_te[idx_plot], fig_dir)
    for feat in importance["feature"].head(3):
        explain.plot_dependence(sv_te[idx_plot], feat, fig_dir / f"shap_dependence_{feat}.png")

    reasons = explain.reason_codes(sv_te, X_te, test_df[ID_COL], pd_te, decline_te)
    reasons.to_csv(out / "reason_codes_test.csv", index=False)

    # Waterfalls: highest-risk declines, a borderline decline, and a clear approval.
    order = np.argsort(-pd_te)
    border = int(np.argmin(np.where(decline_te == 1, pd_te - policy.threshold, np.inf)))
    examples = {"high_risk": list(order[:n_waterfalls]), "borderline": [border], "approved": [int(order[-1])]}
    waterfalls = []
    for label, rows in examples.items():
        for i in rows:
            app_id = int(test_df[ID_COL].iloc[i])
            p = explain.plot_waterfall(
                sv_te[i], fig_dir / f"waterfall_{label}_{app_id}.png",
                f"Applicant {app_id}: PD {pd_te[i]:.1%} ({'decline' if decline_te[i] else 'approve'})",
            )
            waterfalls.append(p.name)

    # --- Fairness ---------------------------------------------------------
    fair_tables, fair_summary = {}, {}
    for attr in AUDIT_ATTRIBUTES:
        t = fairness.group_metrics(y_te, pd_te, decline_te, audit_te[attr])
        fair_tables[attr] = t
        fair_summary[attr] = fairness.summary(t)
        t.to_csv(out / f"fairness_{attr.lower()}.csv")
    fair_summary["CODE_GENDER"]["proxy_auc"] = fairness.proxy_auc(X_te, audit_te["CODE_GENDER"], seed=seed)
    gender_shap = fairness.shap_by_group(sv_te.values, sv_te.feature_names, audit_te["CODE_GENDER"])

    # --- Monitoring -------------------------------------------------------
    monitor = {}
    psi_table = None
    if current_raw is not None:
        cur = engineer(current_raw)
        X_cur, _, _ = split_xy(cur)
        X_cur = align_categories(X_cur, X_tr)
        pd_cur = predict_pd(gbm, X_cur)
        monitor["score"] = monitoring.score_stability(pd_va, pd_cur)
        monitor["current_decline_rate"] = round(float(policy.decide(pd_cur).mean()), 4)
        psi_table = monitoring.stability_report(X_va, X_cur)
        psi_table.to_csv(out / "feature_psi.csv", index=False)
        monitor["features_red"] = psi_table.loc[psi_table.status == "red", "feature"].tolist()
        monitor["features_amber"] = psi_table.loc[psi_table.status == "amber", "feature"].tolist()
        cur_idx = rng.choice(len(X_cur), size=min(shap_sample, len(X_cur)), replace=False)
        attr_drift = monitoring.attribution_drift(
            sv_te.values[idx_plot], exp.explain(X_cur.iloc[cur_idx]).values, sv_te.feature_names
        )
        attr_drift.to_csv(out / "shap_attribution_drift.csv", index=False)

    # An empty frame keeps column order and category levels for scoring new data.
    joblib.dump({"model": gbm, "policy": policy, "schema": X_tr.head(0)}, out / "model.joblib")

    result = {
        "data_source": source,
        "n_train": len(X_tr),
        "n_valid": len(X_va),
        "n_test": len(X_te),
        "metrics": metrics,
        "decision_threshold_pd": round(policy.threshold, 4),
        "target_reject_rate": reject_rate,
        "test_decline_rate": round(float(decline_te.mean()), 4),
        "shap_additivity_max_error": shap_check,
        "fairness": fair_summary,
        "monitoring": monitor,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    _write_report(out, result, importance, calib, fair_tables, gender_shap, psi_table, reasons, waterfalls)
    return result


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _md(df: pd.DataFrame, index: bool = True) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].map("{:,}".format)
    return df.to_markdown(index=index, floatfmt=".4f")


def _write_report(out, r, importance, calib, fair_tables, gender_shap, psi_table, reasons, waterfalls):
    m = r["metrics"]
    lines = [
        "# Credit risk model report",
        "",
        f"Data source: **{r['data_source']}** · train/valid/test = {r['n_train']:,} / {r['n_valid']:,} / {r['n_test']:,}",
        "",
        "## Performance (held-out test set)",
        "",
        _md(pd.DataFrame([m["lightgbm"], m["logistic_baseline"]], index=["LightGBM", "Logistic baseline"])),
        "",
        f"Decision policy: decline if PD ≥ {r['decision_threshold_pd']:.2%} "
        f"(set on validation for a {r['target_reject_rate']:.0%} reject rate; test decline rate {r['test_decline_rate']:.1%}).",
        "",
        "### Calibration by PD decile",
        "",
        _md(calib),
        "",
        "## Explainability (SHAP)",
        "",
        f"SHAP additivity check: max |base + Σ SHAP − logit(PD)| = {r['shap_additivity_max_error']:.2e}",
        "",
        _md(importance.head(15)[["feature", "description", "mean_abs_shap", "share"]], index=False),
        "",
        "![SHAP beeswarm](figures/shap_beeswarm.png)",
        "",
        "### Example reason codes (declined applicants)",
        "",
        _md(reasons[reasons.decision == "decline"].sort_values("pd", ascending=False).head(5), index=False),
        "",
        "Waterfall plots: " + ", ".join(f"[{w}](figures/{w})" for w in waterfalls),
        "",
        "## Fairness audit",
        "",
        "`CODE_GENDER` is excluded from model inputs and used only for this audit.",
        "",
    ]
    for attr, t in fair_tables.items():
        lines += [f"### {attr}", "", _md(t), ""]
    g = r["fairness"]["CODE_GENDER"]
    lines += [
        f"Proxy check: model inputs predict gender with AUC **{g['proxy_auc']:.3f}** (0.5 = no proxy signal).",
        "",
        "Mean SHAP contribution by gender (features driving the PD gap):",
        "",
        _md(gender_shap),
        "",
    ]
    if r["monitoring"]:
        mon = r["monitoring"]
        lines += [
            "## Stability monitoring (validation vs current population)",
            "",
            f"Score PSI **{mon['score']['score_psi']:.3f}** ({mon['score']['status']}); mean PD "
            f"{mon['score']['ref_mean_pd']:.2%} → {mon['score']['cur_mean_pd']:.2%}; "
            f"decline rate at the fixed threshold would be {mon['current_decline_rate']:.1%}.",
            "",
            _md(psi_table.head(12), index=False),
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--reject-rate", type=float, default=0.15)
    ap.add_argument("--n-synthetic", type=int, default=20_000, help="rows to generate if real data is absent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    result = run(args.data_dir, args.out, args.reject_rate, args.n_synthetic, seed=args.seed)
    print(json.dumps({k: result[k] for k in ("data_source", "metrics", "fairness", "monitoring")}, indent=2, default=float))
    print(f"\nReport written to {Path(args.out) / 'REPORT.md'}")


if __name__ == "__main__":
    main()
