"""Train, compare and audit credit-risk models.

    python -m compliance_ai.credit.train

Steps
1. Hold out 20% of applicants as an untouched test set (stratified).
2. Compare four models with 5-fold stratified cross-validation on the rest.
3. Pick each model's cost-minimising threshold from out-of-fold predictions;
   choose the model with the lowest expected cost.
4. Refit the winner, evaluate once on the test set, audit fairness, explain with SHAP.
5. Segment applicants (k-means) and flag unusual applications (Isolation Forest).
Everything is tracked in MLflow; artefacts land in models/ and reports/.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from compliance_ai.config import ROOT, load_config, resolve  # noqa: E402
from compliance_ai.credit import explain, fairness, segment  # noqa: E402
from compliance_ai.credit.data import feature_columns, load_credit_data  # noqa: E402
from compliance_ai.credit.metrics import (  # noqa: E402
    best_threshold,
    classification_report,
    expected_cost,
)
from compliance_ai.credit.models import build_pipeline, candidate_models  # noqa: E402
from compliance_ai.monitoring import build_profile, save_profile  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def _setup_mlflow(experiment: str) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def _plot_roc(oof: dict[str, np.ndarray], y: np.ndarray, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name.replace("_", " "))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (defaulters caught)",
           title="Out-of-fold ROC curves (5-fold CV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost(y, p, chosen: float, c_fa, c_fr, path) -> None:
    grid = np.arange(0.05, 0.951, 0.01)
    costs = [expected_cost(y, p, t, c_fa, c_fr) for t in grid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(chosen, ls="--", color="grey")
    ax.annotate(f"chosen threshold = {chosen:.2f}", (chosen, min(costs)),
                xytext=(10, 20), textcoords="offset points")
    ax.set(xlabel="P(default) threshold for rejection", ylabel="Expected cost per applicant",
           title="Business cost vs decision threshold (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(imp: pd.Series, path) -> None:
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values)
    ax.set(xlabel="Mean |SHAP value| (log-odds of default)", title="Global feature importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    cfg_all = load_config()
    cfg, seed = cfg_all["credit"], cfg_all["seed"]
    c_fa, c_fr = cfg["cost_false_approval"], cfg["cost_false_rejection"]
    model_dir, report_dir = resolve(cfg["model_dir"]), resolve(cfg["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_credit_data()
    numeric, categorical = feature_columns(df)
    X = df[numeric + categorical]
    y = (1 - df[cfg["target"]]).to_numpy()          # 1 = default

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    _setup_mlflow(cfg["mlflow_experiment"])
    comparison, oof = [], {}
    with mlflow.start_run(run_name=f"credit-risk-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as parent:
        mlflow.log_params({"n_train": len(X_tr), "n_test": len(X_te), "cv_folds": cfg["cv_folds"],
                           "cost_false_approval": c_fa, "cost_false_rejection": c_fr,
                           "n_numeric": len(numeric), "n_categorical": len(categorical)})

        for name, model in candidate_models(seed).items():
            pipe = build_pipeline(model, numeric, categorical)
            scores = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=1,
                                    scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                             "brier": "neg_brier_score"})
            p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            oof[name] = p_oof
            thr, cost = best_threshold(y_tr, p_oof, c_fa, c_fr)
            row = {
                "model": name,
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                "cv_brier_mean": -scores["test_brier"].mean(),
                "oof_best_threshold": thr,
                "oof_expected_cost": cost,
            }
            comparison.append(row)
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()
                                   if isinstance(v, (int, float, str, bool)) or v is None})
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model"})
            print(f"{name:20s} AUC {row['cv_roc_auc_mean']:.3f} ± {row['cv_roc_auc_std']:.3f}"
                  f"  cost {cost:.3f} @ {thr:.2f}")

        comp = pd.DataFrame(comparison).sort_values(["oof_expected_cost", "cv_roc_auc_mean"],
                                                    ascending=[True, False])
        best = comp.iloc[0]
        best_name, threshold = best["model"], float(best["oof_best_threshold"])
        print(f"\nSelected: {best_name} (threshold {threshold:.2f})")

        final = build_pipeline(clone(candidate_models(seed)[best_name]), numeric, categorical)
        final.fit(X_tr, y_tr)
        p_te = final.predict_proba(X_te)[:, 1]
        test = classification_report(y_te, p_te, threshold, c_fa, c_fr)
        # Reference points on the same test set.
        test["cost_approve_everyone"] = expected_cost(y_te, np.zeros_like(p_te), 0.5, c_fa, c_fr)
        test["cost_reject_everyone"] = expected_cost(y_te, np.ones_like(p_te), 0.5, c_fa, c_fr)
        test["cost_default_threshold_0_5"] = expected_cost(y_te, p_te, 0.5, c_fa, c_fr)

        audit = fairness.audit(y_te, p_te, df.loc[idx_te], cfg["protected"], threshold)
        # Out-of-fold predictions cover 800 applicants, so group estimates are steadier.
        audit_oof = fairness.audit(y_tr, oof[best_name], df.loc[idx_tr], cfg["protected"], threshold)

        # Fairness experiment: the production model does not use age. How much does adding age
        # back change accuracy, cost and the age gap? (Age still leaks in through proxies.)
        with_age_num = numeric + ["age"]
        X_tr_age = df.loc[idx_tr, with_age_num + categorical]
        pipe_age = build_pipeline(clone(candidate_models(seed)[best_name]), with_age_num, categorical)
        p_age = cross_val_predict(pipe_age, X_tr_age, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr_age, cost_age = best_threshold(y_tr, p_age, c_fa, c_fr)
        age_audit_age = fairness.audit(y_tr, p_age, df.loc[idx_tr], ["age_group"], thr_age)
        mitigation = {
            "experiment": "production model (no age feature) vs the same model with age added (out-of-fold)",
            "with_age": {"roc_auc": float(roc_auc_score(y_tr, p_age)),
                         "expected_cost": cost_age,
                         "age_min_disparate_impact":
                             age_audit_age["age_group"]["min_disparate_impact_ratio"]},
            "without_age": {"roc_auc": float(comp.loc[comp.model == best_name, "cv_roc_auc_mean"].iloc[0]),
                            "expected_cost": float(best["oof_expected_cost"]),
                            "age_min_disparate_impact": audit_oof["age_group"]["min_disparate_impact_ratio"]},
        }
        print(f"Age gap (disparate impact): with age {mitigation['with_age']['age_min_disparate_impact']:.2f}"
              f" -> without age {mitigation['without_age']['age_min_disparate_impact']:.2f}")

        shap_df = explain.shap_values(final, X_te, X_tr.sample(200, random_state=seed))
        importance = explain.global_importance(shap_df)
        example_idx = X_te.index[int(np.argmax(p_te))]
        example = {
            "applicant_index": int(example_idx),
            "p_default": float(p_te.max()),
            "decision": "reject" if p_te.max() >= threshold else "approve",
            "reasons": explain.reason_codes(shap_df.loc[example_idx], X_te.loc[example_idx]),
        }

        print("Explaining the model (SHAP), segmenting applicants...", flush=True)
        seg_labels, seg_report = segment.segment(df, seed)
        anomaly_flags, anomaly_report = segment.anomalies(df, seed)

        # ---- artefacts -------------------------------------------------------------
        comp.round(4).to_csv(report_dir / "model_comparison.csv", index=False)
        importance.round(4).to_csv(report_dir / "shap_importance.csv", header=["mean_abs_shap"])
        _plot_roc(oof, y_tr, report_dir / "roc_curves.png")
        _plot_cost(y_tr, oof[best_name], threshold, c_fa, c_fr, report_dir / "cost_curve.png")
        _plot_importance(importance, report_dir / "shap_importance.png")
        df.assign(segment=seg_labels, anomaly=anomaly_flags)[
            ["segment", "anomaly", "amount", "duration", "age", "credit_risk"]
        ].to_csv(report_dir / "segments.csv", index_label="applicant_index")

        metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": {"rows": len(df), "default_rate": float(y.mean()),
                     "features_numeric": numeric, "features_categorical": categorical,
                     "protected_audit_only": cfg["protected"]},
            "selected_model": best_name,
            "cv_comparison": comp.round(4).to_dict("records"),
            "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()},
            "fairness_test": audit,
            "fairness_out_of_fold": audit_oof,
            "fairness_mitigation": mitigation,
            "explanation_example": example,
            "top_features": importance.head(8).round(4).to_dict(),
            "segmentation": seg_report,
            "anomalies": anomaly_report,
            "mlflow_run_id": parent.info.run_id,
        }
        (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

        bundle = {"pipeline": final, "threshold": threshold, "model_name": best_name,
                  "numeric": numeric, "categorical": categorical,
                  "background": X_tr.sample(200, random_state=seed),
                  "trained_at": metrics["generated_at"]}
        joblib.dump(bundle, model_dir / "credit_model.joblib")
        # Reference profile of the training population, used by drift monitoring.
        save_profile(build_profile(X_tr, final.predict_proba(X_tr)[:, 1]),
                     model_dir / "reference_profile.json")

        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test.items() if isinstance(v, float)})
        for attr, res in audit_oof.items():
            mlflow.log_metric(f"fair_{attr}_min_disparate_impact", res["min_disparate_impact_ratio"])
        mlflow.log_metric("fair_age_di_without_age_feature",
                          mitigation["without_age"]["age_min_disparate_impact"])
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        # Explicit requirements: MLflow otherwise infers them by re-loading the model in a
        # separate process, which can take many minutes on Windows laptops.
        from importlib.metadata import PackageNotFoundError, version

        reqs = []
        for pkg in ("scikit-learn", "xgboost", "pandas", "numpy", "skops"):
            try:
                reqs.append(f"{pkg}=={version(pkg)}")
            except PackageNotFoundError:
                pass
        print("Logging the model to MLflow...", flush=True)
        mlflow.sklearn.log_model(
            final, name="model", input_example=X_te.head(3), pip_requirements=reqs,
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )

    print(f"Test ROC AUC {test['roc_auc']:.3f} | expected cost {test['expected_cost']:.3f} "
          f"(approve-everyone baseline {test['cost_approve_everyone']:.3f})")
    return metrics


if __name__ == "__main__":
    main()
