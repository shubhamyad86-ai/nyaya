# Databricks notebook source
# MAGIC %md
# MAGIC # 3 · Train, register in Unity Catalog, champion vs challenger
# MAGIC
# MAGIC 1. Two candidate models are cross-validated on the training split; the one with the lowest
# MAGIC    **expected cost** wins (approving a defaulter costs 5, rejecting a good payer costs 1), and its
# MAGIC    approval threshold is chosen to minimise that cost.
# MAGIC 2. The winner is logged with MLflow and registered in **Unity Catalog** with the alias
# MAGIC    `challenger`. The threshold and the silver table version it was trained on are stored as
# MAGIC    tags, so every score can be traced back to the exact data.
# MAGIC 3. It becomes `champion` only if it beats the current champion on the same held-out applicants.
# MAGIC    Scoring always uses `@champion`, so a worse model never reaches production by accident.
# MAGIC
# MAGIC Training uses scikit-learn on a pandas copy: 1,000 rows fit in memory, and Spark already did the
# MAGIC data engineering. The full statistical champion-vs-challenger test (power analysis, McNemar,
# MAGIC bootstrap) is `scripts/ab_test.py` in the main project.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "nyaya")
dbutils.widgets.text("model_name", "")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
MODEL = dbutils.widgets.get("model_name") or f"{CATALOG}.{SCHEMA}.nyaya_credit_risk"
SILVER = f"{CATALOG}.{SCHEMA}.credit_silver"

# In the job, skip training when the ingest task loaded nothing new (the champion stays as it is).
# Run by hand, debugValue=1 means it always trains.
new_rows = dbutils.jobs.taskValues.get(taskKey="ingest", key="rows_loaded", default=1, debugValue=1)
if new_rows == 0:
    dbutils.notebook.exit("No new rows since the last run: kept the current champion.")

# COMMAND ----------

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import COST_FALSE_APPROVAL, best_threshold, expected_cost, feature_columns

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

silver_version = spark.sql(f"DESCRIBE HISTORY {SILVER} LIMIT 1").first()["version"]
pdf = spark.read.option("versionAsOf", silver_version).table(SILVER).toPandas()
numeric, categorical = feature_columns(pdf)
features = numeric + categorical
print(f"{len(pdf)} applicants from {SILVER} version {silver_version} | {len(features)} features")

idx_train, idx_test = train_test_split(pdf.index, test_size=0.2, stratify=pdf["default"], random_state=42)
X_train, X_test = pdf.loc[idx_train, features], pdf.loc[idx_test, features]
y_train, y_test = pdf.loc[idx_train, "default"].to_numpy(), pdf.loc[idx_test, "default"].to_numpy()


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])


candidates = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=300, random_state=42),
}

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/nyaya-credit-risk")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
with mlflow.start_run(run_name="train") as parent:
    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"cv_{name}", nested=True):
            oof = cross_val_predict(make_pipeline(model), X_train, y_train, cv=cv,
                                    method="predict_proba")[:, 1]
            threshold, cost = best_threshold(y_train, oof)
            auc = roc_auc_score(y_train, oof)
            mlflow.log_params({"model": name, "cv_folds": 5, "cost_false_approval": COST_FALSE_APPROVAL})
            mlflow.log_metrics({"cv_auc": auc, "cv_expected_cost": cost, "threshold": threshold})
            results[name] = {"auc": auc, "cost": cost, "threshold": threshold}
            print(f"{name:24s} CV AUC {auc:.3f}  cost/applicant {cost:.3f}  threshold {threshold:.2f}")

    winner = min(results, key=lambda n: results[n]["cost"])
    threshold = results[winner]["threshold"]
    final = make_pipeline(candidates[winner]).fit(X_train, y_train)
    p_test = final.predict_proba(X_test)[:, 1]
    reject = p_test >= threshold
    metrics = {
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_expected_cost": expected_cost(y_test, p_test, threshold),
        "test_approval_rate": float(np.mean(~reject)),
        "test_bad_rate_among_approved": float(y_test[~reject].mean()) if (~reject).any() else 0.0,
    }
    mlflow.log_params({"winner": winner, "threshold": threshold, "silver_version": silver_version})
    mlflow.log_metrics(metrics)
    info = mlflow.sklearn.log_model(
        final, name="model", input_example=X_train.head(3),
        signature=infer_signature(X_train, final.predict_proba(X_train.head(3))),
        registered_model_name=MODEL,
    )

version = info.registered_model_version
for key, value in {"threshold": threshold, "algorithm": winner, "silver_version": silver_version,
                   "test_expected_cost": round(metrics["test_expected_cost"], 4)}.items():
    client.set_model_version_tag(MODEL, version, key, str(value))
client.set_registered_model_alias(MODEL, "challenger", version)
print(f"registered {MODEL} version {version} as @challenger | "
      + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote only if better
# MAGIC Both models score the same held-out applicants, each with its own threshold. The paired
# MAGIC bootstrap shows how sure we can be about the difference; with ~200 test applicants it is wide,
# MAGIC which is why `scripts/ab_test.py` computes the sample size a real A/B test would need.
# MAGIC When new data has arrived, the champion may have been trained on some of these test applicants.
# MAGIC That favours the champion, so the comparison is conservative: a challenger has to be clearly better.

# COMMAND ----------

try:
    champ = client.get_model_version_by_alias(MODEL, "champion")
except Exception:  # no champion yet: first run
    champ = None

promote, summary = True, "no champion yet"
if champ is not None and champ.version != version:
    champ_model = mlflow.sklearn.load_model(f"models:/{MODEL}@champion")
    champ_threshold = float(champ.tags.get("threshold", 0.5))
    p_champ = champ_model.predict_proba(X_test[champ_model.feature_names_in_])[:, 1]
    cost_new = expected_cost(y_test, p_test, threshold)
    cost_old = expected_cost(y_test, p_champ, champ_threshold)

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y_test), len(y_test))
        diffs.append(expected_cost(y_test[i], p_test[i], threshold)
                     - expected_cost(y_test[i], p_champ[i], champ_threshold))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    promote = cost_new < cost_old
    summary = (f"challenger v{version} cost {cost_new:.3f} vs champion v{champ.version} {cost_old:.3f} "
               f"(difference 95% CI {lo:+.3f} to {hi:+.3f})")

if promote:
    client.set_registered_model_alias(MODEL, "champion", version)
print(("PROMOTED to @champion: " if promote else "kept the current champion: ") + summary)

client.set_model_version_tag(MODEL, version, "promotion_decision", "promoted" if promote else "not promoted")
dbutils.jobs.taskValues.set(key="promoted", value=bool(promote))
dbutils.jobs.taskValues.set(key="model_version", value=int(version))
