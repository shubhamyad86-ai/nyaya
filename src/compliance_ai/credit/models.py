"""Model definitions: one preprocessing step shared by every candidate model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(seed: int) -> dict[str, object]:
    """Baseline-to-boosted candidates. Positive class = default (bad credit, ~30%)."""
    return {
        "logistic_regression": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=7 / 3, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        ),
    }


def build_pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", model)])
