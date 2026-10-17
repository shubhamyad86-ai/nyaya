"""Hugging Face Hub publishing: the model cards are built from the real result files, and the upload
is called correctly. Nothing here talks to the internet."""
import json
import shutil
import sys
import types

import joblib
import pandas as pd
import pytest

from compliance_ai import config, hub
from tests.conftest import MODEL, require


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A project folder in tmp_path, so the tests never touch the real models/ and reports/."""
    shutil.copy(config.ROOT / "config.yaml", tmp_path / "config.yaml")
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def credit_files(root):
    require(MODEL, "run python -m compliance_ai.credit.train")
    shutil.copy(MODEL, root / "models/credit_model.joblib")
    bundle = joblib.load(MODEL)
    metrics = _metrics(bundle)
    (root / "reports/metrics.json").write_text(json.dumps(metrics))
    return bundle, metrics


def _metrics(bundle) -> dict:
    group = lambda g, n, rate, di: {"group": g, "n": n, "approval_rate": rate, "disparate_impact_ratio": di}  # noqa: E731
    return {
        "selected_model": bundle["model_name"],
        "test": {"roc_auc": 0.7908, "pr_auc": 0.67, "threshold": bundle["threshold"], "expected_cost": 0.645,
                 "recall_defaulters": 0.8333, "approval_rate": 0.355,
                 "confusion": {"tn": 61, "fp": 79, "fn": 10, "tp": 50},
                 "cost_approve_everyone": 1.5, "cost_reject_everyone": 0.7},
        "cv_comparison": [{"model": "xgboost", "cv_roc_auc_mean": 0.795, "cv_roc_auc_std": 0.031,
                           "oof_expected_cost": 0.486}],
        "fairness_out_of_fold": {
            "sex": {"groups": [group("female", 249, 0.33, 0.844), group("male", 551, 0.39, 1.0)],
                    "passes_four_fifths_rule": True},
            "age_group": {"groups": [group("25_plus", 690, 0.40, 1.0), group("under_25", 110, 0.19, 0.4773)],
                          "passes_four_fifths_rule": False}},
        "fairness_mitigation": {"with_age": {"age_min_disparate_impact": 0.4229},
                                "without_age": {"age_min_disparate_impact": 0.4773}},
        "top_features": {"has_checking_account": 0.5, "amount": 0.4},
    }


def test_credit_card_is_built_from_the_results(credit_files):
    bundle, _ = credit_files
    out = hub.publish("credit", "someone/nyaya-credit-risk", dry_run=True)
    card = (hub.Path(out) / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0")
    assert f"Pass mark {round(100 * (1 - bundle['threshold']), 1)}" in card
    assert "| ROC AUC | 0.791 |" in card and "50 of 60" in card
    assert "under_25 approved 19% vs 25_plus 40% (ratio 0.48)" in card
    assert "GDPR Article 22" in card
    files = {p.name for p in hub.Path(out).iterdir()}
    assert {"credit_model.joblib", "predict.py", "requirements.txt", "example_applicants.csv",
            "metrics.json"} <= files
    assert 'REPO_ID = "someone/nyaya-credit-risk"' in (hub.Path(out) / "predict.py").read_text()
    assert "scikit-learn==" in (hub.Path(out) / "requirements.txt").read_text()


def test_credit_predict_script_runs_without_the_project(credit_files, monkeypatch):
    out = hub.Path(hub.publish("credit", "someone/nyaya-credit-risk", dry_run=True))
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda repo, name: str(out / name)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    namespace: dict = {"__name__": "predict"}
    exec(compile((out / "predict.py").read_text(), "predict.py", "exec"), namespace)  # noqa: S102
    scored = namespace["score"](pd.read_csv(out / "example_applicants.csv"))
    assert scored.result.iloc[-1] == "FAIL"                      # the "High risk" example
    assert (scored.requires_human_review == (scored.result == "FAIL")).all()
    assert scored.credit_score.between(0, 100).all()


def test_credit_refuses_results_from_another_model(credit_files, root):
    metrics = json.loads((root / "reports/metrics.json").read_text())
    metrics["test"]["threshold"] += 0.05
    (root / "reports/metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(hub.HubError, match="does not belong"):
        hub.publish("credit", "someone/x", dry_run=True)


def _fake_sentiment_model(root):
    d = root / "models/sentiment/fine_tuned_transformer"
    d.mkdir(parents=True)
    id2label = {"0": "negative", "1": "neutral", "2": "positive"}
    (d / "config.json").write_text(json.dumps({"id2label": id2label}))
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "sentiment_meta.json").write_text(json.dumps({
        "labels": ["negative", "neutral", "positive"], "base_model": "distilbert-base-uncased",
        "max_length": 64, "history": [{"epoch": 1, "train_loss": 0.5, "val_macro_f1": 0.9}]}))
    rows = [("tfidf_baseline", 0.8638, 0.8211), ("fine_tuned_transformer", 0.942, 0.9326)]
    pd.DataFrame([{"method": m, "model": m, "accuracy": a, "macro_f1": f, "weighted_f1": f,
                   "latency_ms_per_sentence": 1.0, "f1_negative": f, "f1_neutral": f, "f1_positive": f,
                   "mlflow_run_id": "x"} for m, a, f in rows]).to_csv(root / "reports/nlp_comparison.csv",
                                                                     index=False)
    (root / "reports/nlp_errors_fine_tuned_transformer.json").write_text(json.dumps(
        [{"text": "Sales grew slowly.", "label": "neutral", "predicted": "positive"}]))


def test_sentiment_card_has_widget_metrics_and_licence(root):
    _fake_sentiment_model(root)
    out = hub.Path(hub.publish("sentiment", "someone/distilbert-financial-sentiment", dry_run=True))
    card = (out / "README.md").read_text()
    assert "license: cc-by-nc-sa-3.0" in card and "pipeline_tag: text-classification" in card
    assert "base_model: distilbert/distilbert-base-uncased" in card
    assert "value: 0.9326" in card and "| **DistilBERT fine-tuned (this model)** | 0.942 | 0.933 |" in card
    assert 'pipeline("text-classification", model="someone/distilbert-financial-sentiment")' in card
    assert "*Sales grew slowly.*" in card
    assert {"config.json", "model.safetensors", "comparison.csv"} <= {p.name for p in out.iterdir()}
    assert "mlflow_run_id" not in (out / "comparison.csv").read_text()


def test_sentiment_needs_a_trained_model(root):
    with pytest.raises(hub.HubError, match="Colab"):
        hub.publish("sentiment", "someone/x", dry_run=True)


class FakeApi:
    calls: list = []

    def whoami(self):
        return {"name": "shubham"}

    def create_repo(self, repo_id, **kw):
        FakeApi.calls.append(("create", repo_id, kw))

    def upload_folder(self, **kw):
        FakeApi.calls.append(("upload", kw["repo_id"], kw))


def test_publish_uploads_to_the_users_account(root, monkeypatch):
    _fake_sentiment_model(root)
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    FakeApi.calls = []
    url = hub.publish("sentiment", private=True)
    assert url == "https://huggingface.co/shubham/distilbert-financial-sentiment"
    (_, repo, create_kw), (_, repo2, upload_kw) = FakeApi.calls
    assert repo == repo2 == "shubham/distilbert-financial-sentiment"
    assert create_kw["private"] is True and create_kw["exist_ok"] is True
    assert upload_kw["folder_path"].endswith("hub/sentiment")


def test_publish_explains_how_to_log_in(root, monkeypatch):
    class NoLogin(FakeApi):
        def whoami(self):
            raise OSError("401")

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = NoLogin
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    with pytest.raises(hub.HubError, match="hf auth login"):
        hub.publish("credit")
