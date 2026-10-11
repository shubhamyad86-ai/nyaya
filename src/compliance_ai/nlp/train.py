"""Train and evaluate a financial-sentiment classifier on Financial PhraseBank.

    python -m compliance_ai.nlp.train --method tfidf_baseline          # seconds, any laptop
    python -m compliance_ai.nlp.train --method frozen_embeddings       # minutes on CPU
    python -m compliance_ai.nlp.train --method fine_tuned_transformer  # use a GPU (Google Colab)

Each run: stratified train/val/test split, fit, evaluate once on the test set,
log to MLflow, save the model to models/sentiment/<method>/ and append the result
to reports/nlp_comparison.csv so methods can be compared side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from compliance_ai.config import ROOT, load_config, resolve  # noqa: E402
from compliance_ai.nlp.data import load_phrasebank, split  # noqa: E402
from compliance_ai.nlp.models import (  # noqa: E402
    EmbeddingClassifier,
    FineTunedTransformer,
    TfidfBaseline,
)


def build(method: str, labels: list[str], cfg: dict, seed: int, **overrides):
    if method == "tfidf_baseline":
        return TfidfBaseline(labels, seed)
    if method == "frozen_embeddings":
        return EmbeddingClassifier(labels, overrides.get("model_name") or cfg["embedding_model"],
                                   cfg["max_length"], seed=seed)
    if method == "fine_tuned_transformer":
        return FineTunedTransformer(labels, overrides.get("model_name") or cfg["transformer_model"],
                                    cfg["max_length"], overrides.get("epochs") or cfg["epochs"],
                                    cfg["batch_size"], cfg["learning_rate"], seed)
    raise ValueError(f"unknown method: {method}")


def evaluate(model, test: pd.DataFrame, labels: list[str]) -> tuple[dict, list[int]]:
    y_true = [labels.index(v) for v in test.label]
    start = time.perf_counter()
    y_pred = model.predict_proba(test.text).argmax(1).tolist()
    latency_ms = (time.perf_counter() - start) / len(test) * 1000
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "latency_ms_per_sentence": latency_ms,
        **{f"f1_{lab}": report[lab]["f1-score"] for lab in labels},
    }
    return metrics, y_pred


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train a financial-sentiment classifier")
    parser.add_argument("--method", default="tfidf_baseline",
                        choices=["tfidf_baseline", "frozen_embeddings", "fine_tuned_transformer"])
    parser.add_argument("--model-name", help="override the Hugging Face model id")
    parser.add_argument("--epochs", type=int, help="override the number of fine-tuning epochs")
    args = parser.parse_args(argv)

    all_cfg = load_config()
    cfg, seed = all_cfg["nlp"], all_cfg["seed"]
    labels = cfg["labels"]
    report_dir = resolve(all_cfg["credit"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_phrasebank()
    train, val, test = split(df, seed)
    print(f"{len(df)} sentences -> train {len(train)} / val {len(val)} / test {len(test)}")

    model = build(args.method, labels, cfg, seed, model_name=args.model_name, epochs=args.epochs)
    model_id = getattr(model, "model_name", "tfidf")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name=f"{args.method}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as run:
        mlflow.log_params({"method": args.method, "model": model_id, "n_train": len(train),
                           "n_val": len(val), "n_test": len(test), "max_length": cfg["max_length"]})
        start = time.perf_counter()
        model.fit(train.text, train.label, val.text, val.label)
        train_seconds = time.perf_counter() - start

        metrics, y_pred = evaluate(model, test, labels)
        metrics["train_seconds"] = train_seconds
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for rec in getattr(model, "history", []):
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=rec["epoch"])

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions([labels.index(v) for v in test.label], y_pred,
                                                display_labels=labels, ax=ax, colorbar=False)
        ax.set_title(f"{args.method} (test set)")
        fig.tight_layout()
        cm_path = report_dir / f"nlp_confusion_{args.method}.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        out_dir = resolve(cfg["model_dir"]) / args.method
        model.save(out_dir)

        # A few test errors are useful for error analysis in interviews and the README.
        errors = test.assign(predicted=[labels[i] for i in y_pred])
        errors = errors[errors.label != errors.predicted].head(8)

        row = {"method": args.method, "model": model_id,
               **{k: round(v, 4) for k, v in metrics.items()},
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "mlflow_run_id": run.info.run_id}
        comp_path = report_dir / "nlp_comparison.csv"
        comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
        comp = pd.concat([comp[comp.get("method", pd.Series(dtype=str)) != args.method]
                          if not comp.empty else comp, pd.DataFrame([row])], ignore_index=True)
        comp.to_csv(comp_path, index=False)
        (report_dir / f"nlp_errors_{args.method}.json").write_text(
            json.dumps(errors.to_dict("records"), indent=2))

    print(f"{args.method}: accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}, "
          f"{metrics['latency_ms_per_sentence']:.2f} ms/sentence, trained in {train_seconds:.0f}s")
    return metrics


if __name__ == "__main__":
    main()
