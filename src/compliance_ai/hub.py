"""Publish the trained models to the Hugging Face Hub, each with a model card built from the real results.

    python scripts/publish_to_hub.py sentiment             # fine-tuned DistilBERT (train it in Colab first)
    python scripts/publish_to_hub.py credit                # XGBoost credit-risk model
    python scripts/publish_to_hub.py credit --dry-run      # only build the folder in dist/hub/, no upload

Nothing in a card is typed in by hand: metrics, fairness results and library versions are read from
reports/ and models/, so the card always matches the model that is uploaded.
"""
from __future__ import annotations

import inspect
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from compliance_ai.config import load_config, resolve

DEFAULT_REPOS = {"sentiment": "distilbert-financial-sentiment", "credit": "nyaya-credit-risk"}
SENTIMENT_EXAMPLES = [
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.",
    "The company reported a pre-tax loss and announced 200 job cuts.",
    "The annual general meeting will be held in Helsinki on 15 March.",
]


class HubError(RuntimeError):
    """Something the user has to fix before publishing (missing model, results out of date)."""


def _versions(*packages: str) -> dict[str, str]:
    out = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            pass
    return out


def _out_dir(kind: str) -> Path:
    out = resolve("dist") / "hub" / kind
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    return out


def _f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- sentiment


def build_sentiment(repo_id: str) -> Path:
    cfg = load_config()
    model_dir = resolve(cfg["nlp"]["model_dir"]) / "fine_tuned_transformer"
    if not (model_dir / "config.json").exists():
        raise HubError(f"No fine-tuned transformer in {model_dir}. Train it first (Colab: "
                       "notebooks/financial_sentiment_colab.ipynb, or "
                       "`python -m compliance_ai.nlp.train --method fine_tuned_transformer`).")
    reports = resolve(cfg["credit"]["report_dir"])
    comp_path = reports / "nlp_comparison.csv"
    if not comp_path.exists():
        raise HubError(f"{comp_path} not found; it is written by the training run.")
    comp = pd.read_csv(comp_path).set_index("method")
    if "fine_tuned_transformer" not in comp.index:
        raise HubError("reports/nlp_comparison.csv has no fine_tuned_transformer row; retrain it.")
    best = comp.loc["fine_tuned_transformer"]
    meta = json.loads((model_dir / "sentiment_meta.json").read_text())
    labels = meta["labels"]

    out = _out_dir("sentiment")
    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    cm = reports / "nlp_confusion_fine_tuned_transformer.png"
    if cm.exists():
        shutil.copy(cm, out / "confusion_matrix.png")
    comp.reset_index().drop(columns=["mlflow_run_id"], errors="ignore").to_csv(
        out / "comparison.csv", index=False)

    errors_path = reports / "nlp_errors_fine_tuned_transformer.json"
    errors = json.loads(errors_path.read_text())[:5] if errors_path.exists() else []

    names = {"tfidf_baseline": "TF-IDF + logistic regression",
             "frozen_embeddings": "Frozen MiniLM embeddings + logistic regression",
             "fine_tuned_transformer": "**DistilBERT fine-tuned (this model)**"}
    rows = [f"| {names.get(m, m)} | {_f(r.accuracy)} | {_f(r.macro_f1)} | "
            + " | ".join(_f(r[f'f1_{lab}']) for lab in labels) + f" | {_f(r.latency_ms_per_sentence, 2)} |"
            for m, r in comp.sort_values("macro_f1").iterrows()]
    history = [f"| {h['epoch']} | {_f(h['train_loss'])} | {_f(h.get('val_macro_f1', float('nan')))} |"
               for h in meta.get("history", [])]
    n_test = _n_test()
    lr = load_config()["nlp"]["learning_rate"]
    epochs = len(meta.get("history", [])) or load_config()["nlp"]["epochs"]
    history_table = "" if not history else "\n".join(
        ["", "| Epoch | Train loss | Validation macro-F1 |", "|---|---|---|", *history, ""])
    error_list = "\n".join(f"- *{e['text']}* — true **{e['label']}**, predicted **{e['predicted']}**"
                            for e in errors) or "-"
    neutral = sum(e["label"] == "neutral" for e in errors)
    error_note = (f"{neutral} of these {len(errors)} are neutral sentences that the model read as positive "
                  "or negative." if errors and neutral * 2 > len(errors) else "")
    widget = "\n".join(f"  - text: \"{t}\"" for t in SENTIMENT_EXAMPLES)

    card = f"""---
language: en
license: cc-by-nc-sa-3.0
library_name: transformers
pipeline_tag: text-classification
base_model: {meta['base_model'] if '/' in meta['base_model'] else 'distilbert/' + meta['base_model']}
datasets:
  - takala/financial_phrasebank
tags:
  - finance
  - sentiment-analysis
  - distilbert
  - pytorch
metrics:
  - f1
  - accuracy
widget:
{widget}
model-index:
  - name: {repo_id.split('/')[-1]}
    results:
      - task:
          type: text-classification
          name: Financial sentiment
        dataset:
          name: Financial PhraseBank (sentences_75agree, held-out test split)
          type: takala/financial_phrasebank
          config: sentences_75agree
        metrics:
          - type: f1
            name: Macro F1
            value: {round(float(best.macro_f1), 4)}
          - type: accuracy
            value: {round(float(best.accuracy), 4)}
---

# DistilBERT for financial news sentiment

Classifies a sentence from financial news or a company report as **{', '.join(labels)}**.
Fine-tuned from `{meta['base_model']}` in plain PyTorch on Financial PhraseBank, as part of the
Nyaya compliance project (credit risk, EU/India rules, regulation search).

## Use it

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{repo_id}")
clf("{SENTIMENT_EXAMPLES[0]}")
# [{{'label': 'positive', 'score': ...}}]
```

## Results

Held-out test set{f' of {n_test:,} sentences' if n_test else ''}, never used for training or model
selection. Macro-F1 is the main metric because about 60% of the sentences are neutral, so accuracy
alone would reward a model that mostly says "neutral".

| Method | Accuracy | Macro-F1 | {' | '.join(f'F1 {lab}' for lab in labels)} | ms / sentence |
|---|---|---|{'---|' * len(labels)}---|
{chr(10).join(rows)}

Every method in the table was trained and tested on the same split, so the gain from fine-tuning is
measured, not assumed.

![Confusion matrix](confusion_matrix.png)

## Training

| Setting | Value |
|---|---|
| Data | Financial PhraseBank, `sentences_75agree` (at least 75% of annotators agree) |
| Split | Stratified train / validation / test, seed {load_config()['seed']} |
| Max length | {meta['max_length']} tokens |
| Optimiser | AdamW, learning rate {lr}, weight decay 0.01, linear warm-up 10% |
| Epochs | {epochs}, keeping the epoch with the best validation macro-F1 |
| Class imbalance | Class-weighted cross-entropy |
{history_table}
Why not FinBERT? FinBERT was trained on this same dataset, so its scores here would be inflated.

## Where it is wrong

Examples from the test set that the model gets wrong:

{error_list}

{error_note}

## Limitations

- English only, short sentences from news about Finnish listed companies. Tweets, earnings calls or
  other languages need new evaluation.
- Sentiment is not a trading signal or an investment recommendation.
- The licence of the training data (CC BY-NC-SA 3.0) carries over: non-commercial use only.

## Data and credit

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). *Good debt or bad debt: Detecting
semantic orientations in economic texts.* Journal of the Association for Information Science and
Technology. Dataset: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def _n_test() -> int | None:
    try:
        from compliance_ai.nlp.data import load_phrasebank, split

        return len(split(load_phrasebank(), load_config()["seed"])[2])
    except Exception:  # noqa: BLE001 - the count is a nice-to-have in the card
        return None


# --------------------------------------------------------------------------- credit

PREDICT_TEMPLATE = '''"""Score loan applicants with the Nyaya credit-risk model from the Hugging Face Hub.

    pip install -r requirements.txt
    python predict.py                      # scores example_applicants.csv
    python predict.py my_applicants.csv

credit_score = 100 x (1 - P(default)). PASS if the score reaches the pass mark; FAIL means
"refer to a human underwriter", never an automatic rejection (GDPR Art. 22).
Only load model files from a source you trust: joblib files can run code when loaded.
"""
import sys

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "{repo_id}"


{add_features}

def score(applicants: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(hf_hub_download(REPO_ID, "credit_model.joblib"))
    X = add_features(applicants)[bundle["numeric"] + bundle["categorical"]]
    p_default = bundle["pipeline"].predict_proba(X)[:, 1].astype(float)
    credit_score = np.round(100 * (1 - p_default), 1)
    pass_mark = round(100 * (1 - bundle["threshold"]), 1)
    return pd.DataFrame({{
        "p_default": np.round(p_default, 4),
        "credit_score": credit_score,
        "pass_mark": pass_mark,
        "result": np.where(credit_score >= pass_mark, "PASS", "FAIL"),
        "requires_human_review": credit_score < pass_mark,
    }}, index=applicants.index)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else hf_hub_download(REPO_ID, "example_applicants.csv")
    df = pd.read_csv(path)
    print(pd.concat([df[["purpose", "amount", "duration"]], score(df)], axis=1).to_string())
'''


def build_credit(repo_id: str) -> Path:
    import joblib

    from compliance_ai.credit import data as credit_data
    from compliance_ai.credit.predict import score
    from compliance_ai.web import PRESETS

    cfg = load_config()
    model_path = resolve(cfg["credit"]["model_dir"]) / "credit_model.joblib"
    metrics_path = resolve(cfg["credit"]["report_dir"]) / "metrics.json"
    if not model_path.exists():
        raise HubError("No credit model. Run `python -m compliance_ai.credit.train` first.")
    if not metrics_path.exists():
        raise HubError(f"{metrics_path} not found; it is written by the training run.")
    bundle = joblib.load(model_path)
    m = json.loads(metrics_path.read_text())
    same_threshold = abs(m["test"]["threshold"] - bundle["threshold"]) < 1e-9
    if m["selected_model"] != bundle["model_name"] or not same_threshold:
        raise HubError("reports/metrics.json does not belong to models/credit_model.joblib (different model "
                       "or threshold). Run `python -m compliance_ai.credit.train` again so both match.")

    out = _out_dir("credit")
    shutil.copy(model_path, out / "credit_model.joblib")
    reports = resolve(cfg["credit"]["report_dir"])
    for name in ("roc_curves.png", "shap_importance.png", "cost_curve.png"):
        if (reports / name).exists():
            shutil.copy(reports / name, out / name)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))

    examples = pd.DataFrame(list(PRESETS.values()), index=list(PRESETS))
    examples.to_csv(out / "example_applicants.csv", index=False)
    scored = score(examples.reset_index(drop=True))

    add_features_src = inspect.getsource(credit_data.add_features)
    (out / "predict.py").write_text(
        PREDICT_TEMPLATE.format(repo_id=repo_id, add_features=add_features_src), encoding="utf-8")
    pins = _versions("scikit-learn", "xgboost", "pandas", "numpy", "joblib")
    (out / "requirements.txt").write_text(
        "# The versions the model was trained with: other versions may fail to load it.\n"
        + "".join(f"{p}=={v}\n" for p, v in pins.items()) + "huggingface_hub\n")

    t = m["test"]
    c = t["confusion"]
    cost_line = (f"{_f(t['expected_cost'])} (approve everyone {_f(t['cost_approve_everyone'], 2)}, "
                 f"reject everyone {_f(t['cost_reject_everyone'], 2)})")
    pass_mark = round(100 * (1 - bundle["threshold"]), 1)
    cv = "\n".join(f"| {r['model']} | {_f(r['cv_roc_auc_mean'])} ± {_f(r['cv_roc_auc_std'])} | "
                   f"{_f(r['oof_expected_cost'])} |" for r in m["cv_comparison"])
    fair_rows = []
    labels = {"sex": "Sex", "age_group": "Age group", "foreign_worker": "Foreign worker"}
    for attr, res in m["fairness_out_of_fold"].items():
        for g in res["groups"]:
            fair_rows.append(f"| {labels.get(attr, attr)} | {g['group']} | {g['n']} | "
                             f"{g['approval_rate']:.0%} | {_f(g['disparate_impact_ratio'], 2)} |")
        verdict = "pass" if res["passes_four_fifths_rule"] else "FAIL"
        fair_rows.append(f"| | *four-fifths rule* | | | **{verdict}** |")
    mit = m.get("fairness_mitigation", {})
    fair_text = _fairness_text(m["fairness_out_of_fold"], mit, labels)
    top = list(m.get("top_features", {}))[:6]
    example_rows = "\n".join(f"| {name} | {r['credit_score']} | {r['result']} | "
                             f"{', '.join(x['label'] for x in r['reasons'][:2])} |"
                             for name, r in zip(PRESETS, scored, strict=True))

    card = f"""---
license: cc-by-4.0
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - credit-risk
  - xgboost
  - explainability
  - fairness
  - gdpr
  - eu-ai-act
---

# Nyaya credit-risk model ({bundle['model_name']})

Estimates the probability that a loan applicant defaults, turns it into a **credit score from 0 to
100** with **PASS or FAIL**, and gives the main reasons. Trained on the Statlog German Credit data
(1,000 applicants, amounts in Deutsche Mark) as part of the Nyaya compliance project.

> **Demo model.** It shows how a credit model can be built, explained, audited and documented for
> the EU AI Act. It is not validated for real lending, and its fairness audit below shows why.

## How the score works

- `credit_score = 100 × (1 − P(default))`
- **Pass mark {pass_mark}**, from the threshold P(default) = {bundle['threshold']} that minimises the
  expected cost: approving an applicant who defaults costs 5, refusing one who repays costs 1
  (the dataset's own cost matrix).
- **FAIL never means an automatic rejection.** Under GDPR Article 22 the application goes to a
  human underwriter, with the reasons.

## Use it

```bash
pip install -r requirements.txt
python predict.py                    # scores example_applicants.csv
```

```python
from predict import score            # predict.py from this repository
import pandas as pd
score(pd.read_csv("example_applicants.csv"))
```

The input needs the 20 columns of the German Credit data (see `example_applicants.csv`).
`requirements.txt` pins the library versions the model was trained with; joblib files can run code
when they are loaded, so only load them from a source you trust.

| Example applicant | Credit score | Result | Main reasons |
|---|---|---|---|
{example_rows}

## Results

Held-out test set: 200 applicants never used for training or for choosing the threshold.

| Metric | Value |
|---|---|
| ROC AUC | {_f(t['roc_auc'])} |
| PR AUC | {_f(t['pr_auc'])} |
| Defaulters caught (recall) | {t['recall_defaulters']:.0%} ({c['tp']} of {c['tp'] + c['fn']}) |
| Expected cost per applicant | {cost_line} |
| Approval rate | {t['approval_rate']:.0%} |

Model choice by 5-fold cross-validation on the training part:

| Model | CV ROC AUC | Expected cost (out-of-fold) |
|---|---|---|
{cv}

![ROC curves](roc_curves.png)

## Fairness audit

Sex, age and foreign-worker status are **not model inputs**; they are kept only for this audit.
Approval rates, out-of-fold over 800 applicants (ratio to the most-approved group; the four-fifths
rule flags a ratio below 0.80):

| Attribute | Group | Applicants | Approved | Ratio |
|---|---|---|---|---|
{chr(10).join(fair_rows)}

{fair_text}

## What drives the score

Mean absolute SHAP value, largest first: {', '.join(f'`{f}`' for f in top)}.
Every score comes with its top reasons (SHAP) in the Nyaya API and web page.

![SHAP importance](shap_importance.png)

## Limitations

- 1,000 applicants from southern Germany, 1973-1975 (Grömping 2019). Not representative of today's
  applicants or of other countries; for India it would need retraining on Indian data in rupees.
- Small test set (200): differences of a few points between models are within noise.
- High-risk AI system under the EU AI Act (Annex III, creditworthiness) if used for real decisions.

## Data and credit

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository,
CC BY 4.0. Corrected codebook: Grömping, U. (2019).

Trained {bundle['trained_at']}; published {datetime.now(timezone.utc):%Y-%m-%d}.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def _fairness_text(fairness: dict, mit: dict, labels: dict) -> str:
    """Plain-language reading of the fairness table, written from the numbers."""
    parts = []
    for attr, res in fairness.items():
        if res["passes_four_fifths_rule"]:
            continue
        low = min(res["groups"], key=lambda g: g["disparate_impact_ratio"])
        high = max(res["groups"], key=lambda g: g["approval_rate"])
        smallest = min(g["n"] for g in res["groups"])
        text = (f"**{labels.get(attr, attr)}: {low['group']} approved {low['approval_rate']:.0%} vs "
                f"{high['group']} {high['approval_rate']:.0%} (ratio {low['disparate_impact_ratio']:.2f}).**")
        if smallest < 50:
            text += f" One group has only {smallest} applicants, so this ratio is uncertain."
        parts.append(text)
    if not parts:
        return "All audited groups pass the four-fifths rule."
    out = " ".join(parts)
    if not fairness.get("age_group", {}).get("passes_four_fifths_rule", True):
        out += (" Age is not an input, so the age gap comes through related features such as housing and "
                "employment length.")
        if mit:
            with_age = mit["with_age"]["age_min_disparate_impact"]
            without = mit["without_age"]["age_min_disparate_impact"]
            out += (f" With age as an input the ratio would be {with_age:.2f}, without it {without:.2f}: "
                    "dropping the attribute alone does not close the gap.")
    return out + (" Before any real use this needs a mitigation step and a documented review "
                  "(EU AI Act, Article 10).")


# --------------------------------------------------------------------------- upload

BUILDERS = {"sentiment": build_sentiment, "credit": build_credit}


def publish(kind: str, repo: str | None = None, private: bool = False, dry_run: bool = False) -> str:
    """Build the folder for `kind` and upload it. Returns the repo URL (or the folder for a dry run)."""
    if kind not in BUILDERS:
        raise HubError(f"unknown model {kind!r}; choose from {sorted(BUILDERS)}")
    api = None
    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            user = api.whoami()["name"]
        except Exception as err:  # noqa: BLE001 - any auth failure means the same thing to the user
            raise HubError("Not logged in to Hugging Face. Run `hf auth login` and paste a token "
                           "with write access (huggingface.co/settings/tokens).") from err
        repo = repo or f"{user}/{DEFAULT_REPOS[kind]}"
    repo = repo or f"<your-hf-name>/{DEFAULT_REPOS[kind]}"
    if "/" not in repo:
        raise HubError("--repo must look like <user>/<name>")

    folder = BUILDERS[kind](repo)
    if dry_run:
        return str(folder)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model",
                      commit_message=f"Publish {kind} model from the Nyaya project")
    return f"https://huggingface.co/{repo}"
