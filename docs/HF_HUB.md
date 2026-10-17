# Publishing the models on the Hugging Face Hub

Two public model pages, each with a model card written from the real results:

| Model | Page | Trained where |
|---|---|---|
| Credit risk (XGBoost) | `huggingface.co/<you>/nyaya-credit-risk` | Your laptop (`python -m compliance_ai.credit.train`) |
| Financial sentiment (DistilBERT) | `huggingface.co/<you>/distilbert-financial-sentiment` | Google Colab, free GPU (about 10 minutes) |

Nothing in a card is typed in by hand. `scripts/publish_to_hub.py` reads `reports/metrics.json`,
`reports/nlp_comparison.csv`, the confusion matrix, the error examples and the installed library
versions, so the card always matches the model it is published with. It refuses to publish the
credit model if `metrics.json` comes from a different training run.

## Once: account and token

1. Sign up at <https://huggingface.co/join> (email only, no card).
2. <https://huggingface.co/settings/tokens> → **Create new token** → token type **Write** → copy it.
   Treat it like a password: never put it in the code or in Git.

## Credit model (from your laptop)

```
pip install -r requirements.txt
hf auth login
python -m compliance_ai.credit.train
python scripts/publish_to_hub.py credit --dry-run
python scripts/publish_to_hub.py credit
```

- `hf auth login` asks for the token (older versions of the library: `huggingface-cli login`).
- Train again right before publishing, so the model and `reports/metrics.json` belong together.
- `--dry-run` only builds `dist\hub\credit\`: open `README.md` there to read the card first.
- The last command prints the link.

The page contains `credit_model.joblib`, a standalone `predict.py` that downloads and runs the
model without this project, `example_applicants.csv`, `requirements.txt` pinned to the versions
you trained with, `metrics.json` and the ROC and SHAP charts.

## Sentiment model (from Google Colab)

The fine-tuned DistilBERT needs a GPU, so it is trained and published from Colab:

1. Open `notebooks/financial_sentiment_colab.ipynb` in Colab (File → Upload notebook, or from GitHub).
   The notebook downloads the project with `git clone`, so the GitHub repository must be **public**
   (GitHub → repository → Settings → General → Danger Zone → Change visibility).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run sections 1-6 (training and results), then section 7: it asks for the token and publishes.

The page gets a **"try it" box** (type a sentence, see negative / neutral / positive) and works with
the standard `pipeline("text-classification", model="<you>/distilbert-financial-sentiment")`. Its
licence is CC BY-NC-SA 3.0 because the training data (Financial PhraseBank) has that licence.

## Options

| Option | What it does |
|---|---|
| `--repo name/other-name` | Choose the repository name |
| `--private` | Create it as private (you can make it public later in Settings) |
| `--dry-run` | Build the folder in `dist/hub/<model>/` without uploading |

Running it again uploads a new version to the same page; the Hub keeps the history.

## What to say about it

"Published both models on the Hugging Face Hub with model cards: intended use, evaluation on a
held-out test set, a fairness audit by age, sex and foreign-worker status, limitations and licences,
the documentation the EU AI Act asks for high-risk credit models."
