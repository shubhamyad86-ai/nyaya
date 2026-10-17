<div align="center">

# ⚖️ Nyaya

**Fair, explainable credit decisions, grounded in the law**

*Nyaya (न्याय) is Sanskrit for justice and sound reasoning.*

[![CI](https://github.com/shubhamyad86-ai/nyaya/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhamyad86-ai/nyaya/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-fine--tuning-EE4C2C?logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20%2B%20agent-1C3C3C)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Delta%20Lake-FF3621?logo=databricks&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

**[Live demo](https://nyaya-ccnp.onrender.com/app)** · [Upload a file](https://nyaya-ccnp.onrender.com/batch) · [API](https://nyaya-ccnp.onrender.com/docs)

<sub>Free hosting: the first visit after a quiet period takes up to a minute while the app wakes up.</sub>

</div>

## Key results

<!-- auto:highlights -->
| **0.79** | **83%** | **0.933** | **89%** |
|---|---|---|---|
| credit ROC AUC (XGBoost) | of defaulters caught | DistilBERT macro-F1 | right law article in top 4 |
<!-- /auto:highlights -->

## Try the credit check

Choose an applicant's details and get a **credit score from 0 to 100**, **PASS or FAIL** against the
pass mark, and the factors that moved the score most. A FAIL is never an automatic rejection: it sends
the case to a human underwriter, as GDPR Article 22 requires.

![Credit check page](docs/screenshots/credit_check_fail.png)

Try it online at **<https://nyaya-ccnp.onrender.com/app>**, or locally: run `start_api.bat` (or
`uvicorn compliance_ai.api:app`) and open `http://127.0.0.1:8000/app`. Three real applicants from the
data are one click away: strong, borderline and high risk. Amounts can be entered and shown in
**euro, rupees or Deutsche Mark** (the model's training currency); see [docs/CURRENCY.md](docs/CURRENCY.md).

**Check a whole file.** At [/batch](https://nyaya-ccnp.onrender.com/batch) (locally
`http://127.0.0.1:8000/batch`), upload a CSV, Excel or PDF with up to
3,000 applicants. Each one gets a score, PASS or FAIL and the action the rules require, rows with
mistakes are listed with what to fix, and the batch is checked against **EU** (GDPR, EU AI Act,
Consumer Credit Directive) and **Indian** (RBI Fair Practices Code, DPDP Act 2023) rules. A 3,000-row
file takes about 3 seconds on a laptop and longer on the free cloud instance. Details: [docs/BATCH_CHECK.md](docs/BATCH_CHECK.md).

![Batch check](docs/screenshots/batch_check.png)

How the score works: credit score = 100 × (1 − probability of default). The pass mark is the approval
threshold that gave the lowest cost in cross-validation, put on the same scale: threshold 0.19 gives a
pass mark of 81. The cost counts a wrongly approved defaulter five times as heavily as a wrongly
rejected good customer, which is why the bar is high.

## Why I built this

When a bank lets an algorithm decide who gets a loan, accuracy is only part of the job. Three other
questions matter just as much:

- **Is it fair?** Does the model treat young people, women or foreign workers worse than it should?
- **Can it explain itself?** Under GDPR Article 22, people have the right not to be judged by a machine
  alone, and the EU AI Act lists credit scoring as high-risk AI (Annex III).
- **Will it still work next month?** Applicants change, and a model trained on last year's customers
  can go wrong without anyone noticing.

I wanted to build one system that takes all three seriously, on real data, and is honest about where it
falls short. Nyaya scores loan applications, explains every decision, audits the results for bias,
watches for drift, and looks up the actual law when a case needs review.

## What's inside

| Part | What it does |
|---|---|
| **Credit-risk model** | Compares four models with 5-fold cross-validation and picks the one with the lowest *business cost*, since approving a defaulter costs five times more than turning away a good customer. |
| **Explanations** | Uses SHAP to give every applicant their top reasons, for example "the loan duration raised the risk the most". |
| **Fairness audit** | Measures approval rates by age, sex and foreign-worker status with the four-fifths rule and equal opportunity. These attributes are never used as model inputs. |
| **Law assistant (RAG)** | Answers questions from the official GDPR and EU AI Act texts (212 articles, 353 recitals, 13 annexes) and checks every citation it gives against what it actually retrieved. |
| **Review agent** | An LLM agent that decides which tools to use: scoring, legal search, the fairness check or sentiment. Whether a human must review a case is decided in code, never by the model. |
| **Drift monitoring** | Compares new applicants with the training data using the Population Stability Index plus a significance test, so small batches don't raise false alarms. |
| **Financial sentiment** | Classifies 3,448 expert-labelled news sentences, comparing TF-IDF, frozen MiniLM embeddings and DistilBERT fine-tuned in PyTorch. |
| **A/B testing** | A champion-vs-challenger experiment for the credit model: power analysis, hash-based random assignment, a sample-ratio check, significance tests with confidence intervals, and a paired comparison on the same applicants. |
| **Credit check page** | A web form for one applicant: every field from the dataset, a 0-100 score with PASS or FAIL, and the SHAP reasons in plain language. Age, sex and foreign-worker status are collected only for the fairness audit. |
| **Batch checks** | Upload a CSV, Excel or PDF of up to 3,000 applicants. Every row gets a score, PASS/FAIL and the action EU or Indian rules require; bad rows are listed with what to fix; the batch runs 16 policy checks (GDPR, EU AI Act, Consumer Credit Directive, RBI Fair Practices Code, DPDP Act) with an audit log that keeps no personal data. |
| **Databricks job** | A 5-task scheduled job deployed as code (Asset Bundle): incremental `COPY INTO` ingestion, a data-quality gate, model registration in Unity Catalog with champion/challenger aliases, batch scoring and drift and fairness monitoring. |
| **Deployment** | Live on Render as a Docker web service built from this repository (the model is trained during the build). Also packaged for Azure Functions with a one-click GitHub Actions deploy (OpenID Connect, no stored passwords), and both models can be published to the Hugging Face Hub with generated model cards. |
| **Engineering** | FastAPI service, 92 tests with GitHub Actions CI, MLflow tracking, Docker. |

## How it fits together

```mermaid
flowchart LR
    A["German Credit: 1,000 applicants"] --> D["4 models, 5-fold CV, cost-based threshold"]
    A --> B["PySpark and Delta: bronze, silver, gold"]
    D -->|MLflow| E["Selected model"]
    E --> F["SHAP reason codes"]
    E --> G["Fairness audit"]
    E --> N["Drift monitor: PSI and chi-square"]
    H["GDPR and EU AI Act texts"] --> I["Article-aware chunks"]
    I --> J["FAISS index"]
    J --> K["RAG assistant with citation check"]
    S["Financial PhraseBank: 3,448 sentences"] --> T["TF-IDF, MiniLM, DistilBERT"]
    F --> L["Review agent with guardrails"]
    G --> L
    K --> L
    T --> L
    L --> M["FastAPI service"]
    E --> O["Docker web app on Render: credit check page, batch upload, EU and India rules"]
    B --> P["Databricks job: ingest, quality gate, train, UC champion, score, monitor"]
    N --> O
```

## Results

All numbers below come from the code in this repository. `python scripts/update_docs.py` writes them
into this README after each training run, so the text always matches the latest results.

### Credit risk and fairness

<!-- auto:credit -->
**Credit model: XGBoost** (held-out test set, 200 applicants)

| Metric | Value |
|---|---|
| ROC AUC | 0.791 (CV 0.795 ± 0.031) |
| Defaulters caught | 83% (50 of 60) |
| Expected cost per applicant | 0.645 vs 0.70 reject-all and 1.50 approve-all |

**Fairness:** age, sex and foreign-worker status are not model inputs. Sex passes the four-fifths rule (0.84). Age fails (0.48, out-of-fold): under-25s are approved 19% of the time versus 40% for older applicants. A controlled experiment shows that removing age as an input helps only partly (disparate impact 0.42 with it, 0.48 without) and does not reach 0.80, because proxies such as housing and employment length carry the signal, so the model card recommends a dedicated mitigation step. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
<!-- /auto:credit -->

![ROC curves](reports/roc_curves.png)

### Finding the right law

<!-- auto:rag -->
**RAG retrieval** (37 hand-labelled questions: 25 GDPR, 12 EU AI Act; offline TF-IDF embeddings)

| Search | hit@1 | hit@4 | primary article in top 4 | MRR@10 |
|---|---|---|---|---|
| Both laws together | 0.54 | 0.81 | 0.76 | 0.67 |
| Filtered to the right law | 0.57 | 0.89 | 0.84 | 0.72 |

The two laws use similar language, so searching both makes GDPR questions harder (hit@4 0.76); filtering by regulation recovers it (0.88). These are baseline numbers from the offline index; `python scripts/compare_retrieval.py` measures the semantic `all-MiniLM-L6-v2` index on the same questions.
<!-- /auto:rag -->

### Drift monitoring

<!-- auto:drift -->
| Batch | Result |
|---|---|
| Held-out test set (same population) | OK |
| Applicants under 25 (new segment) | ALERT: housing and employment duration; mean P(default) 0.39 → 0.49 |
| Largest 25% of loans (product change) | ALERT: amount, amount per month, duration and purpose; mean P(default) 0.39 → 0.51 |
<!-- /auto:drift -->

The test set looks like the training data, so the monitor stays quiet. It does raise an alert for the
two batches that are genuinely different.

### Financial sentiment (690 held-out sentences, 3 classes)

<!-- auto:sentiment -->
| Method | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.864 | **0.821** | 0.767 | 0.909 | 0.787 |
| Frozen MiniLM embeddings + logistic regression | 0.842 | **0.819** | 0.826 | 0.889 | 0.741 |
| DistilBERT fine-tuned (PyTorch) | 0.942 | **0.933** | 0.941 | 0.962 | 0.895 |
<!-- /auto:sentiment -->

I report macro-F1 because 62% of the sentences are neutral, so plain accuracy would flatter a model that
mostly predicts "neutral". The transformer was fine-tuned on a free Colab GPU with
[notebooks/financial_sentiment_colab.ipynb](notebooks/financial_sentiment_colab.ipynb).

### Running on Databricks

I also ran the data pipeline on Databricks Free Edition. It writes Unity Catalog Delta tables
(bronze → silver + quarantine → gold), tracks the models in MLflow and repeats the fairness audit. All
1,000 rows passed the quality checks, and the under-25 approval gap appeared there too (disparate impact
0.525), matching the local results.

For a production setup, `databricks/` defines the whole pipeline as a scheduled five-task job
(ingest → silver → train → score → monitor), deployed with `databricks bundle deploy`. New CSV files
are loaded once each, a load with more than 5% broken rows stops the job, a new model is registered in
Unity Catalog and only replaces the `@champion` if it costs less on the same applicants, and
monitoring compares each new batch with the exact table version the champion was trained on (Delta
time travel). A demo job drops a shifted "new month" of applicants so you can watch the drift alert
fire. I ran both on Databricks Free Edition: the first load trained and registered the champion in
3 minutes, and the shifted month (younger applicants, longer and bigger loans) tripped the alert on
`duration` (PSI 1.83) and `amount` (PSI 0.27), which failed the job and emailed me. Details are in
[docs/DATABRICKS.md](docs/DATABRICKS.md).

### Champion vs challenger A/B test

Before replacing a production model, I test the replacement the way a bank would.
`scripts/ab_test.py` first works out how many applicants a live test needs (power analysis), then
assigns the 200 held-out applicants to arm A or B by hashing their ID, checks the split for a sample
ratio mismatch, and compares approval rate, default rate among approvals and cost per applicant with
confidence intervals. It also runs both models on every applicant (a paired "shadow" comparison with a
McNemar test), which needs far fewer applicants because each applicant is their own control.

The result was to **keep the champion**: the challenger's difference was not significant (p = 0.42).
The bigger lesson is that 100 applicants per arm is far too few. Detecting a 5-point change in approval
rate needs about 1,400 applicants per arm, so a "not significant" result on a small split means "not
enough data", not "no difference". The full results are in `reports/ab_test.md` and in MLflow under
`credit-ab-test`.

## What I learned

- **Removing a sensitive feature does not remove bias.** I took age out of the model, but young
  applicants were still approved far less often. Housing and employment length carry much of the same
  information, and the drift monitor flagged exactly those features for under-25s. Fixing this properly
  would need a dedicated mitigation step and a legal review.
- **The standard drift metric over-reacts on small batches.** PSI alone flagged 70–100% of random
  samples of the training data itself as drifted. Adding a chi-square test cut those false alarms to
  under 2% without missing the real shifts.
- **Fine-tuning is worth it on domain text.** DistilBERT reached macro-F1 0.933, well ahead of TF-IDF
  (0.821). It correctly read "the company reported a pre-tax loss and announced layoffs" as negative,
  which the baseline called neutral.
- **Small LLMs struggle with tool calling.** Rather than failing, the agent falls back to a fixed,
  auditable plan, and every citation is checked against what was actually retrieved.
- **The four credit models are within noise of each other.** Which one wins can change with library
  versions. I saw it happen: my laptop picks XGBoost (pass mark 81), while the cloud build, with newer
  library versions, picks gradient boosting (pass mark 83). Both are within the cross-validation error
  of each other. That is why the README numbers are generated from the results instead of typed by
  hand, and why a real deployment should pin the exact versions it was validated with.
- **I deliberately left out FinBERT.** It was trained on the same PhraseBank data, so testing it here
  would leak training data into the score.

## Design decisions

| Decision | Reason |
|---|---|
| **Cost, not accuracy, picks the model** | A missed defaulter costs five times more than a rejected good customer (Statlog cost matrix). Thresholds are tuned on out-of-fold predictions, and the test set is used once. |
| **The agent proposes, the code decides** | Human review is triggered by a rule based on the score (GDPR Article 22), never by the LLM. Unknown tools are refused and the number of steps is capped. |
| **Chunks follow the structure of the law** | One chunk per article, recital or annex, so every passage carries a precise citation such as `GDPR Article 22`. |
| **Citations are checked, not trusted** | Each citation in an answer is compared with the retrieved passages, and invented ones are flagged. |
| **Drift needs significance, not just size** | A feature is flagged only if PSI is large and the chi-square p-value is below 0.01 divided by the number of features. |
| **Deployment uses the training versions** | The cloud image trains the model during its own build, and the Azure package reads the installed library versions, so a saved model is never loaded by different versions of the libraries than it was trained with. |
| **Graceful fallbacks** | No LLM: the assistant returns the cited passages. Small LLM: the agent uses a fixed plan. No Maven access: Spark writes Parquet. |

## Try it

**On Windows**

1. Double-click `setup.bat`. It installs everything, trains the models and runs the tests.
2. Double-click `start_api.bat`. Your browser opens the credit check page at `http://127.0.0.1:8000/app`;
   all API endpoints are at `http://127.0.0.1:8000/docs`.

Step-by-step help and troubleshooting are in [RUN_GUIDE.md](RUN_GUIDE.md).

**On any platform**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements-nlp.txt
pip install -e .

python scripts/download_data.py                        # German Credit, GDPR, EU AI Act
python -m compliance_ai.credit.train                   # compare, explain, audit, log to MLflow
python -m compliance_ai.rag.build_index --backend tfidf
python -m compliance_ai.rag.evaluate
python -m compliance_ai.nlp.train --method tfidf_baseline
python scripts/drift_demo.py
python scripts/ab_test.py                              # champion vs challenger A/B test
python scripts/check_csv.py my_applicants.pdf          # score a CSV/Excel/PDF and check EU + India rules
python scripts/make_samples.py                         # sample upload files in data/samples/
python scripts/update_docs.py                          # write the results into this README
pytest -q

uvicorn compliance_ai.api:app                         # http://127.0.0.1:8000/app and /docs
```

For AI-written answers, install [Ollama](https://ollama.com) and run `ollama pull llama3.2:1b` (what
`start_api.bat` uses; `llama3.2:3b` gives better answers if your computer can run it), or set
`LLM_PROVIDER=azure` in `.env` to use Azure OpenAI (see `.env.example`). Without an LLM, the assistant
still returns the relevant passages of the law.

**API endpoints**

| Endpoint | What you get |
|---|---|
| `GET /app` | The credit check page |
| `POST /credit/score` | Probability of default, credit score 0-100, PASS or FAIL, and the top four reasons |
| `GET /credit/options` | Allowed values for every applicant field |
| `GET /batch` | The file upload page |
| `GET /batch/template.csv` | An empty file with the right columns |
| `POST /batch/check` | A CSV, Excel or PDF of up to 3,000 applicants → results per applicant and the EU / India policy checks (`?currency=EUR`, `INR` or `DM`) |
| `POST /credit/review` | Score, relevant law and a reviewer memo, with a flag when a human must review |
| `POST /agent/review` | The agent's review, its full tool trace and a citation check |
| `POST /compliance/ask` | An answer from GDPR or the EU AI Act with checked citations |
| `POST /nlp/sentiment` | Positive, neutral or negative for financial news sentences |
| `POST /monitor/drift` | For 30 or more applicants, which features have shifted and by how much |
| `GET /health` | Which models and indexes are loaded |

**Cloud**

- **Databricks:** `cd databricks && databricks bundle deploy && databricks bundle run credit_risk_pipeline`
  deploys and runs the production job. For a quick look, import `notebooks/databricks_credit_pipeline.py`
  and click Run all. See [docs/DATABRICKS.md](docs/DATABRICKS.md).
- **Hugging Face Hub (model pages):** `python scripts/publish_to_hub.py credit` and, from Colab,
  `... sentiment` publish both models with model cards built from the results. See
  [docs/HF_HUB.md](docs/HF_HUB.md).
- **Render (live demo, free):** the credit check page and the batch upload run at
  <https://nyaya-ccnp.onrender.com>, built from `deploy/render/Dockerfile`; every push to `main`
  redeploys. See [docs/RENDER.md](docs/RENDER.md). (A Hugging Face Space works the same way,
  but Docker Spaces now need a paid plan: [docs/HF_SPACE.md](docs/HF_SPACE.md).)
- **Azure Functions:** `build_azure.bat` packages the model, and `deploy_azure.sh` deploys and tests it
  from Azure Cloud Shell. After that, **Actions → Deploy to Azure** retrains, tests and redeploys in one
  click. See [docs/AZURE_DEPLOY.md](docs/AZURE_DEPLOY.md).
- **Docker:** `docker build -t nyaya . && docker run -p 8000:8000 nyaya`

## Project structure

```
src/compliance_ai/
  credit/        data · models · metrics (cost, threshold) · train (CV, MLflow)
                 explain (SHAP) · fairness · segment (k-means, Isolation Forest) · predict
  rag/           ingest (article-aware chunks) · store (FAISS) · llm (Ollama / Azure)
                 assistant (grounded answers) · evaluate
  nlp/           data · models (TF-IDF, embeddings, PyTorch fine-tuning) · train · predict
  agent.py       tool-calling review agent with guardrails
  monitoring.py  PSI + chi-square drift checks
  experiment.py  A/B testing: assignment, power analysis, significance tests
  batch.py       read CSV / Excel / PDF, validate rows, score, fairness, drift
  currency.py    EUR / INR / DM conversion (the model works in DM)
  policies/      EU and India rule packs (YAML) and their automatic checks, audit log
  web/           credit check and upload pages (shared by FastAPI and Azure)
  hub.py         publish models to the Hugging Face Hub with generated model cards
  api.py         FastAPI service
databricks/      Asset Bundle: job and Unity Catalog definitions, five job notebooks, shared rules
notebooks/       Colab GPU run (sentiment) · Databricks quick-start notebook
spark/           bronze → silver (+ quarantine) → gold with Delta Lake
deploy/          Azure Functions app (Cloud Shell, GitHub Actions), Render and Hugging Face Space
scripts/         download_data · eda · drift_demo · ab_test · check_csv · make_samples · compare_retrieval
                 publish_to_hub · update_docs
eval/            37 hand-labelled questions for retrieval evaluation
reports/         metrics, plots, drift report
docs/            model card · batch check · currency · Render · Hugging Face Hub · Databricks · Azure guides
tests/           92 tests (unit, API, web pages, file upload and policies, currency, agent, Spark, Databricks rules,
                 Azure handlers, A/B statistics, Hub model cards)
```

The full model documentation, including limitations, is in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Data

| Dataset | Source | Used for |
|---|---|---|
| Statlog German Credit (1,000 applicants) | UCI, H. Hofmann (1994), decoded CSV via [selva86/datasets](https://github.com/selva86/datasets) | Credit model, fairness, drift |
| GDPR, Regulation (EU) 2016/679 | Official EUR-Lex text via [coolharsh55/GDPRtEXT](https://github.com/coolharsh55/GDPRtEXT) | Law assistant |
| EU AI Act, Regulation (EU) 2024/1689 | Official Journal HTML via [jsiddamsetty/aiactbrowser](https://github.com/jsiddamsetty/aiactbrowser) (pinned) | Law assistant |
| Financial PhraseBank v1.0 | Malo et al. (2014), CC BY-NC-SA 3.0 ([details](data/finance_text/README.md)) | Sentiment |

## Limitations

The credit data is small, old and from one country, so Nyaya demonstrates a method, not a
production lending model. The law assistant is evaluated on 37 questions, its answers still need human
review, and nothing here is legal advice. The policy checks show whether the decision *process* meets
each rule; for Indian lenders the model itself would have to be retrained on Indian data in rupees.
The live demo runs on a free instance: it sleeps when unused and must not receive real customer data.
The Azure Functions package is built and tested but not deployed.

---

<div align="center">

**Shubham Yadav** · Data Scientist

[LinkedIn](https://www.linkedin.com/in/shubham-yadav-a69413200/) · [GitHub](https://github.com/shubhamyad86-ai)

</div>
