# How to run Compliance AI on Windows

This guide assumes Windows 10/11 and Command Prompt. Everything runs on your own laptop; nothing is uploaded.

## 1. One-time installs

| Tool | Where | Notes |
|---|---|---|
| Python 3.10 - 3.13 | python.org | Tick **"Add Python to PATH"** during install |
| Git | git-scm.com | Only needed to push to GitHub |
| Ollama | ollama.com | Optional - only for AI-written answers. Then run `ollama pull llama3.2:1b` |
| Java 17 | adoptium.net | Optional - only for the Spark tests |

## 2. First-time setup (about 20-30 minutes, mostly downloads)

**If this is a new folder:** unzip the project, open the `compliance-ai` folder, and double-click
**`setup.bat`**.

**If you are updating your existing folder:** copy all files from the new zip over your existing
`compliance-ai` folder and choose "Replace the files in the destination". Then double-click **`setup.bat`**.
Your `.venv` and git history are kept.

`setup.bat` does everything in order and stops with a clear message if a step fails:

1. installs the packages (PyTorch is large, be patient)
2. exploratory data analysis -> `reports\eda.md`
3. trains and compares 4 credit-risk models, audits fairness, logs to MLflow
4. builds the search index over the GDPR and the EU AI Act
5. evaluates retrieval on 37 labelled questions (25 GDPR, 12 AI Act)
6. trains the financial-sentiment baseline
7. drift monitoring demo -> `reports\drift_report.md`, then writes your results into the README and model card
8. runs the tests (all should pass; the Spark tests are skipped if Java is not installed)

## 3. Start the app (every time)

1. Make sure Ollama is running if you want AI-written answers (it starts with Windows after install).
2. Double-click **`start_api.bat`**. The browser opens the **credit check page**
   (**http://127.0.0.1:8000/app**) after a few seconds. All endpoints are at **http://127.0.0.1:8000/docs**.
3. Leave the black window open, and **do not click inside it** - Windows "Select" mode pauses the
   server. If the title shows "Select", click it and press **Esc**.
4. To stop: click the window and press **Ctrl+C**.

## 4. The credit check page (/app)

1. Click **Strong**, **Borderline** or **High risk** to load a real applicant from the data.
2. Change any field, for example the loan amount or duration. The score updates by itself.
3. Read the result: the score out of 100, **PASS** or **FAIL** against the pass mark, the chance of
   default, and the four factors that moved the score most (orange raises the risk, blue lowers it).

The last box, *Fairness audit only*, is not used by the model. Changing age or sex does not change the
score; those fields exist so approval rates can be compared across groups.

## 5. What to try in /docs

For each endpoint: click it -> **Try it out** -> **Execute** (example inputs are pre-filled).

| Endpoint | What you see |
|---|---|
| `GET /health` | Which models and indexes are loaded |
| `POST /credit/score` | Probability of default, decision and the top 4 reasons (SHAP) |
| `POST /credit/review` | Score + GDPR passages + reviewer memo (fixed workflow) |
| `POST /agent/review` | **The agent**: it picks its own tools; look at `steps` to see which it called, `mode` (`agent` or `fixed_plan`) and `unsupported_citations` |
| `POST /compliance/ask` | Question answered with GDPR / EU AI Act citations (`mode: llm` when Ollama is used). Set `"regulation": "EU AI Act"` or `"GDPR"` to search only one law |
| `POST /nlp/sentiment` | Positive / negative / neutral for financial news sentences |
| `POST /monitor/drift` | Paste 30+ applicants: shows which features have shifted compared with the training data |

**About the agent and small models:** `llama3.2:1b` is often too small to call tools reliably. Then the
agent switches to its fixed plan (`"mode": "fixed_plan"`) instead of failing - this is by design. With a
bigger model (`llama3.2:3b` or larger, or Azure OpenAI) you will see `"mode": "agent"`. To use another
model, run `set OLLAMA_MODEL=llama3.2:3b` before `start_api.bat`, or edit the line in `start_api.bat`.

## 6. Optional extras

**MLflow dashboard** (experiment tracking):
```
.venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 - experiments `credit-risk` and `financial-sentiment`.

**Transformer sentiment on a free GPU (Google Colab, about 10 minutes)** - needs the project on GitHub first:
1. colab.research.google.com -> File -> Upload notebook -> `notebooks\financial_sentiment_colab.ipynb`
2. Runtime -> Change runtime type -> **T4 GPU**
3. Runtime -> **Run all**
4. At the end it downloads `sentiment_results.zip`. Unzip it inside the project folder; the API then
   uses the fine-tuned model automatically.

**Semantic search** (better GDPR retrieval, downloads PyTorch models):
```
pip install -r requirements-hf.txt
python -m compliance_ai.rag.build_index
python -m compliance_ai.rag.evaluate
```

## 7. Put it on GitHub

First time only: create an **empty** repository named `compliance-ai` at github.com/new (no README), then:
```
git init
git config --global user.name "Shubham Yadav"
git config --global user.email "your-email@example.com"
git remote add origin https://github.com/shubhamyad86-ai/compliance-ai.git
git branch -M main
```
Every time you want to publish changes:
```
git add -A
git commit -m "Describe the change"
git push -u origin main
```
`setup.bat` ends by running `python scripts\update_docs.py`, which rewrites the result sections of
`README.md` and `docs\MODEL_CARD.md` from your own `reports` folder. Your PC may select a different
model than someone else's (the four models are within noise of each other), and the documentation
will still match your files. If you retrain by hand, run that script again before you push.

## 8. Deploy to the cloud (optional)

- **Azure Functions** (the credit check page and the scoring + drift API on a public URL): run
  `build_azure.bat`, then follow [docs/AZURE_DEPLOY.md](docs/AZURE_DEPLOY.md). About 20 minutes; delete
  it afterwards with `bash deploy_azure.sh delete`. Later updates: **Actions → Deploy to Azure** on GitHub.
- **Databricks Free Edition** (scheduled job, Unity Catalog model, monitoring): install the Databricks
  CLI and run `databricks bundle deploy` in the `databricks` folder; see
  [docs/DATABRICKS.md](docs/DATABRICKS.md). About 15 minutes, no card needed.

## 9. Troubleshooting

| You see | Fix |
|---|---|
| `No module named 'compliance_ai'` | Run `pip install -e .` once inside the activated `.venv` (setup.bat does this) |
| Browser shows `{"detail":"Not Found"}` | Use http://127.0.0.1:8000/docs (or update - the new version redirects `/` automatically) |
| Requests keep "LOADING" | The server window is in "Select" mode: click it and press Esc |
| `LLM unavailable: ResponseError` | The Ollama model is not installed: run `ollama list` and set `OLLAMA_MODEL` to one you have |
| `LLM unavailable: ConnectError` | Ollama is not running: start the Ollama app |
| Ollama says the model needs a newer version | Update Ollama from ollama.com/download (your models are kept) |
| `$env:...` gives an error | That is PowerShell syntax; in Command Prompt use `set NAME=value` |
| `'git' is not recognized` | Install Git, then open a new Command Prompt |
| Spark tests "skipped" | Normal without Java; GitHub CI runs them |
