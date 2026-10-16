PY ?= python
export PYTHONPATH := src

.PHONY: setup data eda train sentiment index index-offline eval drift ab-test docs spark azure-package api test lint all dbx-validate dbx-deploy dbx-run

setup:            ## install everything, including semantic embeddings
	$(PY) -m pip install -r requirements-dev.txt -r requirements-hf.txt -r requirements-nlp.txt
	$(PY) -m pip install -e .

data:             ## download the real datasets
	$(PY) scripts/download_data.py --extra

eda:
	$(PY) scripts/eda.py

train:            ## compare models, audit fairness, explain, log to MLflow
	$(PY) -m compliance_ai.credit.train

sentiment:        ## financial sentiment baseline (transformers: see notebooks/financial_sentiment_colab.ipynb)
	$(PY) -m compliance_ai.nlp.train --method tfidf_baseline

index:            ## semantic index (sentence-transformers)
	$(PY) -m compliance_ai.rag.build_index --backend huggingface

index-offline:    ## TF-IDF index, no model download
	$(PY) -m compliance_ai.rag.build_index --backend tfidf

eval:
	$(PY) -m compliance_ai.rag.evaluate

drift:            ## drift monitoring demo -> reports/drift_report.md
	$(PY) scripts/drift_demo.py

ab-test:          ## champion vs challenger A/B test -> reports/ab_test.md
	$(PY) scripts/ab_test.py

docs:             ## rewrite README / model card result sections from reports/
	$(PY) scripts/update_docs.py

spark:
	$(PY) spark/credit_pipeline.py --format delta

azure-package:    ## zip for Azure Functions -> dist/ (see docs/AZURE_DEPLOY.md)
	$(PY) deploy/azure_function/build.py
	cp deploy/azure_function/deploy_azure.sh dist/

api:             ## API + credit check page at http://127.0.0.1:8000/app
	uvicorn compliance_ai.api:app --reload

dbx-validate:    ## Databricks Asset Bundle (needs the Databricks CLI, see docs/DATABRICKS.md)
	cd databricks && databricks bundle validate

dbx-deploy:
	cd databricks && databricks bundle deploy

dbx-run:
	cd databricks && databricks bundle run credit_risk_pipeline

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	pytest -q

lint:
	ruff check src tests spark scripts deploy databricks notebooks/databricks_credit_pipeline.py

all: data eda train sentiment index eval drift docs test
