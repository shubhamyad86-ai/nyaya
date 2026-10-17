# Render.com free web service (no card): the credit check page, the batch upload and the API.
# Built from the repository root (render.yaml sets the context). The model is trained during the build.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src COMPLIANCE_AI_ROOT=/app \
    MPLBACKEND=Agg NYAYA_AUDIT_LOG=/tmp/audit_log.jsonl LLM_PROVIDER=none \
    GIT_PYTHON_REFRESH=quiet

# Only what the web app and training need (no RAG, LLM or Azure packages): fits the 512 MB free plan.
COPY deploy/hf_space/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY data/raw/german_credit.csv data/raw/german_credit.csv

RUN python -m compliance_ai.credit.train

# Render sends traffic to $PORT. One worker: memory peaks at about 370 MB on a 3,000-row upload.
CMD ["sh", "-c", "uvicorn compliance_ai.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
