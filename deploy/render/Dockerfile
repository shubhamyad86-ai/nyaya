# Render.com free web service (no card): the credit check page, the batch upload and the API.
# Built from the repository root (render.yaml sets the context). The model is trained during the build.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src COMPLIANCE_AI_ROOT=/app \
    MPLBACKEND=Agg NYAYA_AUDIT_LOG=/tmp/audit_log.jsonl LLM_PROVIDER=none \
    GIT_PYTHON_REFRESH=quiet

# The web app and training, plus the light law search (TF-IDF + FAISS, no LLM) and the TF-IDF
# sentiment model. No PyTorch, LLM or Azure packages: fits the 512 MB free plan.
COPY deploy/hf_space/requirements.txt deploy/hf_space/requirements-search.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-search.txt

COPY config.yaml .
COPY src/ src/
COPY data/raw/german_credit.csv data/raw/german_credit.csv
COPY data/regulations/eu_ai_act_2024_1689.html data/regulations/gdpr_regulation_2016_679.html data/regulations/
COPY data/finance_text/Sentences_75Agree.txt data/finance_text/

RUN python -m compliance_ai.credit.train \
 && python -m compliance_ai.rag.build_index --backend tfidf \
 && python -m compliance_ai.nlp.train --method tfidf_baseline

# Render sends traffic to $PORT. One worker: with all three models loaded it uses about 340 MB,
# and about 385 MB at the peak of a 3,000-row upload (measured locally).
CMD ["sh", "-c", "uvicorn compliance_ai.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
