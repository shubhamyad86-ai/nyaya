# Hugging Face Spaces (Docker SDK): the credit check page, the batch upload and the API.
# Free CPU hardware, no card needed. The model is trained while the image is built.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src COMPLIANCE_AI_ROOT=/app \
    MPLBACKEND=Agg NYAYA_AUDIT_LOG=/tmp/audit_log.jsonl LLM_PROVIDER=none

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY data/ data/

RUN python -m compliance_ai.credit.train \
 && chmod -R a+rwX /app

# Spaces runs the container as user 1000 and sends traffic to port 7860.
USER 1000
EXPOSE 7860
CMD ["uvicorn", "compliance_ai.api:app", "--host", "0.0.0.0", "--port", "7860"]
