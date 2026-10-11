FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src LLM_PROVIDER=ollama \
    OLLAMA_HOST=http://host.docker.internal:11434

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/
COPY eval/ eval/
COPY data/raw/ data/raw/
COPY data/regulations/ data/regulations/

# Train the model and build an offline (TF-IDF) index at build time so the image is self-contained.
RUN python -m compliance_ai.credit.train \
 && python -m compliance_ai.rag.build_index --backend tfidf

EXPOSE 8000
CMD ["uvicorn", "compliance_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
