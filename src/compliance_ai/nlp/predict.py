"""Classify financial sentences with the best trained sentiment model available.

Preference order: fine-tuned transformer > frozen embeddings > TF-IDF baseline,
or set SENTIMENT_METHOD to force one.
"""
from __future__ import annotations

import os
from functools import lru_cache

from compliance_ai.config import load_config, resolve
from compliance_ai.nlp.models import METHODS

PREFERENCE = ["fine_tuned_transformer", "frozen_embeddings", "tfidf_baseline"]


@lru_cache(maxsize=1)
def load_model():
    base = resolve(load_config()["nlp"]["model_dir"])
    wanted = os.environ.get("SENTIMENT_METHOD")
    for method in ([wanted] if wanted else PREFERENCE):
        path = base / method
        if path.exists() and any(path.iterdir()):
            return METHODS[method].load(path)
    raise FileNotFoundError(
        f"no sentiment model in {base} - run `python -m compliance_ai.nlp.train --method tfidf_baseline`"
    )


def classify(sentences: list[str]) -> list[dict]:
    model = load_model()
    probs = model.predict_proba(sentences)
    return [
        {"text": s, "label": model.labels[int(p.argmax())], "confidence": round(float(p.max()), 4),
         "scores": {lab: round(float(v), 4) for lab, v in zip(model.labels, p, strict=True)},
         "model": model.kind}
        for s, p in zip(sentences, probs, strict=True)
    ]
