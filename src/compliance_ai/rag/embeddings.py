"""Embedding backends behind one LangChain interface.

- huggingface: sentence-transformers model run locally (default, best quality).
- azure: Azure OpenAI embeddings deployment (set AZURE_OPENAI_* env vars).
- tfidf: offline TF-IDF vectors, no download needed (used in CI and air-gapped setups).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEmbeddings(Embeddings):
    """Deterministic, offline embeddings. Must be fitted on the corpus before use."""

    def __init__(self, max_features: int = 8000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True,
            stop_words="english", min_df=1,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> TfidfEmbeddings:
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must be fitted on the corpus first")
        mat = normalize(self.vectorizer.transform(texts)).toarray().astype(np.float32)
        return mat.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.vectorizer, fh)

    @classmethod
    def load(cls, path: Path) -> TfidfEmbeddings:
        obj = cls()
        with open(path, "rb") as fh:
            obj.vectorizer = pickle.load(fh)
        obj._fitted = True
        return obj


def make_embeddings(backend: str, model_name: str | None = None) -> Embeddings:
    if backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
    if backend == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    if backend == "tfidf":
        return TfidfEmbeddings()
    raise ValueError(f"unknown embedding backend: {backend}")
