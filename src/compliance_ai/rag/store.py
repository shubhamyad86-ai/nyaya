"""Build, save and load the FAISS vector index of regulation chunks."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from compliance_ai.config import load_config, resolve
from compliance_ai.rag.embeddings import TfidfEmbeddings, make_embeddings
from compliance_ai.rag.ingest import load_corpus

META_FILE = "index_meta.json"
TFIDF_FILE = "tfidf_vectorizer.pkl"


def build_index(backend: str | None = None, index_dir: Path | None = None) -> FAISS:
    cfg = load_config()["rag"]
    backend = backend or cfg["embedding_backend"]
    index_dir = index_dir or resolve(cfg["index_dir"])
    chunks = load_corpus(resolve(cfg["corpus_dir"]), cfg["chunk_size"], cfg["chunk_overlap"])

    emb = make_embeddings(backend, cfg["embedding_model"])
    if isinstance(emb, TfidfEmbeddings):
        emb.fit([c.page_content for c in chunks])

    # Embeddings are L2-normalised, so inner product = cosine similarity.
    store = FAISS.from_documents(chunks, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    if isinstance(emb, TfidfEmbeddings):
        emb.save(index_dir / TFIDF_FILE)
    sources = sorted({c.metadata["source"] for c in chunks})
    (index_dir / META_FILE).write_text(json.dumps({
        "embedding_backend": backend,
        "embedding_model": cfg["embedding_model"] if backend == "huggingface" else backend,
        "n_chunks": len(chunks),
        "sources": sources,
    }, indent=2))
    print(f"Indexed {len(chunks)} chunks from {len(sources)} file(s) "
          f"with '{backend}' embeddings -> {index_dir}")
    return store


def load_index(index_dir: Path | None = None) -> tuple[FAISS, dict]:
    cfg = load_config()["rag"]
    index_dir = index_dir or resolve(cfg["index_dir"])
    meta_path = index_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"no index at {index_dir} - run `python -m compliance_ai.rag.build_index`")
    meta = json.loads(meta_path.read_text())
    if meta["embedding_backend"] == "tfidf":
        emb = TfidfEmbeddings.load(index_dir / TFIDF_FILE)
    else:
        emb = make_embeddings(meta["embedding_backend"], cfg["embedding_model"])
    # The index files are produced by this project, so local deserialisation is trusted.
    store = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True,
                             distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    return store, meta
