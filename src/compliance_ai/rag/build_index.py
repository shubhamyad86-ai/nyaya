"""CLI: build the regulation vector index.

    python -m compliance_ai.rag.build_index [--backend huggingface|azure|tfidf]
"""
from __future__ import annotations

import argparse

from compliance_ai.rag.store import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index of regulation chunks")
    parser.add_argument("--backend", choices=["huggingface", "azure", "tfidf"], default=None,
                        help="embedding backend (default: config.yaml)")
    args = parser.parse_args()
    build_index(args.backend)


if __name__ == "__main__":
    main()
