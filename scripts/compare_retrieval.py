"""Compare keyword (TF-IDF) and semantic (sentence-transformers) retrieval on the same questions.

    pip install -r requirements-hf.txt      # once; downloads PyTorch + the embedding model
    python scripts/compare_retrieval.py

Builds one index per backend in models/compare/<backend>/, evaluates both on
eval/regulation_retrieval_eval.jsonl and writes reports/rag_comparison.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compliance_ai.config import load_config, resolve  # noqa: E402
from compliance_ai.rag.evaluate import evaluate  # noqa: E402
from compliance_ai.rag.store import build_index  # noqa: E402


def main() -> None:
    cfg = load_config()
    rows, k = [], cfg["rag"]["top_k"]
    for backend in ["tfidf", "huggingface"]:
        index_dir = resolve(cfg["credit"]["model_dir"]) / "compare" / backend
        try:
            build_index(backend=backend, index_dir=index_dir)
        except ImportError as exc:
            print(f"Skipping {backend}: {exc}. Install with: pip install -r requirements-hf.txt")
            continue
        r = evaluate(index_dir=index_dir)
        for scope, scores in [("search all laws", r["overall"]),
                              ("regulation filter", r["with_regulation_filter"])]:
            rows.append(f"| {r['embedding_model']} | {scope} | {scores['hit@1']:.2f} | "
                        f"{scores[f'hit@{k}']:.2f} | {scores['mrr@10']:.2f} |")
    lines = ["# Retrieval comparison: keyword vs semantic", "",
             f"{r['overall']['n_questions']} labelled questions (GDPR + EU AI Act), top {k} passages.", "",
             f"| Embeddings | Search | hit@1 | hit@{k} | MRR@10 |", "|---|---|---|---|---|", *rows]
    out = resolve(cfg["credit"]["report_dir"]) / "rag_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
