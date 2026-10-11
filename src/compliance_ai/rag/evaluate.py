"""Evaluate retrieval quality on a hand-labelled GDPR + EU AI Act question set.

    python -m compliance_ai.rag.evaluate

Each question lists acceptable citations; the first is the primary article.
Metrics
- hit@1 / hit@k: an acceptable citation is ranked first / within the top k
- primary_article@k: the primary article itself is within the top k
- MRR@10: mean reciprocal rank of the first acceptable citation in the top 10
"""
from __future__ import annotations

import json

from compliance_ai.config import load_config, resolve
from compliance_ai.rag.store import load_index


def _rank(store, question: str, regulation: str | None = None) -> list[str]:
    kwargs = {"filter": {"regulation": regulation}, "fetch_k": 400} if regulation else {}
    ranked = []
    for doc in store.similarity_search(question, k=10, **kwargs):
        c = doc.metadata["citation"]
        if c not in ranked:            # several chunks can come from one article
            ranked.append(c)
    return ranked


def _scores(rows: list[dict], ranked_lists: list[list[str]], k: int) -> dict:
    hit1 = hitk = primary = rr = 0.0
    for row, ranked in zip(rows, ranked_lists, strict=True):
        expected = set(row["expected"])
        first = next((i for i, c in enumerate(ranked) if c in expected), None)
        hit1 += first == 0
        hitk += first is not None and first < k
        primary += row["expected"][0] in ranked[:k]
        rr += 1 / (first + 1) if first is not None else 0.0
    n = len(rows)
    return {"n_questions": n, "hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3),
            f"primary_article@{k}": round(primary / n, 3), "mrr@10": round(rr / n, 3)}


def evaluate(k: int | None = None, index_dir=None) -> dict:
    cfg = load_config()["rag"]
    k = k or cfg["top_k"]
    store, meta = load_index(index_dir)
    rows = [json.loads(line) for line in resolve(cfg["eval_path"]).read_text().splitlines() if line.strip()]
    for row in rows:
        row["regulation"] = row["expected"][0].rsplit(" ", 2)[0]     # "GDPR Article 22" -> "GDPR"

    unfiltered = [_rank(store, r["question"]) for r in rows]
    filtered = [_rank(store, r["question"], r["regulation"]) for r in rows]

    result = {
        "embedding_backend": meta["embedding_backend"],
        "embedding_model": meta["embedding_model"],
        "k": k,
        "overall": _scores(rows, unfiltered, k),
        "with_regulation_filter": _scores(rows, filtered, k),
        "by_regulation": {},
        "details": [],
    }
    for reg in sorted({r["regulation"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["regulation"] == reg]
        result["by_regulation"][reg] = {
            "search_all": _scores([rows[i] for i in idx], [unfiltered[i] for i in idx], k),
            "filtered": _scores([rows[i] for i in idx], [filtered[i] for i in idx], k),
        }
    for row, ranked in zip(rows, unfiltered, strict=True):
        first = next((i for i, c in enumerate(ranked) if c in set(row["expected"])), None)
        result["details"].append({"question": row["question"], "expected": row["expected"][0],
                                  "retrieved": ranked[:k],
                                  "rank_of_first_match": None if first is None else first + 1})

    out = resolve(load_config()["credit"]["report_dir"]) / f"rag_eval_{meta['embedding_backend']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = evaluate()
    print(f"{r['embedding_backend']} embeddings, k={r['k']}")
    print("  all questions, search all laws:", r["overall"])
    print("  all questions, regulation filter:", r["with_regulation_filter"])
    for reg, v in r["by_regulation"].items():
        print(f"  {reg}: search all {v['search_all']}")
    for d in r["details"]:
        if d["rank_of_first_match"] is None:
            print(f"  MISS: {d['question']}  (expected {d['expected']}, got {d['retrieved']})")
