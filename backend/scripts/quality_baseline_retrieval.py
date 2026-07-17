"""
quality_baseline_retrieval.py — Increment 3 PRIMARY metric (free, local, zero
Groq tokens, bit-stable): retrieval hit-rate@k + context precision on the frozen
golden set, measured against the BASE ENSEMBLE (BM25 + vector, weights 0.4/0.6),
NOT the multi-query wrapper. This is the deterministic headline and the honest
before/after anchor for Increment 4.

Only single-turn retrieval items (those with acceptable_sources) are scored here;
context-dependent follow-up turns are evaluated end to end in the LLM pass.

Usage:  python -m backend.scripts.quality_baseline_retrieval <run_label>
Writes: results/quality_retrieval_run_<label>.json
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core.config import get_settings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
K = 5


def build_base_ensemble():
    """Replicates get_retriever's ensemble EXACTLY, minus the multi-query wrapper."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    s = get_settings()
    emb = HuggingFaceEmbeddings(model_name=s.embedding_model)
    vs = Chroma(persist_directory=s.vector_db_path, embedding_function=emb)
    with open(s.parent_list_path, "rb") as f:
        parents = pickle.load(f)
    bm25 = BM25Retriever.from_documents(parents)
    chroma_ret = vs.as_retriever(search_kwargs={"k": 5})
    return EnsembleRetriever(retrievers=[bm25, chroma_ret], weights=[0.4, 0.6])


def doc_matches(doc, acc) -> bool:
    md = doc.metadata or {}
    src = md.get("source", "")
    if "section_title" in acc:
        return src == acc["source"] and md.get("section_title", "") == acc["section_title"]
    if "content_contains" in acc:
        return src == acc["source"] and acc["content_contains"].lower() in (doc.page_content or "").lower()
    return False


def first_hit_rank(docs, acceptable) -> int | None:
    for i, d in enumerate(docs[:K]):
        if any(doc_matches(d, a) for a in acceptable):
            return i + 1
    return None


def relevant_count_at_k(docs, acceptable) -> int:
    return sum(1 for d in docs[:K] if any(doc_matches(d, a) for a in acceptable))


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "1"
    gold = json.load(open(REPO / "results/golden_set_v1.json", encoding="utf-8"))
    items = [it for it in gold["items"] if it.get("acceptable_sources") and "turns" not in it]

    retr = build_base_ensemble()
    rows = []
    for it in items:
        docs = retr.invoke(it["question"])
        rank = first_hit_rank(docs, it["acceptable_sources"])
        rows.append({
            "id": it["id"], "category": it["category"],
            "hit_at_1": rank == 1, "hit_at_3": bool(rank and rank <= 3), "hit_at_5": bool(rank and rank <= 5),
            "first_hit_rank": rank,
            "precision_at_5": relevant_count_at_k(docs, it["acceptable_sources"]) / K,
            "top5": [{"source": (d.metadata or {}).get("source"), "section": (d.metadata or {}).get("section_title")} for d in docs[:K]],
        })

    n = len(rows)
    def rate(key): return round(sum(1 for r in rows if r[key]) / n, 4)
    mrr = round(sum((1.0 / r["first_hit_rank"]) if r["first_hit_rank"] else 0.0 for r in rows) / n, 4)
    cp = round(sum(r["precision_at_5"] for r in rows) / n, 4)

    # per-category hit@5
    cats = {}
    for r in rows:
        c = r["category"]; cats.setdefault(c, [0, 0]); cats[c][1] += 1
        if r["hit_at_5"]:
            cats[c][0] += 1
    cat_hit5 = {c: f"{v[0]}/{v[1]}" for c, v in cats.items()}

    result = {
        "run_label": label,
        "metric": "retrieval on BASE ENSEMBLE (bm25 0.4 + vector 0.6, k=5); zero Groq tokens; bit-stable",
        "n_items": n,
        "hit_rate_at_1": rate("hit_at_1"),
        "hit_rate_at_3": rate("hit_at_3"),
        "hit_rate_at_5": rate("hit_at_5"),
        "mrr": mrr,
        "context_precision_at_5": cp,
        "per_category_hit_at_5": cat_hit5,
        "misses_at_5": [r["id"] for r in rows if not r["hit_at_5"]],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "rows": rows,
    }
    out = REPO / f"results/quality_retrieval_run_{label}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run {label} -> {out}")
    print(f"  n={n}  hit@1={result['hit_rate_at_1']}  hit@3={result['hit_rate_at_3']}  "
          f"hit@5={result['hit_rate_at_5']}  MRR={mrr}  ctx_prec@5={cp}")
    print(f"  per-category hit@5: {cat_hit5}")
    print(f"  misses@5: {result['misses_at_5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
