"""
increment4_delta.py — Increment 4 (P3 Rank 1) measured delta: cut multi-query.

Controlled A/B on the RAG answer path, isolating ONE variable (the retriever):
for each question, retrieve context with the base ensemble vs the multi-query
retriever, generate the answer with the SAME LLM + RAG prompt, and record LLM
calls, tokens, timing, the answer text, and a weak-judge faithfulness score.
Retrieval hit@k before/after is already frozen (both 0.913, Increment 3).

Budget-aware (stops on 429, saves partial). Judge = llama-3.1-8b-instant.
Usage:  python -m backend.scripts.increment4_delta
Writes: results/increment4_delta.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core.config import get_settings  # noqa: E402
from backend.core import metrics  # noqa: E402
from backend.scripts._golden import load_frozen, hardware_stamp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
K = 5
JUDGE_MODEL = "llama-3.1-8b-instant"
SUBSET = ["doc-03", "doc-09", "doc-12", "pol-02", "pol-03"]  # from the RAGAS subset (comparable)
ALICE = {"name": "Alice", "owned_products": [
    {"product_name": "Nexora Thermostat Pro", "serial_number": "SN-NTS-PRO-ABC123"},
    {"product_name": "LumiGlow Smart Light", "serial_number": "SN-NLRGB-LMO456"}]}
TOK = {"gen": 0, "judge": 0}


class Quota429(Exception):
    pass


def base_ensemble():
    import pickle
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from backend.core.onnx_embeddings import get_embeddings
    s = get_settings()
    emb = metrics.CountingEmbeddings(get_embeddings(s.embedding_model))
    vs = Chroma(persist_directory=s.vector_db_path, embedding_function=emb)
    with open(s.parent_list_path, "rb") as f:
        parents = pickle.load(f)
    bm25 = BM25Retriever.from_documents(parents)
    return EnsembleRetriever(retrievers=[bm25, vs.as_retriever(search_kwargs={"k": 5})], weights=[0.4, 0.6])


def multiquery(base):
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.prompts import PromptTemplate
    from backend.agent.agent import get_llm
    p = PromptTemplate.from_template(
        "Generate 3 alternative phrasings of this question for vector search. "
        "Return them separated by newlines.\nQuestion: {question}")
    return MultiQueryRetriever.from_llm(retriever=base, llm=get_llm(), prompt=p)


async def answer_with(retriever, question):
    """Replicates the RAG answer path with the given retriever; returns dict."""
    from backend.agent.agent import get_llm, _build_system_message, _format_sources
    from langchain_core.messages import SystemMessage, HumanMessage
    metrics.start()
    t0 = time.perf_counter()
    docs = await retriever.ainvoke(question, config=metrics.callback_config())
    retr_ms = (time.perf_counter() - t0) * 1000
    # Use each retriever's natural output, exactly like the live _retrieve_context.
    ctx = "\n\n".join(f"[Source {i+1}] ({d.metadata.get('source','?')} | {d.metadata.get('section_title','?')})\n{d.page_content}"
                      for i, d in enumerate(docs))
    sys_msg = SystemMessage(content=f"{_build_system_message(ALICE)}\n\n## Retrieved Documentation\n{ctx}")
    t1 = time.perf_counter()
    try:
        resp = await get_llm().ainvoke([sys_msg, HumanMessage(content=question)], config=metrics.callback_config())
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            raise Quota429(str(e))
        raise
    gen_ms = (time.perf_counter() - t1) * 1000
    m = metrics.get()
    TOK["gen"] += m.prompt_tokens + m.completion_tokens
    return {"answer": resp.content, "llm_calls": m.llm_calls, "tokens": m.prompt_tokens + m.completion_tokens,
            "retrieval_ms": round(retr_ms, 1), "gen_ms": round(gen_ms, 1),
            "sources": _format_sources(docs), "context": ctx}


_judge = None
def judge_faithfulness(answer, context):
    global _judge
    if _judge is None:
        from langchain_groq import ChatGroq
        _judge = ChatGroq(model=JUDGE_MODEL, api_key=get_settings().groq_api_key, temperature=0, max_tokens=120)
    try:
        r = _judge.invoke(f"Context:\n{context[:3000]}\n\nAnswer:\n{answer}\n\nHow faithful is the Answer to the Context (is every claim supported)? Reply ONLY a number 0 to 1.")
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            raise Quota429(str(e))
        return None
    import re
    um = getattr(r, "usage_metadata", None) or {}
    TOK["judge"] += um.get("input_tokens", 0) + um.get("output_tokens", 0)
    mt = re.search(r"(0(\.\d+)?|1(\.0+)?)", r.content or "")
    return float(mt.group(1)) if mt else None


async def main():
    import asyncio  # noqa
    gold = load_frozen()
    by_id = {it["id"]: it for it in gold["items"]}
    base = base_ensemble()
    mq = multiquery(base)

    rows = []
    stopped = None
    try:
        for qid in SUBSET:
            it = by_id[qid]
            q = it["question"]
            # BEFORE = multi-query ON; AFTER = base ensemble
            before = await answer_with(mq, q)
            after = await answer_with(base, q)
            # key-fact presence: a distinctive token from the expected answer
            fact_tokens = [w for w in it["expected_answer"].replace(",", " ").split() if w.isdigit() or (w.isupper() and len(w) > 2)]
            fact = fact_tokens[0] if fact_tokens else None
            rows.append({
                "id": qid,
                "before_llm_calls": before["llm_calls"], "after_llm_calls": after["llm_calls"],
                "before_tokens": before["tokens"], "after_tokens": after["tokens"],
                "before_gen_ms": before["gen_ms"], "after_gen_ms": after["gen_ms"],
                "before_retrieval_ms": before["retrieval_ms"], "after_retrieval_ms": after["retrieval_ms"],
                "before_faithfulness": [judge_faithfulness(before["answer"], before["context"]) for _ in range(2)],
                "after_faithfulness": [judge_faithfulness(after["answer"], after["context"]) for _ in range(2)],
                "key_fact": fact,
                "fact_in_before": (fact in before["answer"]) if fact else None,
                "fact_in_after": (fact in after["answer"]) if fact else None,
                "before_cites_source": "[Source" in before["answer"],
                "after_cites_source": "[Source" in after["answer"],
                "before_sources": before["sources"], "after_sources": after["sources"],
                "before_answer": before["answer"], "after_answer": after["answer"],
            })
            time.sleep(1)
    except Quota429 as e:
        stopped = str(e)[:140]

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 3) if xs else None

    all_bf = [v for r in rows for v in r["before_faithfulness"]]
    all_af = [v for r in rows for v in r["after_faithfulness"]]
    out = {
        "increment": "4 (P3 Rank 1) - cut multi-query",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "generator": get_settings().groq_model, "judge_model": JUDGE_MODEL,
        "hardware": hardware_stamp(), "as_of": "2026-07-18",
        "golden_items_sha256": gold.get("items_sha256"),
        "n": len(rows),
        "retrieval_hit_at_5": {"before_multiquery": 0.913, "after_base_ensemble": 0.913,
                               "note": "from the frozen Increment 3 deterministic baseline; identical."},
        "llm_calls_per_doc_query": {"before": mean([r["before_llm_calls"] for r in rows]),
                                    "after": mean([r["after_llm_calls"] for r in rows])},
        "tokens_per_answer": {"before": mean([r["before_tokens"] for r in rows]),
                              "after": mean([r["after_tokens"] for r in rows])},
        "gen_time_ms": {"before": mean([r["before_gen_ms"] for r in rows]),
                        "after": mean([r["after_gen_ms"] for r in rows])},
        "retrieval_time_ms": {"before": mean([r["before_retrieval_ms"] for r in rows]),
                              "after": mean([r["after_retrieval_ms"] for r in rows])},
        "faithfulness": {"before_mean": mean(all_bf), "after_mean": mean(all_af),
                         "note": "weak 8b judge, indicative; compare within its band"},
        "answer_diff": {"fact_preserved_after": sum(1 for r in rows if r["fact_in_after"]),
                        "fact_checked": sum(1 for r in rows if r["key_fact"]),
                        "after_cites_source": sum(1 for r in rows if r["after_cites_source"])},
        "tokens_spent": {"gen": TOK["gen"], "judge": TOK["judge"], "total": TOK["gen"] + TOK["judge"]},
        "stopped_early": stopped,
        "rows": rows,
    }
    (REPO / "results/increment4_delta.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("n:", out["n"], "| stopped:", stopped)
    print("LLM calls/doc:", out["llm_calls_per_doc_query"])
    print("tokens/answer:", out["tokens_per_answer"])
    print("gen_time_ms:", out["gen_time_ms"], "| retrieval_ms:", out["retrieval_time_ms"])
    print("faithfulness:", out["faithfulness"])
    print("answer_diff:", out["answer_diff"])
    print("tokens spent:", out["tokens_spent"])
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
