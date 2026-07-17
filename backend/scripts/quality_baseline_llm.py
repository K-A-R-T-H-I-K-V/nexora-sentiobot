"""
quality_baseline_llm.py — Increment 3 SECONDARY/indicative metrics (spend Groq
tokens): multi-query hit-rate reference, tool-call correctness, RAGAS-style
faithfulness/relevancy (weak 8b judge), and adversarial refusal-correct %.

Budget-aware: every LLM answer comes back with the server's trailing `metrics`
event, so we sum real tokens; on a 429 the run stops gracefully and writes what
it has. RAGAS here is a documented CUSTOM judge (not the ragas library), named
indicative, judge = llama-3.1-8b-instant (distinct from the 70b generator).

Usage:  python -m backend.scripts.quality_baseline_llm
Writes: results/quality_llm.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core.config import get_settings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"
JUDGE_MODEL = "llama-3.1-8b-instant"
TOKENS = {"generation": 0, "judge": 0}
K = 5


def login():
    r = httpx.post(f"{BASE}/auth/login", data={"username": "alice", "password": "password123"}, timeout=30)
    return r.json()["access_token"]


class Quota429(Exception):
    pass


def chat(tok, message):
    """Returns (answer, tools[list of (name,input)], metrics, error)."""
    ans, tools, met, err = [], [], None, None
    with httpx.stream("POST", f"{BASE}/chat/stream",
                      headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                      json={"message": message, "conversation_id": None}, timeout=180) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev["type"]
            if t == "token":
                ans.append(ev["data"])
            elif t == "tool_start":
                tools.append((ev["data"].get("name"), str(ev["data"].get("input", ""))))
            elif t == "metrics":
                met = ev["data"]
            elif t == "error":
                err = ev["data"].get("message")
    if met:
        TOKENS["generation"] += met.get("prompt_tokens", 0) + met.get("completion_tokens", 0)
    if err and not "".join(ans):
        # a pre-answer error is almost certainly the daily 429
        raise Quota429(err)
    return "".join(ans), tools, met, err


_judge = None
def judge_llm():
    global _judge
    if _judge is None:
        from langchain_groq import ChatGroq
        s = get_settings()
        _judge = ChatGroq(model=JUDGE_MODEL, api_key=s.groq_api_key, temperature=0.0, max_tokens=200)
    return _judge


def judge_score(prompt) -> float | None:
    try:
        resp = judge_llm().invoke(prompt)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            raise Quota429(str(e))
        return None
    um = getattr(resp, "usage_metadata", None) or {}
    TOKENS["judge"] += (um.get("input_tokens", 0) + um.get("output_tokens", 0))
    m = re.search(r"(0(\.\d+)?|1(\.0+)?)", resp.content or "")
    return float(m.group(1)) if m else None


def yn_judge(prompt) -> bool | None:
    try:
        resp = judge_llm().invoke(prompt)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            raise Quota429(str(e))
        return None
    um = getattr(resp, "usage_metadata", None) or {}
    TOKENS["judge"] += (um.get("input_tokens", 0) + um.get("output_tokens", 0))
    return (resp.content or "").strip().lower().startswith("y")


def build_mq_retriever():
    from backend.agent.agent import get_retriever
    return get_retriever()


def build_base_ensemble():
    """Free, local base ensemble (no multi-query) for RAGAS context retrieval."""
    import pickle
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
    return EnsembleRetriever(retrievers=[bm25, vs.as_retriever(search_kwargs={"k": 5})], weights=[0.4, 0.6])


def doc_matches(md, content, acc):
    src = md.get("source", "")
    if "section_title" in acc:
        return src == acc["source"] and md.get("section_title", "") == acc["section_title"]
    if "content_contains" in acc:
        return src == acc["source"] and acc["content_contains"].lower() in (content or "").lower()
    return False


def main():
    from backend.scripts._golden import load_frozen
    gold = load_frozen()  # asserts the frozen items hash (R3-1)
    items = gold["items"]
    tok = login()
    out_path = REPO / "results/quality_llm.json"
    if out_path.exists():
        out = json.load(open(out_path, encoding="utf-8"))
        out.pop("stopped_early", None)
    else:
        out = {"judge_model": JUDGE_MODEL, "generator": get_settings().groq_model,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
               "note": "SECONDARY/indicative. RAGAS-style = custom weak-judge (8b grading 70b), not the ragas library.",
               "completed": []}
    out.setdefault("completed", [])

    try:
        # ---- 1) multi-query hit-rate reference (once) ----
        if "multi_query_reference" not in out["completed"]:
            retr = build_mq_retriever()
            rvals = [it for it in items if it.get("acceptable_sources") and "turns" not in it]
            hits = 0
            for it in rvals:
                docs = retr.invoke(it["question"])
                hit = any(any(doc_matches(d.metadata or {}, d.page_content, a) for a in it["acceptable_sources"]) for d in docs[:K])
                hits += 1 if hit else 0
            out["multi_query_reference"] = {"n": len(rvals), "hit_rate_at_5": round(hits / len(rvals), 4),
                                            "note": "full MultiQueryRetriever (adds 1 LLM call/query); compare to base-ensemble deterministic hit@5"}
            out["completed"].append("multi_query_reference")

        # ---- 2) tool-call correctness (sample) ----
        if "tool_call_correctness" not in out["completed"]:
          SAMPLE = ["ord-01", "ord-06", "war-01", "war-03", "war-06", "tkt-01", "tkt-03", "tkt-05"]
          tc = []
          for it in [x for x in items if x["id"] in SAMPLE]:
            ans, tools, met, err = chat(tok, it["question"])
            called = [t[0] for t in tools]
            right_tool = it["expected_tool"] in called
            args_ok = True
            if it.get("expected_args"):
                want = list(it["expected_args"].values())[0]
                args_ok = any(want in t[1] for t in tools if t[0] == it["expected_tool"])
            tc.append({"id": it["id"], "expected_tool": it["expected_tool"], "called": called,
                       "tool_correct": right_tool, "args_correct": bool(right_tool and args_ok)})
            time.sleep(1)
          out["tool_call_correctness"] = {"n": len(tc), "tool_correct": sum(r["tool_correct"] for r in tc),
                                          "args_correct": sum(r["args_correct"] for r in tc), "rows": tc}
          out["completed"].append("tool_call_correctness")

        # ---- 3) RAGAS-style faithfulness + relevancy (8 doc/policy Q, judge 2x) ----
        RAGAS_IDS = ["doc-03", "doc-09", "doc-12", "doc-14", "pol-02", "pol-03", "pol-05", "pol-07"]
        base_retr = build_base_ensemble()
        ragas = []
        for it in [x for x in items if x["id"] in RAGAS_IDS]:
            ans, tools, met, err = chat(tok, it["question"])
            if not ans:
                continue
            # faithfulness judged against the RETRIEVED context (free local retrieval), not the gold answer
            ctx_docs = base_retr.invoke(it["question"])[:K]
            ctx = "\n\n".join((d.page_content or "")[:900] for d in ctx_docs)
            faith, rel = [], []
            for _ in range(2):
                f = judge_score(f"Context:\n{ctx}\n\nAnswer:\n{ans}\n\nHow faithful is the Answer to the Context (is every claim supported)? Reply ONLY a number 0 to 1.")
                r = judge_score(f"Question:\n{it['question']}\n\nAnswer:\n{ans}\n\nHow relevant is the Answer to the Question? Reply ONLY a number 0 to 1.")
                if f is not None: faith.append(f)
                if r is not None: rel.append(r)
            ragas.append({"id": it["id"], "faithfulness": faith, "answer_relevancy": rel})
            time.sleep(1)
        def flat(key):
            xs = [v for r in ragas for v in r[key]]
            return xs
        fa, re_ = flat("faithfulness"), flat("answer_relevancy")
        out["ragas_style"] = {
            "n_questions": len(ragas),
            "faithfulness_mean": round(sum(fa)/len(fa), 3) if fa else None,
            "faithfulness_min_max": [min(fa), max(fa)] if fa else None,
            "answer_relevancy_mean": round(sum(re_)/len(re_), 3) if re_ else None,
            "answer_relevancy_min_max": [min(re_), max(re_)] if re_ else None,
            "rows": ragas,
        }
        out["completed"].append("ragas_style")

        # ---- 4) adversarial refusal-correct % ----
        adv = []
        for it in [x for x in items if x.get("scoring") == "adversarial"]:
            ans, tools, met, err = chat(tok, it["turns"][0]["question"] if "turns" in it else it["question"])
            chk = it["adversarial_check"]
            if chk["method"] == "mechanical":
                passed = not any(bad.lower() in ans.lower() for bad in chk["fail_if_response_contains_any"])
            else:
                passed = yn_judge(f"{chk['judge_question']}\n\nAssistant reply:\n{ans}\n\nReply ONLY 'yes' or 'no'.")
            adv.append({"id": it["id"], "method": chk["method"], "passed": bool(passed)})
            time.sleep(1)
        out["adversarial"] = {"n": len(adv), "refusal_correct": sum(1 for a in adv if a["passed"]),
                              "refusal_correct_pct": round(100*sum(1 for a in adv if a["passed"])/len(adv), 1) if adv else None,
                              "rows": adv}
        out["completed"].append("adversarial")

    except Quota429 as q:
        out["stopped_early"] = f"Groq daily quota hit: {str(q)[:160]}"

    out["tokens_spent"] = {"generation": TOKENS["generation"], "judge": TOKENS["judge"],
                           "total": TOKENS["generation"] + TOKENS["judge"]}
    (REPO / "results/quality_llm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("completed stages:", out["completed"])
    print("tokens spent:", out["tokens_spent"])
    for k in ("multi_query_reference", "tool_call_correctness", "ragas_style", "adversarial"):
        if k in out:
            print(f"  {k}: {json.dumps({kk: vv for kk, vv in out[k].items() if kk != 'rows'})}")
    if out.get("stopped_early"):
        print("STOPPED EARLY:", out["stopped_early"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
