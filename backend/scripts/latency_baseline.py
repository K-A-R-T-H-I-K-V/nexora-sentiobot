"""
latency_baseline.py — Increment 2 baseline: latency + API-call count + tokens.

Drives a small FIXED, deterministic request set through the REAL /chat/stream
endpoint on both routes and records, per request:
  - client-side TTFT (time to first token) and e2e (time to the done event)
  - the server's trailing `metrics` event: LLM calls, embedding ops, Supabase
    round trips, prompt/completion tokens (Groq usage), retrieval_ms, route.

Protocol (token-frugal; Groq free tier is ~100K tokens/day):
  1. warm-up (1 RAG + 1 tool, throwaway) absorbs first-request singleton
     cold-start so the measured set reflects steady state.
  2. COLD set: distinct queries, cache-miss, warm singletons.
  3. WARM set: re-ask two cold queries -> cache hit.

Reproducibility: RESTART the backend before each run so the in-process cache is
cold. The harness asserts cache_hit is False on the cold set and flags
contamination if a "cold" query returns cached (i.e. the backend was not
restarted).

Usage:  python -m backend.scripts.latency_baseline <run_label>
Writes: results/latency_run_<run_label>.json  (stamped: model, commit, date, hw)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core.config import get_settings  # noqa: E402

BASE = os.environ.get("BASELINE_BASE", "http://127.0.0.1:8000")
REPO = Path(__file__).resolve().parents[2]

WARMUP = [
    ("rag", "What is the battery life of the LumiGlow smart bulb?"),
    ("tool", "What is the status of order NX-2025-302?"),
]
COLD = [
    ("rag", "How do I install the LumiGlow smart light bulb?"),
    ("rag", "How do I pair the VisionSphere 360 camera with the app?"),
    ("rag", "How do I mount the Nexora Thermostat Pro on the wall?"),
    ("tool", "What is the status of order NX-2025-301?"),
    ("tool", "Is my Nexora Thermostat Pro still under warranty?"),
    ("tool", "I want to speak to a human agent about an unresolved issue."),
]
WARM_REASK_INDICES = [0, 3]  # re-ask COLD[0] (rag) and COLD[3] (tool) -> cache hit


def login() -> str:
    r = httpx.post(f"{BASE}/auth/login",
                   data={"username": "alice", "password": "password123"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def one_chat(tok: str, message: str) -> dict:
    t0 = time.perf_counter()
    ttft = None
    done_t = None
    metrics = None
    answer_len = 0
    errored = False
    with httpx.stream("POST", f"{BASE}/chat/stream",
                      headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                      json={"message": message, "conversation_id": None}, timeout=300) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev["type"]
            if t == "token":
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                answer_len += len(ev["data"])
            elif t == "done":
                done_t = (time.perf_counter() - t0) * 1000.0
            elif t == "metrics":
                metrics = ev["data"]
            elif t == "error":
                errored = True
    return {"ttft_ms": ttft, "e2e_ms": done_t, "answer_len": answer_len,
            "errored": errored, "metrics": metrics or {}}


def pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 1)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


def aggregate(rows: list[dict]) -> dict:
    ttft = [r["ttft_ms"] for r in rows]
    e2e = [r["e2e_ms"] for r in rows]

    def const(field):
        seen = sorted({r["metrics"].get(field) for r in rows if r["metrics"]})
        return seen[0] if len(seen) == 1 else seen

    tokens = [
        (r["metrics"].get("prompt_tokens", 0) + r["metrics"].get("completion_tokens", 0))
        for r in rows if r["metrics"]
    ]
    return {
        "n": len(rows),
        "ttft_ms_p50": pct(ttft, 0.50), "ttft_ms_p95": pct(ttft, 0.95),
        "e2e_ms_p50": pct(e2e, 0.50), "e2e_ms_p95": pct(e2e, 0.95),
        "retrieval_ms_p50": pct([r["metrics"].get("retrieval_ms") for r in rows if r["metrics"]], 0.50),
        "llm_calls": const("llm_calls"),
        "embedding_ops": const("embedding_ops"),
        "supabase_calls": const("supabase_calls"),
        "tokens_total_p50": pct(tokens, 0.50),
        "tokens_total_max": max(tokens) if tokens else None,
        "prompt_tokens_p50": pct([r["metrics"].get("prompt_tokens") for r in rows if r["metrics"]], 0.50),
        "completion_tokens_p50": pct([r["metrics"].get("completion_tokens") for r in rows if r["metrics"]], 0.50),
    }


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "1"
    s = get_settings()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    tok = login()

    # 1) warm-up (absorb singleton cold-start). Capture the very first TTFT as
    #    the one-time cold-start cost.
    cold_start = one_chat(tok, WARMUP[0][1])
    for _, q in WARMUP[1:]:
        one_chat(tok, q)

    # 2) cold set
    cold_rows = {"rag": [], "tool": []}
    contamination = []
    for route, q in COLD:
        r = one_chat(tok, q)
        r["route"] = route
        r["query"] = q
        if r["metrics"].get("cache_hit"):
            contamination.append(q)
        cold_rows[route].append(r)

    # 3) warm (cache hit) set
    warm_rows = []
    for i in WARM_REASK_INDICES:
        route, q = COLD[i]
        r = one_chat(tok, q)
        r["route"] = route
        r["query"] = q
        r["was_cache_hit"] = bool(r["metrics"].get("cache_hit"))
        warm_rows.append(r)

    result = {
        "run_label": label,
        "stamp": {
            "provider": s.llm_provider,
            "model": s.groq_model,
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "commit": commit,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": __import__("backend.scripts._golden", fromlist=["hardware_stamp"]).hardware_stamp(),
            "note": "TTFT/e2e are client-side over localhost; counts+tokens are server-side (trailing metrics event).",
        },
        "first_request_cold_start_ttft_ms": round(cold_start["ttft_ms"], 1) if cold_start["ttft_ms"] else None,
        "cache_contamination": contamination,
        "rag_cold": aggregate(cold_rows["rag"]),
        "tool_cold": aggregate(cold_rows["tool"]),
        "cache_warm": {
            "n": len(warm_rows),
            "all_were_cache_hits": all(r["was_cache_hit"] for r in warm_rows),
            "ttft_ms_p50": pct([r["ttft_ms"] for r in warm_rows], 0.50),
            "e2e_ms_p50": pct([r["e2e_ms"] for r in warm_rows], 0.50),
            "supabase_calls": sorted({r["metrics"].get("supabase_calls") for r in warm_rows}),
        },
        "raw": {"rag_cold": cold_rows["rag"], "tool_cold": cold_rows["tool"], "warm": warm_rows},
    }

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"latency_run_{label}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nRun {label} written to {out_path}")
    print(f"  first-request cold-start TTFT: {result['first_request_cold_start_ttft_ms']} ms")
    if contamination:
        print(f"  WARNING cache contamination (restart the backend!): {contamination}")
    for route in ("rag_cold", "tool_cold"):
        a = result[route]
        print(f"  {route}: TTFT p50 {a['ttft_ms_p50']} / p95 {a['ttft_ms_p95']} ms | "
              f"e2e p50 {a['e2e_ms_p50']} ms | LLM {a['llm_calls']} embeds {a['embedding_ops']} "
              f"supabase {a['supabase_calls']} | tokens/req p50 {a['tokens_total_p50']}")
    cw = result["cache_warm"]
    print(f"  cache_warm: TTFT p50 {cw['ttft_ms_p50']} ms e2e p50 {cw['e2e_ms_p50']} ms "
          f"(all hits: {cw['all_were_cache_hits']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
