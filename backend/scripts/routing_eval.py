"""
routing_eval.py - Feature F1 A/B: keyword router vs embedding intent classifier.

Free and deterministic: no LLM call, no Supabase, no paid tokens. It loads the
local ONNX MiniLM embedding model once (a free, public download, same as the
retrieval eval) and runs both routers over results/routing_set_v1.json, then
reports routing accuracy (route matches the labeled ideal route) for each router,
a per-intent breakdown, and the specific keyword-misroute cases.

It also VERIFIES the dataset's own claims: every item tagged keyword_misroute=true
must actually be misrouted by the keyword router (else the label is stale), and
the confidence/fallback stats are reported so the threshold can be sanity-checked.

Usage:  python -m backend.scripts.routing_eval
Writes: results/routing_eval.json and results/routing_eval.md
        (stamped: commit, model, date, hardware). Exit 1 if embedding does not
        at least match the keyword router, or a dataset claim is stale.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agent.intent_router import (  # noqa: E402
    INTENT_PROTOTYPES, INTENT_ROUTE, classify_intent, keyword_route, route_message,
)
from backend.core.config import get_settings  # noqa: E402
from backend.core.onnx_embeddings import get_embeddings  # noqa: E402
from backend.scripts._golden import hardware_stamp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("SENTIOBOT_RESULTS_DIR", str(REPO / "results")))

# F1-R1 leakage guard: an eval question this close to a prototype is measuring
# memorization, not generalization, so it must not be in the set.
LEAK_COSINE = 0.90


def _load_set() -> dict:
    return json.load(open(REPO / "results" / "routing_set_v1.json", encoding="utf-8"))


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def leakage_report(items: list[dict]) -> dict:
    """For each eval question, its max cosine to any prototype and whether it is a
    normalized string copy of one. Flags anything >= LEAK_COSINE or an exact copy:
    such an item grades the classifier against its own prototype text (F1-R1)."""
    emb = get_embeddings()
    protos = [(intent, p) for intent, ps in INTENT_PROTOTYPES.items() for p in ps]
    pv = np.array(emb.embed_documents([p for _, p in protos]), dtype=np.float32)
    pv = pv / np.linalg.norm(pv, axis=1, keepdims=True)
    pnorm = {_normalize(p) for _, p in protos}

    per_item, leaks = {}, []
    for it in items:
        q = it["question"]
        qv = np.array(emb.embed_query(q), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        sims = pv @ qv
        j = int(np.argmax(sims))
        max_cos = round(float(sims[j]), 4)
        str_copy = _normalize(q) in pnorm
        per_item[it["id"]] = {"max_prototype_cosine": max_cos,
                              "nearest_prototype": protos[j][1], "string_copy": str_copy}
        if max_cos >= LEAK_COSINE or str_copy:
            leaks.append({"id": it["id"], "max_cosine": max_cos, "string_copy": str_copy})
    return {"leak_cosine_threshold": LEAK_COSINE,
            "max_cosine_over_set": max(v["max_prototype_cosine"] for v in per_item.values()),
            "leaks": leaks, "per_item": per_item}


def evaluate() -> dict:
    data = _load_set()
    items = data["items"]
    s = get_settings()

    rows = []
    kw_correct = emb_correct = 0
    by_intent_total: dict[str, int] = defaultdict(int)
    by_intent_emb_ok: dict[str, int] = defaultdict(int)
    fallback_used = 0
    stale_labels = []
    gate_failures = []

    for it in items:
        q = it["question"]
        ideal = it["route"]

        kw = keyword_route(q)
        dec = route_message(q, router="embedding")
        intent, conf, scores = classify_intent(q)

        kw_ok = kw == ideal
        emb_ok = dec.route == ideal
        kw_correct += kw_ok
        emb_correct += emb_ok
        by_intent_total[it["intent"]] += 1
        by_intent_emb_ok[it["intent"]] += emb_ok
        if dec.intent == "low_confidence_fallback":
            fallback_used += 1

        # Dataset integrity: a claimed misroute must actually be misrouted by keyword.
        if it.get("keyword_misroute") and kw_ok:
            stale_labels.append({"id": it["id"], "reason": "labeled keyword_misroute but keyword routed correctly"})
        if it.get("keyword_misroute") is None and not kw_ok:
            stale_labels.append({"id": it["id"], "reason": "keyword misrouted but not labeled keyword_misroute"})

        # Gate items must route correctly under the embedding router.
        if it.get("gate_must_route_correct") and not emb_ok:
            gate_failures.append({"id": it["id"], "question": q, "ideal": ideal, "got": dec.route, "intent": dec.intent})

        rows.append({
            "id": it["id"], "question": q, "ideal_intent": it["intent"], "ideal_route": ideal,
            "keyword_route": kw, "keyword_ok": kw_ok,
            "embedding_route": dec.route, "embedding_intent": dec.intent,
            "embedding_ok": emb_ok, "confidence": round(conf, 4),
            "top_scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]},
            "keyword_misroute_claimed": bool(it.get("keyword_misroute")),
        })

    n = len(items)
    kw_acc = round(kw_correct / n, 4)
    emb_acc = round(emb_correct / n, 4)

    # Focused accuracy on the keyword-misroute subset (where F1 must earn its keep).
    mis = [r for r in rows if r["keyword_misroute_claimed"]]
    mis_fixed = sum(1 for r in mis if r["embedding_ok"])

    # F1-R1: prove the eval measures generalization, not memorization.
    leakage = leakage_report(items)

    out = {
        "test": "F1 routing A/B: keyword router vs embedding intent classifier (free, deterministic, 0 tokens)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "as_of": "2026-07-18",
        "hardware": hardware_stamp(),
        "embedding_model": s.embedding_model + " (ONNX Runtime via fastembed)",
        "router_config": {"threshold": s.intent_confidence_threshold, "intents": list(INTENT_ROUTE.keys())},
        "routing_set_version": data.get("version"),
        "total": n,
        "keyword_accuracy": kw_acc,
        "embedding_accuracy": emb_acc,
        "delta": round(emb_acc - kw_acc, 4),
        "keyword_misroute_subset": {"total": len(mis), "fixed_by_embedding": mis_fixed},
        "embedding_by_intent": {k: f"{by_intent_emb_ok[k]}/{by_intent_total[k]}" for k in by_intent_total},
        "low_confidence_fallbacks": fallback_used,
        "leakage_guard": {"threshold": leakage["leak_cosine_threshold"],
                          "max_cosine_over_set": leakage["max_cosine_over_set"],
                          "leaks": leakage["leaks"]},
        "limitations": data.get("limitations", []),
        "stale_labels": stale_labels,
        "gate_failures": gate_failures,
        "rows": rows,
        "passed": (emb_acc >= kw_acc and not gate_failures and not stale_labels
                   and not leakage["leaks"]),
    }
    return out


def _write_md(out: dict) -> None:
    lines = [
        "# Feature F1 - Intent routing A/B (keyword vs embedding classifier)",
        "",
        f"- Commit: `{out['commit']}`  |  As-of: {out['as_of']}",
        f"- Embedding model: {out['embedding_model']} (local, 0 LLM calls, 0 tokens)",
        f"- Hardware: {out['hardware']}",
        f"- Confidence threshold: {out['router_config']['threshold']}",
        "",
        "## Headline",
        f"- Keyword router accuracy:   **{out['keyword_accuracy']:.3f}** ({out['total']} queries)",
        f"- Embedding router accuracy: **{out['embedding_accuracy']:.3f}**  (delta {out['delta']:+.3f})",
        f"- Keyword-misroute cases fixed by embedding: "
        f"**{out['keyword_misroute_subset']['fixed_by_embedding']}/{out['keyword_misroute_subset']['total']}**",
        f"- Low-confidence fallbacks to keyword: {out['low_confidence_fallbacks']}/{out['total']}",
        "",
        "## Leakage guard (F1-R1: measures generalization, not memorization)",
        f"- No eval question copies or near-copies a prototype: "
        f"**{'PASS' if not out['leakage_guard']['leaks'] else 'FAIL'}** "
        f"(max cosine to any prototype over the set = {out['leakage_guard']['max_cosine_over_set']:.3f}, "
        f"threshold {out['leakage_guard']['threshold']})",
        "",
        "## Embedding accuracy by intent",
    ]
    for k, v in out["embedding_by_intent"].items():
        lines.append(f"- {k}: {v}")
    if out["limitations"]:
        lines += ["", "## Known limitations"]
        for lim in out["limitations"]:
            lines.append(f"- {lim}")
    lines += ["", "## Per-query", "", "| id | ideal | keyword | emb intent | emb route | ok | conf |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in out["rows"]:
        ok = "ok" if r["embedding_ok"] else "MISS"
        lines.append(
            f"| {r['id']} | {r['ideal_route']} | {r['keyword_route']}"
            f"{' (wrong)' if not r['keyword_ok'] else ''} | {r['embedding_intent']} | "
            f"{r['embedding_route']} | {ok} | {r['confidence']:.3f} |"
        )
    (RESULTS_DIR / "routing_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out = evaluate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "routing_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(out)

    print(f"keyword accuracy:   {out['keyword_accuracy']:.3f}")
    print(f"embedding accuracy: {out['embedding_accuracy']:.3f}  (delta {out['delta']:+.3f})")
    print(f"misroutes fixed:    {out['keyword_misroute_subset']['fixed_by_embedding']}"
          f"/{out['keyword_misroute_subset']['total']}")
    print(f"fallbacks:          {out['low_confidence_fallbacks']}/{out['total']}")
    print(f"leakage guard:      {'PASS' if not out['leakage_guard']['leaks'] else 'FAIL'}"
          f" (max cosine {out['leakage_guard']['max_cosine_over_set']:.3f}"
          f" < {out['leakage_guard']['threshold']})")
    if out["leakage_guard"]["leaks"]:
        print("LEAKAGE:", json.dumps(out["leakage_guard"]["leaks"], indent=2))
    if out["stale_labels"]:
        print("STALE DATASET LABELS:", json.dumps(out["stale_labels"], indent=2))
    if out["gate_failures"]:
        print("GATE FAILURES:", json.dumps(out["gate_failures"], indent=2))
    if not out["passed"]:
        print("FAIL: embedding did not beat/match keyword, or a dataset/gate check failed.")
        return 1
    print("PASS: embedding router >= keyword router; all gate items and dataset labels consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
