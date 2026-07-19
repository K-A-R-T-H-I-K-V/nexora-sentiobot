"""
sentiment_eval.py - Feature F4 gate: validate emotion detection and, more
importantly, the NO-FALSE-ESCALATION property (F2's no-false-green applied to
emotion). Free, deterministic, 0 tokens (local ONNX MiniLM only).

Reports:
- Detection accuracy per label (calm/confused/frustrated/angry).
- FALSE-ESCALATION rate: fraction of no-escalate controls (calm/confused/
  emphatic_calm/positive single messages) that wrongly escalate. This is the
  load-bearing number; a proactive "want a human?" to a calm/happy user is the
  cardinal sin. Must be 0.
- Escalation on multi-turn scenarios: fires on sustained frustration, not on calm/
  resolved.
- Leakage guard vs the prototypes (like the routing set).

Usage:  python -m backend.scripts.sentiment_eval
Writes: results/sentiment_eval.json (+ .md), stamped. Exit 1 on any false
escalation, an escalation-scenario miss, or a leak.
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
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("JWT_SECRET", "sentiment-eval-secret")
os.environ.setdefault("GROQ_API_KEY", "")

from backend.core import sentiment as S               # noqa: E402
from backend.core.config import get_settings          # noqa: E402
from backend.core.onnx_embeddings import get_embeddings  # noqa: E402
from backend.scripts._golden import hardware_stamp     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("SENTIOBOT_RESULTS_DIR", str(REPO / "results")))
LEAK_COSINE = 0.90
# Controls whose ideal single-message decision is NOT to escalate.
NO_ESCALATE_CONTROLS = {"calm", "confused"}
NO_ESCALATE_CONTROL_TAGS = {"emphatic_calm", "positive"}


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def leakage(messages: list[dict]) -> dict:
    emb = get_embeddings()
    protos = [p for ps in S.EMOTION_PROTOTYPES.values() for p in ps]
    pv = np.array(emb.embed_documents(protos), dtype=np.float32)
    pv = pv / np.linalg.norm(pv, axis=1, keepdims=True)
    pnorm = {_normalize(p) for p in protos}
    leaks, mx = [], 0.0
    for m in messages:
        qv = np.array(emb.embed_query(m["text"]), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        c = float(np.max(pv @ qv))
        mx = max(mx, c)
        if c >= LEAK_COSINE or _normalize(m["text"]) in pnorm:
            leaks.append({"id": m["id"], "max_cosine": round(c, 4)})
    return {"threshold": LEAK_COSINE, "max_cosine_over_set": round(mx, 4), "leaks": leaks}


def evaluate() -> dict:
    s = get_settings()
    data = json.load(open(REPO / "results/sentiment_set_v1.json", encoding="utf-8"))
    messages, scenarios = data["messages"], data["escalation_scenarios"]

    rows, correct = [], 0
    by_label_total, by_label_ok = defaultdict(int), defaultdict(int)
    false_escalations, abuse_missed = [], []
    for m in messages:
        r = S.analyze(m["text"], [], s)                # single message, no history
        ok = r.label == m["label"]
        correct += ok
        by_label_total[m["label"]] += 1
        by_label_ok[m["label"]] += ok
        tag = m.get("control")
        is_no_escalate_control = m["label"] in NO_ESCALATE_CONTROLS or tag in NO_ESCALATE_CONTROL_TAGS
        if is_no_escalate_control and r.escalate:
            false_escalations.append({"id": m["id"], "text": m["text"], "label": m["label"]})
        if m.get("expect_escalate") and not r.escalate:
            abuse_missed.append({"id": m["id"], "text": m["text"]})
        rows.append({"id": m["id"], "text": m["text"], "expected": m["label"],
                     "got": r.label, "ok": ok, "frust": r.score, "ema": r.ema,
                     "escalate": r.escalate, "control": tag})

    control_ct = sum(1 for m in messages
                     if m["label"] in NO_ESCALATE_CONTROLS or m.get("control") in NO_ESCALATE_CONTROL_TAGS)

    scen_rows, scen_fail = [], []
    for sc in scenarios:
        turns = sc["turns"]
        r = S.analyze(turns[-1], turns[:-1], s)
        ok = r.escalate == sc["expect_escalate"]
        if not ok:
            scen_fail.append({"id": sc["id"], "expect": sc["expect_escalate"], "got": r.escalate})
        scen_rows.append({"id": sc["id"], "turns": len(turns), "ema": r.ema,
                          "expect_escalate": sc["expect_escalate"], "got_escalate": r.escalate, "ok": ok})

    leak = leakage(messages)
    n = len(messages)
    out = {
        "test": "F4 sentiment: detection accuracy + NO-FALSE-ESCALATION (free, deterministic, 0 tokens)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "as_of": "2026-07-19",
        "hardware": hardware_stamp(),
        "embedding_model": s.embedding_model + " (ONNX Runtime via fastembed)",
        "config": {"escalation_threshold": s.sentiment_escalation_threshold,
                   "ema_alpha": s.sentiment_ema_alpha, "history_turns": s.sentiment_history_turns},
        "n_messages": n,
        "detection_accuracy": round(correct / n, 4),
        "accuracy_by_label": {k: f"{by_label_ok[k]}/{by_label_total[k]}" for k in by_label_total},
        "no_escalate_controls": control_ct,
        "false_escalations": false_escalations,
        "false_escalation_rate": round(len(false_escalations) / control_ct, 4) if control_ct else 0.0,
        "abuse_override_missed": abuse_missed,
        "escalation_scenarios": {"rows": scen_rows, "failures": scen_fail},
        "leakage_guard": {"threshold": leak["threshold"], "max_cosine": leak["max_cosine_over_set"],
                          "leaks": leak["leaks"]},
        "rows": rows,
        "passed": (not false_escalations and not abuse_missed and not scen_fail and not leak["leaks"]),
    }
    return out


def _write_md(out: dict) -> None:
    lines = [
        "# Feature F4 - Sentiment detection + no-false-escalation",
        "",
        f"- Commit: `{out['commit']}`  |  As-of: {out['as_of']}",
        f"- Embedding: {out['embedding_model']} (local, 0 tokens)",
        f"- Escalation threshold {out['config']['escalation_threshold']}, EMA alpha "
        f"{out['config']['ema_alpha']}, window {out['config']['history_turns']}",
        "",
        "## Headline",
        f"- Detection accuracy: **{out['detection_accuracy']:.3f}** ({out['n_messages']} messages)",
        f"- FALSE-ESCALATION rate on no-escalate controls: "
        f"**{out['false_escalation_rate']:.3f}** "
        f"({len(out['false_escalations'])}/{out['no_escalate_controls']}) [load-bearing; must be 0]",
        f"- Abuse-override misses: {len(out['abuse_override_missed'])}",
        f"- Escalation-scenario failures: {len(out['escalation_scenarios']['failures'])}",
        f"- Leakage guard: {'PASS' if not out['leakage_guard']['leaks'] else 'FAIL'} "
        f"(max cosine {out['leakage_guard']['max_cosine']:.3f} < {out['leakage_guard']['threshold']})",
        "",
        "## Accuracy by label",
    ]
    for k, v in out["accuracy_by_label"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Escalation scenarios", "", "| id | turns | ema | expect | got | ok |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in out["escalation_scenarios"]["rows"]:
        lines.append(f"| {r['id']} | {r['turns']} | {r['ema']:.2f} | {r['expect_escalate']} "
                     f"| {r['got_escalate']} | {'ok' if r['ok'] else 'MISS'} |")
    (RESULTS_DIR / "sentiment_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out = evaluate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sentiment_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(out)

    print(f"detection accuracy:     {out['detection_accuracy']:.3f}")
    print(f"false-escalation rate:  {out['false_escalation_rate']:.3f} "
          f"({len(out['false_escalations'])}/{out['no_escalate_controls']})  [must be 0]")
    print(f"abuse-override misses:  {len(out['abuse_override_missed'])}")
    print(f"escalation failures:    {len(out['escalation_scenarios']['failures'])}")
    print(f"leakage guard:          {'PASS' if not out['leakage_guard']['leaks'] else 'FAIL'}"
          f" (max cosine {out['leakage_guard']['max_cosine']:.3f})")
    if out["false_escalations"]:
        print("FALSE ESCALATIONS:", json.dumps(out["false_escalations"], indent=2))
    if out["escalation_scenarios"]["failures"]:
        print("SCENARIO FAILURES:", json.dumps(out["escalation_scenarios"]["failures"], indent=2))
    print("PASS" if out["passed"] else "FAIL")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
