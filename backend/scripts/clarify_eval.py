"""
clarify_eval.py - Feature F5 gate: NO OVER-ASK (Convention 10), tested on the HARD
cases (queries that LOOK ambiguous but are resolvable from profile/history).

Free, deterministic, 0 tokens (the clarify decision is a pure function; no embedder,
no LLM). Reports:
- OVER-ASK: asking when the slot is resolvable (well_specified / profile_resolvable /
  history_resolvable / interaction buckets). This is the load-bearing number; it must
  be 0.
- UNDER-ASK: failing to ask on truly_ambiguous.
- The frustration interaction: DEFER when F4 escalation is active; a warm lead when the
  user is mildly frustrated.

Usage:  python -m backend.scripts.clarify_eval
Writes: results/clarify_eval.json (+ .md), stamped. Exit 1 on any over-ask, under-ask,
or interaction failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("JWT_SECRET", "clarify-eval-secret")
os.environ.setdefault("GROQ_API_KEY", "")

from backend.core import clarify as C                 # noqa: E402
from backend.core.config import get_settings          # noqa: E402
from backend.scripts._golden import hardware_stamp     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("SENTIOBOT_RESULTS_DIR", str(REPO / "results")))
RESOLVABLE_BUCKETS = {"well_specified", "profile_resolvable", "history_resolvable",
                      "interaction", "not_slot"}


def evaluate() -> dict:
    s = get_settings()
    data = json.load(open(REPO / "results/clarify_set_v1.json", encoding="utf-8"))
    products = data["products"]

    rows, over_ask, under_ask, warm_fail = [], [], [], []
    by_bucket = defaultdict(lambda: {"n": 0, "ok": 0})
    for it in data["items"]:
        owned = products[it["owned"]]
        d = C.decide(it["intent"], it["message"], {"owned_products": owned},
                     it.get("history", []), it.get("sentiment"), s)
        expect = it["expect_ask"]
        ok = d.ask == expect
        by_bucket[it["bucket"]]["n"] += 1
        by_bucket[it["bucket"]]["ok"] += ok
        if d.ask and not expect:
            over_ask.append({"id": it["id"], "bucket": it["bucket"], "reason": d.reason})
        if not d.ask and expect:
            under_ask.append({"id": it["id"], "bucket": it["bucket"], "reason": d.reason})
        if it.get("expect_warm") and not d.question.startswith(C._WARM_LEAD):
            warm_fail.append({"id": it["id"], "question": d.question})
        rows.append({"id": it["id"], "bucket": it["bucket"], "intent": it["intent"],
                     "expect_ask": expect, "ask": d.ask, "ok": ok, "reason": d.reason,
                     "hint": bool(d.hint), "question": d.question})

    n = len(rows)
    out = {
        "test": "F5 clarify: NO OVER-ASK on resolvable/hard cases + correct-ask (0 tokens)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "as_of": "2026-07-20",
        "hardware": hardware_stamp(),
        "n": n,
        "accuracy": round(sum(r["ok"] for r in rows) / n, 4),
        "over_ask": over_ask,
        "over_ask_rate": round(len(over_ask) / n, 4),
        "under_ask": under_ask,
        "warm_lead_failures": warm_fail,
        "by_bucket": {b: f"{v['ok']}/{v['n']}" for b, v in by_bucket.items()},
        "rows": rows,
        "passed": not over_ask and not under_ask and not warm_fail,
    }
    return out


def _write_md(out: dict) -> None:
    lines = [
        "# Feature F5 - Clarify: no-over-ask + correct-ask",
        "",
        f"- Commit: `{out['commit']}`  |  As-of: {out['as_of']}  |  0 tokens (pure function)",
        "",
        "## Headline",
        f"- Decision accuracy: **{out['accuracy']:.3f}** ({out['n']} items)",
        f"- OVER-ASK (load-bearing; must be 0): **{len(out['over_ask'])}** "
        f"(rate {out['over_ask_rate']:.3f})",
        f"- UNDER-ASK: {len(out['under_ask'])}  |  warm-lead failures: {len(out['warm_lead_failures'])}",
        "",
        "## By bucket",
    ]
    for b, v in out["by_bucket"].items():
        lines.append(f"- {b}: {v}")
    lines += ["", "| id | bucket | expect_ask | ask | reason | ok |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in out["rows"]:
        lines.append(f"| {r['id']} | {r['bucket']} | {r['expect_ask']} | {r['ask']} "
                     f"| {r['reason']} | {'ok' if r['ok'] else 'MISS'} |")
    (RESULTS_DIR / "clarify_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out = evaluate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "clarify_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(out)

    print(f"decision accuracy: {out['accuracy']:.3f}")
    print(f"OVER-ASK (must be 0): {len(out['over_ask'])}  (rate {out['over_ask_rate']:.3f})")
    print(f"under-ask: {len(out['under_ask'])}  warm-lead failures: {len(out['warm_lead_failures'])}")
    print("by bucket:", out["by_bucket"])
    if out["over_ask"]:
        print("OVER-ASK:", json.dumps(out["over_ask"], indent=2))
    if out["under_ask"]:
        print("UNDER-ASK:", json.dumps(out["under_ask"], indent=2))
    print("PASS" if out["passed"] else "FAIL")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
