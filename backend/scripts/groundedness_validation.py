"""
groundedness_validation.py - Feature F2 gate: validate the local groundedness
label BEFORE shipping its wording/threshold (planner ruling).

Free and deterministic (0 tokens). It has two parts:

1. LOCAL vs RAGAS faithfulness on REAL answers. It reuses the 5 human-verified
   doc/policy answers committed in results/increment4_delta.json (each carries the
   real 70B answer AND the 8B RAGAS faithfulness the Increment 3/4 harness scored),
   re-retrieves the context locally, and runs the local groundedness pass. It
   reports local label vs RAGAS faithfulness so a FALSE GREEN (local says grounded
   where the answer is genuinely unfaithful) is visible. Per the Increment 4
   lesson, where local disagrees with the WEAK 8B judge, the judge is often the one
   erring (it gave spurious 0.0s to answers a human verified correct); those are
   judge noise, not false greens, and are flagged for a human read, not
   auto-counted against local.

2. NO-FALSE-GREEN, synthetic + deterministic. Each real answer is POISONED with a
   fabricated claim absent from the corpus; the guard must drop it out of
   "grounded". This is the unambiguous safety property (no dependence on the noisy
   judge).

Usage:  python -m backend.scripts.groundedness_validation
Writes: results/groundedness_validation.json (+ .md), stamped.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("JWT_SECRET", "groundedness-validation-secret")
os.environ.setdefault("GROQ_API_KEY", "")

from backend.core.config import get_settings          # noqa: E402
from backend.core.groundedness import analyze          # noqa: E402
from backend.scripts._golden import hardware_stamp     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("SENTIOBOT_RESULTS_DIR", str(REPO / "results")))

# A fabricated claim absent from the Nexora corpus, for the no-false-green probe.
POISON = ("This product also ships with a complimentary lifetime international "
          "shipping plan and a free smart speaker in every box.")


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


async def _retrieve_source_texts(question: str) -> list[dict]:
    from backend.agent.agent import _retrieve_context
    _, _sources, source_texts = await _retrieve_context(question)
    return source_texts


async def main() -> int:
    s = get_settings()
    delta = json.load(open(REPO / "results/increment4_delta.json", encoding="utf-8"))
    golden = json.load(open(REPO / "results/golden_set_v1.json", encoding="utf-8"))
    q_by_id = {it["id"]: (it.get("question") or it.get("turns", [{}])[0].get("question"))
               for it in golden["items"]}

    rows, false_greens, poison_failures = [], [], []
    for r in delta["rows"]:
        rid = r["id"]
        question = q_by_id.get(rid)
        answer = r["after_answer"]
        faith = _mean(r.get("after_faithfulness", []))
        src = await _retrieve_source_texts(question)

        real = analyze(answer, src, s)
        poisoned = analyze(answer + " " + POISON, src, s)

        rl = real["grounded"]["label"]
        # A FALSE GREEN worth worrying about = local grounded where RAGAS is low.
        # Flagged for a human read (the weak judge gave spurious 0.0s per Increment 4).
        disagree = rl == "grounded" and faith is not None and faith < 0.5
        if disagree:
            false_greens.append({"id": rid, "local": rl, "ragas_faithfulness": faith,
                                 "note": "local grounded but weak-judge faithfulness low; read the answer"})
        if poisoned["grounded"]["label"] == "grounded":
            poison_failures.append({"id": rid})

        rows.append({
            "id": rid, "question": question,
            "local_label": rl, "local_score": real["grounded"]["score"],
            "supported": real["grounded"]["supported"], "total": real["grounded"]["total"],
            "ragas_faithfulness": faith,
            "citations": len(real["citations"]),
            "poisoned_label": poisoned["grounded"]["label"],
            "local_vs_ragas_disagree": disagree,
        })

    out = {
        "test": "F2 groundedness validation: local label vs RAGAS faithfulness + no-false-green (0 tokens)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "as_of": "2026-07-19",
        "hardware": hardware_stamp(),
        "embedding_model": s.embedding_model + " (ONNX Runtime via fastembed)",
        "threshold": s.groundedness_threshold,
        "source": "increment4_delta.json (human-verified real answers + committed 8B RAGAS faithfulness)",
        "n": len(rows),
        "no_false_green_synthetic": {
            "poisoned_answers_that_stayed_grounded": poison_failures,
            "passed": not poison_failures,
        },
        "local_vs_ragas_disagreements": false_greens,
        "rows": rows,
        # The build gate is the SYNTHETIC no-false-green (unambiguous). The RAGAS
        # disagreements are surfaced for a human read, not auto-failed, because the
        # 8B judge is documented-noisy (Increment 4).
        "passed": not poison_failures,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "groundedness_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(out)

    print(f"threshold: {s.groundedness_threshold}")
    for r in rows:
        print(f"  {r['id']:7s} local={r['local_label']:10s} "
              f"({r['supported']}/{r['total']}, score {r['local_score']}) "
              f"ragas_faith={r['ragas_faithfulness']} "
              f"poisoned={r['poisoned_label']}"
              f"{'  <-- disagree (read answer)' if r['local_vs_ragas_disagree'] else ''}")
    print(f"no-false-green (synthetic poison): {'PASS' if not poison_failures else 'FAIL'}")
    if false_greens:
        print(f"local vs weak-RAGAS disagreements (human-read): {[f['id'] for f in false_greens]}")
    return 0 if out["passed"] else 1


def _write_md(out: dict) -> None:
    lines = [
        "# Feature F2 - Groundedness validation (local vs RAGAS + no-false-green)",
        "",
        f"- Commit: `{out['commit']}`  |  As-of: {out['as_of']}",
        f"- Embedding: {out['embedding_model']} (local, 0 tokens)  |  Threshold: {out['threshold']}",
        f"- Source: {out['source']}",
        "",
        f"No-false-green (synthetic poison): **{'PASS' if out['no_false_green_synthetic']['passed'] else 'FAIL'}**",
        "",
        "| id | local label | supported | RAGAS faith | poisoned label | disagree |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in out["rows"]:
        lines.append(f"| {r['id']} | {r['local_label']} | {r['supported']}/{r['total']} "
                     f"| {r['ragas_faithfulness']} | {r['poisoned_label']} "
                     f"| {'yes' if r['local_vs_ragas_disagree'] else ''} |")
    lines += ["",
              "Disagreements are local=grounded where the WEAK 8B judge scored low "
              "faithfulness. Per Increment 4, the judge gave spurious 0.0s to "
              "human-verified-correct answers; these are judge noise, not false "
              "greens. The build gate is the synthetic no-false-green above."]
    (RESULTS_DIR / "groundedness_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
