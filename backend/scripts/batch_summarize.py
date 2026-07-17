"""
scripts/batch_summarize.py — Production-grade RAG summarisation pipeline.

Provider : Groq  (free forever, no billing, no credit card)
Model    : llama-3.3-70b-versatile
  • Beats Gemini 1.5 Flash on summarisation benchmarks
  • 1,000 req/day free  — we only need ~17 calls for 85 docs
  • 30 RPM free
  • Runs on Groq's custom LPU hardware — faster than GPU inference

Quality decisions
-----------------
1. Batches of 5 with the 70b model: better coherence per summary than
   batches of 10 with a small model.
2. Prompt engineering: we tell the model exactly what to extract
   (specs, part numbers, procedures, policy terms) and what NOT to include
   (filler, markdown, opinions).
3. Output validation: each summary is checked for minimum length and
   suspicious patterns (e.g. the model returning "Summary 1:" headers).
4. Per-doc fallback: if a batch partially fails validation, we retry
   individual docs with a stricter single-doc prompt rather than losing
   the whole batch.
5. Quality scores logged: word count per summary is logged so you can
   spot thin summaries before they hurt retrieval.

Setup (one time)
----------------
  1. https://console.groq.com  →  free signup  →  API Keys  →  Create
  2. Add to .env:   GROQ_API_KEY=gsk_xxxxxxxxxxxx
  3. pip install groq
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL      = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MIN_GAP    = float(os.getenv("MIN_GAP",  "2.5"))   # ~24 RPM, under 30 RPM limit
MAX_RETRIES= int(os.getenv("MAX_RETRIES","4"))
MIN_WORDS  = int(os.getenv("MIN_WORDS",  "30"))     # reject summaries shorter than this

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
PARENT_STORE_PATH = PROJECT_ROOT / "parent_docstore"
SUMMARIES_PATH    = PROJECT_ROOT / "summaries"
FAILED_LOG        = PROJECT_ROOT / "failed_docs.log"

SUMMARIES_PATH.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_client():
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        raise ImportError(
            "\ngroq not installed. Run:  pip install groq\n"
        )
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "\nGROQ_API_KEY missing.\n"
            "  1. Sign up free at https://console.groq.com\n"
            "  2. Create an API key\n"
            "  3. Add  GROQ_API_KEY=gsk_xxx  to your .env file\n"
        )
    return Groq(api_key=key)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert technical writer building a semantic search index for a \
product support AI assistant.

Rules:
- Return ONLY a valid JSON array of strings — one string per document.
- Each string is a dense, self-contained summary of that document section.
- Every summary MUST include all of: product names, model numbers, numeric \
  specifications, step-by-step procedures (condensed), policy terms, error \
  codes, and troubleshooting keywords.
- Write in plain prose. No markdown, no bullet points, no headers, \
  no "Summary:" prefix.
- Minimum 40 words per summary. Be thorough.
- Do NOT omit any factual detail — these summaries are the only thing the \
  search engine will see.
"""

BATCH_USER = """\
Summarise each of the following {n} document sections.
Return a JSON array of exactly {n} strings, one per document, same order.

{docs}
"""

SINGLE_USER = """\
Summarise the following document section into a single dense paragraph.
Include every product name, spec, procedure step, and policy detail.
Return ONLY the summary text — no labels, no markdown.

{content}
"""


# ── Output validation ─────────────────────────────────────────────────────────

_BAD_PATTERNS = re.compile(
    r"^(summary\s*\d*[:：]|document\s*\d*[:：]|here\s+is|here\'s|sure[\s,])",
    re.IGNORECASE,
)


def _validate(text: str) -> str | None:
    """
    Returns cleaned summary string, or None if it fails quality checks.
    """
    text = text.strip()

    # Remove accidental markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    # Reject if model added a label prefix
    if _BAD_PATTERNS.match(text):
        log.debug("  Quality: rejected prefix pattern → '%s'", text[:60])
        return None

    # Reject if too short
    if len(text.split()) < MIN_WORDS:
        log.debug("  Quality: too short (%d words)", len(text.split()))
        return None

    return text


# ── API calls ─────────────────────────────────────────────────────────────────

def _parse_json_array(raw: str) -> list[str] | None:
    """Extract a JSON array from the model response, tolerating minor wrapping."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    # Find the first [ ... ] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except json.JSONDecodeError:
        pass
    return None


def _call_batch(client, docs: list) -> list[str | None]:
    """
    Call the 70b model with a batch.
    Returns a list of validated summary strings (or None for failed items).
    """
    user_msg = BATCH_USER.format(n=len(docs), docs=_format_docs(docs))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",  "content": SYSTEM},
                    {"role": "user",    "content": user_msg},
                ],
                temperature=0.05,   # low temp = consistent structured output
                max_tokens=3000,
            )
            raw      = resp.choices[0].message.content
            summaries = _parse_json_array(raw)

            if summaries is None:
                log.warning("  Attempt %d/%d — JSON parse failed, retrying", attempt, MAX_RETRIES)
                time.sleep(3)
                continue

            if len(summaries) != len(docs):
                log.warning("  Attempt %d/%d — count mismatch (%d vs %d), retrying",
                            attempt, MAX_RETRIES, len(summaries), len(docs))
                time.sleep(3)
                continue

            # Validate each summary
            validated = [_validate(s) for s in summaries]
            bad = sum(1 for v in validated if v is None)
            if bad:
                log.warning("  %d/%d summaries failed quality check in batch", bad, len(docs))

            return validated

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate" in exc_str.lower():
                wait = 65
                log.warning("  Rate limited (attempt %d/%d) — waiting %ds", attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                log.warning("  API error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc_str[:120])
                time.sleep(5 * attempt)

    return [None] * len(docs)


def _call_single(client, doc) -> str | None:
    """Fallback: summarise one doc at a time with a stricter prompt."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": SINGLE_USER.format(content=doc.page_content)},
                ],
                temperature=0.05,
                max_tokens=600,
            )
            text = resp.choices[0].message.content
            validated = _validate(text)
            if validated:
                return validated
            log.warning("  Single-doc fallback quality check failed (attempt %d)", attempt)
            time.sleep(3)

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate" in exc_str.lower():
                time.sleep(65)
            else:
                time.sleep(5)

    return None


def _format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs):
        src     = doc.metadata.get("source", "?")
        section = doc.metadata.get("section_title", "?")
        parts.append(f"=== DOCUMENT {i+1}  [{src} › {section}] ===\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Persist ───────────────────────────────────────────────────────────────────

def _save(doc_id: str, text: str) -> None:
    (SUMMARIES_PATH / f"{doc_id}.txt").write_text(text, encoding="utf-8")


def _log_failed(ids: list[str], reason: str) -> None:
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "doc_ids":   ids,
            "reason":    reason[:200],
        }) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("═" * 60)
    log.info("SentioBot Summarisation Pipeline")
    log.info("Provider : Groq (free)   Model : %s", MODEL)
    log.info("═" * 60)

    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore

    byte_store   = LocalFileStore(str(PARENT_STORE_PATH))
    docstore     = create_kv_docstore(byte_store)
    existing_ids = {f.stem for f in SUMMARIES_PATH.glob("*.txt")}
    all_ids      = list(byte_store.yield_keys())
    pending_ids  = [i for i in all_ids if i not in existing_ids]

    if not pending_ids:
        log.info("✅  All %d documents already summarised.", len(all_ids))
        log.info("Next step:  python scripts/ingest.py")
        return

    total     = len(pending_ids)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    log.info("Progress  : %d / %d done,  %d remaining", len(existing_ids), len(all_ids), total)
    log.info("Batches   : %d  (%d docs each)", n_batches, BATCH_SIZE)
    log.info("Est. time : ~%.0f min", (n_batches * MIN_GAP) / 60)
    log.info("Tip       : Ctrl-C anytime — each saved batch is kept")
    log.info("─" * 60)

    client    = _get_client()
    succeeded = 0
    failed    = 0

    try:
        # Use a while loop so pending_ids refresh stays in sync with iteration
        while pending_ids:
            batch_ids      = pending_ids[:BATCH_SIZE]
            batch_docs_raw = docstore.mget(batch_ids)
            valid = [(bid, doc)
                     for bid, doc in zip(batch_ids, batch_docs_raw)
                     if doc is not None]

            if not valid:
                # All in this slice were None — skip and advance
                pending_ids = pending_ids[BATCH_SIZE:]
                continue

            v_ids, v_docs = zip(*valid)
            batch_num = (len(all_ids) - len(pending_ids)) // BATCH_SIZE + 1
            total_batches = (len(pending_ids) + BATCH_SIZE - 1) // BATCH_SIZE

            log.info("[Batch %d]  %d docs", batch_num, len(v_docs))

            t0       = time.time()
            results  = _call_batch(client, list(v_docs))
            elapsed  = time.time() - t0

            for doc_id, doc, summary in zip(v_ids, v_docs, results):
                # CSV docs have "Category", markdown docs have "section_title"
                section = (
                    doc.metadata.get("section_title")
                    or doc.metadata.get("Category")
                    or doc.metadata.get("source", "?")
                )[:40]
                if summary:
                    _save(doc_id, summary)
                    wc = len(summary.split())
                    log.info("  ✅  %-42s  %d words", section, wc)
                    succeeded += 1
                else:
                    log.info("  🔄  %-42s  retrying individually…", section)
                    single = _call_single(client, doc)
                    if single:
                        _save(doc_id, single)
                        succeeded += 1
                        log.info("     ✅  fallback succeeded  (%d words)", len(single.split()))
                    else:
                        log.error("     ❌  failed permanently")
                        _log_failed([doc_id], "batch + single fallback both failed")
                        failed += 1

            log.info("  ─  batch done in %.1fs", elapsed)

            # Refresh pending list — now correctly drives the while loop
            existing_ids = {f.stem for f in SUMMARIES_PATH.glob("*.txt")}
            pending_ids  = [i for i in all_ids if i not in existing_ids]

            if pending_ids:
                gap = MIN_GAP - elapsed
                if gap > 0:
                    time.sleep(gap)

    except KeyboardInterrupt:
        done_now = len(list(SUMMARIES_PATH.glob("*.txt")))
        log.info("")
        log.info("⚠️  Interrupted.  %d / %d total done.", done_now, len(all_ids))
        log.info("   Re-run to continue from where you left off.")
        return

    log.info("═" * 60)
    if failed == 0:
        log.info("🎉  All %d documents summarised successfully!", succeeded)
        log.info("Quality note : all summaries passed minimum %d-word check", MIN_WORDS)
        log.info("Next step    : python scripts/ingest.py")
    else:
        log.info("Done. %d ok,  %d failed → see %s", succeeded, failed, FAILED_LOG)
        log.info("Re-run to retry failed docs, then: python scripts/ingest.py")
    log.info("═" * 60)


if __name__ == "__main__":
    main()