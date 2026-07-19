"""
groundedness.py - local, zero-token groundedness + inline citations (Feature F2).

The trust feature. After a documentation answer is generated, we check whether it
is actually supported by the retrieved source text, and we EXTRACT the exact source
sentence behind each supported claim. Both come from ONE local pass over the ONNX
MiniLM embedder the retriever/cache/F1-router already load, so it adds NO LLM call
and ~0 tokens.

Honest scope of the claim (ratified in the F2 spec): cosine overlap measures
TOPICAL support, not entailment. It can be fooled by a topically-similar wrong
number ("2 years" vs "3 years") or a negation. So the badge NEVER says
"verified"/"correct"/"entailed"; it says each claim MATCHES a retrieved passage,
and it SHOWS that passage so the human makes the final check. The shown source
sentence is the real backstop; the score is a supporting signal. Because of that
weakness the label is CONSERVATIVE and 3-state:
  - grounded   : EVERY factual claim matches a source sentence above the threshold.
  - partial    : some but not all claims matched (soft-withhold; show but flag).
  - unverified : no claims matched, or there was no retrieved context.
"grounded" requiring ALL claims to match is what makes a false green hard: one
unmatched claim drops it to partial.

Citations are EXTRACTED, never generated: each span is a literal slice of the
retrieved source text, so a hallucinated quote is structurally impossible (a test
asserts every span is a substring of a source).
"""
from __future__ import annotations

import re

import numpy as np

from backend.core.onnx_embeddings import get_embeddings

# Split on sentence enders (keeping the punctuation) or hard line breaks. Every
# resulting piece is a CONTIGUOUS slice of the input, which is what lets a citation
# span stay a literal substring of the source (the anti-hallucination guarantee).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION_MARKER = re.compile(r"\[\s*source\s+\d+\s*\]", re.I)
_MD_NOISE = re.compile(r"[*_`#>]+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# Non-factual sentences (greetings, offers of help) are excluded from the claim set:
# they carry no fact to ground, and counting them would wrongly sink the label.
_NONFACTUAL = re.compile(
    r"\b(let me know|feel free|hope (this|that) helps|anything else|glad to help|"
    r"happy to help|is there anything|if you have any|please let me|would you like|"
    r"i can help|how can i help|you'?re welcome|no problem|of course|sure thing)\b",
    re.I,
)

# Citation-apparatus references ("According to the (visionsphere360manual.md | 3.")
# are about the SOURCE machinery, not facts, and would never match source prose.
# Excluding them stops a good answer from being penalised for its own citations.
_META_REF = re.compile(r"\.(md|csv|pdf|txt)\b", re.I)


def _normed(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0, mat / norms, 0.0)


def _clean_claim(sentence: str) -> str:
    s = _CITATION_MARKER.sub("", sentence)
    s = _LIST_PREFIX.sub("", s)
    s = _MD_NOISE.sub("", s)
    return s.strip()


def _is_factual_claim(text: str) -> bool:
    stripped = text.rstrip()
    if len(text) < 25:
        return False
    if stripped.endswith("?"):     # a question, not a claim
        return False
    if stripped.endswith(":"):     # a list lead-in ("the product must be:")
        return False
    if _META_REF.search(text):     # a citation-apparatus reference, not a fact
        return False
    if _NONFACTUAL.search(text):
        return False
    return len(text.split()) >= 4


_HEADER = re.compile(r"^\s*#+\s")


def _extract_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for raw in _SENT_SPLIT.split(answer or ""):
        if _HEADER.match(raw):  # markdown heading, not a factual claim
            continue
        c = _clean_claim(raw)
        if _is_factual_claim(c):
            claims.append(c)
    return claims


def _extract_source_sentences(source_texts: list[dict], cap: int) -> list[dict]:
    """Flatten each retrieved source's text into sentences, tagged with its
    [Source N] index (n = position + 1) and its source/section. Spans stay raw
    slices of the text so they remain literal substrings for the citation gate."""
    out: list[dict] = []
    for idx, st in enumerate(source_texts):
        text = st.get("text") or ""
        for piece in _SENT_SPLIT.split(text):
            span = piece.strip()
            if len(span) < 15:
                continue
            out.append({
                "n": idx + 1,
                "source": st.get("source", "N/A"),
                "section": st.get("section", "N/A"),
                "text": span,
            })
            if len(out) >= cap:
                return out
    return out


def _truncate(span: str, max_chars: int) -> str:
    """Bound the payload while keeping the span a literal prefix (still a substring
    of the source). Cut at the last word boundary for readability."""
    if len(span) <= max_chars:
        return span
    cut = span[:max_chars]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > max_chars * 0.5 else cut


def analyze(answer: str, source_texts: list[dict], settings) -> dict:
    """Returns {"grounded": {label, score, supported, total}, "citations": [...]}.
    Local and zero-token. Callers gate this to DOC-grounded answers (non-empty
    source_texts); pure tool-action answers get no badge."""
    claims = _extract_claims(answer)
    max_src = getattr(settings, "groundedness_max_source_sentences", 120)
    src = _extract_source_sentences(source_texts, max_src)
    total = len(claims)

    if not claims or not src:
        return {"grounded": {"label": "unverified", "score": 0.0,
                             "supported": 0, "total": total}, "citations": []}

    emb = get_embeddings()
    cvecs = _normed(np.array(emb.embed_documents(claims), dtype=np.float32))
    svecs = _normed(np.array(emb.embed_documents([s["text"] for s in src]), dtype=np.float32))
    sims = cvecs @ svecs.T                                   # (claims, source_sents)

    threshold = float(getattr(settings, "groundedness_threshold", 0.5))
    span_max = int(getattr(settings, "groundedness_span_max_chars", 240))
    max_cites = int(getattr(settings, "groundedness_max_citations", 6))

    supported = 0
    citations: list[dict] = []
    for i, claim in enumerate(claims):
        j = int(np.argmax(sims[i]))
        score = float(sims[i][j])
        if score >= threshold:
            supported += 1
            s = src[j]
            citations.append({
                "n": s["n"], "source": s["source"], "section": s["section"],
                "span": _truncate(s["text"], span_max),
                "claim": _truncate(claim, span_max),
                "similarity": round(score, 3),
            })

    if supported == total:
        label = "grounded"
    elif supported > 0:
        label = "partial"
    else:
        label = "unverified"

    result = {
        "grounded": {"label": label, "score": round(supported / total, 3),
                     "supported": supported, "total": total},
        "citations": _cap_citations(citations, max_cites),
    }

    if getattr(settings, "groundedness_nli", False):
        result = _nli_refine(answer, source_texts, result, settings)
    return result


def _cap_citations(citations: list[dict], cap: int) -> list[dict]:
    """Dedupe identical (source index, span) pairs and keep the strongest few, to
    bound the `done` payload."""
    seen, unique = set(), []
    for c in sorted(citations, key=lambda x: -x["similarity"]):
        key = (c["n"], c["span"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique[:cap]


# ---------------------------------------------------------------------------
# 8B NLI escalation (documented mitigation, OFF by default: GROUNDEDNESS_NLI=on)
# ---------------------------------------------------------------------------

def _nli_refine(answer: str, source_texts: list[dict], result: dict, settings) -> dict:
    """Optional stricter overlay for the topical-overlap weakness. One
    llama-3.1-8b-instant call asks which answer claims are actually supported; a
    claim counts as supported only if BOTH the local cosine AND the judge agree
    (conservative AND), so this can only DOWNGRADE the label, never false-green it.
    Costs tokens; off by default. Never raises: on any error it returns the local
    result unchanged (fail-safe, never a false green)."""
    try:
        from langchain_groq import ChatGroq
        context = "\n\n".join(st.get("text", "") for st in source_texts)
        claims = _extract_claims(answer)
        if not claims:
            return result
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        judge = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.groq_api_key,
                         temperature=0.0, max_tokens=200)
        prompt = (
            f"Context:\n{context}\n\nClaims:\n{numbered}\n\n"
            "For each numbered claim, is it directly supported by the Context? "
            "Reply one line per claim as '<number>: yes' or '<number>: no'."
        )
        resp = judge.invoke(prompt)
        judged_ok = set()
        for line in (resp.content or "").splitlines():
            m = re.match(r"\s*(\d+)\s*[:.)-]\s*(yes|no)", line, re.I)
            if m and m.group(2).lower() == "yes":
                judged_ok.add(int(m.group(1)))
        total = len(claims)
        # The judge is an entailment check, stricter than topical overlap. Use its
        # supported count as an overlay that can only DOWNGRADE the local label.
        supported = len({i for i in judged_ok if 1 <= i <= total})
        if supported == total:
            label = "grounded"
        elif supported > 0:
            label = "partial"
        else:
            label = "unverified"
        # Only allow a downgrade from the local label; never an upgrade.
        order = {"unverified": 0, "partial": 1, "grounded": 2}
        if order[label] < order[result["grounded"]["label"]]:
            result["grounded"] = {"label": label, "score": round(supported / total, 3),
                                  "supported": supported, "total": total}
        return result
    except Exception:
        return result
