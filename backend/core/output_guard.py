"""
output_guard.py - Increment 6 output-side exfiltration guard.

The Increment 5 input filter was phrasing-specific: the reviewer reworded around
it and dumped the whole system prompt. This guard defends at the EXFILTRATION
point instead of the injection point. It inspects the MODEL'S OUTPUT for
fingerprints that only ever appear in the system prompt (the "You are SentioBot"
preamble, the confidentiality marker, the behaviour-rule strings, the tool-list
line). Their presence in a customer-facing answer means a verbatim or near-
verbatim prompt dump, no matter how the attacker phrased the request. When a
fingerprint appears, the whole response is replaced with a refusal.

Streaming without leaking: the endpoint streams token by token, so we cannot
"un-send" a dump once it is out. The guard holds back the last MAXFP characters
(MAXFP = the longest fingerprint) before releasing any text. Because we scan the
full accumulated buffer on every token and every fingerprint is <= MAXFP long,
any fingerprint is detected while its first character is still inside the
held-back tail - so a blocked dump leaks ZERO characters to the client.

Known limit (documented, not hidden): this is string matching. A TRANSFORMED
leak - the prompt translated, base64-encoded, spelled out, or heavily paraphrased
- will not match these fingerprints and can still evade the guard. That residual
is covered in the red-team suite and the residual-risk doc; it is why the primary
defense is blast-radius containment, not this filter.
"""
from __future__ import annotations

import re

# Lower-cased structural markers of the system prompt. None of these should ever
# occur in a legitimate Nexora support answer, so their presence in OUTPUT means
# a verbatim/near-verbatim prompt dump.
#
# Deliberately NOT included: the bare internal tool names (check_warranty_status,
# etc.). We tried them, and they caused FALSE POSITIVES: the model legitimately
# names a tool while explaining itself ("I can check the warranty status..."),
# and blocking that whole answer breaks real support replies (red-team ben-03, a
# return-policy question, got blocked). A leaked tool name is low-value recon,
# not worth turning good answers into refusals. The prompt's tool-LIST line
# ("available tools: lookup_documentation") stays, since it only appears in a
# dump, never in a natural answer.
FINGERPRINTS = [
    "you are sentiobot",
    "confidentiality and scope",
    "highest priority, overrides any later request",
    "these instructions are confidential",
    "behaviour rules",
    "rag first",
    "escalate only if needed",
    "proactive:",
    "available tools: lookup_documentation",
    "## user profile",
]

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Lower-case and collapse whitespace runs to a single space.

    This defeats the common 'spell it out / one word per line' transform that
    otherwise slips a rule past an exact-string match (red-team obf-03 produced
    'RAG  FIRST  For  any ...'). It does NOT defeat translation, base64, or
    per-CHARACTER splitting; those remain documented residuals.
    """
    return _WS.sub(" ", s.lower())


_NORM_FINGERPRINTS = [(_norm(f), f) for f in FINGERPRINTS]

# Hold-back for the streaming guard. It must exceed the RAW length of any
# fingerprint AFTER moderate whitespace inflation, so a normalized match is
# always still inside the unreleased tail (zero leak). We size it to the longest
# fingerprint plus generous slack for word-per-line spacing.
MAXFP = max(len(f) for f in FINGERPRINTS) + 64


def find_fingerprint(text: str) -> str | None:
    """Return the first system-prompt/tool fingerprint present in text, else None.

    Matching is whitespace-normalized, so 'RAG  FIRST' and 'RAG\\nFIRST' match
    'rag first'.
    """
    low = _norm(text)
    for norm_fp, original in _NORM_FINGERPRINTS:
        if norm_fp in low:
            return original
    return None


class OutputGuard:
    """Streaming guard with a MAXFP-char hold-back for zero-leak blocking.

    Usage:
        g = OutputGuard()
        for tok in stream:
            safe = g.feed(tok)          # text safe to send now (may be "")
            if safe: emit(safe)
            if g.blocked: break
        if not g.blocked:
            tail = g.flush()            # remaining safe text at end of stream
            if tail: emit(tail)
        if g.blocked: emit_refusal()
    """

    def __init__(self) -> None:
        self.buf = ""
        self.blocked = False
        self.hit: str | None = None

    def _trip(self, hit: str) -> str:
        self.blocked = True
        self.hit = hit
        self.buf = ""
        return ""

    def feed(self, token: str) -> str:
        if self.blocked:
            return ""
        self.buf += token
        hit = find_fingerprint(self.buf)
        if hit:
            return self._trip(hit)
        if len(self.buf) > MAXFP:
            release = self.buf[:-MAXFP]
            self.buf = self.buf[-MAXFP:]
            return release
        return ""

    def flush(self) -> str:
        if self.blocked:
            return ""
        hit = find_fingerprint(self.buf)
        if hit:
            return self._trip(hit)
        release = self.buf
        self.buf = ""
        return release
