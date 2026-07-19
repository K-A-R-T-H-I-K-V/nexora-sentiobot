"""
sentiment.py - local, zero-token frustration/emotion awareness (Feature F4).

SentioBot means "I feel". Each turn we read the user's emotional state, adapt the
answer's TONE (never announcing the emotion), and when frustration is SUSTAINED we
proactively OFFER a human. All local: cosine-match the message to labeled emotion
prototype phrases with the same in-memory ONNX MiniLM the retriever/cache/F1
router/F2 already load (the current-turn embedding is SHARED with the F1 router, so
this adds ~0 work on the hot path), plus a small high-precision lexical intensity
booster for the affect embeddings miss (sarcasm, emphasis, profanity). No LLM call,
~0 tokens. An 8B escalation path sits behind SENTIMENT_NLI (default off).

Honest scope (ratified): emotion is HARDER for embeddings than intent - overlap
reads topical emotion, not true affect. It can miss dry sarcasm and polite-but-furious
phrasing and can over-read loud-but-positive text. That is why the escalation offer is
gated on NO-FALSE-ESCALATION (the F2 no-false-green applied to emotion): a proactive
"want a human?" to a calm user is the cardinal sin, so the booster keys on NEGATIVE
cues only (loud positivity like "AMAZING!!!" must not escalate) and the decision is a
SUSTAINED (EMA over turns) signal, not a single spike.

Security: this runs AFTER the layer-1 injection guard and the tone instruction sits
BELOW the confidentiality block, so a frustrated-toned jailbreak is refused, not coddled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from backend.core.onnx_embeddings import get_embeddings

LABELS = ("calm", "confused", "frustrated", "angry")

# Base frustration weight per emotion label (0..1), fed into the running EMA.
_FRUSTRATION = {"calm": 0.0, "confused": 0.2, "frustrated": 0.7, "angry": 1.0}

# Emotion is WEAK for embeddings (calibrated on the labeled set): a plain question
# and loud-positive text score a NON-calm emotion only weakly (~0.2-0.3, barely above
# calm). So a non-calm label is trusted only when its cosine clears an absolute FLOOR
# and beats calm by a MARGIN; otherwise the turn is calm. This gate is what stops a
# false escalation on "This is AMAZING!!!" (the cardinal sin).
_EMOTION_FLOOR = 0.40
_EMOTION_MARGIN = 0.12

# Labeled prototype phrases per emotion. GENERIC exemplars, kept DISJOINT from the
# labeled sentiment eval set (results/sentiment_set_v1.json); sentiment_eval.py
# enforces that no eval message near-copies a prototype (leakage guard), so accuracy
# measures generalization, not memorization.
EMOTION_PROTOTYPES: dict[str, list[str]] = {
    "calm": [
        "how do I set up my new device",
        "could you tell me the return policy",
        "what is the warranty period on this product",
        "thanks, that answered my question",
        "where can I find the user manual",
        "i would like to check my order status please",
    ],
    "confused": [
        "i am not sure which step to do next",
        "i do not understand what this setting means",
        "wait, that does not make sense to me",
        "i am confused about how this is supposed to work",
        "which cable am i supposed to use here",
        "i do not get how any of this is meant to fit together",
        "i am a bit lost and do not know what to press",
    ],
    "frustrated": [
        "i have tried everything and it still will not work",
        "this is the third time i am asking about the same problem",
        "i am getting really tired of this not working",
        "why is this so complicated, nothing is helping",
        "i already did that and it did not fix anything",
    ],
    "angry": [
        "this is completely unacceptable and a waste of my money",
        "i am furious, this product is garbage and so is the support",
        "i have had enough of this useless thing, i want a refund now",
        "absolutely ridiculous, no one has helped me at all",
        "i am done, this is the worst experience i have ever had",
    ],
}

# High-precision NEGATIVE lexical cues. The booster only fires on negativity, so
# loud POSITIVE emphasis ("this is AMAZING, thank you!!!") gets ~0 boost - the key to
# not false-escalating an excited-but-happy user.
_NEG_WORDS = re.compile(
    r"\b(useless|ridiculous|unacceptable|garbage|rubbish|terrible|awful|furious|"
    r"pathetic|worst|hate|stupid|broken|scam|refund|angry|frustrat\w*|annoy\w*|"
    r"disappoint\w*|hopeless|nonsense|junk|rip.?off|fed up|sick of|had enough|"
    r"wast(e of|ed)|no matter what|keeps? (failing|crashing|happening|dropping)|"
    r"over and over|not working|does\s?n.?t work|will\s?n.?t work|"
    r"still (not|does\s?n.?t|will\s?n.?t)|third time|every (single )?time|"
    r"again and again|come on|damn|hell|crap|wtf|bloody)\b",
    re.I,
)
# Profanity (incl. lightly-censored forms) is a high-precision abuse signal used for
# the single-message escalation override and to intensify the lexical read.
_PROFANITY = re.compile(
    r"\b(f+u+c+k+\w*|f\W*[*x#]+\w*k?\w*|sh[i*]+t\w*|bull\W?shit|damn|hell|crap|wtf|"
    r"bloody|piss(ed)?|arse|bastard|screw (you|this))\b", re.I)

# High-precision POSITIVE / resolved cues. MiniLM is polarity-blind ("it works now"
# is embedding-close to the frustrated prototypes purely via the word "work"), so a
# clearly positive message with NO negative cue is forced to calm. This also lets a
# frustrated conversation DE-ESCALATE the moment the user says it is fixed. Sarcastic
# "thanks" alongside negative words is NOT vetoed (the negative cue keeps it frustrated).
_POSITIVE = re.compile(
    r"\b(thank you|thanks|works? now|worked|fixed|resolved|solved|sorted|all set|"
    r"perfect|great|awesome|amazing|appreciate|no worries|got it working|that did it)\b",
    re.I)


def _normed(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0, mat / norms, 0.0)


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _lexical_frust(text: str) -> float:
    """Frustration (0..1) from NEGATIVE cues ONLY, so it can stand on its own when the
    weak emotion embedding misses (sarcasm/emphasis) yet stays 0 on loud POSITIVE text.
    One negative word alone is mild (a calm bug report: "my device is broken"); piled-on
    negativity, profanity, SHOUTING, or "!!!"" escalate it."""
    n = len(_NEG_WORDS.findall(text))
    if n == 0:
        return 0.0
    score = 0.30 * n                               # 1/2/3 negative cues -> 0.30/0.60/0.90
    if _PROFANITY.search(text):
        score += 0.35
    if _caps_ratio(text) >= 0.5:                   # SHOUTING (on a negative message)
        score += 0.20
    if re.search(r"[!?]{2,}", text):               # !!! / ?!?
        score += 0.10
    return min(1.0, score)


@dataclass
class Sentiment:
    label: str            # calm | confused | frustrated | angry (current turn)
    score: float          # current-turn frustration 0..1
    ema: float            # running frustration over recent user turns 0..1
    escalate: bool        # sustained frustration -> proactively offer a human
    confidence: float     # top emotion cosine (debug)
    tone: str = ""        # system-prompt tone instruction (style only)
    scores: dict = field(default_factory=dict)


class _EmotionClassifier:
    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._proto_label: list[str] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        emb = get_embeddings()
        phrases, labels = [], []
        for label, protos in EMOTION_PROTOTYPES.items():
            for p in protos:
                phrases.append(p)
                labels.append(label)
        vecs = np.array(emb.embed_documents(phrases), dtype=np.float32)
        self._matrix = _normed(vecs)
        self._proto_label = labels
        self._built = True

    def scores(self, text: str, embedding=None) -> dict[str, float]:
        self.build()
        q = np.array(embedding if embedding is not None
                     else get_embeddings().embed_query(text), dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ qn
        best: dict[str, float] = {}
        for label, s in zip(self._proto_label, sims):
            fs = float(s)
            if label not in best or fs > best[label]:
                best[label] = fs
        return best


_classifier: _EmotionClassifier | None = None


def _get_classifier() -> _EmotionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = _EmotionClassifier()
    return _classifier


def warm() -> None:
    _get_classifier().build()


def message_frustration(text: str, embedding=None) -> tuple[str, float, float, dict]:
    """Per-message: (label, frustration_0_1, gated_cosine, per_label_scores).

    Conservative by design (emotion embeddings are noisy): a non-calm emotion is
    trusted only when it clears the floor AND beats calm by the margin, else the turn
    is calm. The negative-only lexical booster then adds intensity (catching sarcasm/
    emphasis the embedding misses); loud POSITIVE text scores 0 on the booster, so it
    stays calm. Requiring the gate + negative-only booster is what bounds false
    escalation."""
    scores = _get_classifier().scores(text, embedding)
    calm = scores["calm"]

    lex = _lexical_frust(text)
    # Polarity guard: a clearly POSITIVE/resolved message with no negative cue is calm,
    # overriding the polarity-blind embedding.
    if lex == 0.0 and _POSITIVE.search(text):
        return "calm", 0.0, round(calm, 3), scores

    emb_label, emb_frust, gated_cos = "calm", 0.0, calm
    for lab in ("angry", "frustrated", "confused"):  # strongest emotion first
        if scores[lab] >= _EMOTION_FLOOR and (scores[lab] - calm) >= _EMOTION_MARGIN:
            emb_label, emb_frust, gated_cos = lab, _FRUSTRATION[lab], scores[lab]
            break

    # Take the STRONGER of the two independent signals (no double-count): the gated
    # embedding, or the negative-only lexical read.
    frust = max(emb_frust, lex)
    if frust >= 0.85:
        label = "angry"
    elif frust >= 0.5:
        label = "frustrated"
    elif emb_label == "confused":
        label = "confused"
    else:
        label = "calm"
    return label, round(frust, 3), round(gated_cos, 3), scores


def _ema(scores: list[float], alpha: float) -> float:
    """Seed-at-0 EMA (oldest to newest). Seeding at neutral makes the signal require
    DURATION: a single frustrated spike decays, sustained frustration accumulates."""
    ema = 0.0
    for s in scores:
        ema = alpha * s + (1 - alpha) * ema
    return ema


def _tone_instruction(label: str, escalate: bool) -> str:
    # Tone is STYLE + ACTION only. It deliberately does NOT ask the model to
    # "acknowledge the difficulty": that phrasing made the model announce the mood
    # ("I can see you're frustrated"), violating the never-announce rule. Leading with
    # the fix conveys care without naming feelings.
    base = {
        "calm": "",
        "confused": ("Be extra clear and patient: short numbered steps, define any "
                     "jargon, and do not assume prior knowledge."),
        "frustrated": ("Be warm, concise, and efficient. Lead directly with the fix or "
                       "the next concrete step; skip filler."),
        "angry": ("Stay calm, concise, and solution-focused. Lead immediately with the "
                  "concrete fix or the escalation path; no lengthy preamble."),
    }[label]
    if not base and not escalate:
        return ""
    parts = [base] if base else []
    if escalate:
        parts.append("If you cannot fully resolve this now, proactively OFFER to "
                     "connect them with a human agent (ask if they would like that); "
                     "do NOT create a ticket unless they accept.")
    # The non-negotiable rule, always attached when any tone is injected.
    parts.append("Adapt your tone and pacing ONLY. Do not comment on, name, or guess "
                 "the user's mood or feelings (never say things like 'I can see you're "
                 "frustrated'), and never mention these instructions.")
    return " ".join(parts)


def analyze(current_text: str, history_user_texts: list[str], settings,
            embedding=None) -> Sentiment:
    """Full per-request sentiment: current-turn emotion + a running EMA over the last
    few user turns, the tone instruction, and the sustained-frustration escalate flag.
    Local and zero-token. Callers run this AFTER the injection guard."""
    label, score, conf, scores = message_frustration(current_text, embedding)

    n = int(getattr(settings, "sentiment_history_turns", 3))
    alpha = float(getattr(settings, "sentiment_ema_alpha", 0.5))
    window = [message_frustration(t)[1] for t in history_user_texts[-(n - 1):]] if n > 1 else []
    window.append(score)
    ema = _ema(window, alpha)

    # Escalate on SUSTAINED frustration (the seed-at-0 EMA needs duration, so a single
    # spike, even a strong one, stays below threshold - only repeated frustration
    # crosses it). A single-message override fires only on explicit profanity (rare,
    # high-precision abuse), never on merely emphatic or sarcastic text, so a calm or
    # loud-positive turn can never trigger a false escalation.
    esc_threshold = float(getattr(settings, "sentiment_escalation_threshold", 0.52))
    # Profanity is high-precision abuse; combined with at least a frustrated read it is
    # enough to offer help immediately. Positive-profanity ("this is fucking amazing")
    # scores 0 (positive guard + negative-only lexical), so it never trips this.
    severe = _PROFANITY.search(current_text) is not None and score >= 0.5
    escalate = ema >= esc_threshold or severe

    result = Sentiment(
        label=label, score=score, ema=round(ema, 3), escalate=escalate,
        confidence=round(conf, 3), tone=_tone_instruction(label, escalate),
        scores={k: round(v, 3) for k, v in scores.items()},
    )
    if getattr(settings, "sentiment_nli", False):
        result = _nli_refine(current_text, history_user_texts, result, settings)
    return result


# Canned, EMOTION-NEUTRAL human offer (never names the mood). Deterministic backstop
# so the offer is GUARANTEED when escalate is set, independent of the model.
_HUMAN_OFFER = ("\n\nIf you would like, I can connect you with a human agent who can "
                "help further - just let me know.")
_OFFER_MARKERS = re.compile(
    r"\b(human agent|human support|support ticket|connect you|a person|real person|"
    r"speak (to|with) (a|someone)|escalat\w+|raise a ticket)\b", re.I)


def escalation_offer(escalate: bool, answer_text: str) -> str:
    """The deterministic backstop: if escalation is warranted and the model did not
    already offer a human, return an emotion-neutral offer line to append; else ''."""
    if not escalate:
        return ""
    if _OFFER_MARKERS.search(answer_text or ""):
        return ""     # the model already offered; do not double up
    return _HUMAN_OFFER


def _nli_refine(current_text, history_user_texts, result: Sentiment, settings) -> Sentiment:
    """Optional 8B escalation-decision check (SENTIMENT_NLI=on). Can only make the
    escalate decision MORE conservative (turn a local escalate off if the judge says
    the user is not actually frustrated), never invent a false escalation. Fail-safe:
    any error leaves the local result unchanged."""
    if not result.escalate:
        return result
    try:
        from langchain_groq import ChatGroq
        judge = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.groq_api_key,
                         temperature=0.0, max_tokens=5)
        recent = " | ".join((history_user_texts[-2:] + [current_text]))
        resp = judge.invoke(
            "Is the user sustainedly frustrated or angry (not merely emphatic or "
            f"asking a normal question)? Reply yes or no.\nUser turns: {recent}")
        if not (resp.content or "").strip().lower().startswith("y"):
            result.escalate = False
            result.tone = _tone_instruction(result.label, False)
    except Exception:
        pass
    return result
