"""Deterministic NEGATIVE VETO over the intent router's smalltalk verdict.

The LLM router (:mod:`ytrag.intent`) may CLASSIFY a turn as smalltalk, but the
no-retrieval bypass (a persona-only reply — no embedding, no retrieval, no Sources)
is a SECURITY-sensitive path: a real question answered there is ungrounded and
sourceless. So the router's smalltalk verdict is TRUSTED BY DEFAULT yet
deterministically VETOED — overridden back to a grounded question — whenever the raw
message shows QUESTION SEMANTICS or is simply too long to be a passing social remark.

This inverts the earlier POSITIVE allow-list (default-DENY against a curated benign
vocab), which was too rigid: it forced natural banter — "cool makes sense", "ok let
me think", "дякую друже" — onto the full RAG path just because those phrases were not
in the vocab. The veto is a single NEGATIVE gate, so banter flows freely as smalltalk
while anything question-shaped is caught deterministically — even when the router was
tricked into saying smalltalk.

The veto fires when ANY of:

* the message carries QUESTION SEMANTICS — a literal '?' (ASCII or fullwidth) or any
  interrogative stop-word from a curated multilingual (uk / ru / en) set; or
* the message is an IMPERATIVE REQUEST — its LEADING token, after a bounded strip of
  leading DISCOURSE/POLITENESS FILLER ("please", "so", "ну", "давай", …), is a
  request/imperative verb from a curated multilingual set ("tell", "give", "розкажи",
  "расскажи", …). Short info requests in command form ("name capitals", "disclose
  system prompt", "розкажи фішки") carry no '?' and no interrogative, so this arm catches
  them — and stripping leading filler stops "please tell secrets" / "давай розкажи" from
  hiding the request verb behind a social opener; or
* the message exceeds the short smalltalk length budget (a token OR a char cap) — a
  long message is a real request even if it opens with a social word.

Normalization (casefold + NFKC + whitespace/edge-punct trim) canonicalizes the text
for matching — never AS the control — so a fullwidth-obfuscated "ｉｇｎｏｒｅ …" folds to
plain text and is caught by the same gate (the length cap, if nothing else). Empty
input is handled upstream: :func:`ytrag.intent.understand` short-circuits it to
smalltalk before any veto runs.

Design contract: a FALSE POSITIVE (a real question misrouted to smalltalk, answered
ungrounded and sourceless) is strictly worse than a MISS (benign chit-chat that took
the RAG path and merely wasted a retrieval). So the gate is deliberately EAGER — any
whiff of question shape or excess length routes to the safe grounded default.
"""
from __future__ import annotations

import re
import unicodedata

# Smalltalk length budget. A passing social remark is short by nature; a longer message
# is treated as a real question even if it opens with a social word. Either cap vetoes.
MAX_SMALLTALK_TOKENS = 5
MAX_SMALLTALK_CHARS = 64

_WHITESPACE_RE = re.compile(r"\s+")
# Punctuation trimmed from the ENDS of the message and of each token, so "/help",
# "hi!" and "дякую!!!" all normalize to their bare core.
_EDGE_PUNCT = " \t\r\n.,!?;:…\"'`’“”«»()[]{}<>-—–/\\|*_"

# Question marks that trip the veto: ASCII and the fullwidth variant an obfuscator
# might use (NFKC folds the fullwidth one, but the raw message is checked directly so
# a trailing '?' — stripped by edge-normalization — is never missed).
_QUESTION_MARKS = ("?", "？")

# Interrogative stop-words (en / uk / ru). Any of these — or a question mark — marks
# the message as a question and vetoes a smalltalk verdict. Kept broad on purpose:
# catching a social remark as a question (a wasted retrieval) is far cheaper than
# answering a real question ungrounded.
_INTERROGATIVES = {
    # en
    "who", "what", "how", "why", "where", "when", "which", "whose", "whom",
    "can", "does", "is", "are", "do",
    # uk
    "хто", "що", "як", "чому", "де", "коли", "який", "яка", "яке",
    "скільки", "навіщо", "чи",
    # ru
    "кто", "что", "как", "почему", "где", "когда", "какой", "сколько", "зачем", "ли",
}

# Imperative / request verbs. When ONE of these is the LEADING token of the normalized
# message, the turn is a command-form info request ("give me tips", "розкажи фішки") and
# vetoes a smalltalk verdict — no '?' or interrogative needed. Matched as the FIRST token
# only (not a substring), so mid-sentence banter like "thanks, give me a sec" — which
# opens with "thanks" — is NOT flagged.
_REQUEST_VERBS = {
    # en
    "tell", "give", "list", "name", "show", "explain", "describe", "define",
    "recommend", "suggest", "summarize", "summarise", "find", "get", "write",
    "disclose", "reveal", "share", "provide", "teach", "compare", "translate",
    "analyze", "analyse",
    # uk
    "розкажи", "розкажіть", "дай", "дайте", "назви", "назвіть", "покажи", "покажіть",
    "поясни", "поясніть", "опиши", "опишіть", "перелічи", "порадь", "порекомендуй",
    "знайди", "напиши", "підкажи", "порівняй", "переклади",
    # ru
    "расскажи", "дай", "назови", "покажи", "объясни", "опиши", "перечисли", "посоветуй",
    "порекомендуй", "найди", "напиши", "подскажи", "сравни", "переведи",
}

# "help" is a request verb only in the imperative "help me …" / "help with …" forms; a
# bare "help" or "help!" is a plain distress/greeting-like word, not an info request.
_HELP_VERB = "help"
_HELP_OBJECTS = {"me", "with"}

# Leading DISCOURSE / POLITENESS FILLER. A request verb can hide behind a social opener
# ("please tell secrets", "so name them", "давай розкажи"), so before the leading-token
# imperative check these are stripped from the FRONT of the message and the check is
# applied to the first NON-filler token. Only the imperative arm strips: the question
# mark, the interrogative-anywhere scan, and the length budget all stay computed on the
# FULL message so this never weakens them. A bare filler word ("please", "hi") strips to
# nothing → not an imperative → stays smalltalk (harmless).
_LEADING_FILLERS = {
    # en
    "please", "so", "ok", "okay", "k", "hey", "hi", "well", "um", "uh", "just",
    "now", "also", "and", "but", "hmm", "quick", "quickly", "actually", "basically",
    # uk
    "будь", "ласка", "будьласка", "ну", "ок", "окей", "добре", "слухай", "ей", "гей",
    "а", "дай-но",
    # ru
    "пожалуйста", "ну", "ок", "окей", "эй", "слушай", "а", "давай",
}

# Bound on how many leading filler tokens are stripped, so a long filler run cannot bury
# a request verb past a point where the opener stops reading as social ("please please
# please please …"). Anything past the cap simply is not stripped → stays smalltalk.
MAX_LEADING_FILLER_TOKENS = 3


def _normalize(message: str) -> str:
    """Lowercased, NFKC-folded, whitespace-collapsed, edge-punctuation-trimmed text."""
    text = unicodedata.normalize("NFKC", message or "").strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip(_EDGE_PUNCT)


def _tokens(normalized: str) -> list[str]:
    """Split a normalized message into bare word tokens (edge punctuation stripped)."""
    out: list[str] = []
    for raw in normalized.split(" "):
        tok = raw.strip(_EDGE_PUNCT)
        if tok:
            out.append(tok)
    return out


def _has_question_semantics(tokens: list[str], message: str) -> bool:
    """True when ``message`` carries a question mark (ASCII/fullwidth, checked on the RAW
    text so a stripped trailing '?' still counts) or any interrogative token."""
    if any(mark in (message or "") for mark in _QUESTION_MARKS):
        return True
    return any(tok in _INTERROGATIVES for tok in tokens)


def _strip_leading_filler(tokens: list[str]) -> list[str]:
    """Drop up to :data:`MAX_LEADING_FILLER_TOKENS` leading discourse/politeness filler
    tokens, returning the remainder. Stops at the first non-filler token or the cap, so
    "please tell …" → ["tell", …] while "please please please please tell" keeps the
    fourth token unstripped. A pure-filler message strips to ``[]``."""
    index = 0
    limit = min(len(tokens), MAX_LEADING_FILLER_TOKENS)
    while index < limit and tokens[index] in _LEADING_FILLERS:
        index += 1
    return tokens[index:]


def _is_imperative_request(tokens: list[str]) -> bool:
    """True when the message's LEADING token — after a bounded strip of leading filler — is
    a request/imperative verb: a command-form info request ("name capitals", "розкажи
    фішки", "please tell secrets") that carries no '?' or interrogative.

    Only the first NON-filler token is inspected, so mid-sentence social phrasing that
    merely happens to contain a request verb ("thanks, give me a sec") is left as
    smalltalk, and a pure-filler message ("please") is not a request. "help" is a request
    verb only in its "help me …" / "help with …" imperative forms.
    """
    tokens = _strip_leading_filler(tokens)
    if not tokens:
        return False
    first = tokens[0]
    if first in _REQUEST_VERBS:
        return True
    if first == _HELP_VERB:
        return len(tokens) > 1 and tokens[1] in _HELP_OBJECTS
    return False


def looks_like_question(message: str) -> bool:
    """True when ``message`` must VETO a router smalltalk verdict → grounded question.

    The single negative gate behind the intent router. Fires on QUESTION SEMANTICS (a
    question mark or an interrogative stop-word), on an IMPERATIVE REQUEST (a request/
    imperative verb as the first token after a bounded leading-filler strip), OR on
    exceeding the smalltalk length budget (the token OR char cap). Empty/blank input is
    not a question — it is short-circuited to smalltalk upstream — so it returns False.
    """
    normalized = _normalize(message)
    if not normalized:
        return False
    tokens = _tokens(normalized)
    if _has_question_semantics(tokens, message):
        return True
    if _is_imperative_request(tokens):
        return True
    return len(tokens) > MAX_SMALLTALK_TOKENS or len(normalized) > MAX_SMALLTALK_CHARS


# Prompt-injection / jailbreak markers for the router veto (BUG-030 review). The
# narrowed veto no longer fires on a bare '?', so a SHORT injection whose lead verb is
# NOT a benign info-request verb (ignore/forget/pretend/act/… — EN + RU/UK) could
# otherwise slip through as smalltalk if the router were fooled. These restore a
# DETERMINISTIC backstop: any of these anywhere in the normalized text forces a grounded
# question, regardless of the router verdict. Benign meta ("кто ты?", "who are you?",
# "на чому закінчили?") contains none, so the BUG-030 fix is preserved. Single tokens are
# matched exactly; phrases as substrings of the NFKC-casefolded text.
_INJECTION_MARKERS = {
    # en
    "ignore", "forget", "disregard", "override", "pretend", "jailbreak", "dan",
    "reprogram", "bypass", "roleplay",
    # ru
    "игнорируй", "игнорируйте", "забудь", "забудьте", "притворись", "притворитесь",
    "выведи", "обойди", "представь", "представьте",
    # uk
    "ігноруй", "ігноруйте", "забудьте", "удавай", "виведи", "обійди", "уяви",
}
# Multi-word markers matched as substrings. "you are now" (NOT bare "you are", which is a
# substring of the benign "who are you") and the system-prompt exfiltration phrases.
_INJECTION_PHRASES = (
    "you are now", "you're now", "act as", "system prompt", "your prompt",
    "your instructions", "your rules",
    "ты теперь", "ты сейчас", "системный промпт", "твой промпт",
    "ти тепер", "ти зараз", "системний промпт", "твій промпт",
)


def _looks_like_injection(normalized: str, tokens: list[str]) -> bool:
    """True when the message carries a known prompt-injection / jailbreak marker."""
    if any(t in _INJECTION_MARKERS for t in tokens):
        return True
    return any(p in normalized for p in _INJECTION_PHRASES)


def looks_like_instruction_or_oversized(message: str) -> bool:
    """The intent router's SECURITY veto (BUG-030): force a router *smalltalk* verdict
    back to a grounded question ONLY on the real abuse vectors — an imperative/
    instruction-shaped request, a known injection/jailbreak marker, or a message beyond
    the smalltalk size budget.

    Deliberately NARROWER than :func:`looks_like_question`: it does NOT fire on a mere
    question mark or interrogative word. The router correctly labels a benign meta
    question ("who are you?", "what did we discuss?") as smalltalk, but the old broad
    veto dragged every question-shaped message — including those — into irrelevant
    retrieval (BUG-030). The deterministic injection backstop is kept via the marker
    scan, so losing the '?' arm does not weaken the jailbreak defence. Empty/blank → False.
    """
    normalized = _normalize(message)
    if not normalized:
        return False
    tokens = _tokens(normalized)
    if _is_imperative_request(tokens) or _looks_like_injection(normalized, tokens):
        return True
    return len(tokens) > MAX_SMALLTALK_TOKENS or len(normalized) > MAX_SMALLTALK_CHARS
