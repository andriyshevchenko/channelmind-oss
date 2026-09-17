"""Untrusted-data fencing primitive: mark attacker-controllable text as DATA-only.

Two callers share this technique. The RAG context builder (:mod:`ytrag.rag`) wraps
every retrieved excerpt, and the intent router (:mod:`ytrag.intent`) wraps the
conversation history and the newest user message. In both cases the text is
attacker-controllable, so it is placed between OPEN/CLOSE markers and the model is
told to treat everything inside strictly as DATA, never as instructions.

:func:`neutralize` rewrites any literal fence marker that appears INSIDE that text to
an inert lookalike, so the text can neither OPEN nor CLOSE the fence and smuggle
instructions back into the trusted region. The tampering is rewritten, not stripped,
so it stays visible to the model as quoted content it may report on but never obey.
"""
from __future__ import annotations

# Inert lookalike a literal fence marker is rewritten to. Not stripped — the tampering
# stays visible to the model as quoted content, never as an active marker.
INERT_MARKER = "<<<_>>>"


def neutralize(text: str, *markers: str) -> str:
    """Rewrite each fence ``marker`` found in ``text`` to :data:`INERT_MARKER`.

    ``text`` is attacker-controllable; rewriting its embedded markers means it can
    neither open nor close the untrusted fence it will be wrapped in.
    """
    for marker in markers:
        text = text.replace(marker, INERT_MARKER)
    return text
