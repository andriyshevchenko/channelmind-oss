"""Split transcripts into overlapping word-based chunks for embedding.

Each chunk carries the start/end offset (seconds) of its underlying captions so
retrieval hits can deep-link into the video at the right moment (`&t=<start>`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    video_id: str
    title: str
    url: str
    index: int
    start: float
    end: float
    author: str = ""
    source: str = ""


def _timed_words(doc: dict) -> list[tuple[str, float, float]]:
    """Flatten a doc into (word, start, end) triples using segment timing when
    available, falling back to untimed prose (start/end = 0.0)."""
    words: list[tuple[str, float, float]] = []
    segments = doc.get("segments") or []
    if segments:
        for seg in segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            for w in str(seg.get("text", "")).split():
                words.append((w, start, end))
    else:
        for w in doc.get("text", "").split():
            words.append((w, 0.0, 0.0))
    return words


def chunk_transcript(
    doc: dict, chunk_words: int = 350, overlap_words: int = 60
) -> list[Chunk]:
    timed = _timed_words(doc)
    if not timed:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[Chunk] = []
    for i, start_idx in enumerate(range(0, len(timed), step)):
        piece = timed[start_idx : start_idx + chunk_words]
        if not piece:
            break
        chunks.append(
            Chunk(
                id=f"{doc['id']}::{i}",
                text=" ".join(w for w, _, _ in piece),
                video_id=doc["id"],
                title=doc["title"],
                url=doc["url"],
                index=i,
                start=piece[0][1],
                end=piece[-1][2],
            )
        )
        if start_idx + chunk_words >= len(timed):
            break
    return chunks


def fragments_for_chunk(doc: dict, chunk: "Chunk") -> list[dict]:
    """The timed caption segments that make up one chunk, in order (BUG-028).

    Each fragment = ``{"id", "start", "end", "text"}`` where ``id`` is the
    fragment's ordinal WITHIN the chunk. This is the id-tagged unit we feed the
    LLM: the model cites a fragment id, and we map that id straight back to its
    real caption ``start`` (seconds) for the deep-link — so a citation lands on
    the exact spoken moment, not the head of the whole ~350-word chunk (whose
    ``start`` is near-zero on short videos).

    Selected by caption start-time within the chunk's ``[start, end]`` window (the
    same segment timing the chunk was built from), so fragment timecodes are the
    transcript's OWN values — never computed or guessed.
    """
    segs = doc.get("segments") or []
    out: list[dict] = []
    for s in segs:
        st = float(s.get("start", 0.0) or 0.0)
        if chunk.start <= st <= chunk.end:
            txt = str(s.get("text", "")).strip()
            if not txt:
                continue
            out.append({"id": len(out), "start": st,
                        "end": float(s.get("end", st) or st), "text": txt})
    return out


def timecode_for_fragment(fragments: list[dict], frag_id) -> float | None:
    """Map an LLM-returned fragment id to its caption start-seconds.

    Returns ``None`` when the id is not an int or is out of range — the caller
    then falls back to the chunk head, so a hallucinated/garbled id can never
    produce a wrong deep-link, only the current (safe) behaviour."""
    try:
        i = int(frag_id)
    except (TypeError, ValueError):
        return None
    if 0 <= i < len(fragments):
        return float(fragments[i]["start"])
    return None
