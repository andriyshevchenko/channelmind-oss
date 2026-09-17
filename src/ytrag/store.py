"""Persistent Chroma vector store for transcript chunks.

One collection per corpus gives hard isolation: deleting an agent's corpus is a
single `drop()` of its collection. The default collection ("transcripts") backs
the local single-corpus CLI/web flow.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from .chunk import Chunk

DEFAULT_COLLECTION = "transcripts"


def collection_name(corpus_id: str) -> str:
    return f"corpus_{corpus_id}"


class VectorStore:
    def __init__(self, path: Path, collection: str = DEFAULT_COLLECTION):
        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._name = collection
        self._client = chromadb.PersistentClient(path=str(path))
        # Embeddings are supplied explicitly; disable Chroma's default embedder.
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "video_id": c.video_id,
                    "title": c.title,
                    "url": c.url,
                    "index": c.index,
                    "start": float(c.start),
                    "end": float(c.end),
                    "author": c.author,
                    "source": c.source,
                }
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int = 6) -> list[dict]:
        res = self._col.query(query_embeddings=[embedding], n_results=top_k)
        hits: list[dict] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "meta": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self._col.count()

    def video_chunk_counts(self) -> dict[str, int]:
        """Map each stored ``video_id`` to how many of its chunks are in the collection.

        The incremental rebuild path (BUG-021) uses this to decide, per video,
        whether it is fully indexed: a video is skipped only when its PRESENT chunk
        count equals its EXPECTED chunk count, so a video left partial by an
        interrupted embed (F1) re-embeds and self-heals on the next pass instead of
        being skipped forever. Only ``video_id`` metadata is fetched (no
        documents/embeddings) to keep the scan cheap."""
        got = self._col.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for m in got.get("metadatas") or []:
            vid = m.get("video_id") if m else None
            if vid:
                counts[vid] = counts.get(vid, 0) + 1
        return counts

    def delete_video(self, video_id: str) -> None:
        """Delete every chunk of ONE video from the collection (scoped, not a drop).

        The incremental rebuild calls this before re-embedding a video whose chunk
        set changed, so a SHORTENED transcript leaves no orphan ``{vid}::i`` chunks
        past the new count — the video then matches present == expected and stops
        re-embedding on later passes. Other videos are untouched."""
        if not video_id:
            return
        self._col.delete(where={"video_id": video_id})

    def drop(self) -> None:
        """Delete this corpus's collection entirely (hard isolation cleanup)."""
        self._client.delete_collection(self._name)
