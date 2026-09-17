"""Corpus lifecycle operations shared by the admin plane and MCP search tool.

Every read/write for a corpus uses that corpus's PINNED embedding model — the
query must embed with the same model the corpus was built with, or the vectors
are incompatible. All Chroma access is scoped to the corpus's own collection.
"""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .chunk import chunk_transcript
from .config import Config
from .corpora import STATUS_BUILDING, STATUS_READY, Corpus, CorpusStore
from .embed import make_embedder
from .ingest import download_transcripts, load_transcripts
from .store import VectorStore
from .throttle import ThrottleConfig

ProgressFn = Callable[[dict], None]


def _corpus_cfg(cfg: Config, corpus: Corpus) -> Config:
    """A view of cfg with the embed provider/model pinned to the corpus."""
    return replace(
        cfg, embed_provider=corpus.embed_provider, embed_model=corpus.embed_model
    )


def _deep_link(url: str, start: float) -> str:
    if not url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(start)}"


def format_hits(hits: list[dict]) -> list[dict]:
    """Shape raw store hits into the API-contract search result form."""
    out: list[dict] = []
    for h in hits:
        m = h.get("meta", {})
        start = float(m.get("start", 0.0) or 0.0)
        end = float(m.get("end", 0.0) or 0.0)
        out.append(
            {
                "text": h.get("text", ""),
                "video_title": m.get("title", ""),
                "video_id": m.get("video_id", ""),
                "url": _deep_link(m.get("url", ""), start),
                "start": start,
                "end": end,
                "score": round(1.0 - float(h.get("distance", 0.0) or 0.0), 4),
            }
        )
    return out


def ingest_corpus(
    cfg: Config,
    cstore: CorpusStore,
    corpus: Corpus,
    channel: str,
    langs=None,
    limit: int | None = None,
    throttle: ThrottleConfig | None = None,
    max_retries: int | None = None,
    cookies_browser: str | None = None,
    progress: ProgressFn | None = None,
) -> list[dict]:
    return download_transcripts(
        channel,
        cstore.transcripts_dir(corpus.id),
        langs=langs,
        limit=limit,
        throttle=throttle,
        max_retries=max_retries,
        cookies_browser=cookies_browser,
        progress=progress,
        proxy=cfg.transcript_proxy or None,
    )


def build_corpus(
    cfg: Config,
    cstore: CorpusStore,
    corpus: Corpus,
    chunk_words: int = 350,
    overlap: int = 60,
) -> int:
    """Chunk + embed this corpus's transcripts into its own collection."""
    docs = load_transcripts(cstore.transcripts_dir(corpus.id))
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_transcript(doc, chunk_words, overlap))
    cstore.update(corpus.id, status=STATUS_BUILDING)
    store = VectorStore(cfg.chroma_dir, corpus.collection)
    if all_chunks:
        embedder = make_embedder(_corpus_cfg(cfg, corpus))
        embeddings = embedder.embed_documents([c.text for c in all_chunks])
        store.add(all_chunks, embeddings)
    count = store.count()
    cstore.update(corpus.id, status=STATUS_READY, chunk_count=count)
    return count


def search_corpus(
    cfg: Config, corpus: Corpus, query: str, top_k: int = 6
) -> list[dict]:
    embedder = make_embedder(_corpus_cfg(cfg, corpus))
    store = VectorStore(cfg.chroma_dir, corpus.collection)
    hits = store.query(embedder.embed_query(query), top_k=top_k)
    return format_hits(hits)


def delete_corpus(cfg: Config, cstore: CorpusStore, corpus_id: str) -> Corpus | None:
    """Drop the collection, remove transcript files, then delete the record."""
    corpus = cstore.get(corpus_id)
    if corpus is None:
        return None
    try:
        VectorStore(cfg.chroma_dir, corpus.collection).drop()
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    corpus_dir = cstore.transcripts_dir(corpus_id).parent
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir, ignore_errors=True)
    return cstore.delete(corpus_id)
