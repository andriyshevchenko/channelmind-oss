"""Rebuild the single shared vector index from every registered source.

A full drop-and-rebuild (rather than incremental add) keeps the index a faithful
mirror of the registry: deleting a source or re-ingesting a channel can never
leave orphan vectors behind. Each chunk is stamped with its source's author so
the assistant can attribute and compare advice across pickup artists and docs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .chunk import chunk_transcript
from .config import Config
from .documents import document_to_doc
from .embed import make_embedder
from .sources import TYPE_DOCUMENT, TYPE_YOUTUBE, Source, SourceStore
from .store import VectorStore

ProgressFn = Callable[[dict], None]


def _load_source_docs(sstore: SourceStore, source: Source) -> list[dict]:
    if source.type == TYPE_YOUTUBE:
        tdir = sstore.transcripts_dir(source.id)
        if not tdir.exists():
            return []
        return [
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(tdir.glob("*.json"))
        ]
    if source.type == TYPE_DOCUMENT:
        ddir = sstore.documents_dir(source.id)
        if not ddir.exists():
            return []
        docs = []
        for f in sorted(ddir.iterdir()):
            if f.is_file():
                docs.append(document_to_doc(source.id, f, source.label))
        return docs
    return []


def rebuild_index(
    cfg: Config, sstore: SourceStore, progress: ProgressFn | None = None
) -> int:
    """Drop and rebuild the shared index from all sources. Returns total chunks."""
    def emit(**kw):
        if progress:
            progress(kw)

    store = VectorStore(cfg.chroma_dir)
    try:
        store.drop()
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    store = VectorStore(cfg.chroma_dir)

    embedder = make_embedder(cfg)
    total = 0
    for source in sstore.list():
        docs = _load_source_docs(sstore, source)
        chunks = []
        for d in docs:
            for c in chunk_transcript(d):
                c.author = source.author
                c.source = source.label
                chunks.append(c)
        if chunks:
            embeddings = embedder.embed_documents([c.text for c in chunks])
            store.add(chunks, embeddings)
        sstore.update(source.id, item_count=len(docs), chunk_count=len(chunks))
        total += len(chunks)
        emit(
            stage="indexed_source",
            source=source.label,
            author=source.author,
            chunks=len(chunks),
            message=f"Indexed {len(chunks)} chunks from {source.label} ({source.author})",
        )
    return store.count()
