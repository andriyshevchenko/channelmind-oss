"""Corpus registry: per-agent transcript collections and their bearer tokens.

Each corpus owns one Chroma collection (hard isolation) and pins the embedding
provider/model it was built with — a query MUST embed with the same model, else
vectors are incompatible. The opaque bearer token an agent presents resolves to
exactly one corpus, so an agent physically cannot address another's data. Only
the sha256 of a token is stored at rest; the raw token is shown once at mint.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json
from .store import collection_name

TOKEN_PREFIX = "ytr_"

STATUS_EMPTY = "empty"
STATUS_BUILDING = "building"
STATUS_READY = "ready"


@dataclass
class Corpus:
    id: str
    name: str
    collection: str
    embed_provider: str
    embed_model: str
    token_hash: str
    status: str
    chunk_count: int
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def mint_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


class CorpusStore:
    """JSON-backed registry of corpora, indexed by id and by token hash."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "corpora.json"

    # ---- layout -------------------------------------------------------
    def transcripts_dir(self, corpus_id: str) -> Path:
        return self._root / "corpora" / corpus_id / "transcripts"

    # ---- reads --------------------------------------------------------
    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def list(self) -> list[Corpus]:
        return [Corpus(**rec) for rec in self._load().values()]

    def get(self, corpus_id: str) -> Corpus | None:
        rec = self._load().get(corpus_id)
        return Corpus(**rec) if rec else None

    def resolve_token(self, token: str) -> Corpus | None:
        if not token:
            return None
        th = hash_token(token)
        for rec in self._load().values():
            if secrets.compare_digest(rec.get("token_hash", ""), th):
                return Corpus(**rec)
        return None

    # ---- writes -------------------------------------------------------
    def create(
        self, name: str, embed_provider: str, embed_model: str
    ) -> tuple[Corpus, str]:
        """Create a corpus and mint its bearer token. Returns (corpus, raw_token)."""
        corpus_id = uuid.uuid4().hex[:12]
        token = mint_token()
        corpus = Corpus(
            id=corpus_id,
            name=name,
            collection=collection_name(corpus_id),
            embed_provider=embed_provider,
            embed_model=embed_model,
            token_hash=hash_token(token),
            status=STATUS_EMPTY,
            chunk_count=0,
            created_at=_now(),
        )
        with file_lock(self._registry):
            data = self._load()
            data[corpus_id] = asdict(corpus)
            atomic_write_json(self._registry, data)
        return corpus, token

    def update(
        self, corpus_id: str, *, status: str | None = None, chunk_count: int | None = None
    ) -> Corpus | None:
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(corpus_id)
            if rec is None:
                return None
            if status is not None:
                rec["status"] = status
            if chunk_count is not None:
                rec["chunk_count"] = chunk_count
            data[corpus_id] = rec
            atomic_write_json(self._registry, data)
            return Corpus(**rec)

    def delete(self, corpus_id: str) -> Corpus | None:
        """Remove the registry record. Collection/files are dropped by the caller."""
        with file_lock(self._registry):
            data = self._load()
            rec = data.pop(corpus_id, None)
            if rec is None:
                return None
            atomic_write_json(self._registry, data)
            return Corpus(**rec)
