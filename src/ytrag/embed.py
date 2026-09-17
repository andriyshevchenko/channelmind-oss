"""Embedding backends: Voyage AI (native) or any OpenAI-compatible API (OpenAI, OpenRouter)."""
from __future__ import annotations

import asyncio
import threading
import weakref
from typing import Protocol, Sequence

from .config import Config


# Process-wide Voyage clients, keyed by API key.  A fresh embedder is created for
# each chat request, so these keep Voyage's connection pools warm between requests.
_VOYAGE_CLIENTS: dict[str, object] = {}
_VOYAGE_CLIENTS_LOCK = threading.Lock()


def shared_voyage_client(api_key: str):
    """Return the process-wide synchronous Voyage client for ``api_key``."""
    with _VOYAGE_CLIENTS_LOCK:
        client = _VOYAGE_CLIENTS.get(api_key)
        if client is None:
            import voyageai

            client = _VOYAGE_CLIENTS[api_key] = voyageai.Client(api_key=api_key)
    return client


# Async clients are loop-scoped: this preserves a warm pool across requests while
# avoiding reuse of an event-loop-bound client when tests or workers use many loops.
_ASYNC_VOYAGE_CLIENTS: "weakref.WeakKeyDictionary[object, dict[str, object]]" = (
    weakref.WeakKeyDictionary()
)
_ASYNC_VOYAGE_CLIENTS_LOCK = threading.Lock()


def shared_async_voyage_client(api_key: str):
    """Return the current loop's shared asynchronous Voyage client for ``api_key``."""
    loop = asyncio.get_running_loop()
    with _ASYNC_VOYAGE_CLIENTS_LOCK:
        per_loop = _ASYNC_VOYAGE_CLIENTS.get(loop)
        if per_loop is None:
            per_loop = _ASYNC_VOYAGE_CLIENTS[loop] = {}
        client = per_loop.get(api_key)
        if client is None:
            import voyageai

            client = per_loop[api_key] = voyageai.AsyncClient(api_key=api_key)
    return client


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    # Async twins (PLAN §5b): same vectors and batching as the sync methods, awaited
    # on the event loop so the chat request path never blocks it.
    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def aembed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str):
        self._client = shared_voyage_client(api_key)
        # The async client remains lazy: resolve it only from _aembed, inside a
        # running loop, so sync-only CLI paths never construct one.
        self._api_key = api_key
        self.model = model

    def embed_documents(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        return self._embed(texts, "document", batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        # Batch the QUERY-side embeddings in one request (BUG-029): same input_type
        # as embed_query, so retrieval vectors are byte-for-byte the per-query path —
        # only the number of network round-trips changes (N → 1).
        texts = list(texts)
        if not texts:
            return []
        return self._embed(texts, "query")

    def _embed(self, texts: Sequence[str], input_type: str, batch_size: int = 128) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._client.embed(batch, model=self.model, input_type=input_type)
            out.extend(resp.embeddings)
        return out

    # ---- async twins (PLAN §5b) — same requests/vectors, awaited ----------
    async def aembed_documents(
        self, texts: Sequence[str], batch_size: int = 128
    ) -> list[list[float]]:
        return await self._aembed(texts, "document", batch_size)

    async def aembed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        return await self._aembed(texts, "query")

    async def _aembed(
        self, texts: Sequence[str], input_type: str, batch_size: int = 128
    ) -> list[list[float]]:
        client = shared_async_voyage_client(self._api_key)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = await client.embed(batch, model=self.model, input_type=input_type)
            out.extend(resp.embeddings)
        return out


class OpenAICompatEmbedder:
    """Works with OpenAI and OpenRouter (both expose /v1/embeddings)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        # Share a warm, pooled client across requests (BUG-029): a fresh embedder is
        # built per chat request, so a cold handshake here was on the critical path.
        from .llm import shared_openai_client

        self._client = shared_openai_client(api_key, base_url)
        # Kept for the async path: the per-LOOP async client is resolved at call
        # time (inside a running loop), never at construction (CLI has no loop).
        self._api_key = api_key
        self._base_url = base_url
        self.model = model

    def embed_documents(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        # OpenAI/OpenRouter embeddings have no query/document distinction, so a batched
        # query embed is just the batched document call — one request for all queries.
        texts = list(texts)
        if not texts:
            return []
        return self.embed_documents(texts)

    # ---- async twins (PLAN §5b) — same requests/vectors, awaited ----------
    async def aembed_documents(
        self, texts: Sequence[str], batch_size: int = 128
    ) -> list[list[float]]:
        from .llm import shared_async_openai_client

        client = shared_async_openai_client(self._api_key, self._base_url)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = await client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    async def aembed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        return await self.aembed_documents(texts)


def make_embedder(cfg: Config) -> Embedder:
    from .config import test_mode

    if test_mode():
        from .testmode import FakeEmbedder

        return FakeEmbedder()
    if cfg.embed_provider == "voyage":
        return VoyageEmbedder(cfg.voyage_api_key, cfg.embed_model)
    return OpenAICompatEmbedder(
        cfg.embed_key(), cfg.embed_model, cfg.openai_compat_base_url(cfg.embed_provider)
    )
