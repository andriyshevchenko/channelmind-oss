"""RAG search exposed as ONE MCP server over streamable HTTP.

The corpus token rides as a standard `Authorization: Bearer <token>` header, so
this server is host-agnostic (any MCP client can use it). The bearer resolves to
exactly one corpus; the tool returns transcript CHUNKS (with deep-links), never a
synthesized answer — the calling agent's own model composes the reply.

Failures are returned as MCP tool errors (isError, never a transport 500) whose
text is a typed code the host can branch on: INVALID_TOKEN, CORPUS_NOT_FOUND,
CORPUS_EMPTY, EMBED_MODEL_UNAVAILABLE.
"""
from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from .config import Config
from .corpora import CorpusStore
from .corpus_service import _corpus_cfg, format_hits
from .embed import make_embedder
from .store import VectorStore

MAX_TOP_K = 8

SEARCH_DESCRIPTION = (
    "Search this agent's YouTube-transcript knowledge base by meaning. Returns the "
    "most relevant transcript chunks, each with the source video title, a deep-link "
    "to the exact moment (url with &t=start), start/end offsets in seconds, and a "
    "similarity score. These are raw excerpts — reason over them and compose your "
    "own answer, citing the videos."
)


def _bearer_from_ctx(ctx: Context) -> str:
    try:
        request = ctx.request_context.request
    except Exception:  # noqa: BLE001 - no HTTP request context
        return ""
    if request is None:
        return ""
    auth = request.headers.get("authorization", "") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def build_mcp(cfg: Config, cstore: CorpusStore) -> FastMCP:
    mcp = FastMCP(
        "youtube-rag",
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool(description=SEARCH_DESCRIPTION)
    def search(query: str, top_k: int = 6, ctx: Context = None) -> dict:  # type: ignore[assignment]
        token = _bearer_from_ctx(ctx)
        corpus = cstore.resolve_token(token)
        if corpus is None:
            raise ValueError("INVALID_TOKEN")
        # Guard a delete that races an in-flight query.
        if cstore.get(corpus.id) is None:
            raise ValueError("CORPUS_NOT_FOUND")

        store = VectorStore(cfg.chroma_dir, corpus.collection)
        if store.count() == 0:
            raise ValueError("CORPUS_EMPTY")

        k = max(1, min(int(top_k), MAX_TOP_K))
        try:
            embedder = make_embedder(_corpus_cfg(cfg, corpus))
            q_emb = embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001 - provider/model failure
            raise ValueError("EMBED_MODEL_UNAVAILABLE") from exc

        hits = store.query(q_emb, top_k=k)
        return {"hits": format_hits(hits)}

    return mcp
