"""End-to-end check of the RAG<->sensorium API contract v1.

Boots the combined server in-process (uvicorn thread), provisions a corpus via
the admin plane, seeds a synthetic transcript with real timecodes, builds the
index, then drives the MCP endpoint as an agent would: streamable HTTP with an
`Authorization: Bearer <corpus-token>` header. Verifies the search hit shape,
deep-links, and the typed error path for a bad token.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import httpx
import uvicorn

os.environ.setdefault("YTRAG_ADMIN_KEY", "test-admin-key")
# Isolate test artifacts from the real data/chroma dirs.
ROOT = Path(__file__).resolve().parent.parent
os.environ["YTRAG_DATA_DIR"] = str(ROOT / "_e2e" / "data")
os.environ["YTRAG_CHROMA_DIR"] = str(ROOT / "_e2e" / "chroma")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from ytrag.config import load_config  # noqa: E402
from ytrag.corpora import CorpusStore  # noqa: E402
from ytrag.server import build_server  # noqa: E402

# Provider-independent verification: a deterministic bag-of-words hash embedder so
# the contract mechanics can be proven without a live embedding API. Enable with
# YTRAG_FAKE_EMBED=1. (The real path uses the corpus's pinned provider/model.)
if os.getenv("YTRAG_FAKE_EMBED") == "1":
    import hashlib
    import re

    DIM = 256

    class _FakeEmbedder:
        def _vec(self, text: str):
            v = [0.0] * DIM
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            return [x / norm for x in v]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    def _fake_make_embedder(cfg):
        return _FakeEmbedder()

    import ytrag.corpus_service as _cs
    import ytrag.mcp_server as _ms
    _cs.make_embedder = _fake_make_embedder
    _ms.make_embedder = _fake_make_embedder

HOST, PORT = "127.0.0.1", 8077
ADMIN = f"http://{HOST}:{PORT}/admin"
MCP_URL = f"http://{HOST}:{PORT}/mcp"
HK = {"x-admin-key": os.environ["YTRAG_ADMIN_KEY"]}


def start_server():
    app = build_server()
    cfg = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        try:
            httpx.get(f"http://{HOST}:{PORT}/mcp", timeout=1)
            return server
        except Exception:
            time.sleep(0.2)
    return server


def seed_transcript(corpus_id: str):
    cfg = load_config()
    cstore = CorpusStore(cfg.data_dir)
    tdir = cstore.transcripts_dir(corpus_id)
    tdir.mkdir(parents=True, exist_ok=True)
    doc = {
        "id": "abc123",
        "title": "How GPS jamming works",
        "url": "https://youtu.be/abc123",
        "segments": [
            {"start": 0.0, "end": 8.0,
             "text": "Welcome. Today we explain how GPS jamming disrupts satellite navigation signals."},
            {"start": 8.0, "end": 20.0,
             "text": "A jammer broadcasts noise on the same L1 frequency so receivers lose lock and cannot compute a position fix."},
            {"start": 20.0, "end": 33.0,
             "text": "Spoofing is different: it transmits counterfeit signals to trick the receiver into reporting a false location."},
        ],
    }
    doc["text"] = " ".join(s["text"] for s in doc["segments"])
    (tdir / "abc123.json").write_text(json.dumps(doc), encoding="utf-8")


async def run_mcp(token: str, query: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(MCP_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("search", {"query": query, "top_k": 3})
            return tools, result


def main():
    start_server()
    print("server up")

    # 1) create corpus -> mint token
    r = httpx.post(f"{ADMIN}/corpora", json={"name": "gps-demo"}, headers=HK, timeout=30)
    r.raise_for_status()
    corpus = r.json()
    token = corpus["token"]
    cid = corpus["corpusId"]
    print("minted corpus:", cid, "embed:", corpus["embed_provider"], corpus["embed_model"])
    assert token.startswith("ytr_")

    # 2) seed + build
    seed_transcript(cid)
    rb = httpx.post(f"{ADMIN}/corpora/{cid}/build", headers=HK, timeout=30)
    rb.raise_for_status()
    job_id = rb.json()["job_id"]
    for _ in range(120):
        js = httpx.get(f"{ADMIN}/jobs/{job_id}", headers=HK, timeout=10).json()
        if js["status"] in ("done", "error"):
            break
        time.sleep(1)
    print("build job:", js["status"], "indexed:", js.get("indexed"), js.get("error") or "")
    assert js["status"] == "done", js

    # 3) MCP search with valid bearer
    tools, result = asyncio.run(run_mcp(token, "How does a jammer stop a GPS receiver?"))
    tool_names = [t.name for t in tools.tools]
    print("tools:", tool_names)
    assert tool_names == ["search"], tool_names
    payload = result.structuredContent if result.structuredContent else json.loads(result.content[0].text)
    hits = payload["hits"]
    print("isError:", result.isError, "num hits:", len(hits))
    top = hits[0]
    print("top hit keys:", sorted(top.keys()))
    print("top url:", top["url"], "| start:", top["start"], "| score:", top["score"])
    assert not result.isError
    assert {"text", "video_title", "video_id", "url", "start", "end", "score"} <= set(top)
    assert "&t=" in top["url"] or "?t=" in top["url"]

    # 4) typed error: bad token -> INVALID_TOKEN, isError, no 500
    _, bad = asyncio.run(run_mcp("ytr_totally-wrong", "anything"))
    err_text = bad.content[0].text if bad.content else ""
    print("bad-token isError:", bad.isError, "| text:", err_text)
    assert bad.isError and "INVALID_TOKEN" in err_text

    print("\nCONTRACT E2E: PASS")


if __name__ == "__main__":
    main()
