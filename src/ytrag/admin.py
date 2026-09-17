"""Admin plane: create/build/delete corpora and mint their bearer tokens.

Separate auth from the agent-facing MCP server (a static admin key), and
unreachable by an agent's corpus token. This is where a dashboard provisions a
corpus: create it, ingest a channel into it, build its index, and receive the
one-time corpus token to stamp into that agent's mcp-config.
"""
from __future__ import annotations

import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .config import Config, load_config
from .corpora import Corpus, CorpusStore
from .corpus_service import build_corpus, delete_corpus, ingest_corpus
from .throttle import ThrottleConfig


@dataclass
class CorpusJob:
    id: str
    corpus_id: str
    status: str = "running"  # running | done | error
    stage: str = "starting"
    total: int = 0
    processed: int = 0
    saved: int = 0
    indexed: int = 0
    log: list[str] = field(default_factory=list)
    error: str | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "corpus_id": self.corpus_id,
            "status": self.status,
            "stage": self.stage,
            "total": self.total,
            "processed": self.processed,
            "saved": self.saved,
            "indexed": self.indexed,
            "log": self.log[-200:],
            "error": self.error,
        }


class CorpusJobManager:
    def __init__(self, cfg: Config, cstore: CorpusStore):
        self._cfg = cfg
        self._cstore = cstore
        self._jobs: dict[str, CorpusJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> CorpusJob | None:
        return self._jobs.get(job_id)

    def start(self, corpus: Corpus, *, channel: str | None, langs, limit,
              base_delay: float, max_delay: float, do_build: bool,
              max_retries: int | None = None,
              cookies_browser: str | None = None) -> CorpusJob:
        job = CorpusJob(id=uuid.uuid4().hex[:12], corpus_id=corpus.id)
        with self._lock:
            self._jobs[job.id] = job
        t = threading.Thread(
            target=self._run,
            args=(job, corpus, channel, langs, limit, base_delay, max_delay, do_build,
                  max_retries, cookies_browser),
            daemon=True,
        )
        t.start()
        return job

    def _run(self, job, corpus, channel, langs, limit, base_delay, max_delay, do_build,
             max_retries=None, cookies_browser=None):
        try:
            if channel:
                def on_progress(ev: dict):
                    job.stage = ev.get("stage", job.stage)
                    if "total" in ev:
                        job.total = ev["total"]
                    if "index" in ev:
                        job.processed = ev["index"]
                    if ev.get("stage") == "saved":
                        job.saved += 1
                    msg = ev.get("message")
                    if msg:
                        job.log.append(msg)

                ingest_corpus(
                    self._cfg, self._cstore, corpus, channel,
                    langs=langs, limit=limit,
                    throttle=ThrottleConfig(base_delay=base_delay, max_delay=max_delay),
                    max_retries=max_retries,
                    cookies_browser=cookies_browser,
                    progress=on_progress,
                )

            if do_build:
                job.stage = "building"
                job.log.append("Building corpus index...")
                job.indexed = build_corpus(self._cfg, self._cstore, corpus)
                job.log.append(f"Indexed {job.indexed} chunks.")

            job.stage = "done"
            job.status = "done"
            job.log.append("All done.")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.log.append(f"ERROR: {exc}")


def _public(corpus: Corpus) -> dict:
    """Corpus record safe to return (never exposes the token hash)."""
    return {
        "corpusId": corpus.id,
        "name": corpus.name,
        "collection": corpus.collection,
        "embed_provider": corpus.embed_provider,
        "embed_model": corpus.embed_model,
        "status": corpus.status,
        "chunk_count": corpus.chunk_count,
        "created_at": corpus.created_at,
    }


def build_admin_app(cfg: Config, cstore: CorpusStore) -> FastAPI:
    app = FastAPI(title="youtube-rag admin")
    jobs = CorpusJobManager(cfg, cstore)

    def _guard(x_admin_key: str | None):
        expected = os.getenv("YTRAG_ADMIN_KEY", "")
        if not expected:
            return JSONResponse(
                {"error": "admin plane disabled: set YTRAG_ADMIN_KEY"}, status_code=503
            )
        if not secrets.compare_digest(x_admin_key or "", expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return None

    @app.get("/corpora")
    def list_corpora(x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        return {"corpora": [_public(c) for c in cstore.list()]}

    @app.post("/corpora")
    async def create_corpus(request: Request, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        data = await request.json()
        name = (data.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        embed_provider = data.get("embed_provider") or cfg.embed_provider
        embed_model = data.get("embed_model") or cfg.embed_model
        corpus, token = cstore.create(name, embed_provider, embed_model)
        # token is returned exactly once, at mint time.
        return {**_public(corpus), "token": token}

    @app.get("/corpora/{corpus_id}")
    def get_corpus(corpus_id: str, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        corpus = cstore.get(corpus_id)
        if corpus is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return _public(corpus)

    @app.post("/corpora/{corpus_id}/ingest")
    async def ingest(corpus_id: str, request: Request, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        corpus = cstore.get(corpus_id)
        if corpus is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = await request.json()
        channel = (data.get("channel") or "").strip()
        if not channel:
            return JSONResponse({"error": "channel is required"}, status_code=400)
        # Blank/"auto" => auto-detect every subtitle language the channel offers.
        langs = [x.strip() for x in (data.get("langs") or "").split(",") if x.strip()]
        limit = data.get("limit")
        limit = int(limit) if limit not in (None, "", 0, "0") else None
        # Blank / 0 / negative => patient unbounded retries (never fail on a block).
        raw_retries = data.get("max_retries")
        max_retries = int(raw_retries) if str(raw_retries).strip() not in ("", "None") else None
        if max_retries is not None and max_retries <= 0:
            max_retries = None
        job = jobs.start(
            corpus, channel=channel, langs=langs, limit=limit,
            base_delay=float(data.get("base_delay", 2.0)),
            max_delay=float(data.get("max_delay", 900.0)),
            do_build=bool(data.get("build", True)),
            max_retries=max_retries,
            cookies_browser=(data.get("cookies_browser") or "").strip() or None,
        )
        return {"job_id": job.id}

    @app.post("/corpora/{corpus_id}/build")
    def build(corpus_id: str, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        corpus = cstore.get(corpus_id)
        if corpus is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        job = jobs.start(
            corpus, channel=None, langs=["en"], limit=None,
            base_delay=2.0, max_delay=900.0, do_build=True,
        )
        return {"job_id": job.id}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return job.snapshot()

    @app.delete("/corpora/{corpus_id}")
    def remove(corpus_id: str, x_admin_key: str = Header(default=None)):
        guard = _guard(x_admin_key)
        if guard:
            return guard
        corpus = delete_corpus(cfg, cstore, corpus_id)
        if corpus is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"deleted": corpus_id}

    return app
