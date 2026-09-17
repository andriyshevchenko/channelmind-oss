"""Per-bot lifecycle: create/delete a bot, ingest its sources, rebuild its
isolated index, and chat against it.

A bot owns one ``Corpus`` (one Chroma collection = hard isolation) plus many
sources. Each source keeps its own transcript/document files under
``<data>/corpora/<corpus_id>/sources/<source_id>/`` so a rebuild can attribute
every chunk to the right author and a deleted source's files are trivially
dropped. Rebuild has two modes (see :func:`rebuild_bot`): the routine INCREMENTAL
mode appends only the chunks of NEW videos to the existing collection (BUG-021 —
auto-sync / "index more" / re-import never re-embed unchanged content), while the
FULL mode drops and rebuilds the whole collection and is reserved for the cases
that semantically require it (a reconcile after a source is removed, or an
explicit "Rebuild" that must mirror the registry exactly).

Retrieval/embedding always use the corpus's PINNED embed model — a query must
embed with the same model the corpus was built with, or the vectors mismatch.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator

from .bots import (
    SRC_BUILDING,
    SRC_ERROR,
    SRC_READY,
    TYPE_DOCUMENT,
    TYPE_YOUTUBE,
    Bot,
    BotSource,
    BotStore,
)
from .chunk import chunk_transcript, fragments_for_chunk
from .config import Config
from .corpora import Corpus, CorpusStore, STATUS_BUILDING, STATUS_READY
from .documents import document_to_doc
from .embed import make_embedder
from .ingest import download_transcripts, load_transcripts
from .metrics import MetricsStore
from .rag import Assistant
from .store import VectorStore, collection_name
from .throttle import ThrottleConfig
from .transcript_ingest import write_items
from .usage import Usage, UsageStore

ProgressFn = Callable[[dict], None]


# ---- per-bot rebuild serialization (shared lock registry) -------------
# A rebuild is a destructive drop() + rebuild-from-sources on ONE Chroma
# collection. There are TWO independent drivers that trigger it — the FIFO
# ingest worker (``ingest_queue``) and the user-facing ``JobManager`` — and the
# whole app is a single process (single replica is the invariant), so an
# ingest-triggered rebuild and a user-triggered "Rebuild" for the SAME bot can
# otherwise race and drop+rebuild one collection at once (→ half-dropped /
# duplicated / empty vectors). This registry hands out ONE ``threading.Lock``
# per bot id; ``rebuild_bot`` acquires it around the whole drop+rebuild critical
# section, so EVERY caller of ``rebuild_bot`` — regardless of driver — is
# mutually exclusive for a given bot. Different bots get different locks, so they
# still rebuild independently. A caller only ever holds this ONE lock (never
# nested with another), and it is always released in ``finally`` (the
# ``with`` block), so no deadlock is possible.
_REBUILD_LOCKS: dict[str, threading.Lock] = {}
_REBUILD_LOCKS_GUARD = threading.Lock()


def _rebuild_lock_for(bot_id: str) -> threading.Lock:
    """Return the process-wide singleton rebuild lock for ``bot_id``."""
    with _REBUILD_LOCKS_GUARD:
        lock = _REBUILD_LOCKS.get(bot_id)
        if lock is None:
            lock = _REBUILD_LOCKS[bot_id] = threading.Lock()
        return lock


@contextmanager
def rebuild_lock(bot_id: str) -> Iterator[None]:
    """Serialize the drop+rebuild of a single bot's collection across all drivers.

    Both rebuild drivers (the ingest FIFO worker and ``JobManager``) reach the
    real rebuild through ``rebuild_bot``, which enters this context manager, so
    they contend for the SAME per-bot lock. Exposed so a caller can also assert /
    coordinate on the exact same lock if needed.
    """
    lock = _rebuild_lock_for(bot_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


# ---- stores & layout --------------------------------------------------
def corpus_store(cfg: Config) -> CorpusStore:
    return CorpusStore(cfg.data_dir)


def bot_store(cfg: Config) -> BotStore:
    return BotStore(cfg.data_dir)


def _corpus_root(cfg: Config, corpus_id: str) -> Path:
    return cfg.data_dir / "corpora" / corpus_id


def source_dir(cfg: Config, corpus_id: str, source_id: str) -> Path:
    return _corpus_root(cfg, corpus_id) / "sources" / source_id


def source_transcripts_dir(cfg: Config, corpus_id: str, source_id: str) -> Path:
    return source_dir(cfg, corpus_id, source_id) / "transcripts"


def source_documents_dir(cfg: Config, corpus_id: str, source_id: str) -> Path:
    return source_dir(cfg, corpus_id, source_id) / "documents"


def completed_video_count(cfg: Config, bot: Bot, source: BotSource) -> int:
    """How many of this source's videos already have a transcript on disk.

    The durable, ABSOLUTE count of completed downloads from prior passes. Used to
    seed a resumed job's ``videos_done`` so progress reflects real on-disk state
    and never visibly regresses on a reload/restart (BUG-005). Non-youtube sources
    return 0 (the seed only applies to the youtube download path).
    """
    if source.type != TYPE_YOUTUBE:
        return 0
    tdir = source_transcripts_dir(cfg, bot.corpus_id, source.id)
    if not tdir.exists():
        return 0
    return len(load_transcripts(tdir))


def _corpus_cfg(cfg: Config, corpus: Corpus) -> Config:
    return replace(
        cfg, embed_provider=corpus.embed_provider, embed_model=corpus.embed_model
    )


def _owner_capability_cfg(
    cfg: Config, base_cfg: Config, owner_id: str, capabilities: tuple[str, ...]
) -> Config:
    """Swap the bot OWNER's BYOK key(s) into ``base_cfg`` for each capability in
    ``capabilities`` (subset of ``embed`` / ``vision`` / ``transcription``).

    Isolated here so the ingest/rebuild embed path and the chat retrieval/vision
    path resolve the owner's key identically. The owner in Managed mode — or with
    no stored key for a capability's provider — leaves ``base_cfg`` unchanged (the
    server key, i.e. the documented graceful fallback). ``base_cfg`` is what the
    key is swapped into (e.g. a ``_corpus_cfg`` so the corpus-pinned embed provider
    governs); ``cfg`` supplies only the data dir for the owner/settings lookup.
    Never raises — any lookup failure degrades to ``base_cfg`` (server key).
    """
    try:
        from .accounts import UserStore
        from .plans import plan_for
        from .user_settings import UserSettingsStore, apply_byok_capability_key

        owner = UserStore(cfg.data_dir).get(owner_id) if owner_id else None
        if owner is None:
            return base_cfg
        plan = plan_for(owner)
        store = UserSettingsStore(cfg.data_dir)
        eff = base_cfg
        for cap in capabilities:
            eff = apply_byok_capability_key(eff, owner, plan, cap, store=store)
        return eff
    except Exception:  # noqa: BLE001 - a resolution hiccup must never break ingest/chat
        return base_cfg


def _owner_embed_pin(cfg: Config, owner_id: str) -> tuple[str, str]:
    """The ``(embed_provider, embed_model)`` a NEW corpus should be pinned to.

    A BYOK owner who selected an embed provider (and whose plan enables BYOK) pins
    new corpora to THAT provider's default embed model; everyone else gets the
    server default. This pins at CREATE time ONLY — a corpus keeps its embed provider
    for the life of its vectors, so changing an existing bot's embedding provider is
    deliberately NOT done here: it requires a NEW bot / full rebuild against a fresh
    corpus (otherwise old and new vectors would be mutually incompatible). Best-
    effort — any lookup hiccup falls back to the server default and never raises.
    """
    default = (cfg.embed_provider, cfg.embed_model)
    try:
        from .accounts import UserStore
        from .config import default_embed_model
        from .plans import plan_for
        from .user_settings import EMBED_PROVIDERS, MODE_BYOK, UserSettingsStore

        owner = UserStore(cfg.data_dir).get(owner_id) if owner_id else None
        if owner is None or not plan_for(owner).byok_enabled:
            return default
        us = UserSettingsStore(cfg.data_dir).get(owner.id)
        if us.mode != MODE_BYOK:
            return default
        chosen = (us.embed_provider or "").strip().lower()
        if chosen and chosen in EMBED_PROVIDERS:
            return chosen, default_embed_model(chosen)
        return default
    except Exception:  # noqa: BLE001 - corpus pinning must never break bot creation
        return default


# ---- bot lifecycle ----------------------------------------------------
def create_bot(
    cfg: Config, owner_id: str, name: str, persona: str, description: str,
    language: str = "",
) -> Bot:
    """Create a bot and its own isolated corpus (pinned to an embed provider/model).

    A BYOK owner's chosen embed provider pins the new corpus (see
    :func:`_owner_embed_pin`); otherwise the server default is used. The pin is
    permanent for this corpus — switching an existing bot's embed provider requires
    a new bot / full rebuild, never an in-place change.
    """
    cstore = corpus_store(cfg)
    embed_provider, embed_model = _owner_embed_pin(cfg, owner_id)
    corpus, _token = cstore.create(
        name=name or "bot",
        embed_provider=embed_provider,
        embed_model=embed_model,
    )
    return bot_store(cfg).create(
        owner_id=owner_id,
        name=name,
        persona=persona,
        description=description,
        corpus_id=corpus.id,
        language=language,
    )


def delete_bot(cfg: Config, bot: Bot) -> None:
    """Drop the bot's collection + all its files, then the registry records."""
    cstore = corpus_store(cfg)
    corpus = cstore.get(bot.corpus_id)
    if corpus is not None:
        try:
            VectorStore(cfg.chroma_dir, corpus.collection).drop()
        except Exception:  # noqa: BLE001 - collection may not exist
            pass
        cstore.delete(corpus.id)
    root = _corpus_root(cfg, bot.corpus_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    bot_store(cfg).delete(bot.id)


def sweep_corpus(cfg: Config, corpus_id: str) -> None:
    """Best-effort re-drop of a corpus's Chroma collection + on-disk dir.

    The account-erasure cascade calls this AFTER the registry records are gone:
    a worker that was mid-rebuild may have re-created a just-dropped collection
    between our cancel and the bot delete, so we drop it once more so no
    registry-invisible orphan survives erasure. Idempotent — safe to call when
    nothing is left."""
    try:
        VectorStore(cfg.chroma_dir, collection_name(corpus_id)).drop()
    except Exception:  # noqa: BLE001 - "already gone" is the normal case
        pass
    root = _corpus_root(cfg, corpus_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---- source registration ---------------------------------------------
def add_youtube_source(
    cfg: Config, bot: Bot, key: str, label: str, author: str, kind: str = "channel"
) -> BotSource:
    return bot_store(cfg).add_source(bot.id, TYPE_YOUTUBE, key, label, author, kind=kind)


def add_document_source(
    cfg: Config, bot: Bot, key: str, filename: str, author: str, data: bytes
) -> BotSource:
    src = bot_store(cfg).add_source(
        bot.id, TYPE_DOCUMENT, key, filename, author, kind="document"
    )
    ddir = source_documents_dir(cfg, bot.corpus_id, src.id)
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / filename).write_bytes(data)
    return src


def remove_source(cfg: Config, bot: Bot, source_id: str) -> BotSource | None:
    removed = bot_store(cfg).remove_source(bot.id, source_id)
    if removed is not None:
        shutil.rmtree(source_dir(cfg, bot.corpus_id, source_id), ignore_errors=True)
    return removed


# ---- ingestion (both server yt-dlp and the client-text seam) ----------
def ingest_channel(
    cfg: Config,
    bot: Bot,
    source: BotSource,
    langs=None,
    limit: int | None = None,
    throttle: ThrottleConfig | None = None,
    max_retries: int | None = None,
    cookies_browser: str | None = None,
    progress: ProgressFn | None = None,
    job_id: str | None = None,
    budget_probe=None,
) -> int:
    """Server-side path: scrape a channel with yt-dlp, then funnel the results
    through the SAME entrypoint the HTTP endpoint uses.

    yt-dlp is just an internal client of ``ingest_text_items`` here — it scrapes
    into a staging dir, the scraped docs are read back as transcript items, and
    ``ingest_text_items`` writes the canonical, author-stamped files. One code
    path, one schema: identical to what the browser-extension endpoint produces.
    Returns the number of transcripts written."""
    scrape_dir = source_dir(cfg, bot.corpus_id, source.id) / "scrape"
    download_transcripts(
        source.key,
        scrape_dir,
        langs=langs,
        limit=limit,
        throttle=throttle,
        max_retries=max_retries,
        cookies_browser=cookies_browser,
        progress=progress,
        proxy=cfg.transcript_proxy or None,
        job_id=job_id,
        budget_probe=budget_probe,
        # BUG-009: a bot excludes YouTube Shorts by default; the per-bot toggle
        # re-includes them. ``getattr`` keeps pre-BUG-009 Bot records working.
        include_shorts=getattr(bot, "include_shorts", False),
    )
    items = load_transcripts(scrape_dir)
    return ingest_text_items(cfg, bot, source, items)


def ingest_text_items(
    cfg: Config, bot: Bot, source: BotSource, items: list[dict]
) -> int:
    """Client seam: accept already-extracted transcript items and store them.

    This is what the future browser extension calls (over HTTP). Returns count
    written."""
    tdir = source_transcripts_dir(cfg, bot.corpus_id, source.id)
    return write_items(tdir, items, source.author, source.label)


# ---- rebuild ----------------------------------------------------------
def _load_source_docs(cfg: Config, bot: Bot, source: BotSource) -> list[dict]:
    if source.type == TYPE_YOUTUBE:
        tdir = source_transcripts_dir(cfg, bot.corpus_id, source.id)
        return load_transcripts(tdir) if tdir.exists() else []
    if source.type == TYPE_DOCUMENT:
        ddir = source_documents_dir(cfg, bot.corpus_id, source.id)
        if not ddir.exists():
            return []
        return [
            document_to_doc(source.id, f, source.label)
            for f in sorted(ddir.iterdir())
            if f.is_file()
        ]
    return []


def rebuild_bot(
    cfg: Config,
    bot: Bot,
    progress: ProgressFn | None = None,
    should_abort: Callable[[], bool] | None = None,
    *,
    full: bool = False,
) -> int:
    """(Re)build this bot's collection from all its sources. Returns chunk count.

    ``full`` selects the mode (BUG-021):

    * ``full=False`` (default, the routine path) — INCREMENTAL: keep the existing
      collection and embed + append ONLY the chunks of videos not already indexed.
      Auto-sync, "index more" and a channel re-import all take this path, so
      unchanged content is never re-embedded (no wasted work, and no re-charge once
      embedding is metered). On-disk dedup means a re-import only ever downloads new
      videos, so incremental embedding is the natural match.
    * ``full=True`` — drop the whole collection and re-embed every source. Used only
      where the collection must be reconciled to the registry as a whole: after a
      source is removed (its vectors must disappear) or an explicit "Rebuild".

    ``should_abort`` is an optional cheap probe (e.g. "is this ingest job
    cancel_requested?"). Together with a re-fetch of the bot/corpus from their
    stores it lets a rebuild bail cleanly if the account/bot was erased — or the
    job cancelled — mid-flight, instead of re-creating a registry-invisible
    Chroma collection that would survive erasure. Returns 0 on such an abort.

    The whole build runs under the shared per-bot ``rebuild_lock`` so no two
    rebuilds of the same collection ever overlap, regardless of which driver
    (ingest worker or ``JobManager``) initiated them."""
    with rebuild_lock(bot.id):
        return _rebuild_bot_locked(cfg, bot, progress, should_abort, full=full)


def _rebuild_bot_locked(
    cfg: Config,
    bot: Bot,
    progress: ProgressFn | None,
    should_abort: Callable[[], bool] | None,
    *,
    full: bool,
) -> int:
    """The actual build. MUST be called while holding ``rebuild_lock``."""
    def emit(**kw):
        if progress:
            progress(kw)

    cstore = corpus_store(cfg)
    bstore = bot_store(cfg)

    def _corpus_erased() -> bool:
        # Bot or corpus record deleted — an account/bot erasure raced this rebuild.
        return bstore.get(bot.id) is None or cstore.get(bot.corpus_id) is None

    def _aborted() -> bool:
        # A deleted bot/corpus, or an explicit cancel — either way, stop.
        if _corpus_erased():
            return True
        if should_abort is not None:
            try:
                return bool(should_abort())
            except Exception:  # noqa: BLE001 - a bad probe must not wedge rebuild
                return False
        return False

    def _drop_quietly(vs: VectorStore) -> None:
        try:
            vs.drop()
        except Exception:  # noqa: BLE001 - collection may already be gone
            pass

    def _abort_cleanup(store: VectorStore) -> None:
        # Drop the collection our handle points at ONLY when leaving it would orphan
        # it: the corpus record is gone (erasure), or this was a FULL rebuild that
        # already dropped the prior vectors and holds only a partial. An INCREMENTAL
        # rebuild that was merely cancelled keeps its pre-existing vectors — dropping
        # them would wipe a working bot (the whole point of BUG-021).
        if full or _corpus_erased():
            _drop_quietly(store)

    # Guard #1: bail before touching Chroma if the bot/corpus was already erased
    # (or a cancel was requested) — no collection has been (re)created yet, so
    # there is nothing to clean up. This runs BEFORE the corpus-None check so a
    # deleted corpus aborts cleanly instead of raising.
    if _aborted():
        return 0

    corpus = cstore.get(bot.corpus_id)
    if corpus is None:
        raise RuntimeError(f"Bot {bot.id} has no corpus")

    cstore.update(corpus.id, status=STATUS_BUILDING)
    store = VectorStore(cfg.chroma_dir, corpus.collection)
    if full:
        # FULL: drop so the rebuilt collection mirrors the registry exactly.
        try:
            store.drop()
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        store = VectorStore(cfg.chroma_dir, corpus.collection)
        present_counts: dict[str, int] = {}
    else:
        # INCREMENTAL: keep the collection and skip only FULLY-indexed videos.
        present_counts = store.video_chunk_counts()
    # BYOK: embed with the OWNER's own key for the corpus's PINNED provider when
    # they're in BYOK mode and have one stored; else the server key (fallback).
    # Only the key is swapped — provider/model stay corpus-pinned so vectors match.
    embed_cfg = _owner_capability_cfg(cfg, _corpus_cfg(cfg, corpus), bot.owner_id, ("embed",))
    embedder = make_embedder(embed_cfg)

    # Re-read the bot so we see freshly-registered sources.
    bot = bstore.get(bot.id) or bot
    for source in bot.sources:
        bstore.update_source(bot.id, source.id, status=SRC_BUILDING)
        try:
            docs = _load_source_docs(cfg, bot, source)
            # Chunking is free CPU, so we chunk every video and group by video id:
            # the group size IS the video's EXPECTED chunk count, and per-video
            # embedding bounds an interrupted upsert to a single video (F1).
            by_video: dict[str, list] = {}
            chunk_total = 0
            for d in docs:
                for c in chunk_transcript(d):
                    c.author = source.author
                    c.source = source.label
                    by_video.setdefault(c.video_id, []).append(c)
                    chunk_total += 1
            added = 0
            for vid, vchunks in by_video.items():
                # Skip only a FULLY-indexed video (present == expected). Any mismatch
                # re-embeds: present < expected heals a crash-partial video; present
                # > expected reindexes a SHORTENED transcript. FULL mode starts from
                # empty counts, so every video embeds.
                present = present_counts.get(vid, 0)
                if present == len(vchunks):
                    continue
                embeddings = embedder.embed_documents([c.text for c in vchunks])
                # Guard #2: re-check IMMEDIATELY before the add — the bot/corpus may
                # have been erased while we were embedding. If so, clean up the
                # collection our VectorStore handle may have re-created and abort.
                if _aborted():
                    _abort_cleanup(store)
                    return 0
                # Scoped per-video reindex: drop THIS video's old chunks first so a
                # shortened transcript leaves no orphan ``{vid}::i`` tail past the new
                # count, then add the fresh full set (delete+add back-to-back, never
                # a whole-collection drop, never touching another video).
                if present:
                    store.delete_video(vid)
                store.add(vchunks, embeddings)
                added += len(vchunks)
            bstore.update_source(
                bot.id, source.id, status=SRC_READY,
                item_count=len(docs), chunk_count=chunk_total,
            )
            emit(stage="indexed_source", source=source.label, author=source.author,
                 chunks=added,
                 message=f"Indexed {added} new chunk(s) from "
                         f"{source.label} ({source.author})")
        except Exception as exc:  # noqa: BLE001
            bstore.update_source(bot.id, source.id, status=SRC_ERROR)
            emit(stage="source_error", source=source.label,
                 message=f"Error indexing {source.label}: {exc}")

    # Guard #3: a deletion during the loop (e.g. a zero-chunk rebuild, or right
    # after the last add) would otherwise leave a collection our handle re-created
    # as an orphan. Clean up and abort before finalizing.
    if _aborted():
        _abort_cleanup(store)
        return 0

    count = store.count()
    cstore.update(corpus.id, status=STATUS_READY, chunk_count=count)
    return count


# ---- persona precedence (Phase J) -------------------------------------
def effective_persona(bot: Bot) -> str:
    """The persona body folded into the system prompt for THIS bot.

    Phase J precedence: a non-empty ``custom_prompt`` OVERRIDES the builder
    ``persona``; an empty custom prompt falls back to the builder persona. This is
    the SINGLE place precedence is decided so every chat surface (web, guest,
    Telegram) behaves identically — they all reach the model through ``chat`` /
    ``chat_with_image`` below.

    Grounding is deliberately NOT handled here: ``rag.build_system_prompt`` always
    layers ``GROUNDING_RULES`` on top of whatever persona it is given, so a custom
    prompt can shape voice/identity but can never switch off citation grounding.
    """
    custom = (getattr(bot, "custom_prompt", "") or "").strip()
    return custom or bot.persona


# ---- chat -------------------------------------------------------------
# BUG-007: the friendly message every surface (web, guest, Telegram) shows when a
# user tries to chat before ANY source has been indexed — a calm "still indexing"
# note, never an error or a hallucinated answer over an empty corpus.
CHAT_NOT_READY_MESSAGE = (
    "I'm still indexing this channel — no videos are ready to answer from yet. "
    "Please check back in a few minutes and try again."
)


def chat_ready(cfg: Config, bot: Bot, answer_mode: str | None = None) -> bool:
    """True once this bot may answer in ``answer_mode`` (BUG-007 gate).

    Retrieval-backed modes («Довідник», no-mode) must be blocked until the bot has
    real, retrievable content — otherwise the model answers from an empty corpus and
    hallucinates. The corpus's durable ``chunk_count`` is the single source of truth:
    it is stamped by every (re)build and survives a restart, so the gate is consistent
    across a reload. A missing corpus is treated as not-ready.

    «Мислення» v2 (``answer_mode == "thinking"``) is the deliberate exception (H1): it
    is a general advisor that reasons from world knowledge + the dialogue with NO
    retrieval, so a corpus-less / still-indexing bot may use it — the gate is OPEN for
    it regardless of corpus. This is the single mode-aware seam every public chat entry
    point (chat / chat_stream / achat / achat_stream) shares, sync ≡ async; the default
    ``answer_mode=None`` keeps every other caller (Telegram, image, retrieval modes)
    byte-for-byte unchanged."""
    if answer_mode == "thinking":
        return True
    corpus = corpus_store(cfg).get(bot.corpus_id)
    return bool(corpus is not None and getattr(corpus, "chunk_count", 0) >= 1)


def _not_ready_result() -> dict:
    """The gate's response dict: the friendly "still indexing" note, no model call.

    Shaped like an :class:`Assistant.answer` result (``answer``/``sources``) plus
    an ``indexing`` flag the web surfaces propagate so the 1.5 UI can render an
    indexing state. No ``usage`` — nothing was spent, and the caller returns before
    any accounting."""
    return {
        "answer": CHAT_NOT_READY_MESSAGE,
        "sources": [],
        "grounded": False,
        "indexing": True,
    }


def _chat_assistant(cfg: Config, bot: Bot, answer_mode: str | None = None) -> Assistant:
    """Build the grounded Assistant for this bot (shared by chat + chat_stream).

    Retrieval embeds the query — use the OWNER's BYOK embed key (corpus-pinned
    provider) when set, else the server key. The chat LLM key was already resolved
    into ``cfg`` upstream (web/telegram _resolve_chat). The no-corpus precondition is
    enforced ONLY for retrieval-backed modes: «Мислення» (answer_mode=="thinking")
    reasons from world knowledge + the dialogue with NO retrieval, so a corpus-less
    bot can still use it. «Довідник» / no-mode raise, same caller-visible
    misconfiguration as before."""
    corpus = corpus_store(cfg).get(bot.corpus_id)
    if corpus is None:
        if answer_mode != "thinking":
            raise RuntimeError(f"Bot {bot.id} has no corpus")
        # «Мислення» v2 never retrieves — build a corpus-less Assistant (no vector
        # collection, no fragment provider, server cfg since there is no BYOK pin).
        return Assistant(
            cfg, persona=effective_persona(bot),
            collection=None, language=bot.language,
            suggest_followups=getattr(bot, "suggest_followups", True),
            fragment_provider=None,
            answer_mode=answer_mode,
        )
    eff = _owner_capability_cfg(cfg, _corpus_cfg(cfg, corpus), bot.owner_id, ("embed",))
    return Assistant(
        eff, persona=effective_persona(bot),
        collection=corpus.collection, language=bot.language,
        suggest_followups=getattr(bot, "suggest_followups", True),
        fragment_provider=_fragment_provider_for(cfg, bot),
        answer_mode=answer_mode,
    )


def _chat_assistant_for_mode(
    cfg: Config, bot: Bot, answer_mode: str | None,
) -> Assistant:
    """Keep the long-standing two-argument seam usable by test/custom patches."""
    if answer_mode is None:
        return _chat_assistant(cfg, bot)
    return _chat_assistant(cfg, bot, answer_mode)


def _fragment_provider_for(cfg: Config, bot: Bot) -> "Callable[[dict], list[dict]]":
    """Build a per-chat provider mapping a retrieval hit → its ordered caption
    fragments (BUG-028), by reconstructing the hit's chunk from the on-disk
    transcript (no re-index needed). Caches (doc, chunks) per video for the request.

    Returns ``[]`` for a document hit (no url/timecode) or when the transcript json
    isn't on disk — the answer path then keeps the chunk-head timecode. All failures
    are swallowed: citations are best-effort.

    Reads the CANONICAL indexed transcripts (``sources/*/transcripts/<vid>.json``,
    written by ``write_items``) — the exact docs ``rebuild_bot`` chunks — so a
    reconstructed ``chunk_transcript(doc)[index]`` aligns with the stored chunk's
    ``meta.index``/timecodes, and it works for the extension seam too (which never
    writes a yt-dlp ``scrape/`` dir)."""
    root = _corpus_root(cfg, bot.corpus_id)
    cache: dict[str, tuple[dict, list] | None] = {}

    def _load(vid: str):
        for path in root.glob(f"sources/*/transcripts/{vid}.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                return doc, chunk_transcript(doc)
            except Exception:  # noqa: BLE001 - a bad/absent transcript => no fragments
                return None
        return None

    def provider(hit: dict) -> list[dict]:
        meta = (hit or {}).get("meta") or {}
        vid = meta.get("video_id")
        if not vid or not (meta.get("url") or "").strip():
            return []  # no video / a document source has no per-fragment timecode
        try:
            idx = int(meta.get("index"))
        except (TypeError, ValueError):
            return []
        if vid not in cache:
            cache[vid] = _load(vid)
        loaded = cache[vid]
        if not loaded:
            return []
        doc, chunks = loaded
        if not (0 <= idx < len(chunks)):
            return []
        ch = chunks[idx]
        # Alignment guard: the reconstructed chunk MUST match the one that was
        # indexed (same start/end as stored in ``meta``). This protects against two
        # things the video_id lookup alone can't: a DIFFERENT source's copy of the
        # same video winning the glob, and a transcript that CHANGED since indexing
        # (incremental rebuild can skip a re-embed when the chunk count is unchanged).
        # On any mismatch we return no fragments → the answer keeps the chunk-head
        # timecode (safe fallback), never a wrong deep-link.
        ms, me = meta.get("start"), meta.get("end")
        if ms is not None and abs(float(ch.start) - float(ms)) > 0.01:
            return []
        if me is not None and abs(float(ch.end) - float(me)) > 0.01:
            return []
        return fragments_for_chunk(doc, ch)

    return provider


def _record_chat_usage(cfg: Config, bot: Bot, model: str | None, usage) -> None:
    """Passive owner-billed accounting for a chat answer. Never breaks a chat."""
    try:
        UsageStore(cfg.data_dir).record(bot.owner_id, model or cfg.llm_model, usage)
    except Exception:  # noqa: BLE001 - accounting must never surface to the user
        pass


def _record_chat_metric(cfg: Config, bot: Bot, kind: str | None, usage: Usage) -> None:
    """Passive per-bot beta metric for ONE completed chat turn. Never breaks a chat.

    Counts the turn under its route (``question`` / ``smalltalk`` / ``decline``) with
    the turn's total tokens, alongside owner billing at the single chat choke point. A
    turn with no derivable ``kind`` (e.g. the "still indexing" gate, which is not a
    turn) records nothing; unlike billing it does NOT skip a zero-token turn — a turn
    happened, so it counts regardless of tokens spent."""
    if not kind:
        return
    try:
        MetricsStore(cfg.data_dir).record_turn(
            bot.id, kind=kind, tokens=(usage or Usage()).total
        )
    except Exception:  # noqa: BLE001 - metrics must never surface to the user
        pass


class _BillOnce:
    """One-shot owner billing for a STREAMED answer (Inc1.4b N1).

    A stream ends one of two ways and BOTH must bill the produced tokens, but never
    together:

    * normally — the terminal ``done`` event carries the authoritative usage; or
    * via ``GeneratorExit`` — the SSE client disconnected mid-answer, so ``done``
      never arrives and only the ``finally`` fallback runs.

    Without this guard a fully-generated-then-torn-down stream could bill twice (the
    done-path AND the finally), and a client that always disconnects mid-answer
    could evade billing entirely. :meth:`record` bills AT MOST once and ignores a
    zero-token call (nothing was produced → nothing to charge), so the done-path and
    the interruption ``finally`` are safely idempotent."""

    def __init__(self) -> None:
        self._done = False

    def record(self, cfg: Config, bot: Bot, model: str | None, usage: Usage) -> None:
        if self._done:
            return
        usage = usage or Usage()
        if usage.total <= 0:
            return
        self._done = True
        _record_chat_usage(cfg, bot, model, usage)


class _MetricOnce:
    """One-shot per-turn beta-metric recording for a STREAMED answer.

    Mirrors :class:`_BillOnce` so metrics and billing share the same exactly-once
    discipline: the terminal ``done`` event carries the turn's ``kind`` and
    authoritative usage, and the turn is recorded AT MOST once. Unlike billing there
    is no ``finally`` fallback — the ``kind`` is only known once ``done`` arrives, so a
    stream torn down before it (SSE client disconnect mid-answer) records no metric: a
    turn is only *counted* once its answer completed. The guard therefore makes the
    done path safe to call idempotently and the metric can never double."""

    def __init__(self) -> None:
        self._done = False

    def record(self, cfg: Config, bot: Bot, kind: str | None, usage: Usage) -> None:
        if self._done or not kind:
            return
        self._done = True
        _record_chat_metric(cfg, bot, kind, usage)


def chat(
    cfg: Config, bot: Bot, message: str, history: list[dict] | None = None,
    top_k: int | None = None, answer_mode: str | None = None,
) -> dict:
    """Answer grounded ONLY in this bot's corpus, in this bot's persona.

    BUG-007: this is the single choke point every chat surface (web, guest,
    Telegram text/audio) funnels through, so the "not indexed yet" gate lives
    HERE — before any retrieval or model call — and every surface is covered at
    once. Until at least one chunk is indexed, it returns the friendly note —
    EXCEPT «Мислення» v2 (answer_mode=="thinking"), the corpus-less general advisor
    the gate lets through (H1)."""
    if not chat_ready(cfg, bot, answer_mode):
        return _not_ready_result()
    assistant = _chat_assistant_for_mode(cfg, bot, answer_mode)
    result = assistant.answer(message, history=history, top_k=top_k)
    _record_chat_usage(cfg, bot, result.get("model"), result["usage"])
    _record_chat_metric(cfg, bot, result.get("kind"), result["usage"])
    return result


def chat_stream(
    cfg: Config, bot: Bot, message: str, history: list[dict] | None = None,
    top_k: int | None = None, answer_mode: str | None = None,
) -> Iterator[dict]:
    """Streaming twin of :func:`chat` (Inc1.4b) — yields answer event dicts.

    Same single choke point and gate as :func:`chat`: when the bot is not yet
    indexed it yields the friendly note as one ``delta`` plus a terminal ``done``
    (carrying ``indexing``), never touching retrieval or the model. Otherwise it
    proxies :meth:`Assistant.answer_stream`.

    Billing (N1): the owner is charged EXACTLY once for the tokens produced, even if
    the SSE client disconnects mid-answer. A live ``accrued`` usage is kept mirrored
    by the streaming layer; the normal path bills the authoritative ``done`` usage,
    and a ``finally`` bills whatever ``accrued`` if a ``GeneratorExit`` tore the
    stream down before ``done`` arrived. A shared :class:`_BillOnce` makes the two
    paths mutually exclusive, so a partial answer is neither double-charged nor free.
    Event shape: ``{"type": "delta", "text": ...}`` … then ``{"type": "done", ...}``."""
    if not chat_ready(cfg, bot, answer_mode):
        note = _not_ready_result()
        yield {"type": "delta", "text": note["answer"]}
        yield {"type": "done", "sources": [], "grounded": False, "indexing": True}
        return
    assistant = _chat_assistant_for_mode(cfg, bot, answer_mode)
    accrued = Usage()
    once = _BillOnce()
    metric_once = _MetricOnce()
    model: str | None = None
    try:
        # INVARIANT: answer_stream(...) is passed INLINE into the for-loop on purpose.
        # On a client disconnect, GeneratorExit unwinds this frame; CPython finalizes
        # the (unreferenced) inner generator — running its own finally, which folds the
        # mid-stream OUTPUT-token estimate into ``accrued`` (see rag._stream_deltas) —
        # BEFORE this frame's ``finally`` bills ``accrued``. Binding the generator to a
        # local first would reverse that order and silently bill router tokens only,
        # dropping the produced-answer estimate. Keep it inline.
        for event in assistant.answer_stream(
            message, history=history, top_k=top_k, usage_sink=accrued,
        ):
            if event.get("type") == "done":
                model = event.get("model")
                once.record(cfg, bot, model, event.get("usage") or accrued)
                # Metric on the SAME authoritative done event as billing, once — the
                # done event is the only frame carrying the turn's ``kind``. A stream
                # abandoned before ``done`` bills its tokens in the finally below but
                # records no metric (no kind → an incomplete turn isn't counted).
                metric_once.record(cfg, bot, event.get("kind"), event.get("usage") or accrued)
            yield event
    finally:
        # Client disconnect (GeneratorExit) or any exit that skipped ``done``: bill
        # the tokens produced so far, once. Zero tokens produced → nothing recorded.
        once.record(cfg, bot, model, accrued)


async def achat(
    cfg: Config, bot: Bot, message: str, history: list[dict] | None = None,
    top_k: int | None = None, answer_mode: str | None = None,
) -> dict:
    """Async twin of :func:`chat` (PLAN §5b) for the web request path.

    Same single choke point, same BUG-007 gate ordering (BEFORE any retrieval or
    model call) and the same passive accounting — but the model/embedding calls are
    awaited on the event loop, so a ~7s answer no longer serializes every other
    request behind it. The gate probe and Assistant construction read local stores
    (small-file + Chroma client init), so they run on a worker thread too. The
    usage/metric writes are tiny local file appends, kept inline like the sync path."""
    if not await asyncio.to_thread(chat_ready, cfg, bot, answer_mode):
        return _not_ready_result()
    assistant = await asyncio.to_thread(_chat_assistant_for_mode, cfg, bot, answer_mode)
    result = await assistant.aanswer(message, history=history, top_k=top_k)
    _record_chat_usage(cfg, bot, result.get("model"), result["usage"])
    _record_chat_metric(cfg, bot, result.get("kind"), result["usage"])
    return result


async def achat_stream(
    cfg: Config, bot: Bot, message: str, history: list[dict] | None = None,
    top_k: int | None = None, answer_mode: str | None = None,
) -> AsyncIterator[dict]:
    """Async twin of :func:`chat_stream` — same gate, events and exactly-once billing.

    The sync path's disconnect billing rests on a CPython finalization-order
    INVARIANT (the inline-generator comment in :func:`chat_stream`). Here the
    ordering is EXPLICIT instead: the ``finally`` first acloses the inner answer
    generator — which runs rag's disconnect-estimate fold into ``accrued`` — and
    only then bills ``accrued``. Never rely on GC order for async generators."""
    if not await asyncio.to_thread(chat_ready, cfg, bot, answer_mode):
        note = _not_ready_result()
        yield {"type": "delta", "text": note["answer"]}
        yield {"type": "done", "sources": [], "grounded": False, "indexing": True}
        return
    assistant = await asyncio.to_thread(_chat_assistant_for_mode, cfg, bot, answer_mode)
    accrued = Usage()
    once = _BillOnce()
    metric_once = _MetricOnce()
    model: str | None = None
    agen = assistant.aanswer_stream(
        message, history=history, top_k=top_k, usage_sink=accrued,
    )
    try:
        async for event in agen:
            if event.get("type") == "done":
                model = event.get("model")
                once.record(cfg, bot, model, event.get("usage") or accrued)
                # Metric on the SAME authoritative done event as billing (see
                # chat_stream): a stream abandoned before ``done`` records no metric.
                metric_once.record(cfg, bot, event.get("kind"), event.get("usage") or accrued)
            yield event
    finally:
        # Client disconnect (GeneratorExit/CancelledError) or any exit that skipped
        # ``done``: aclose the inner generator FIRST so the produced-so-far estimate
        # is folded into ``accrued``, then bill it once. Zero tokens → no record.
        # The billing sits in a nested ``finally`` so it runs EVEN IF ``aclose`` raises
        # (an httpx/provider close error during cancellation) — otherwise a raising
        # aclose would skip billing entirely and lose the disconnected turn's tokens.
        # (Sync ``chat_stream`` gets this for free: it folds during GC finalization
        # where close-exceptions are swallowed; awaiting aclose here makes it explicit.)
        try:
            await agen.aclose()
        finally:
            once.record(cfg, bot, model, accrued)


def _record_image_usage(cfg: Config, bot: Bot, result: dict) -> None:
    """Passive owner-billed accounting for a vision answer. Never breaks a chat.

    Bills against the model the turn actually ran on (``result["model"]`` —
    the vision model on both the grounded and «Мислення» v2 vision paths), falling
    back to the server vision model."""
    try:
        UsageStore(cfg.data_dir).record(
            bot.owner_id, result.get("model") or cfg.vision_model, result["usage"]
        )
    except Exception:  # noqa: BLE001 - accounting must never surface to the user
        pass


def effective_vision_key(cfg: Config, bot: Bot) -> str:
    """The vision API key that a photo turn for ``bot`` would actually run on.

    Resolves the OWNER's BYOK vision credential first (falling back to the server
    vision key), using the SAME ``_owner_capability_cfg(..., ("vision",))`` seam that
    both the «Мислення» v2 and grounded/«Довідник» photo paths use to swap the key.
    The vision-key portion of that resolution is identical regardless of the base cfg
    (only the embed provider is corpus-pinned), so a single check governs both
    branches. Callers gate the "no vision" early-return on this being empty — true
    ONLY when NEITHER the owner's BYOK vision key NOR the server vision key exists.
    Never raises (``_owner_capability_cfg`` degrades to the server key)."""
    return _owner_capability_cfg(cfg, cfg, bot.owner_id, ("vision",)).vision_key()


def _chat_with_image_thinking(
    cfg: Config, bot: Bot, message: str, image_bytes: bytes, image_mime: str,
    history: list[dict] | None,
) -> dict:
    """«Мислення» v2 vision turn: the model reasons over the image + the full
    dialogue from world knowledge — NO retrieval, NO corpus needed (H1), NO
    citations. Only the VISION capability's OWNER BYOK key is swapped (there is no
    retrieval, so no embed key is used); a corpus-less bot is fully supported, so
    the Assistant is built with ``collection=None`` and no fragment provider, exactly
    like the text v2 base."""
    eff = _owner_capability_cfg(cfg, cfg, bot.owner_id, ("vision",))
    assistant = Assistant(
        eff, persona=effective_persona(bot),
        collection=None, language=bot.language,
        suggest_followups=getattr(bot, "suggest_followups", True),
        fragment_provider=None,
        answer_mode="thinking",
    )
    result = assistant.answer_with_image_thinking(
        message, image_bytes, image_mime, history=history,
    )
    _record_image_usage(cfg, bot, result)
    return result


def chat_with_image(
    cfg: Config, bot: Bot, message: str, image_bytes: bytes, image_mime: str,
    top_k: int | None = None, history: list[dict] | None = None,
    answer_mode: str | None = None,
) -> dict:
    """Answer a photo message in this bot's persona, honoring the per-chat mode.

    «Мислення» (``answer_mode == "thinking"``): a VISION + v2 BASE turn — the model
    sees the image + the full ``history`` and reasons from world knowledge with NO
    retrieval and NO citations/Sources, so it works even on a corpus-less / still-
    indexing bot (the same H1 exception the text/gate seams share).

    «Довідник» / no-mode: the grounded vision path — retrieval over this bot's corpus
    plus the vision model, byte-for-byte as before (``answer_mode`` only special-cases
    ``"thinking"`` in :func:`chat_ready`, so both keep the BUG-007 gate and the
    grounded Assistant)."""
    if not chat_ready(cfg, bot, answer_mode):  # BUG-007 gate; thinking bypasses (H1)
        return _not_ready_result()
    if answer_mode == "thinking":
        return _chat_with_image_thinking(
            cfg, bot, message, image_bytes, image_mime, history
        )
    corpus = corpus_store(cfg).get(bot.corpus_id)
    if corpus is None:
        raise RuntimeError(f"Bot {bot.id} has no corpus")
    # The grounded image path uses BOTH embeddings (retrieval) and the VISION model —
    # swap the OWNER's BYOK key for each (corpus-pinned embed provider; server vision
    # provider) when set, else the server key (fallback).
    eff = _owner_capability_cfg(
        cfg, _corpus_cfg(cfg, corpus), bot.owner_id, ("embed", "vision")
    )
    assistant = Assistant(
        eff, persona=effective_persona(bot),
        collection=corpus.collection, language=bot.language,
        suggest_followups=getattr(bot, "suggest_followups", True),
    )
    result = assistant.answer_with_image(
        message, image_bytes, image_mime, top_k=top_k
    )
    _record_image_usage(cfg, bot, result)
    return result
