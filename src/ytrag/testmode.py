"""Offline, deterministic test-mode fakes for the three non-deterministic edges.

This module is imported LAZILY and ONLY when :func:`ytrag.config.test_mode` is
True (which is itself fail-closed: forced off in production, default off). Nothing
here is loaded or instantiated on a production path. The fakes are swapped in at
the FACTORY boundaries — ``llm.make_llm`` / ``bot._builder_llm`` (Seam A),
``embed.make_embedder`` (Seam B), and ``ingest.list_video_ids`` /
``ingest.download_transcripts`` (Seam C) — so no ``if test_mode`` branching leaks
into business logic. Everything runs with ZERO network and ZERO API keys:

* :class:`FakeLLM` returns canned-but-context-derived text so the persona
  interview ({done,question,persona}) and grounded chat (an answer that quotes the
  retrieved excerpt and emits a citation) contracts are exercised deterministically.
* :class:`FakeEmbedder` maps text to a hashed n-gram, L2-normalized fixed-dim
  vector so Chroma add/query works offline with stable nearest-neighbours.
* The YouTube source returns canned video ids per kind and reads tiny in-repo
  transcript fixtures under ``e2e/fixtures/transcripts/`` — the import then flows
  through the REAL queue -> chunk -> embed(FakeEmbedder) -> Chroma pipeline, so
  job lifecycle/counts/statuses are all exercised for real, just with fake inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Iterator, Sequence

from .config import DEFAULT_MAX_OUTPUT_TOKENS, test_mode
from .llm import Completion, JsonCompletion, ToolCallRequest, ToolCompletion
from .usage import Usage

# In-repo fixtures: <repo>/e2e/fixtures/transcripts/<video_id>.json. testmode.py
# lives at src/ytrag/testmode.py, so parents[2] is the repo root.
_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "e2e" / "fixtures" / "transcripts"

# Canned video ids per source KIND. A channel/playlist enumerates a few fixture
# videos; a single-video source resolves to exactly one. These map to the fixture
# filenames under _FIXTURES_DIR.
_CHANNEL_VIDEO_IDS = ["tm_open_3sec", "tm_confidence", "tm_texting"]
_PLAYLIST_VIDEO_IDS = ["tm_open_3sec", "tm_confidence"]
_SINGLE_VIDEO_ID = "tm_open_3sec"

# A stable channel display name for the ingest "channel" progress event.
_CHANNEL_TITLE = "Test Mode Coaching"

_fixture_cache: dict[str, dict] = {}


# ---- E2E fault-injection sentinels (TEST_MODE ONLY) ------------------------
# The offline fakes ALWAYS succeed, which hides three real UI states from specs:
# a FAILED ingest job, a CANCELLED ingest job, and a chat ERROR. These sentinels
# are deterministic triggers to reach those states. Each is FAIL-CLOSED: every
# branch that acts on a sentinel is gated on ``config.test_mode()`` (which
# production forces OFF), so a channel/handle/url/question that merely contains
# one of these tokens has ZERO special effect outside test mode. The tokens are
# mutually non-overlapping, so match order does not matter. Contract (documented
# here for specs + future devs):
#   * a YouTube source whose key/handle/url contains INGEST_FAIL_SENTINEL ->
#     ingest raises on EVERY attempt, so the job ends 'error' (retry stays error).
#   * ...contains INGEST_FAIL_ONCE_SENTINEL -> ingest raises the FIRST time that
#     key is seen this process, then SUCCEEDS on retry (retry -> success/done is
#     testable). Per-process state; the e2e harness uses a fresh app per run.
#   * ...contains INGEST_HANG_SENTINEL -> ingest marks the job 'running' then
#     holds in a COOPERATIVELY CANCELLABLE wait: each tick calls the progress
#     callback, whose queue-side probe raises IngestCancelled on cancel, so a
#     spec can cancel a running job and observe 'cancelled' with no sleep race.
#     Bounded, so a run that never cancels still terminates (normal import).
#   * a chat QUESTION containing CHAT_ERROR_SENTINEL -> the fake chat LLM raises,
#     so the web chat returns its error state. Ask it against an indexed bot (or
#     append it to a groundable phrase) so retrieval reaches the LLM.
INGEST_FAIL_SENTINEL = "e2e-fail"
INGEST_FAIL_ONCE_SENTINEL = "e2e-retry"
INGEST_HANG_SENTINEL = "e2e-hang"
CHAT_ERROR_SENTINEL = "__E2E_CHAT_ERROR__"
# A chat QUESTION containing AGENT_LOOP_SENTINEL makes the fake tool-calling LLM
# request ANOTHER search on every round instead of finishing — the deterministic
# way to reach the agentic loop's round CAP (and multi-round billing) offline.
# FAIL-CLOSED like the others: the branch is gated on test_mode().
AGENT_LOOP_SENTINEL = "__E2E_AGENT_LOOP__"

# A YouTube source whose key/handle/url contains SYNC_GROWTH_SENTINEL enumerates
# ONE EXTRA "new upload" in its LISTING (``list_video_ids``/``canned_video_ids``)
# beyond the set its kind actually downloads (``_ids_for_kind``). A static fixture
# channel never "grows", so the incremental "Sync now" diff would always find 0
# new and never enqueue anything — this sentinel is the deterministic way to make
# the auto-sync ENQUEUE path reachable offline: after a full import ingests the
# base set, the diff sees the extra id as new and enqueues one ``origin='auto'``
# ingest. FAIL-CLOSED like the fault sentinels: the growth branch is gated on
# ``test_mode()`` (production forces it OFF), and the synthetic id is deliberately
# ABSENT from ``_ids_for_kind`` so it is never downloaded (no fixture needed) —
# the enqueued auto job re-lists the base set, finds it all on disk, and completes.
SYNC_GROWTH_SENTINEL = "e2e-syncgrow"

# Per-process memory for the fail-once sentinel: source keys that have already
# failed once (so their retry succeeds).
_fail_once_seen: set[str] = set()

# Cooperative-cancel hold for the hang sentinel: poll the cancel flag in small
# ticks up to a ceiling so a no-cancel run can never wedge the single worker.
_HANG_TICKS = 600
_HANG_TICK_SEC = 0.05  # ~30s total ceiling


def _ingest_fault(channel: str) -> str | None:
    """Classify a source key into an E2E ingest fault, or None.

    FAIL-CLOSED: returns None (no fault) unless ``test_mode()`` is true, so the
    sentinels are inert on any real path even if this were somehow reached."""
    if not test_mode():
        return None
    c = (channel or "").lower()
    if INGEST_HANG_SENTINEL in c:
        return "hang"
    if INGEST_FAIL_ONCE_SENTINEL in c:
        return "fail-once"
    if INGEST_FAIL_SENTINEL in c:
        return "fail"
    return None


# ---- Seam C: canned YouTube source -----------------------------------
def _kind_of(channel: str) -> str:
    """Classify a source key/URL into channel | playlist | video.

    ``download_transcripts``/``list_video_ids`` receive the source's normalized
    de-dup key (``@handle`` / ``channel/UC…`` / ``playlist/<id>`` / ``video/<id>``)
    or, defensively, a raw URL. The prefixes are enough to pick the canned set.
    """
    c = (channel or "").strip()
    low = c.lower()
    if c.startswith("video/") or "watch?v=" in low or "youtu.be/" in low or "/shorts/" in low:
        return "video"
    if c.startswith("playlist/") or "list=" in low:
        return "playlist"
    return "channel"


def _ids_for_kind(kind: str) -> list[str]:
    if kind == "video":
        return [_SINGLE_VIDEO_ID]
    if kind == "playlist":
        return list(_PLAYLIST_VIDEO_IDS)
    return list(_CHANNEL_VIDEO_IDS)


def _fixture(video_id: str) -> dict:
    """Load (and cache) one transcript fixture, matching download_transcripts' shape."""
    if video_id not in _fixture_cache:
        path = _FIXTURES_DIR / f"{video_id}.json"
        _fixture_cache[video_id] = json.loads(path.read_text(encoding="utf-8"))
    return _fixture_cache[video_id]


def canned_video_ids(channel: str) -> list[dict]:
    """Test-mode ``list_video_ids``: a flat list of {id, title, url} per kind."""
    out: list[dict] = []
    for vid in _ids_for_kind(_kind_of(channel)):
        doc = _fixture(vid)
        out.append({"id": doc["id"], "title": doc["title"], "url": doc["url"]})
    # E2E: simulate a channel that gained one new video since import so the
    # incremental "Sync now" diff has exactly one new id to enqueue as an
    # origin='auto' job (see SYNC_GROWTH_SENTINEL). Fail-closed: test-mode only,
    # and this id is never in _ids_for_kind so it is listed but never downloaded.
    if test_mode() and SYNC_GROWTH_SENTINEL in (channel or "").lower():
        out.append({
            "id": "tm_sync_new",
            "title": "Newly uploaded video",
            "url": "https://youtube.com/watch?v=tm_sync_new",
        })
    return out


def fake_download_transcripts(
    channel: str,
    out_dir: Path,
    langs=None,
    limit: int | None = None,
    progress=None,
) -> list[dict]:
    """Test-mode ``download_transcripts``: write fixtures to disk, emit the SAME
    progress stages the real fetcher does so the queue's counts/statuses fire.

    Writes one ``<video_id>.json`` per canned video into ``out_dir`` using the exact
    on-disk schema ``load_transcripts`` expects, then returns the ``{id,title,url}``
    summaries. No network, no yt-dlp.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(**kw):
        if progress:
            progress(kw)

    # E2E fault injection (TEST_MODE only — see the sentinel block above). Raise
    # BEFORE any progress/emit so the job never reports partial work: the failure
    # propagates through ingest_channel to the queue's error handler (job ->
    # 'error'). fail-once raises only the first time this key is seen, so its
    # retry falls through to a normal import.
    fault = _ingest_fault(channel)
    if fault == "fail":
        raise RuntimeError(f"[test-mode] {INGEST_FAIL_SENTINEL} sentinel: forced ingest failure")
    if fault == "fail-once" and channel not in _fail_once_seen:
        _fail_once_seen.add(channel)
        raise RuntimeError(
            f"[test-mode] {INGEST_FAIL_ONCE_SENTINEL} sentinel: forced first-attempt failure"
        )

    ids = _ids_for_kind(_kind_of(channel))
    if limit is not None:
        ids = ids[:limit]
    total = len(ids)

    emit(stage="listing", message=f"[test-mode] listing videos for {channel}")
    emit(stage="listed", total=total, message=f"[test-mode] found {total} videos")
    emit(stage="channel", title=_CHANNEL_TITLE, avatar="")

    # Hang sentinel: hold the (now 'running') job in a cooperatively cancellable
    # wait BEFORE saving any transcript, so a spec can cancel a running job. Each
    # tick calls progress(); the queue-side probe raises IngestCancelled on cancel
    # and the queue marks the job 'cancelled'. Bounded so a no-cancel run still
    # terminates (falls through to a normal import -> 'done'). No effect if there
    # is no progress callback to carry the cancel probe.
    if fault == "hang" and progress is not None:
        for _ in range(_HANG_TICKS):
            emit(stage="waiting", message="[test-mode] holding for cancel")
            time.sleep(_HANG_TICK_SEC)

    saved: list[dict] = []
    for i, vid in enumerate(ids, 1):
        doc = _fixture(vid)
        summary = {"id": doc["id"], "title": doc["title"], "url": doc["url"]}
        dest = out_dir / f"{vid}.json"
        if dest.exists():
            saved.append(summary)
            emit(stage="skip", index=i, total=total, video_id=vid, title=doc["title"],
                 message=f"[{i}/{total}] skip (already have) {doc['title']}")
            continue
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(summary)
        emit(stage="saved", index=i, total=total, video_id=vid, title=doc["title"],
             message=f"[{i}/{total}] saved {doc['title']}")

    emit(stage="done", total=total, saved=len(saved),
         message=f"[test-mode] done. {len(saved)} transcripts available.")
    return saved


# ---- Seam B: deterministic offline embedder --------------------------
_EMBED_DIM = 256


def _ngrams(text: str) -> list[str]:
    """Word tokens plus char 3/4-grams — stable, cheap, and enough overlap for a
    deterministic nearest-neighbour on a small corpus."""
    low = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", low)
    grams: list[str] = list(words)
    joined = " ".join(words)
    for n in (3, 4):
        for i in range(len(joined) - n + 1):
            grams.append(joined[i : i + n])
    return grams or ["_empty_"]


class FakeEmbedder:
    """Maps text to a deterministic hashed-n-gram, L2-normalized fixed-dim vector.

    Same dimension everywhere so Chroma add/query works offline; the vector is a
    pure function of the input text, so nearest-neighbour is stable for a corpus.
    """

    def embed_documents(self, texts: Sequence[str], batch_size: int = 128) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    # Async twins (PLAN §5b): identical deterministic vectors so the async chat
    # path is byte-for-byte the sync one under TEST_MODE.
    async def aembed_documents(
        self, texts: Sequence[str], batch_size: int = 128
    ) -> list[list[float]]:
        return self.embed_documents(texts, batch_size)

    async def aembed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_queries(texts)

    def _vector(self, text: str) -> list[float]:
        # The chat-error token is a TEST_MODE control, not user meaning.  Strip it
        # before embedding so an otherwise groundable fault-injection question
        # reaches FakeLLM rather than falling below the retrieval floor.
        if test_mode():
            text = text.replace(CHAT_ERROR_SENTINEL, " ")
        vec = [0.0] * _EMBED_DIM
        for tok in _ngrams(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
            idx = h % _EMBED_DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


# ---- Seam A: deterministic offline LLM -------------------------------
def _sink_final(usage_sink: Usage | None, usage: Usage) -> None:
    """Mirror a round's final usage into the caller-owned sink (in place)."""
    if usage_sink is None:
        return
    usage_sink.prompt_tokens = usage.prompt_tokens
    usage_sink.completion_tokens = usage.completion_tokens


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            # Multimodal content parts: concatenate any text parts.
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


def _extract_after(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    rest = text[idx + len(marker) :]
    # Stop at the next blank-line-delimited section header.
    return rest.split("\n\n", 1)[0].strip()


_INTERVIEW_QUESTIONS = [
    "What tone and personality should this bot have — for example warm and "
    "encouraging, formal and factual, or witty and playful?",
    "Which language should the bot reply in, and is there any topic it should "
    "gently steer away from?",
]


def _fake_persona(user_content: str) -> str:
    desc = " ".join(_extract_after(user_content, "Original description:").split())[:180]
    focus = desc or "answering viewers' questions about this YouTube channel"
    return (
        "You are a knowledgeable, friendly assistant representing this YouTube "
        f"channel. Your focus: {focus}. Speak clearly and stay in character, greet "
        "users warmly, and steer off-topic messages back to the channel's subjects. "
        "Keep answers concise and helpful."
    )


def _first_excerpt(user_content: str) -> dict | None:
    """Parse the first retrieved excerpt block out of a rag chat user message.

    ``rag._format_context`` renders each hit as::

        [1]
        <<<UNTRUSTED_EXCERPT_DATA>>>
        author: X | title: Y | timecode: T | link: L
        <chunk text>
        <<<END_UNTRUSTED_EXCERPT_DATA>>>

    (documents use ``document: Y | (no link)`` instead). Returns the parsed fields
    of the FIRST block, or None when there are no excerpts.
    """
    m = re.search(r"Transcript excerpts:\s*(.*?)\n\nQuestion:", user_content, re.S)
    body = m.group(1) if m else user_content
    blocks = [b for b in re.split(r"\n\n+", body) if b.strip().startswith("[")]
    if not blocks:
        return None
    lines = blocks[0].splitlines()
    # The trusted source index is now on its own line; metadata and the chunk
    # live inside the untrusted-data fence.  Parse the first non-marker line as
    # the header, exactly as the production formatter emits it.
    if not lines or not re.fullmatch(r"\[\d+\]", lines[0].strip()):
        return None
    content_lines = [
        ln for ln in lines[1:]
        if not re.fullmatch(r"<<<.*?>>>", ln.strip())
    ]
    if not content_lines:
        return None
    header, text_lines = content_lines[0], content_lines[1:]
    text = "\n".join(text_lines)
    # The excerpt body is fenced in <<<UNTRUSTED_EXCERPT_DATA>>> markers
    # (rag._format_context); drop those marker lines so the echoed snippet is the
    # raw chunk text, not the isolation scaffolding.
    fields: dict[str, str] = {}
    for part in header.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            fields[k.strip().lower()] = v.strip()
    author = fields.get("author") or "the channel"
    title = fields.get("title") or fields.get("document") or "the source"
    link = fields.get("link", "")
    return {
        "author": author,
        "title": title,
        "timecode": fields.get("timecode", ""),
        "link": link,
        "has_link": bool(link),
        "snippet": " ".join(text.split())[:200],
    }


def _source_block(ex: dict) -> str:
    """Render one retrieved excerpt as the production source block (BUG-032):
    a bold ``**Author — Title**`` heading (a document adds ``_(документ)_``),
    the actual excerpt in a Markdown blockquote (every line prefixed ``> ``),
    and — for a video only — a ``▶️ [Дивитись з mm:ss](link)`` watch line right
    after the quote (intentionally folded into the Telegram quote block)."""
    document = not ex["has_link"]
    head = f"**{ex['author']} — {ex['title']}**" + (" _(документ)_" if document else "")
    quote = "\n".join(
        f"> {ln}" for ln in ex["snippet"].splitlines() if ln.strip()
    ) or f"> {ex['snippet']}"
    block = f"{head}\n{quote}"
    if ex["has_link"]:
        tc = ex["timecode"] or "0:00"
        block += f"\n▶️ [Дивитись з {tc}]({ex['link']})"
    return block


class FakeLLM:
    """Deterministic stand-in for a chat/persona LLM (no network, no key).

    Distinguishes the two flows by the SYSTEM prompt: the persona interview prompt
    (from ``bot._INTERVIEW_PROMPT``) mentions a "prompt engineer"; anything else is
    treated as a grounded chat (``rag.GROUNDING_RULES`` system prompt).
    """

    def complete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        # ``reasoning_effort`` accepted for interface parity with the real LLMs; the
        # deterministic reply does not depend on it.
        _ = reasoning_effort
        text = self._reply(system or "", messages or [])
        return Completion(text=text, usage=Usage(prompt_tokens=64, completion_tokens=32))

    def stream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Yield the deterministic reply in word-sized deltas, then return usage.

        Chunking on whitespace (not one big blob) is what lets the offline SSE and
        Telegram-progressive tests observe real incremental delivery. Raising for
        the error sentinel is preserved so the streaming path exercises the same
        failure surface as :meth:`complete`.

        ``usage_sink`` models a provider that reports RUNNING usage as tokens
        accrue: it is advanced one completion token per delta and finalized to the
        full count, so an interrupted stream still exposes a non-zero produced-token
        count for the caller to bill (the N1 mid-stream-billing path)."""
        _ = reasoning_effort  # interface parity; reply is deterministic
        text = self._reply(system or "", messages or [])
        words = text.split(" ")
        total = len(words)
        for i, word in enumerate(words):
            if usage_sink is not None:
                usage_sink.prompt_tokens = 64
                usage_sink.completion_tokens = i + 1
            yield word if i == 0 else " " + word
        final = Usage(prompt_tokens=64, completion_tokens=total or 32)
        if usage_sink is not None:
            usage_sink.prompt_tokens = final.prompt_tokens
            usage_sink.completion_tokens = final.completion_tokens
        return final

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        _ = reasoning_effort  # interface parity; reply is deterministic
        """Deterministic structured reply for the grounded-citation path.

        Returns ``{"answer": <deterministic reply>, "citations": [...]}``. To make
        the fragment-timecode wiring testable, it scans the fed context for
        ``[source.fragment]`` markers and cites the LATEST-indexed fragment it finds
        (simulating a model that grounded on a passage deeper than the chunk head);
        with no markers it cites source 1, fragment 0.
        """
        user = _last_user_text(messages)
        self._raise_chat_error(system, user)
        pairs = [(int(s), int(f)) for s, f in re.findall(r"\[(\d+)\.(\d+)\]", user)]
        if pairs:
            s, f = max(pairs, key=lambda p: (p[1], p[0]))
            citations = [{"source": s, "fragment": f}]
        else:
            citations = [{"source": 1, "fragment": 0}]
        data = {"answer": self._chat_reply(user), "citations": citations}
        return JsonCompletion(
            data=data, usage=Usage(prompt_tokens=64, completion_tokens=32), ok=True,
        )

    # ---- async twins (PLAN §5b) — same deterministic replies, awaited --------
    async def acomplete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        return self.complete(
            system, messages, max_tokens=max_tokens, reasoning_effort=reasoning_effort
        )

    async def astream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ):
        """Async twin of :meth:`stream`: the same word-sized deltas and running
        ``usage_sink`` advance (final usage travels via the sink only — an async
        generator cannot ``return`` a value)."""
        _ = reasoning_effort  # interface parity; reply is deterministic
        text = self._reply(system or "", messages or [])
        words = text.split(" ")
        total = len(words)
        for i, word in enumerate(words):
            if usage_sink is not None:
                usage_sink.prompt_tokens = 64
                usage_sink.completion_tokens = i + 1
            yield word if i == 0 else " " + word
        if usage_sink is not None:
            usage_sink.prompt_tokens = 64
            usage_sink.completion_tokens = total or 32

    async def acomplete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        return self.complete_json(
            system, messages, schema, schema_name=schema_name,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )

    # ---- tool calling (agentic RAG) — deterministic offline twin ----------
    def complete_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> ToolCompletion:
        """Deterministic agent round: request a search until a tool result is
        present, then answer citing the first excerpt in the last tool result.

        Mirrors a well-behaved tool-calling model: a round with no gathered
        material asks for exactly one ``search_channel`` call (query = the user's
        question); once a ``role="tool"`` result exists it produces the final
        grounded answer with a clickable deep-link citation. The AGENT_LOOP
        sentinel (test-mode only) keeps requesting searches so the round CAP is
        reachable; ``tool_choice="none"`` always forces a final answer."""
        _ = reasoning_effort
        question = self._first_user_text(messages)
        if tool_choice != "none" and self._wants_search(question, messages):
            n = sum(1 for m in messages if m.get("role") == "tool")
            return ToolCompletion(
                text="",
                tool_calls=[ToolCallRequest(
                    id=f"call_{n + 1}", name="search_channel",
                    arguments=json.dumps({"query": f"{question} details {n + 1}"}),
                )],
                usage=Usage(prompt_tokens=64, completion_tokens=8),
            )
        return ToolCompletion(
            text=self._agent_final_answer(messages),
            usage=Usage(prompt_tokens=64, completion_tokens=32),
        )

    def stream_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ):
        """Streaming twin: a tool round yields NO text; a final round yields the
        answer in word deltas with a running ``usage_sink`` (like stream())."""
        res = self.complete_tools(
            system, messages, tools, tool_choice=tool_choice,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )
        if res.tool_calls:
            _sink_final(usage_sink, res.usage)
            return res
        words = res.text.split(" ")
        for i, word in enumerate(words):
            if usage_sink is not None:
                usage_sink.prompt_tokens = 64
                usage_sink.completion_tokens = i + 1
            yield word if i == 0 else " " + word
        res.usage = Usage(prompt_tokens=64, completion_tokens=len(words) or 32)
        _sink_final(usage_sink, res.usage)
        return res

    async def acomplete_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> ToolCompletion:
        return self.complete_tools(
            system, messages, tools, tool_choice=tool_choice,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )

    async def astream_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        completion_sink: ToolCompletion | None = None,
        reasoning_effort: str | None = None,
    ):
        """Async twin: same deltas; the final ToolCompletion is delivered by
        filling ``completion_sink`` in place (async generators cannot return)."""
        res = self.complete_tools(
            system, messages, tools, tool_choice=tool_choice,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )
        if not res.tool_calls:
            words = res.text.split(" ")
            for i, word in enumerate(words):
                if usage_sink is not None:
                    usage_sink.prompt_tokens = 64
                    usage_sink.completion_tokens = i + 1
                yield word if i == 0 else " " + word
            res.usage = Usage(prompt_tokens=64, completion_tokens=len(words) or 32)
        _sink_final(usage_sink, res.usage)
        if completion_sink is not None:
            completion_sink.text = res.text
            completion_sink.tool_calls = res.tool_calls
            completion_sink.usage = res.usage

    @staticmethod
    def _first_user_text(messages: list[dict]) -> str:
        for m in messages or []:
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
        return ""

    @staticmethod
    def _wants_search(question: str, messages: list[dict]) -> bool:
        """Request (another) search when nothing is gathered yet, or when the
        loop sentinel (test-mode only) forces an endless gather phase."""
        if test_mode() and AGENT_LOOP_SENTINEL in question:
            return True
        return not any(m.get("role") == "tool" for m in messages)

    def _agent_final_answer(self, messages: list[dict]) -> str:
        """Final grounded reply with a real-excerpt source block from the last tool result."""
        last_tool = ""
        for m in messages or []:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                last_tool = m["content"]
        author_m = re.search(r"author:\s*([^|\n]+)", last_tool)
        title_m = re.search(r"(?:title|document):\s*([^|\n]+)", last_tool)
        video_m = re.search(r"video:\s*(\S+)", last_tool)
        frag_m = re.search(r"\[t=(\d+) \| ([^\]]+)\]\s*([^\n]*)", last_tool)
        if not (author_m and title_m):
            return (
                "I could not find anything in the channel's material on that. "
                "Try asking about a topic the channel covers."
            )
        text = frag_m.group(3).strip() if frag_m else ""
        if not text:
            # A document has no timecoded fragments. Its first body line is still
            # retrieved material and is therefore the excerpt to quote.
            body = re.split(r"(?:\n|^)<<<END_UNTRUSTED_EXCERPT_DATA>>>", last_tool, 1)[0]
            header_end = body.find("\n")
            text = body[header_end + 1:].strip() if header_end >= 0 else ""
        if not text:
            return (
                "I could not find anything in the channel's material on that. "
                "Try asking about a topic the channel covers."
            )
        author = author_m.group(1).strip()
        title = title_m.group(1).strip()
        quote = "\n".join(f"> {line}" for line in text.splitlines() if line.strip())
        document = video_m is None
        source_block = f"**{author} — {title}**" + (" _(документ)_" if document else "")
        source_block += f"\n{quote}"
        if video_m and frag_m:
            url = video_m.group(1).strip()
            t, tc = frag_m.group(1), frag_m.group(2).strip()
            sep = "&" if "?" in url else "?"
            source_block += f"\n▶️ [Дивитись з {tc}]({url}{sep}t={t})"
        return f"The channel's take: {text[:160]}\n\n{source_block}"

    def _reply(self, system: str, messages: list[dict]) -> str:
        user = _last_user_text(messages)
        self._raise_chat_error(system, user)
        if "prompt engineer" in system.lower():
            return self._interview_reply(user)
        return self._chat_reply(user)

    @staticmethod
    def _raise_chat_error(system: str, user: str) -> None:
        # Chat-error sentinel (TEST_MODE only — see the sentinel block above).
        # FAIL-CLOSED: gated on test_mode() so a question that merely contains the
        # token is inert on any real path. Raising here surfaces through
        # rag.answer -> the web chat's LLM-failure error state. Guarded to the
        # grounded-chat flow (not the persona interview), matching its "magic
        # question" contract.
        if (
            test_mode()
            and "prompt engineer" not in system.lower()
            and CHAT_ERROR_SENTINEL in user
        ):
            raise RuntimeError(
                f"[test-mode] {CHAT_ERROR_SENTINEL} sentinel: forced chat LLM error"
            )

    def _interview_reply(self, user_content: str) -> str:
        # The builder formats prior turns as "N. Q: …". Count them to decide whether
        # to ask another tailored-looking question or finish with a persona. We ask
        # up to two questions, then return a persona derived from the description.
        answered = len(re.findall(r"(?m)^\s*\d+\.\s*Q:", user_content))
        if answered < len(_INTERVIEW_QUESTIONS):
            return json.dumps(
                {"done": False, "question": _INTERVIEW_QUESTIONS[answered], "persona": None}
            )
        return json.dumps({"done": True, "question": None, "persona": _fake_persona(user_content)})

    def _chat_reply(self, user_content: str) -> str:
        ex = _first_excerpt(user_content)
        if ex is None:
            return (
                "I could not find anything in the channel's transcripts to answer that. "
                "Try asking about a topic the channel covers."
            )
        return (
            f"According to {ex['author']}, {ex['snippet']}\n\n{_source_block(ex)}"
        ).strip()
