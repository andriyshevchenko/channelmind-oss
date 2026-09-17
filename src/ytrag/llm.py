"""Chat backends: Anthropic (native, with prompt caching) or any OpenAI-compatible API."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Protocol

from .config import DEFAULT_MAX_OUTPUT_TOKENS, Config
from .usage import Usage

# Process-wide OpenAI/OpenRouter client cache, keyed by (api_key, base_url) (BUG-029).
# A fresh Assistant — and thus fresh clients — is built per chat request, so without
# this each request paid a cold TCP+TLS handshake to the provider on every call
# (router, embed, answer). One shared client per credential keeps httpx's connection
# pool warm ACROSS requests, so subsequent calls reuse the connection.
_OPENAI_CLIENTS: "dict[tuple[str, str | None], object]" = {}
_OPENAI_CLIENTS_LOCK = threading.Lock()


def shared_openai_client(api_key: str, base_url: str | None = None):
    """Return the process-wide OpenAI client for ``(api_key, base_url)``, created once.

    The client is safe to share across threads; the cache is keyed by credentials so
    BYOK keys never mix. NOTE: the cache never evicts or closes clients — it grows by one
    persistent client (httpx pool) per distinct credential for the process lifetime. Fine
    for single-tenant / shared-key deploys; under multi-tenant BYOK with many distinct
    keys this should become an LRU that calls ``client.close()`` on eviction (deferred to
    the async rework — see PLAN §5b)."""
    key = (api_key, base_url)
    client = _OPENAI_CLIENTS.get(key)
    if client is None:
        with _OPENAI_CLIENTS_LOCK:
            client = _OPENAI_CLIENTS.get(key)
            if client is None:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=base_url)
                _OPENAI_CLIENTS[key] = client
    return client


# Async twin of _OPENAI_CLIENTS (PLAN §5b async migration). An AsyncOpenAI client
# wraps an httpx.AsyncClient whose transport/pool belongs to the event loop it is
# first used on — sharing one across loops raises "attached to a different loop".
# So the cache is keyed per RUNNING LOOP first, then per credential. In production
# there is exactly ONE loop (the single uvicorn worker), so this behaves like the
# sync cache: one warm client per credential for the process lifetime. The
# WeakKeyDictionary drops a dead loop's clients with the loop object itself.
_ASYNC_OPENAI_CLIENTS: (
    "weakref.WeakKeyDictionary[object, dict[tuple[str, str | None], object]]"
) = weakref.WeakKeyDictionary()
_ASYNC_OPENAI_CLIENTS_LOCK = threading.Lock()


def shared_async_openai_client(api_key: str, base_url: str | None = None):
    """Return the per-loop shared AsyncOpenAI client for ``(api_key, base_url)``.

    Mirror of :func:`shared_openai_client` for the async chat path: one pooled
    client per credential so the httpx connection pool stays warm ACROSS requests,
    but scoped to the current event loop (see the cache comment above) so it is
    safe even if several loops exist (e.g. tests). MUST be called from a running
    event loop. Same non-eviction caveat as the sync cache."""
    loop = asyncio.get_running_loop()
    key = (api_key, base_url)
    with _ASYNC_OPENAI_CLIENTS_LOCK:
        per_loop = _ASYNC_OPENAI_CLIENTS.get(loop)
        if per_loop is None:
            per_loop = _ASYNC_OPENAI_CLIENTS[loop] = {}
        client = per_loop.get(key)
        if client is None:
            from openai import AsyncOpenAI

            client = per_loop[key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client


@dataclass
class Completion:
    text: str
    usage: Usage


@dataclass
class ToolCallRequest:
    """One tool invocation the model asked for (OpenAI tool-calling shape).

    ``arguments`` is the RAW JSON string exactly as the provider sent it — the
    caller parses it (and treats a malformed blob as empty args) so a model that
    emits broken JSON degrades gracefully instead of failing in this layer."""
    id: str
    name: str
    arguments: str


@dataclass
class ToolCompletion:
    """A chat reply that may carry tool calls (the agentic-RAG loop primitive).

    ``tool_calls`` non-empty => the model wants tools executed and the
    conversation continued; empty => ``text`` is a normal (final) assistant
    message. Instances are also used as caller-owned SINKS by ``astream_tools``
    (filled in place, like ``usage_sink``), so the dataclass is mutable."""
    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class JsonCompletion:
    """A structured (schema-validated) model reply plus its token usage.

    ``data`` is the parsed object the model returned against the requested JSON
    schema — never free-text the caller must regex. ``ok`` is False when the
    provider/SDK failed to produce parseable structured output, so the caller can
    fall back to its non-structured path instead of trusting an empty ``data``."""
    data: dict
    usage: Usage
    ok: bool = True


class LLM(Protocol):
    def complete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion: ...

    def stream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Yield answer text deltas as they arrive.

        Implementations MUST ``return`` the final :class:`Usage` from the generator
        (accessible via ``StopIteration.value``) so a streaming caller can still do
        the same passive spend accounting as the non-stream path.

        ``usage_sink`` (optional) is a caller-owned :class:`Usage` that the
        implementation keeps mirrored with the latest usage the provider has
        reported. It lets a caller bill what was PRODUCED even when the stream is
        torn down early (a client disconnect raises ``GeneratorExit`` before the
        generator can ``return`` its final usage), without waiting for that return.

        ``reasoning_effort`` (optional) turns on the provider's thinking/reasoning
        mode for THIS call only (``low``/``medium``/``high``); ``None`` means OFF —
        the request is byte-for-byte the non-reasoning one. The gate lives in the
        caller (see :class:`ytrag.rag.Assistant`), so the model only "thinks" where
        it pays off.
        """
        ...

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        """Return a schema-validated structured reply (parsed dict), not free text.

        Used where the caller needs a machine-readable answer it must NOT regex out
        of prose (e.g. an answer plus a list of cited fragment ids). Implementations
        force the provider's structured mode (OpenAI ``response_format`` json_schema
        / Anthropic forced tool-use) and parse the result. On any provider/parse
        failure they return ``ok=False`` so the caller can fall back cleanly.

        ``reasoning_effort`` behaves as in :meth:`complete` (OpenRouter-only, per
        call, gated by the caller); ``None`` keeps the request unchanged.
        """
        ...

    # ---- async twins (PLAN §5b) -------------------------------------------
    # Same contracts as the sync methods above, awaited on the event loop so a
    # chat request never blocks it. One difference: an ASYNC generator cannot
    # ``return`` a value, so :meth:`astream` reports its final usage ONLY via
    # ``usage_sink`` (which the sync contract already mirrors continuously) —
    # callers snapshot the sink after exhaustion instead of StopIteration.value.
    async def acomplete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion: ...

    def astream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]: ...

    async def acomplete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._api_key = api_key
        # Lazy async twin (PLAN §5b): created on first async use so the sync-only
        # paths (CLI) never pay for an extra httpx.AsyncClient they won't use.
        self._async_client = None
        self.model = model

    def _aclient(self):
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    def complete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        # ``reasoning_effort`` is accepted for a uniform LLM interface but ignored
        # here: it drives the OpenRouter ``reasoning`` field, so the native
        # Anthropic request stays byte-for-byte unchanged.
        _ = reasoning_effort
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text=text, usage=_anthropic_usage(resp))

    def stream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        # See complete(): reasoning is OpenRouter-only, so it is ignored here and the
        # native Anthropic stream request is unchanged.
        _ = reasoning_effort
        # The ``with`` block closes the underlying HTTP stream deterministically —
        # including on the GeneratorExit raised when a client disconnects mid-answer.
        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
            usage = _anthropic_usage(stream.get_final_message())
            _mirror_usage(usage_sink, usage)
            return usage

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        # reasoning is OpenRouter-only (see complete()); ignored on native Anthropic.
        _ = reasoning_effort
        # Force a single tool call whose input_schema IS the requested schema —
        # Anthropic then returns a validated ``tool_use`` block whose ``input`` is
        # the parsed object (no prose to regex).
        tool = {"name": schema_name, "description": "Return the structured result.",
                "input_schema": schema}
        usage = Usage()  # captured before parse so a mid-parse failure still bills
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
            )
            usage = _anthropic_usage(resp)
            for b in resp.content:
                # Require OUR tool + a non-empty dict input. A different tool name, a
                # non-dict/empty input (e.g. a max_tokens-truncated forced call) is a
                # malformed reply → ok=False so the caller falls back, never a trusted
                # empty result.
                if (getattr(b, "type", None) == "tool_use"
                        and getattr(b, "name", None) == schema_name):
                    data = b.input if isinstance(b.input, dict) else {}
                    ok = bool(data)
                    return JsonCompletion(data=data if ok else {}, usage=usage, ok=ok)
            return JsonCompletion(data={}, usage=usage, ok=False)
        except Exception:  # noqa: BLE001 - any SDK/parse failure => clean fallback
            return JsonCompletion(data={}, usage=usage, ok=False)

    # ---- async twins (PLAN §5b) — same requests, awaited on the event loop ----
    async def acomplete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        _ = reasoning_effort  # OpenRouter-only, ignored on native Anthropic (see complete)
        resp = await self._aclient().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text=text, usage=_anthropic_usage(resp))

    async def astream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        _ = reasoning_effort  # OpenRouter-only, ignored on native Anthropic (see stream)
        # ``async with`` closes the HTTP stream deterministically, including on the
        # GeneratorExit/CancelledError raised when a client disconnects mid-answer.
        # Final usage is reported via usage_sink only (async generators can't return).
        async with self._aclient().messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
            _mirror_usage(usage_sink, _anthropic_usage(await stream.get_final_message()))

    async def acomplete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        _ = reasoning_effort  # OpenRouter-only (see complete_json)
        tool = {"name": schema_name, "description": "Return the structured result.",
                "input_schema": schema}
        usage = Usage()  # captured before parse so a mid-parse failure still bills
        try:
            resp = await self._aclient().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
            )
            usage = _anthropic_usage(resp)
            for b in resp.content:
                # Same malformed-reply guard as the sync path (see complete_json).
                if (getattr(b, "type", None) == "tool_use"
                        and getattr(b, "name", None) == schema_name):
                    data = b.input if isinstance(b.input, dict) else {}
                    ok = bool(data)
                    return JsonCompletion(data=data if ok else {}, usage=usage, ok=ok)
            return JsonCompletion(data={}, usage=usage, ok=False)
        except Exception:  # noqa: BLE001 - any SDK/parse failure => clean fallback
            return JsonCompletion(data={}, usage=usage, ok=False)


def _mirror_usage(sink: Usage | None, usage: Usage) -> None:
    """Copy the latest observed usage into a caller's running holder (in place).

    Lets a streaming caller bill the tokens PRODUCED SO FAR when the stream is torn
    down early (client disconnect → GeneratorExit) instead of only at the
    generator's final ``return``. No-op when the caller passed no sink.
    """
    if sink is None:
        return
    sink.prompt_tokens = usage.prompt_tokens
    sink.completion_tokens = usage.completion_tokens


def _anthropic_usage(resp) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    # Prompt caching reports cached tokens separately from input_tokens; fold
    # them back in so a warm cache doesn't make spend look smaller than it is.
    prompt = (
        (getattr(u, "input_tokens", 0) or 0)
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        + (getattr(u, "cache_read_input_tokens", 0) or 0)
    )
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=getattr(u, "output_tokens", 0) or 0,
    )


def _reasoning_body(reasoning_effort: str | None) -> dict:
    """OpenRouter ``reasoning`` request field for the current effort, or ``{}``.

    OFF (``None``) returns an empty dict so the request is identical to today's;
    an effort adds ``{"reasoning": {"effort": ...}}`` passed via the OpenAI SDK's
    ``extra_body`` (OpenRouter's unified thinking interface)."""
    if not reasoning_effort:
        return {}
    return {"extra_body": {"reasoning": {"effort": reasoning_effort}}}


def _strict_schema(schema: dict) -> dict:
    """Return a copy of ``schema`` in OpenAI Structured-Outputs *strict* form.

    OpenAI's strict json_schema rejects any object that doesn't set
    ``additionalProperties: false`` AND list every declared property in
    ``required``. Callers pass a natural schema; this deep-copies it and injects
    both on every nested object so a normal schema doesn't 400 (which the caller's
    ``except`` would otherwise swallow into a silent ok=False no-op). Anthropic's
    ``input_schema`` has no such rule, so only the OpenAI path normalizes."""
    def _norm(node):
        if isinstance(node, dict):
            out = {k: _norm(v) for k, v in node.items()}
            if out.get("type") == "object" and isinstance(out.get("properties"), dict):
                out["additionalProperties"] = False
                out["required"] = list(out["properties"].keys())
            return out
        if isinstance(node, list):
            return [_norm(x) for x in node]
        return node

    return _norm(deepcopy(schema))


class OpenAICompatLLM:
    """Works with OpenAI and OpenRouter (both expose /v1/chat/completions)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        # Reuse a warm, pooled client across requests (BUG-029) instead of a fresh
        # cold-handshake client per Assistant.
        self._client = shared_openai_client(api_key, base_url)
        # Kept for the async path: the per-LOOP async client must be resolved at
        # call time (inside a running loop), not here — CLI paths have no loop.
        self._api_key = api_key
        self._base_url = base_url
        self.model = model

    def _async_client(self):
        return shared_async_openai_client(self._api_key, self._base_url)

    def complete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        # Reasoning is decided PER CALL by the caller (None => today's request,
        # byte-for-byte); the gate lives in rag.Assistant, not here.
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            **_reasoning_body(reasoning_effort),
        )
        return Completion(
            text=resp.choices[0].message.content or "",
            usage=_openai_usage(resp),
        )

    def stream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full, stream=True,
            # Ask the provider to emit a final usage-only chunk so streaming spend
            # is metered exactly like the non-stream path.
            stream_options={"include_usage": True},
            # Reasoning is per-call (None => unchanged request); gated by the caller.
            **_reasoning_body(reasoning_effort),
        )
        usage = Usage()
        try:
            for chunk in resp:
                if getattr(chunk, "usage", None):
                    usage = _openai_usage(chunk)
                    _mirror_usage(usage_sink, usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                text = getattr(choices[0].delta, "content", None)
                if text:
                    yield text
            return usage
        finally:
            # Close the underlying HTTP stream deterministically rather than leaving
            # it to GC. On a client disconnect the generator is torn down with
            # GeneratorExit mid-iteration, so without this the socket could linger.
            close = getattr(resp, "close", None)
            if callable(close):
                close()

    def complete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        # response_format=json_schema (strict) makes the provider emit ONLY a JSON
        # object matching ``schema`` in message.content — parse, don't regex.
        # OpenAI strict mode REQUIRES additionalProperties:false + a complete
        # ``required`` on every object, so normalize the caller's natural schema
        # (else the API 400s and the feature silently no-ops behind the except).
        full = [{"role": "system", "content": system}, *messages]
        usage = Usage()  # captured before parse so a mid-parse failure still bills
        try:
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, messages=full,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name,
                                    "schema": _strict_schema(schema),
                                    "strict": True},
                },
                **_reasoning_body(reasoning_effort),
            )
            usage = _openai_usage(resp)
            msg = resp.choices[0].message
            # A safety refusal or empty content is NOT a valid structured reply —
            # fail so the caller falls back instead of trusting {} as ok.
            if getattr(msg, "refusal", None):
                return JsonCompletion(data={}, usage=usage, ok=False)
            raw = msg.content or ""
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict) or not data:
                return JsonCompletion(data={}, usage=usage, ok=False)
            return JsonCompletion(data=data, usage=usage, ok=True)
        except Exception:  # noqa: BLE001 - provider/parse failure => clean fallback
            return JsonCompletion(data={}, usage=usage, ok=False)

    # ---- tool calling (agentic RAG) ---------------------------------------
    # OPTIONAL LLM capability, discovered by the caller via hasattr (only the
    # OpenAI-compatible providers and the test-mode FakeLLM implement it; the
    # native Anthropic path doesn't, so the agentic flag falls back to the
    # classic pipeline there). Not part of the LLM Protocol on purpose — the
    # Protocol stays the universal contract every backend satisfies.

    def complete_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> ToolCompletion:
        """One tool-enabled chat round: the model may answer OR request tool calls.

        ``messages`` may already contain assistant ``tool_calls`` turns and
        ``role="tool"`` results from earlier rounds — they are passed through
        verbatim. ``tool_choice="none"`` keeps the earlier tool turns valid in the
        request but forbids NEW calls (the agent loop's round-cap "answer now")."""
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            tools=tools, tool_choice=tool_choice,
            **_reasoning_body(reasoning_effort),
        )
        msg = resp.choices[0].message
        return ToolCompletion(
            text=msg.content or "",
            tool_calls=_openai_tool_calls(getattr(msg, "tool_calls", None)),
            usage=_openai_usage(resp),
        )

    def stream_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Streaming twin of :meth:`complete_tools`.

        Yields TEXT deltas as they arrive (a pure tool-call round yields nothing)
        and RETURNS the final :class:`ToolCompletion` via ``StopIteration.value``,
        with the accumulated tool calls, full text and authoritative usage. The
        provider streams tool calls as indexed fragments (id/name once, arguments
        in pieces), which are reassembled here. ``usage_sink`` is kept mirrored
        like :meth:`stream` so a torn-down round still bills what it produced."""
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            tools=tools, tool_choice=tool_choice, stream=True,
            stream_options={"include_usage": True},
            **_reasoning_body(reasoning_effort),
        )
        usage = Usage()
        parts: list[str] = []
        pending: dict[int, dict] = {}
        try:
            for chunk in resp:
                if getattr(chunk, "usage", None):
                    usage = _openai_usage(chunk)
                    _mirror_usage(usage_sink, usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                _accumulate_tool_deltas(pending, getattr(delta, "tool_calls", None))
                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
                    yield text
            return ToolCompletion(
                text="".join(parts),
                tool_calls=_assemble_tool_calls(pending),
                usage=usage,
            )
        finally:
            # Deterministic HTTP-stream close, incl. on GeneratorExit (same reason
            # as stream()).
            close = getattr(resp, "close", None)
            if callable(close):
                close()

    # ---- async twins (PLAN §5b) — same requests, awaited on the event loop ----
    async def acomplete(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> Completion:
        full = [{"role": "system", "content": system}, *messages]
        resp = await self._async_client().chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            **_reasoning_body(reasoning_effort),
        )
        return Completion(
            text=resp.choices[0].message.content or "",
            usage=_openai_usage(resp),
        )

    async def acomplete_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> ToolCompletion:
        """Async twin of :meth:`complete_tools` — same request, awaited."""
        full = [{"role": "system", "content": system}, *messages]
        resp = await self._async_client().chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            tools=tools, tool_choice=tool_choice,
            **_reasoning_body(reasoning_effort),
        )
        msg = resp.choices[0].message
        return ToolCompletion(
            text=msg.content or "",
            tool_calls=_openai_tool_calls(getattr(msg, "tool_calls", None)),
            usage=_openai_usage(resp),
        )

    async def astream_tools(
        self, system: str, messages: list[dict], tools: list[dict], *,
        tool_choice: str = "auto",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        completion_sink: ToolCompletion | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        """Async twin of :meth:`stream_tools`. An async generator cannot
        ``return`` a value, so the final result is delivered by filling the
        caller-owned ``completion_sink`` IN PLACE (text/tool_calls/usage) —
        the same convention ``usage_sink`` already uses for usage."""
        full = [{"role": "system", "content": system}, *messages]
        resp = await self._async_client().chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full,
            tools=tools, tool_choice=tool_choice, stream=True,
            stream_options={"include_usage": True},
            **_reasoning_body(reasoning_effort),
        )
        usage = Usage()
        parts: list[str] = []
        pending: dict[int, dict] = {}
        try:
            async for chunk in resp:
                if getattr(chunk, "usage", None):
                    usage = _openai_usage(chunk)
                    _mirror_usage(usage_sink, usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                _accumulate_tool_deltas(pending, getattr(delta, "tool_calls", None))
                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
                    yield text
        finally:
            # Fill the sink EVEN on a mid-stream teardown so the caller sees the
            # partial text/usage gathered so far, then close deterministically.
            if completion_sink is not None:
                completion_sink.text = "".join(parts)
                completion_sink.tool_calls = _assemble_tool_calls(pending)
                completion_sink.usage = usage
            close = getattr(resp, "close", None)
            if callable(close):
                await close()

    async def astream(
        self, system: str, messages: list[dict],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        usage_sink: Usage | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        full = [{"role": "system", "content": system}, *messages]
        resp = await self._async_client().chat.completions.create(
            model=self.model, max_tokens=max_tokens, messages=full, stream=True,
            stream_options={"include_usage": True},
            **_reasoning_body(reasoning_effort),
        )
        try:
            async for chunk in resp:
                if getattr(chunk, "usage", None):
                    # Final usage travels via usage_sink only — an async generator
                    # cannot ``return`` a value (see the LLM protocol note).
                    _mirror_usage(usage_sink, _openai_usage(chunk))
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                text = getattr(choices[0].delta, "content", None)
                if text:
                    yield text
        finally:
            # Close the underlying HTTP stream deterministically (same reason as the
            # sync path): a disconnect tears this generator down mid-iteration.
            close = getattr(resp, "close", None)
            if callable(close):
                await close()

    async def acomplete_json(
        self, system: str, messages: list[dict], schema: dict, *,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str | None = None,
    ) -> JsonCompletion:
        # Same strict json_schema request + malformed-reply guards as complete_json.
        full = [{"role": "system", "content": system}, *messages]
        usage = Usage()  # captured before parse so a mid-parse failure still bills
        try:
            resp = await self._async_client().chat.completions.create(
                model=self.model, max_tokens=max_tokens, messages=full,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name,
                                    "schema": _strict_schema(schema),
                                    "strict": True},
                },
                **_reasoning_body(reasoning_effort),
            )
            usage = _openai_usage(resp)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                return JsonCompletion(data={}, usage=usage, ok=False)
            raw = msg.content or ""
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict) or not data:
                return JsonCompletion(data={}, usage=usage, ok=False)
            return JsonCompletion(data=data, usage=usage, ok=True)
        except Exception:  # noqa: BLE001 - provider/parse failure => clean fallback
            return JsonCompletion(data={}, usage=usage, ok=False)


def _openai_tool_calls(raw) -> list[ToolCallRequest]:
    """Normalize a non-stream response's ``message.tool_calls`` (or None)."""
    calls: list[ToolCallRequest] = []
    for i, tc in enumerate(raw or []):
        fn = getattr(tc, "function", None)
        calls.append(ToolCallRequest(
            id=getattr(tc, "id", None) or f"call_{i}",
            name=getattr(fn, "name", None) or "",
            arguments=getattr(fn, "arguments", None) or "",
        ))
    return calls


def _accumulate_tool_deltas(pending: dict[int, dict], deltas) -> None:
    """Fold one streamed chunk's tool-call fragments into ``pending`` (by index).

    Providers stream a tool call as indexed fragments: the id and name arrive
    once, the JSON ``arguments`` in pieces — each piece is appended here."""
    for tc in deltas or []:
        slot = pending.setdefault(
            getattr(tc, "index", 0) or 0, {"id": "", "name": "", "arguments": ""},
        )
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments


def _assemble_tool_calls(pending: dict[int, dict]) -> list[ToolCallRequest]:
    """Finalize accumulated stream fragments into ordered ToolCallRequests."""
    return [
        ToolCallRequest(
            id=slot["id"] or f"call_{i}",
            name=slot["name"],
            arguments=slot["arguments"],
        )
        for i, slot in sorted(pending.items())
    ]


def _openai_usage(resp) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
    )


def make_llm(cfg: Config) -> LLM:
    from .config import test_mode

    if test_mode():
        from .testmode import FakeLLM

        return FakeLLM()
    if cfg.llm_provider == "anthropic":
        return AnthropicLLM(cfg.anthropic_api_key, cfg.llm_model)
    # Reasoning is applied PER CALL by rag.Assistant (hardness-gated), not baked into
    # the client here, so the router/smalltalk/simple paths stay reasoning-free.
    return OpenAICompatLLM(
        cfg.llm_key(), cfg.llm_model, cfg.openai_compat_base_url(cfg.llm_provider),
    )


def make_router_llm(cfg: Config) -> LLM:
    """LLM for the pre-answer understanding/router call (BUG-029 item 4).

    Same provider and key as :func:`make_llm`, but uses ``cfg.router_model`` when set so
    the router — a cheap classify+rewrite-to-JSON call that runs BEFORE the first token
    on EVERY turn — can be a small, fast model (e.g. gemini-3.5-flash-lite, ~0.5s and
    steady) instead of the full answer model (~2.4s and highly variable). Empty
    ``router_model`` falls back to the answer model, i.e. today's behavior unchanged."""
    from .config import test_mode

    if test_mode():
        from .testmode import FakeLLM

        return FakeLLM()
    model = (cfg.router_model or "").strip() or cfg.llm_model
    if cfg.llm_provider == "anthropic":
        return AnthropicLLM(cfg.anthropic_api_key, model)
    return OpenAICompatLLM(
        cfg.llm_key(), model, cfg.openai_compat_base_url(cfg.llm_provider),
    )


class OpenAICompatVisionLLM:
    """Multimodal chat via OpenAI-compatible image_url parts (OpenRouter Sonnet, OpenAI gpt-4o)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        # Reuse a warm, pooled client across requests (BUG-029) instead of a fresh
        # cold-handshake client per Assistant.
        self._client = shared_openai_client(api_key, base_url)
        self.model = model

    def complete_with_image(
        self,
        system: str,
        prompt: str,
        image_bytes: bytes,
        image_mime: str,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        history: list[dict] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict | None = None,
        schema_name: str = "result",
    ) -> Completion:
        """Answer a single-image turn. ``history`` (optional) is prepended as the
        prior dialogue window so a multimodal turn can reason over the conversation
        (the «Мислення» v2 vision path); it is None for the stateless grounded path,
        which then sends only the image turn — byte-for-byte as before.
        ``reasoning_effort`` behaves as in :meth:`OpenAICompatLLM.complete`
        (OpenRouter-only, ignored elsewhere) — None keeps the plain call unchanged.

        ``json_schema`` (optional) switches the provider into strict Structured-Outputs
        mode (``response_format`` json_schema — the SAME seam as
        :meth:`OpenAICompatLLM.complete_json`), so ``text`` comes back as a JSON object
        string matching the schema. Used by the «Мислення» v2 image path to get a clean
        ``{answer, image_facts}`` in ONE non-streaming call (no marker to scrape, no leak
        risk). None keeps the plain free-text call unchanged."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
        ]
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": content})
        # Strict json_schema requires additionalProperties:false + a complete
        # ``required`` on every object (see _strict_schema / complete_json).
        structured = (
            {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name,
                                "schema": _strict_schema(json_schema),
                                "strict": True}}}
            if json_schema is not None else {}
        )
        if json_schema is None:
            # Plain free-text image turn — byte-for-byte unchanged: no refusal check,
            # no degrade wrapper, a provider error propagates exactly as before.
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                **_reasoning_body(reasoning_effort),
            )
            return Completion(
                text=resp.choices[0].message.content or "",
                usage=_openai_usage(resp),
            )
        # Structured (strict json_schema) path — mirror complete_json's graceful degrade
        # (#62): a safety refusal, empty content, OR a provider that REJECTS strict
        # json_schema must NOT raise or leak. Return EMPTY content (usage still billed)
        # so ``_parse_image_reply`` turns it into the clean localized fallback instead of
        # "(no answer)" / a blank memory turn.
        usage = Usage()  # captured before parse so a mid-call failure still bills
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                **structured,
                **_reasoning_body(reasoning_effort),
            )
            usage = _openai_usage(resp)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                return Completion(text="", usage=usage)
            return Completion(text=msg.content or "", usage=usage)
        except Exception:  # noqa: BLE001 - provider/parse failure => clean fallback
            return Completion(text="", usage=usage)


def make_vision_llm(cfg: Config) -> OpenAICompatVisionLLM:
    return OpenAICompatVisionLLM(
        cfg.vision_key(), cfg.vision_model, cfg.openai_compat_base_url(cfg.vision_provider)
    )
