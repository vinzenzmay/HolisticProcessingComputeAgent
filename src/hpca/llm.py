"""OpenAI-compatible chat client (§2): one code path for vLLM, llama.cpp, …

Design notes for the small-model reality (§1):

* The backend model may be a *reasoning* model (e.g. Qwen3.6). vLLM exposes
  thinking tokens in a separate ``reasoning`` field, which the TUI shows in
  the collapsible thinking box. Whether to think at all — and, on a model that
  grades it, how hard — is decided by the caller per request and comes from
  the session's effort level (hpca.thinking); ``llm.enable_thinking`` is only
  the fallback when nobody says. Thinking buys better decisions for roughly
  15x the latency and 15x the completion tokens (measured on Qwen3.6-27B).
* Constrained decoding (``response_format`` with a JSON schema) makes tool
  calls syntactically valid *by construction*. Support is probed once against
  the backend when settings say ``auto``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from hpca.config import LLMSettings

Message = dict[str, Any]

# A message may carry an API-only sidecar: recalled memory context (redesign
# Phase 1) is appended to what the *model* sees without polluting the stored
# transcript. The checkpointer round-trips the key untouched; only the wire
# encoding below substitutes it for the content.
API_CONTENT_KEY = "api_content"

# When a message was added to the thread, ISO-8601 UTC. Stamped once, by the
# reducer that appends it (`hpca.agent.graph`), so the chat can say when each
# turn happened and a session can say when it was last worked in.
#
# On the message rather than in a record anchored to its index — the way
# reasoning and tool calls are kept — because a stamp is *of* the message
# rather than about the turn around it, and because an anchored record would
# have to be trimmed in step with every rollback and fork that moves a message.
# Safe there because `wire_messages` below whitelists what a backend sees: an
# extra key on a stored message cannot reach a model.
STAMP_KEY = "at"


# Keys the native tool-calling protocol needs on the wire: the assistant's
# calls, and the id + name that tie a tool-role result back to one of them.
# Passed through only when present, so an envelope-protocol history still
# sends exactly role and content.
TOOL_KEYS = ("tool_calls", "tool_call_id", "name")


def wire_messages(messages: list[Message]) -> list[Message]:
    """Messages as sent to the backend: sidecar applied, extras dropped.

    Backends vary in how strictly they validate message objects, so only
    ``role``, ``content`` and the tool-protocol keys go on the wire.
    """
    wire = []
    for message in messages:
        out = {
            "role": message["role"],
            "content": message.get(API_CONTENT_KEY) or message["content"],
        }
        for key in TOOL_KEYS:
            if message.get(key) is not None:
                out[key] = message[key]
        wire.append(out)
    return wire

# Read timeout for a request that asked the model to think (LLMClient._timeout_for).
#
# Sized off the ceiling rather than off an average, because the ceiling is
# knowable: a generation is bounded by max_tokens, so the longest a request can
# take is max_tokens ÷ the backend's decode rate. The agent's own cap is
# MAX_DECISION_TOKENS (4096), and the slowest rate measured on the cluster's
# Qwen3.8-27B-FP8 is 17 tok/s (a loaded box: 4039 completion tokens in 233.6s,
# stopping just short of the cap) — so ~240s is what a decision that runs to
# the cap costs there. 600s is 2.5x that: the headroom for a busier hour on a
# shared GPU, and still short enough that a genuinely dead backend is noticed
# within the session rather than at the end of the day.
#
# Erring long is the cheap direction. A deadline that fires has already spent
# every one of those seconds and thrown the generation away.
THINKING_TIMEOUT_S = 600

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class LLMError(Exception):
    """Any failure talking to the LLM backend."""


class TruncatedOutput(LLMError):
    """Structured output stopped at max_tokens, so there is nothing to parse.

    Its own type because the caller can act on this one, unlike an HTTP 500.
    Two very different things share the shape: constrained decoding looping
    (unbounded digit runs), and a call that was simply too long to finish —
    which is what writing a file's content into a tool call looks like when it
    does not fit. Both were seen live on the 27B; the caller decides what to
    do about it. ``partial`` carries the whole cut-off generation, because a
    too-long file write is salvageable: everything up to the last complete
    line is real work (hpca.agent.middleware._salvage_truncated_call), and
    without the fragment the only option is to throw the tokens away and
    retry.
    """

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial


@dataclass
class ChatResponse:
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # Native tool calls, when the request offered tools and the model used
    # them. Empty under the envelope protocol, where the call is in ``content``.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamDelta:
    content: str = ""
    reasoning: str = ""


class LLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10),
            transport=transport,
        )
        self._constrained_supported: bool | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------- requests

    def _thinking(self, enable_thinking: bool | None) -> bool:
        """Whether this request will actually think: what the caller asked for,
        else the client's setting. One resolution, read by both the payload and
        the deadline below — they must not disagree about it."""
        if enable_thinking is None:
            return self._settings.enable_thinking
        return enable_thinking

    def _timeout_for(self, enable_thinking: bool | None) -> httpx.Timeout:
        """The read timeout for one request: longer while the model is thinking.

        ``request_timeout_s`` (120) sizes a non-thinking request and is not
        moved, because it is also what makes a wedged backend fail fast. It
        does not size a thinking one: measured against Qwen3.8-27B-FP8 on a
        loaded box, one real agent decision on a hard question took 74s with
        thinking off and MEASURE_PLACEHOLDER — so a thinking turn would have
        died on an httpx read timeout, mid-generation, having already burned
        every one of those seconds.

        Keyed on thinking rather than on the effort level, so the backend that
        was already slow before the dial existed — a reasoning model with
        ``llm.enable_thinking`` on and no level chosen — gets the same
        headroom instead of the deadline that never fitted it.

        ``max`` rather than a plain constant, so a site that raised
        ``request_timeout_s`` past this still gets what it configured.
        """
        if not self._thinking(enable_thinking):
            return self._client.timeout
        return httpx.Timeout(
            max(self._settings.request_timeout_s, THINKING_TIMEOUT_S), connect=10
        )

    def _payload(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None,
        schema_name: str,
        max_tokens: int | None,
        temperature: float | None,
        enable_thinking: bool | None,
        stream: bool,
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": wire_messages(messages),
            "chat_template_kwargs": {
                "enable_thinking": self._thinking(enable_thinking)
            },
        }
        if reasoning_effort is not None:
            # Top level rather than inside chat_template_kwargs, although vLLM
            # accepts it in both places (measured on Qwen3.8-27B-FP8, both
            # produce the same injected prefix). Top level is the OpenAI
            # parameter, so vLLM validates it against the model's own enum and
            # a wrong level comes back as a 400 naming the accepted ones; the
            # same string smuggled through chat_template_kwargs is handed to
            # the Jinja template unchecked, where a typo means "no effort
            # prefix" and looks exactly like it worked. There is no per-session
            # default to fall back on here — the client is shared by every
            # session on this backend — so the field appears only when the
            # caller asks for it, and a backend that has never heard of
            # reasoning_effort sees the request it always saw.
            payload["reasoning_effort"] = reasoning_effort
        if tools is not None:
            # Native protocol: the backend's parser produces the call, so
            # there is no envelope to constrain — the two are alternatives,
            # and vLLM rejects a response_format sent alongside tools.
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        elif json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if stream:
            payload["stream"] = True
        # Last, so a backend entry can correct anything decided above — which
        # is the point of it: `chat_template_kwargs.enable_thinking` is a vLLM
        # extension, and a server that ignores it needs its own spelling of
        # the same instruction (`config.ExtraBody`). It cannot reach `model`,
        # `messages`, `stream` or `tools`; the settings model refuses those at
        # load, so nothing here has to defend them.
        payload.update(self._settings.extra_body)
        return payload

    async def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        schema_name: str = "output",
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        payload = self._payload(
            messages,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            stream=False,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "chat/completions",
                json=payload,
                timeout=self._timeout_for(enable_thinking),
            )
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise LLMError(
                f"LLM request failed ({response.status_code}): {response.text[:500]}"
            )
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        truncated = choice.get("finish_reason") == "length"
        if tools is not None and truncated and not message.get("tool_calls"):
            # Native protocol: the backend's parser emits nothing when the
            # call was cut off mid-arguments, so there is no fragment to
            # salvage — only the feedback-and-retry path applies.
            raise TruncatedOutput(
                "Tool call truncated at max_tokens: "
                f"{message.get('content') or '':.120}",
                partial=message.get("content") or "",
            )
        if json_schema is not None and truncated:
            # Constrained decoding can loop (e.g. unbounded digit runs) until
            # max_tokens; the truncated output cannot be valid JSON.
            raise TruncatedOutput(
                "Structured output truncated at max_tokens "
                f"(looping, or too long to finish?): "
                f"{message.get('content') or '':.120}",
                partial=message.get("content") or "",
            )
        # "request_seconds" is ours, not the backend's: OpenAI-style bodies
        # carry token counts but no timing, and without streaming the wall
        # clock around the request is the only speed measurement there is.
        # It rides in usage so the existing per-decision reporting (decide ->
        # on_usage -> context bar, and the chat logs) carries it for free.
        usage = dict(data.get("usage") or {})
        usage["request_seconds"] = elapsed
        return ChatResponse(
            content=message.get("content") or "",
            reasoning=message.get("reasoning"),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            tool_calls=list(message.get("tool_calls") or []),
        )

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[StreamDelta]:
        payload = self._payload(
            messages,
            json_schema=None,
            schema_name="output",
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            stream=True,
        )
        try:
            async with self._client.stream(
                "POST", "chat/completions", json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    raise LLMError(
                        f"LLM request failed ({response.status_code}): {body[:500]}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    yield StreamDelta(
                        content=delta.get("content") or "",
                        reasoning=delta.get("reasoning") or "",
                    )
        except httpx.HTTPError as e:
            raise LLMError(f"LLM stream failed: {e}") from e

    # ------------------------------------------------------------ discovery

    async def _model_entries(self) -> list[dict]:
        try:
            response = await self._client.get("models")
        except httpx.HTTPError as e:
            raise LLMError(f"Listing models failed: {e}") from e
        if response.status_code != 200:
            raise LLMError(
                f"Listing models failed ({response.status_code}): {response.text[:500]}"
            )
        return response.json().get("data", [])

    async def models(self) -> list[str]:
        return [entry["id"] for entry in await self._model_entries()]

    async def context_window(self) -> int | None:
        """The served model's context length, if the backend advertises it.

        vLLM puts ``max_model_len`` on each /v1/models entry, which is the
        honest number: it reflects how the server was actually launched, not
        what the model card claims. Returns None when the backend does not
        say, and the caller falls back to the configured value.
        """
        try:
            entries = await self._model_entries()
        except LLMError:
            return None
        for entry in entries:
            if entry.get("id") == self._settings.model:
                window = entry.get("max_model_len")
                return int(window) if window else None
        return None

    def uses_native_tools(self) -> bool:
        """Whether calls travel on the backend's own tool-calling channel.

        A setting, not a probe: unlike constrained decoding there is nothing
        to fall back to mid-session — the two protocols shape the whole
        conversation, not one request — so the choice is made once, up front.
        """
        return self._settings.tool_protocol == "native"

    async def supports_constrained_decoding(self) -> bool:
        """Whether tool calls can use JSON-schema constrained decoding (§2).

        Honors the ``constrained_decoding`` setting; ``auto`` probes the
        backend once with a trivial schema and caches the verdict.
        """
        if self._settings.constrained_decoding == "on":
            return True
        if self._settings.constrained_decoding == "off":
            return False
        if self._constrained_supported is None:
            try:
                await self.chat(
                    [{"role": "user", "content": 'Return {"ok": true}'}],
                    json_schema=PROBE_SCHEMA,
                    schema_name="probe",
                    max_tokens=20,
                    # Never think here: reasoning would blow the 20-token cap,
                    # and a truncation error reads as "unsupported".
                    enable_thinking=False,
                )
                self._constrained_supported = True
            except LLMError:
                self._constrained_supported = False
        return self._constrained_supported
