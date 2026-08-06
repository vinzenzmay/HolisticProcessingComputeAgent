"""OpenAI-compatible chat client (§2): one code path for vLLM, llama.cpp, Ollama, …

Design notes for the small-model reality (§1):

* The backend model may be a *reasoning* model (e.g. Qwen3.6). vLLM exposes
  thinking tokens in a separate ``reasoning`` field, which the TUI shows in
  the collapsible thinking box. Whether to think at all is the
  ``llm.enable_thinking`` setting — it buys better decisions for roughly 15x
  the latency and 15x the completion tokens (measured on Qwen3.6-27B) — and
  callers can override it per request.
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


def wire_messages(messages: list[Message]) -> list[Message]:
    """Messages as sent to the backend: sidecar applied, extras dropped.

    Backends vary in how strictly they validate message objects, so only
    ``role`` and ``content`` go on the wire.
    """
    return [
        {
            "role": message["role"],
            "content": message.get(API_CONTENT_KEY) or message["content"],
        }
        for message in messages
    ]

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
    does not fit. Measured on the live 27B backend with thinking on, a 30-line
    document costs ~1200 completion tokens, so a document of a few hundred
    lines reaches any sane cap honestly.
    """


@dataclass
class ChatResponse:
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


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
    ) -> dict[str, Any]:
        if enable_thinking is None:
            enable_thinking = self._settings.enable_thinking
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": wire_messages(messages),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if json_schema is not None:
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
    ) -> ChatResponse:
        payload = self._payload(
            messages,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            stream=False,
        )
        started = time.perf_counter()
        try:
            response = await self._client.post("chat/completions", json=payload)
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
        if json_schema is not None and choice.get("finish_reason") == "length":
            # Constrained decoding can loop (e.g. unbounded digit runs) until
            # max_tokens; the truncated output cannot be valid JSON.
            raise TruncatedOutput(
                "Structured output truncated at max_tokens "
                f"(looping, or too long to finish?): "
                f"{message.get('content') or '':.120}"
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
