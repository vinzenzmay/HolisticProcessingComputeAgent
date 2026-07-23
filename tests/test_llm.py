"""Tests for hpca.llm: OpenAI-compatible client, streaming, constrained decoding.

Unit tests run against a mocked httpx transport. Integration tests
(`-m integration`) run against the live backend configured via
$HPCA_TEST_LLM_URL (default http://localhost:20001/v1) and are skipped when it
is unreachable.
"""

import json
import os

import httpx
import pytest

from hpca.config import LLMSettings
from hpca.llm import ChatResponse, LLMClient, LLMError

# ---------------------------------------------------------------- unit tests


def make_client(handler, **settings_kwargs) -> LLMClient:
    settings = LLMSettings(base_url="http://test/v1", **settings_kwargs)
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def completion_body(content="hi", reasoning=None, finish_reason="stop"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


class TestChat:
    async def test_sends_model_and_messages(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion_body())

        client = make_client(handler, model="test-model")
        await client.chat([{"role": "user", "content": "hi"}])
        assert seen["model"] == "test-model"
        assert seen["messages"] == [{"role": "user", "content": "hi"}]


    async def test_parses_response(self):
        def handler(request):
            return httpx.Response(
                200, json=completion_body(content="answer", reasoning="thoughts")
            )

        client = make_client(handler)
        resp = await client.chat([{"role": "user", "content": "q"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "answer"
        assert resp.reasoning == "thoughts"
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 3

    async def test_null_content_becomes_empty_string(self):
        def handler(request):
            return httpx.Response(200, json=completion_body(content=None))

        client = make_client(handler)
        resp = await client.chat([{"role": "user", "content": "q"}])
        assert resp.content == ""

    async def test_no_auth_header_without_api_key(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=completion_body())

        client = make_client(handler)
        await client.chat([{"role": "user", "content": "q"}])
        assert seen["auth"] is None

    async def test_bearer_header_with_api_key(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=completion_body())

        client = make_client(handler, api_key="sekrit")
        await client.chat([{"role": "user", "content": "q"}])
        assert seen["auth"] == "Bearer sekrit"

    async def test_json_schema_sets_response_format(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion_body(content="{}"))

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        client = make_client(handler)
        await client.chat(
            [{"role": "user", "content": "q"}], json_schema=schema, schema_name="thing"
        )
        assert seen["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "thing", "schema": schema},
        }

    async def test_truncated_structured_output_raises(self):
        # Small models can loop under constrained decoding (e.g. endless digits)
        # until max_tokens; the result is invalid JSON and must fail loudly so
        # retry middleware can act.
        def handler(request):
            return httpx.Response(
                200, json=completion_body(content='{"x": 123', finish_reason="length")
            )

        client = make_client(handler)
        with pytest.raises(LLMError, match="truncat"):
            await client.chat(
                [{"role": "user", "content": "q"}], json_schema={"type": "object"}
            )

    async def test_truncated_plain_text_does_not_raise(self):
        def handler(request):
            return httpx.Response(
                200, json=completion_body(content="partial", finish_reason="length")
            )

        client = make_client(handler)
        resp = await client.chat([{"role": "user", "content": "q"}])
        assert resp.content == "partial"
        assert resp.finish_reason == "length"

    async def test_http_error_raises_llm_error_with_body(self):
        def handler(request):
            return httpx.Response(400, json={"error": {"message": "bad schema"}})

        client = make_client(handler)
        with pytest.raises(LLMError, match="bad schema"):
            await client.chat([{"role": "user", "content": "q"}])

    async def test_connection_error_raises_llm_error(self):
        def handler(request):
            raise httpx.ConnectError("boom")

        client = make_client(handler)
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "q"}])


class TestThinking:
    def payload_of(self, seen, **settings_kwargs):
        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=completion_body())

        return make_client(handler, **settings_kwargs)

    async def test_setting_drives_the_default(self):
        for enabled in (True, False):
            seen = {}
            client = self.payload_of(seen, enable_thinking=enabled)
            await client.chat([{"role": "user", "content": "hi"}])
            assert seen["chat_template_kwargs"] == {"enable_thinking": enabled}

    async def test_per_call_override_wins(self):
        seen = {}
        client = self.payload_of(seen, enable_thinking=True)
        await client.chat([{"role": "user", "content": "hi"}], enable_thinking=False)
        assert seen["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_streaming_follows_the_setting(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, text="data: [DONE]\n")

        client = make_client(handler, enable_thinking=True)
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass
        assert seen["chat_template_kwargs"] == {"enable_thinking": True}

    async def test_capability_probe_never_thinks(self):
        seen = {}
        client = self.payload_of(seen, enable_thinking=True)
        await client.supports_constrained_decoding()
        assert seen["chat_template_kwargs"] == {"enable_thinking": False}

class TestChatStream:
    async def test_yields_content_deltas(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "he"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "hm"}}]},
            {"choices": [{"index": 0, "delta": {"content": "llo"}}]},
        ]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        sse += "data: [DONE]\n\n"

        def handler(request):
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"}
            )

        client = make_client(handler)
        content, reasoning = "", ""
        async for delta in client.chat_stream([{"role": "user", "content": "q"}]):
            content += delta.content
            reasoning += delta.reasoning
        assert content == "hello"
        assert reasoning == "hm"

    async def test_stream_error_raises_llm_error(self):
        def handler(request):
            return httpx.Response(500, content=b"boom")

        client = make_client(handler)
        with pytest.raises(LLMError):
            async for _ in client.chat_stream([{"role": "user", "content": "q"}]):
                pass


class TestConstrainedDecodingProbe:
    async def test_auto_probes_and_caches(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=completion_body(content='{"ok": true}'))

        client = make_client(handler, constrained_decoding="auto")
        assert await client.supports_constrained_decoding() is True
        assert await client.supports_constrained_decoding() is True
        assert len(calls) == 1  # probe result cached

    async def test_auto_detects_unsupported_backend(self):
        def handler(request):
            body = json.loads(request.content)
            if "response_format" in body:
                return httpx.Response(400, json={"error": "unsupported"})
            return httpx.Response(200, json=completion_body())

        client = make_client(handler, constrained_decoding="auto")
        assert await client.supports_constrained_decoding() is False

    async def test_settings_on_skips_probe(self):
        def handler(request):
            raise AssertionError("no request expected")

        client = make_client(handler, constrained_decoding="on")
        assert await client.supports_constrained_decoding() is True

    async def test_settings_off_skips_probe(self):
        def handler(request):
            raise AssertionError("no request expected")

        client = make_client(handler, constrained_decoding="off")
        assert await client.supports_constrained_decoding() is False


class TestModels:
    async def test_lists_model_ids(self):
        def handler(request):
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "a"}, {"id": "b"}]}
            )

        client = make_client(handler)
        assert await client.models() == ["a", "b"]


# --------------------------------------------------------- integration tests

from tests.live_backend import LIVE_KEY, LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@pytest.fixture
def live_client():
    settings = LLMSettings(
            base_url=LIVE_URL,
            model=LIVE_MODEL,
            api_key=LIVE_KEY,
            request_timeout_s=120,
            # these test routing, not reasoning; thinking is ~15x slower
            enable_thinking=False,
        )
    return LLMClient(settings)


@integration
class TestLiveBackend:
    async def test_models(self, live_client):
        assert LIVE_MODEL in await live_client.models()

    async def test_chat(self, live_client):
        resp = await live_client.chat(
            [{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=10
        )
        assert "pong" in resp.content.lower()

    async def test_stream(self, live_client):
        collected = ""
        async for delta in live_client.chat_stream(
            [{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=10
        ):
            collected += delta.content
        assert "pong" in collected.lower()

    async def test_constrained_decoding_supported(self, live_client):
        assert await live_client.supports_constrained_decoding() is True

    async def test_json_schema_output_validates(self, live_client):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        }
        resp = await live_client.chat(
            [{"role": "user", "content": "Invent a person as JSON."}],
            json_schema=schema,
            schema_name="person",
            max_tokens=200,
            temperature=0,
        )
        data = json.loads(resp.content)
        assert isinstance(data["name"], str)
        assert isinstance(data["age"], int)


@integration
class TestLiveThinking:
    """Reasoning is a separate channel that survives constrained decoding.

    Probed on Qwen3.6-27B/vLLM: guided decoding applies to the content after
    the reasoning block, so a thinking model still emits schema-valid JSON.
    The cost is the reason `enable_thinking` is a setting: ~17s and ~340
    completion tokens per decision, against ~1s and ~20 without.
    """

    QUESTION = [
        {
            "role": "user",
            "content": "A cohort has 4 BAMs of 8 GB each. Disk for a 2x copy?",
        }
    ]

    def client(self, **overrides):
        return LLMClient(
            LLMSettings(
                base_url=LIVE_URL, model=LIVE_MODEL, api_key=LIVE_KEY, request_timeout_s=180, **overrides
            )
        )

    async def test_thinking_off_by_setting_returns_no_reasoning(self):
        async with self.client(enable_thinking=False) as llm:
            resp = await llm.chat(self.QUESTION, max_tokens=2000, temperature=0)
            assert not resp.reasoning

    async def test_thinking_on_by_setting_returns_reasoning(self):
        async with self.client(enable_thinking=True) as llm:
            resp = await llm.chat(self.QUESTION, max_tokens=2000, temperature=0)
            assert resp.reasoning
            assert resp.content  # the answer stays in the content channel

    async def test_per_call_override_beats_the_setting(self):
        async with self.client(enable_thinking=True) as llm:
            resp = await llm.chat(
                self.QUESTION, max_tokens=2000, temperature=0, enable_thinking=False
            )
            assert not resp.reasoning

    async def test_reasoning_and_json_schema_coexist(self):
        schema = {
            "type": "object",
            "properties": {"action": {"const": "respond"},
                           "response": {"type": "string"}},
            "required": ["action", "response"],
            "additionalProperties": False,
        }
        async with self.client(enable_thinking=True) as llm:
            resp = await llm.chat(
                self.QUESTION,
                json_schema=schema,
                schema_name="decision",
                max_tokens=4096,
                temperature=0,
            )
            assert resp.reasoning
            assert json.loads(resp.content)["action"] == "respond"

    async def test_capability_probe_never_thinks(self):
        # thinking would blow the probe's 20-token cap and read as "unsupported"
        async with self.client(enable_thinking=True) as llm:
            assert await llm.supports_constrained_decoding() is True
