"""The thinking-effort dial: the vocabulary, and what each level puts on the wire.

The mapping is the whole point of the feature, and one of its four cases is
asymmetric: ``off`` must send NO ``reasoning_effort`` at all, so a backend that
has never heard of the parameter behaves exactly as it did before the dial
existed. The levels themselves are not ours to choose — vLLM validates
``reasoning_effort`` against the model's own enum, and Qwen3.8 answers a level
outside it with a 400 (*"Supported types are xhigh (default), medium, and
low"*), which is why "high" is tested as absent rather than as a fourth level.
"""

from __future__ import annotations

import httpx
import pytest

from hpca.agent.middleware import decide
from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import LLMSettings, Settings
from hpca.llm import THINKING_TIMEOUT_S, LLMClient
from hpca.thinking import (
    EFFORT_HINTS,
    EFFORTS,
    normalize_effort,
    wire_thinking,
)
from pydantic import BaseModel


class TestVocabulary:
    def test_the_four_levels(self):
        assert EFFORTS == ("off", "low", "medium", "xhigh")

    def test_high_is_not_a_level(self):
        # The server rejects it outright; accepting it here would turn a typo
        # into a 400 on a real turn instead of a quiet fallback.
        assert "high" not in EFFORTS
        assert normalize_effort("high") == "off"

    def test_unknown_and_empty_fall_back_to_off(self):
        # A session stores "" for "use the configured default", and settings
        # are hand-edited; neither may reach the wire.
        assert normalize_effort("") == "off"
        assert normalize_effort(None) == "off"
        assert normalize_effort("XHIGH") == "off"

    def test_every_level_carries_a_hint(self):
        # The chooser indexes the dict directly (`overlays/thinking.py`), so a
        # level without one is a KeyError on opening the screen.
        assert set(EFFORT_HINTS) == set(EFFORTS)
        assert all(EFFORT_HINTS[level] for level in EFFORTS)

    def test_no_level_is_flagged_as_unusable(self):
        # xhigh was once announced as not working. It does work, and the hints
        # say what each level does rather than warning off one of them.
        assert not any(
            "NOT USABLE" in hint or "does not work" in hint
            for hint in EFFORT_HINTS.values()
        )


class TestWireMapping:
    def test_off_disables_thinking_and_sends_no_level(self):
        assert wire_thinking("off") == (False, None)

    def test_every_other_level_enables_thinking_and_sends_itself(self):
        for level in ("low", "medium", "xhigh"):
            assert wire_thinking(level) == (True, level)

    def test_an_unknown_level_is_off_rather_than_a_400(self):
        assert wire_thinking("turbo") == (False, None)


def _client(seen: dict, **overrides) -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.clear()
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    settings = LLMSettings(base_url="http://x/v1", model="m", **overrides)
    return LLMClient(settings, transport=httpx.MockTransport(handler))


class TestPayload:
    @pytest.mark.asyncio
    async def test_the_level_rides_top_level_not_in_template_kwargs(self):
        # Top level is the OpenAI parameter, so vLLM validates it against the
        # model's enum; the same string inside chat_template_kwargs is passed
        # to the Jinja template unchecked, where a typo silently means "no
        # effort" instead of an error.
        seen: dict = {}
        async with _client(seen) as llm:
            await llm.chat([{"role": "user", "content": "q"}],
                           enable_thinking=True, reasoning_effort="medium")
        assert seen["reasoning_effort"] == "medium"
        assert "reasoning_effort" not in seen["chat_template_kwargs"]

    @pytest.mark.asyncio
    async def test_the_field_is_absent_when_no_level_is_given(self):
        seen: dict = {}
        async with _client(seen) as llm:
            await llm.chat([{"role": "user", "content": "q"}])
        assert "reasoning_effort" not in seen

    @pytest.mark.asyncio
    async def test_thinking_still_travels_in_template_kwargs(self):
        # The two are separate switches on the wire: the level says how hard,
        # the template kwarg says whether at all.
        seen: dict = {}
        async with _client(seen) as llm:
            await llm.chat([{"role": "user", "content": "q"}],
                           enable_thinking=True, reasoning_effort="low")
        assert seen["chat_template_kwargs"] == {"enable_thinking": True}


class Params(BaseModel):
    pass


def _tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(name="noop", description="d", params=Params,
             handler=lambda a, c: "ok")
    )
    return registry


class RecordingLLM:
    """A client that records the thinking arguments each decision asked for."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def supports_constrained_decoding(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):
        from hpca.llm import ChatResponse

        self.calls.append(kwargs)
        return ChatResponse(content='{"action": "respond", "response": "hi"}')


class TestDecideCarriesTheLevel:
    @pytest.mark.asyncio
    async def test_off_sends_thinking_false_and_no_level(self):
        llm = RecordingLLM()
        await decide(llm, [{"role": "user", "content": "q"}], _tools(),
                     effort="off")
        assert llm.calls[0]["enable_thinking"] is False
        assert "reasoning_effort" not in llm.calls[0]

    @pytest.mark.asyncio
    async def test_a_level_sends_both(self):
        llm = RecordingLLM()
        await decide(llm, [{"role": "user", "content": "q"}], _tools(),
                     effort="xhigh")
        assert llm.calls[0]["enable_thinking"] is True
        assert llm.calls[0]["reasoning_effort"] == "xhigh"

    @pytest.mark.asyncio
    async def test_no_effort_says_nothing_about_thinking_at_all(self):
        # Every caller predating the dial — and every bare graph in a test —
        # must keep getting the request it always got, so the client's own
        # settings stay in charge.
        llm = RecordingLLM()
        await decide(llm, [{"role": "user", "content": "q"}], _tools())
        assert "enable_thinking" not in llm.calls[0]
        assert "reasoning_effort" not in llm.calls[0]


class TestTimeout:
    def test_a_thinking_request_gets_the_longer_deadline(self):
        # 120s sizes a non-thinking turn and is what makes a wedged backend
        # fail fast; a thinking turn measurably outruns it, so the deadline
        # moves per request rather than for everyone.
        client = _client({})
        assert client._timeout_for(False).read == 120
        assert client._timeout_for(True).read == THINKING_TIMEOUT_S

    def test_a_client_configured_to_think_gets_it_without_a_level(self):
        # The pre-dial case: a reasoning backend with llm.enable_thinking on
        # and no level chosen was always the slow one, and the 120s never
        # fitted it either.
        client = _client({}, enable_thinking=True)
        assert client._timeout_for(None).read == THINKING_TIMEOUT_S

    def test_a_configured_timeout_above_the_floor_still_wins(self):
        client = _client({}, request_timeout_s=THINKING_TIMEOUT_S + 60)
        assert client._timeout_for(True).read == THINKING_TIMEOUT_S + 60


class TestSettingsDefault:
    def test_new_sessions_start_off(self):
        # Preserves exactly what every turn did before the dial existed: any
        # level above off costs a thinking pass on every decision of a turn,
        # and the server's own default once thinking is on is the slowest one.
        assert Settings().agent.default_thinking == "off"

    def test_it_is_configurable(self):
        settings = Settings.model_validate({"agent": {"default_thinking": "low"}})
        assert settings.agent.default_thinking == "low"


class TestGraphResolvesItPerSession:
    """``effort_fn`` is a per-thread_id resolver, like ``mode_fn``: two turns
    running at once must each put their own session's level on the wire, and
    the level is read per round so ``/reasoning`` lands on the next decision
    rather than the next turn."""

    def _graph(self, llm, effort_fn):
        from langgraph.checkpoint.memory import InMemorySaver

        from hpca.agent.graph import build_graph

        return build_graph(
            llm=llm,
            tools=ToolRegistry(),
            checkpointer=InMemorySaver(),
            effort_fn=effort_fn,
        )

    async def _run(self, graph, session_id):
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "q"}]},
            {"configurable": {"thread_id": session_id}},
        )

    @pytest.mark.asyncio
    async def test_each_session_gets_its_own_level(self):
        levels = {"a": "low", "b": "xhigh"}
        llm = RecordingLLM()
        graph = self._graph(llm, lambda sid: levels[sid])
        await self._run(graph, "a")
        await self._run(graph, "b")
        assert llm.calls[0]["reasoning_effort"] == "low"
        assert llm.calls[1]["reasoning_effort"] == "xhigh"

    @pytest.mark.asyncio
    async def test_no_resolver_leaves_the_wire_untouched(self):
        llm = RecordingLLM()
        graph = self._graph(llm, None)
        await self._run(graph, "a")
        assert "enable_thinking" not in llm.calls[0]
        assert "reasoning_effort" not in llm.calls[0]


class TestCoreServiceResolver:
    def test_it_prefers_the_session_then_the_setting(self):
        from hpca.core.service import _effort_for

        class Store:
            def __init__(self, thinking):
                self._thinking = thinking

            def get(self, session_id):
                return type("S", (), {"thinking": self._thinking})()

        settings = Settings()
        settings.agent.default_thinking = "medium"
        assert _effort_for(Store("xhigh"), settings, "s1") == "xhigh"
        assert _effort_for(Store(""), settings, "s1") == "medium"
