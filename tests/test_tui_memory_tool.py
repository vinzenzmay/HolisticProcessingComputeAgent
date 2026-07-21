"""TUI wiring for the deferred `memory` tool.

The agent flags durable facts mid-conversation, but nothing is written then:
the flagged batch is queued and reviewed together at the next /conclude.
"""

import json

import pytest

from hpca.llm import ChatResponse
from hpca.profiles import MemoryScope, Profile
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.memory_screens import MemoryBatchScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.calls.append(list(messages))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def memory_json(operations):
    return json.dumps(
        {
            "action": "tool_call",
            "tool": "memory",
            "arguments": {"operations": operations},
        }
    )


def no_reflections():
    """A /conclude self-review that proposes nothing, so the drain of the
    flagged batch is the only thing left to review."""
    return json.dumps({"proposals": []})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def submit(app, pilot, text, *, expect_modal=False):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    if expect_modal:
        for _ in range(20):
            await pilot.pause()
    else:
        await app.workers.wait_for_complete()
        await pilot.pause()


ADD_OP = [
    {"op": "add", "scope": "system-prompt", "text": "The user prefers R over Python.", "match": ""}
]


class TestDeferredFlagging:
    async def test_flagging_writes_nothing_until_conclude(self, hpca_home):
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            # no approval modal appears mid-conversation
            assert not isinstance(app.screen, MemoryBatchScreen)
            # and nothing is saved yet
            assert Profile.load("default").memories == []
            # the model is told the fact was queued for /conclude, not saved
            assert any(
                "conclude" in str(m.get("content", "")).lower()
                for call in llm.calls
                for m in call
            )

    async def test_conclude_reviews_and_saves_flagged(self, hpca_home):
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted"), no_reflections()])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert [m.text for m in memories] == ["The user prefers R over Python."]
            assert memories[0].scope is MemoryScope.SYSTEM_PROMPT

    async def test_conclude_rejecting_flagged_saves_nothing(self, hpca_home):
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted"), no_reflections()])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []


class TestBudgetFeedback:
    async def test_full_scope_rejects_the_flagged_batch(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x " * 6000, scope=MemoryScope.SYSTEM_PROMPT)  # over 2400 tokens
        profile.save()
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted"), no_reflections()])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            await submit(app, pilot, "/conclude")
            # over budget: the drain never reaches an approval dialog
            assert not isinstance(app.screen, MemoryBatchScreen)
            assert len(Profile.load("default").memories) == 1

    async def test_batch_can_free_room_and_add_in_one_call(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("y " * 6000, scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        operations = [
            {"op": "remove", "scope": "system-prompt", "match": "y y y y", "text": ""},
            {"op": "add", "scope": "system-prompt", "text": "The user prefers R.", "match": ""},
        ]
        llm = FakeLLM(
            [memory_json(operations), respond_json("condensed"), no_reflections()]
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "condense my memory")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert [m.text for m in memories] == ["The user prefers R."]


class TestDriftGuard:
    async def test_hand_edit_during_conclude_is_not_clobbered(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Cluster is cubi.", scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted"), no_reflections()])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            # the user edits the file by hand while the dialog is up
            edited = Profile.load("default")
            edited.add_memory("Scratch is on /fast.", scope=MemoryScope.SYSTEM_PROMPT)
            edited.save()
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = [m.text for m in Profile.load("default").memories]
            assert "Scratch is on /fast." in texts  # the hand edit survived
            assert "The user prefers R over Python." not in texts
            backups = list((hpca_home / "profiles").glob("default.bak.*"))
            assert backups


class TestGuidance:
    async def test_memory_guidance_in_prompt(self, hpca_home):
        llm = FakeLLM([respond_json("hi")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            system = llm.calls[0][0]["content"]
            assert "declarative FACTS" in system
            assert "stale in a week" in system
            # the guidance now tells the model it flags for /conclude, not writes
            assert "conclude" in system.lower()


class TestDecisionSchemaIsSendable:
    """The schema the app actually sends must be self-contained.

    This is the shape that failed in the field: the first tool with a nested
    model made every turn fail at the backend with
    "Grammar error: Pointer '/$defs/MemoryOperation' does not exist",
    because the tool's $defs do not survive being embedded in the envelope.
    A FakeLLM ignores json_schema, so no behavioural test can catch it.
    """

    async def test_app_registry_produces_a_schema_with_no_dangling_pointers(
        self, hpca_home
    ):
        from hpca.agent.middleware import decision_schema

        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            blob = json.dumps(decision_schema(app._tools))
            assert "$ref" not in blob
            assert "$defs" not in blob

    async def test_a_turn_runs_with_the_memory_tool_registered(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("hello")]))
        async with app.run_test(size=(120, 40)) as pilot:
            assert "memory" in app._tools.names()
            await submit(app, pilot, "hi")
            assert any("hello" == t for t in app.chat_log_texts())
