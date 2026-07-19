"""TUI wiring for the agent-writable `memory` tool (redesign Phase 3)."""

import json

import pytest

from hpca.llm import ChatResponse
from hpca.profiles import Profile
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


ADD_OP = [{"op": "add", "tier": 2, "text": "The user prefers R over Python.", "match": ""}]


class TestApprovalGate:
    async def test_approved_batch_is_saved(self, hpca_home):
        llm = FakeLLM([memory_json(ADD_OP), respond_json("noted")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert len(memories) == 1
            assert memories[0].text == "The user prefers R over Python."
            assert memories[0].tier == 2

    async def test_rejected_batch_is_not_saved(self, hpca_home):
        llm = FakeLLM([memory_json(ADD_OP), respond_json("ok, not saving")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R", expect_modal=True)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []
            # the model is told plainly, so it does not claim it saved
            assert any(
                "rejected" in str(m.get("content", ""))
                for call in llm.calls
                for m in call
            )


class TestBudgetFeedback:
    async def test_full_tier_returns_inventory_not_a_write(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 3190, tier=2)
        profile.save()
        llm = FakeLLM([memory_json(ADD_OP), respond_json("memory is full")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R")
            # no approval dialog: the batch never got that far
            assert len(Profile.load("default").memories) == 1
            tool_results = [
                str(m.get("content", ""))
                for call in llm.calls
                for m in call
                if "Memory unchanged" in str(m.get("content", ""))
            ]
            assert tool_results
            assert "Tier 2 is full" in tool_results[0]

    async def test_batch_can_free_room_and_add_in_one_call(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("y" * 3190, tier=2)
        profile.save()
        operations = [
            {"op": "remove", "tier": 2, "match": "y" * 40, "text": ""},
            {"op": "add", "tier": 2, "text": "The user prefers R.", "match": ""},
        ]
        llm = FakeLLM([memory_json(operations), respond_json("condensed")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "condense my memory", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert [m.text for m in memories] == ["The user prefers R."]


class TestDriftGuard:
    async def test_hand_edit_during_the_turn_is_not_clobbered(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Cluster is cubi.", tier=1)
        profile.save()

        llm = FakeLLM([memory_json(ADD_OP), respond_json("could not save")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I prefer R", expect_modal=True)
            assert isinstance(app.screen, MemoryBatchScreen)
            # the user edits the file by hand while the dialog is up
            edited = Profile.load("default")
            edited.add_memory("Scratch is on /fast.", tier=1)
            edited.save()
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = [m.text for m in Profile.load("default").memories]
            assert "Scratch is on /fast." in texts  # the hand edit survived
            assert "The user prefers R over Python." not in texts
            backups = list((hpca_home / "profiles").glob("default.bak.*"))
            assert backups
            # the model is told why, rather than left thinking it saved
            assert any(
                "changed on disk" in str(m.get("content", ""))
                for call in llm.calls
                for m in call
            )


class TestGuidance:
    async def test_memory_guidance_in_prompt(self, hpca_home):
        llm = FakeLLM([respond_json("hi")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            system = llm.calls[0][0]["content"]
            assert "declarative FACTS" in system
            assert "stale in a week" in system
