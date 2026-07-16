"""Tests for the memory workflows: \\memorize, \\conclude, caps, editor (§6.3, §6.4)."""

import json
from contextlib import contextmanager

import pytest
from textual.widgets import Input

from hpca.editor import resolve_editor
from hpca.llm import ChatResponse
from hpca.profiles import Profile
from hpca.tui.app import HpcaApp
from hpca.tui.memory_screens import MemoryProposalScreen, TierSelectScreen


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def proposals_json(*proposals):
    return json.dumps({"proposals": list(proposals)})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def type_and_submit(app, pilot, text):
    chat_input = app.query_one("#chat-input", Input)
    chat_input.focus()
    chat_input.value = text
    await pilot.press("enter")
    await pilot.pause()


class TestMemorize:
    async def test_memorize_opens_tier_modal_then_saves(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\memorize STAR needs 40G here")
            assert isinstance(app.screen, TierSelectScreen)
            await pilot.press("2")
            await pilot.pause()
            loaded = Profile.load("default")
            tier2 = [m for m in loaded.memories if m.tier == 2]
            assert any("STAR needs 40G" in m.text for m in tier2)

    async def test_memorize_tier1(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\memorize Cluster is cubi")
            await pilot.press("1")
            await pilot.pause()
            loaded = Profile.load("default")
            assert loaded.memories[0].tier == 1

    async def test_memorize_escape_saves_nothing(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\memorize forget me")
            await pilot.press("escape")
            await pilot.pause()
            assert Profile.load("default").memories == []

    async def test_no_session_created_by_slash_command(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\memorize something")
            await pilot.press("escape")
            assert app.session_store.list(profile="default") == []

    async def test_unknown_command_is_reported_not_sent(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\frobnicate now")
            assert app.session_store.list(profile="default") == []
            assert app.chat_log_texts() == []


class TestConclude:
    async def test_approve_and_reject_proposals(self, hpca_home):
        llm = FakeLLM(
            [
                respond_json("hi"),
                proposals_json(
                    {"tier": 2, "kind": "learning", "text": "STAR needs 40G."},
                    {"tier": 1, "kind": "fact", "text": "Cluster is cubi."},
                ),
            ]
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "hello")
            await app.workers.wait_for_complete()
            await type_and_submit(app, pilot, r"\conclude")
            await pilot.pause()
            assert isinstance(app.screen, MemoryProposalScreen)
            await pilot.press("y")  # keep the first
            await pilot.pause()
            assert isinstance(app.screen, MemoryProposalScreen)
            await pilot.press("n")  # discard the second
            await app.workers.wait_for_complete()
            await pilot.pause()
            loaded = Profile.load("default")
            assert len(loaded.memories) == 1
            assert loaded.memories[0].text == "STAR needs 40G."
            assert loaded.memories[0].tier == 2
            assert loaded.memories[0].kind == "learning"

    async def test_conclude_without_session_warns(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\conclude")
            assert Profile.load("default").memories == []


class TestMemoryCaps:
    async def test_over_cap_reported(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 8000, tier=2)  # far over the 800-token cap
        profile.save()
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert app.check_memory_caps() == [2]

    async def test_under_cap_quiet(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert app.check_memory_caps() == []


class TestResolveEditor:
    def test_settings_first(self):
        assert resolve_editor("code --wait", {"VISUAL": "vim"}) == ["code", "--wait"]

    def test_visual_then_editor(self):
        assert resolve_editor(None, {"VISUAL": "vim", "EDITOR": "nano"}) == ["vim"]
        assert resolve_editor(None, {"EDITOR": "emacs -nw"}) == ["emacs", "-nw"]

    def test_nano_fallback(self):
        assert resolve_editor(None, {}) == ["nano"]


class TestEditProfileAction:
    async def test_edit_reloads_profile(self, hpca_home, monkeypatch):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            called = {}

            @contextmanager
            def fake_suspend():
                yield

            def fake_call(argv):
                called["argv"] = argv
                # simulate the user adding a memory in their editor
                path = Profile.path_for("default")
                content = path.read_text()
                path.write_text(content + "\nedited-in-editor memory\n")
                return 0

            monkeypatch.setattr(app, "suspend", fake_suspend)
            import subprocess

            monkeypatch.setattr(subprocess, "call", fake_call)
            app.action_edit_profile()
            await pilot.pause()
            assert called["argv"][-1].endswith("default.md")
            assert any(
                "edited-in-editor" in m.text for m in app.profile_memory.memories
            )
