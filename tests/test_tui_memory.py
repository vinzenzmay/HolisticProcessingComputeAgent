"""Tests for the memory workflows: /memorize, /conclude, caps, editor (§6.3, §6.4)."""

import asyncio
import json
from time import monotonic
from contextlib import contextmanager

import pytest
from textual.widgets import ListView

from hpca.editor import resolve_editor
from hpca.llm import ChatResponse
from hpca.profiles import MemoryScope, Profile
from hpca.tui.app import ChatInput, HpcaApp, UNTITLED_SESSION, WorkingIndicator
from hpca.tui.memory_screens import MemoryProposalScreen, ReflectionScreen


def is_title_request(json_schema):
    """The app names a session by asking the model (§3 sessions column); that
    call is not one of the queued decisions."""
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})

class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


class RecordingLLM(FakeLLM):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if not is_title_request(json_schema):
            self.calls.append(list(messages))
        return await super().chat(messages, json_schema=json_schema, **kwargs)


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def proposals_json(*proposals):
    return json.dumps({"proposals": list(proposals)})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def type_and_submit(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def wait_for_screen(app, pilot, screen_type, *, timeout_s=10.0):
    """Pause until the modal is actually up, rather than counting pauses.

    A fixed number of pauses is a guess about scheduling. The worker behind
    /memorize runs a model round trip and then pushes a screen, and six pauses
    is sometimes not enough on a box running the suite across every core —
    which is why this only ever failed in the parallel run and never when the
    test was run on its own. Waiting on the condition makes the test say what
    it means and stops the result depending on how busy the machine is.

    The worker cannot simply be awaited: it pushes the screen and then blocks
    on the user's answer, so ``wait_for_complete`` would deadlock against the
    very modal this is waiting for.
    """
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause()
        # Yield properly rather than spinning: the whole point is not to make
        # a loaded machine any busier.
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"{screen_type.__name__} never appeared within {timeout_s}s; "
        f"on screen: {type(app.screen).__name__}"
    )


MEMORIZE_REPLY = proposals_json(
    {"scope": "system-prompt", "kind": "fact", "text": "STAR needs 40G on this cluster."}
)


class TestMemorize:
    """/memorize <note>: the model forms memories from the note and the
    conversation so far; each one still needs approval."""

    async def test_the_model_forms_the_memory_from_the_note(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([MEMORIZE_REPLY]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/memorize STAR needed 40G here")
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            saved = Profile.load("default").memories
            assert len(saved) == 1
            assert saved[0].text == "STAR needs 40G on this cluster."
            # the model chose the scope, not a picker
            assert saved[0].scope is MemoryScope.SYSTEM_PROMPT

    async def test_the_note_and_the_conversation_reach_the_model(self, hpca_home):
        llm = RecordingLLM([respond_json("ok"), MEMORIZE_REPLY])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "my STAR job was killed")
            await app.workers.wait_for_complete()
            await type_and_submit(app, pilot, "/memorize that was a memory limit")
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            prompt = llm.calls[-1][-1]["content"]
            assert "that was a memory limit" in prompt  # the user's note
            assert "my STAR job was killed" in prompt  # the conversation context
            await pilot.press("n")
            await app.workers.wait_for_complete()

    async def test_rejecting_saves_nothing(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([MEMORIZE_REPLY]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/memorize forget me")
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []

    async def test_memorize_without_a_note_explains_itself(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/memorize")
            await pilot.pause()
            assert not isinstance(app.screen, MemoryProposalScreen)
            assert Profile.load("default").memories == []

    async def test_backslash_still_works(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([MEMORIZE_REPLY]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\memorize STAR needed 40G")
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()

    async def test_slash_command_is_not_the_sessions_topic(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([MEMORIZE_REPLY]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/memorize something")
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            # a command is not a message: it must not name the session
            titles = [s.title for s in app.session_store.list(profile="default")]
            assert titles == [UNTITLED_SESSION]

    async def test_unknown_command_is_reported_not_sent(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/frobnicate now")
            assert app.chat_log_texts() == []
            assert app.active_session.title == UNTITLED_SESSION

    async def test_spinner_visible_while_forming_memories(self, hpca_home):
        """The silent /memorize LLM call shows the chat-bottom spinner while it
        runs, labelled "forming memories", and clears it once the call returns."""
        gate = asyncio.Event()

        class GatedLLM(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if kwargs.get("schema_name") == "memory_proposals":
                    await gate.wait()
                return await super().chat(
                    messages, json_schema=json_schema, **kwargs
                )

        app = HpcaApp(llm=GatedLLM([MEMORIZE_REPLY]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "/memorize STAR needed 40G here")
            for _ in range(50):  # let the worker park on the gated call
                if app.query(WorkingIndicator):
                    break
                await pilot.pause()
            indicators = app.query(WorkingIndicator)
            assert indicators, "spinner should be visible while forming memories"
            assert indicators.first(WorkingIndicator).activity == "forming memories"
            gate.set()
            await wait_for_screen(app, pilot, MemoryProposalScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.query(WorkingIndicator)


class TestCommandMenu:
    """Typing the prefix lists the commands: /memorize should be discoverable
    rather than folklore."""

    async def test_slash_lists_the_commands(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            menu = app.query_one("#command-menu")
            assert not menu.display  # quiet until a command is started

            app.query_one("#chat-input", ChatInput).focus()
            await pilot.press("/")
            await pilot.pause()
            assert menu.display
            listed = str(menu.content)
            assert "/memorize" in listed
            assert "/conclude" in listed

    async def test_the_list_narrows_as_the_name_is_typed(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.query_one("#chat-input", ChatInput).focus()
            await pilot.press("/", "m", "e")
            await pilot.pause()
            listed = str(app.query_one("#command-menu").content)
            assert "/memorize" in listed
            assert "/conclude" not in listed

    async def test_the_menu_goes_when_the_draft_is_ordinary_text(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            await pilot.press("/")
            await pilot.pause()
            assert app.query_one("#command-menu").display
            chat_input.text = "which BAMs are in the cohort?"
            await pilot.pause()
            assert not app.query_one("#command-menu").display

    async def test_backslash_lists_them_too(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.query_one("#chat-input", ChatInput).text = "\\"
            await pilot.pause()
            assert app.query_one("#command-menu").display


class TestConclude:
    """/conclude runs a full self-review of the conversation — memories,
    struggle notes and skills — each approved via the reflection dialog."""

    async def test_approve_and_reject_proposals(self, hpca_home):
        llm = FakeLLM(
            [
                respond_json("hi"),
                proposals_json(
                    {"kind": "memory", "scope": "system-prompt", "text": "STAR needs 40G."},
                    {"kind": "memory", "scope": "rag", "text": "Cluster is cubi."},
                ),
            ]
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "hello")
            await app.workers.wait_for_complete()
            await type_and_submit(app, pilot, r"\conclude")
            await pilot.pause()
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")  # keep the first
            await pilot.pause()
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("n")  # discard the second
            await app.workers.wait_for_complete()
            await pilot.pause()
            loaded = Profile.load("default")
            assert len(loaded.memories) == 1
            assert loaded.memories[0].text == "STAR needs 40G."
            assert loaded.memories[0].scope is MemoryScope.SYSTEM_PROMPT
            assert loaded.memories[0].kind == "learning"

    async def test_spinner_visible_while_reviewing(self, hpca_home):
        """The silent /conclude LLM call shows the chat-bottom spinner while it
        runs, labelled for the step, and clears it once the call returns."""
        gate = asyncio.Event()

        class GatedLLM(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if kwargs.get("schema_name") == "reflection":
                    await gate.wait()
                return await super().chat(
                    messages, json_schema=json_schema, **kwargs
                )

        llm = GatedLLM(
            [
                respond_json("hi"),
                proposals_json(
                    {"kind": "memory", "scope": "system-prompt", "text": "STAR needs 40G."},
                ),
            ]
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, "hello")
            await app.workers.wait_for_complete()
            await type_and_submit(app, pilot, r"\conclude")
            for _ in range(50):  # let the worker park on the gated call
                if app.query(WorkingIndicator):
                    break
                await pilot.pause()
            indicators = app.query(WorkingIndicator)
            assert indicators, "spinner should be visible during the review call"
            assert indicators.first(WorkingIndicator).activity == "reviewing conversation"
            gate.set()
            await pilot.pause()
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")  # keep the proposal
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.query(WorkingIndicator)

    async def test_conclude_without_a_conversation_warns(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await type_and_submit(app, pilot, r"\conclude")
            await pilot.pause()
            assert Profile.load("default").memories == []


class TestMemoryCaps:
    async def test_over_cap_reported(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x " * 6000, scope=MemoryScope.SYSTEM_PROMPT)  # over 2400 tokens
        profile.save()
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert app.check_memory_caps() is True

    async def test_under_cap_quiet(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert app.check_memory_caps() is False


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


class TestHardWriteBudget:
    """A full system-prompt scope rejects new writes; injection never
    truncates what is already in the file. RAG has no budget."""

    async def test_full_scope_blocks_new_writes(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x " * 6000, scope=MemoryScope.SYSTEM_PROMPT)  # over 2400 tokens
        profile.save()
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert app._memory_write_blocked(MemoryScope.SYSTEM_PROMPT, "a new learning")

    async def test_room_available_allows_write(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert not app._memory_write_blocked(
                MemoryScope.SYSTEM_PROMPT, "a short site note"
            )

    async def test_rag_writes_never_blocked(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert not app._memory_write_blocked(MemoryScope.RAG, "x" * 10000)


class TestCuratorWiring:
    """Redesign Phase 6: the curator runs at startup, when the app is idle
    by definition, at most once every few days."""

    def old_note(self):
        from datetime import date, timedelta

        profile = Profile.load("default")
        memory = profile.add_memory("an old struggle note", scope=MemoryScope.RAG)
        memory.created = (date.today() - timedelta(days=200)).isoformat()
        profile.save()

    async def test_startup_archives_old_rag(self, hpca_home):
        from hpca import curator

        self.old_note()
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            assert Profile.load("default").memories == []
            assert curator.archive_path("default").exists()

    async def test_second_start_does_not_re_run(self, hpca_home):
        from hpca import curator

        self.old_note()
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)):
            pass
        first = curator.load_state()["last_run"]
        app2 = HpcaApp(llm=FakeLLM([]))
        async with app2.run_test(size=(120, 40)):
            assert app2.run_curator_if_due() == {}
        assert curator.load_state()["last_run"] == first

    async def test_can_be_disabled(self, hpca_home):
        from hpca import curator

        self.old_note()
        app = HpcaApp(llm=FakeLLM([]))
        app.settings.memory.curator_interval_days = 0
        async with app.run_test(size=(120, 40)):
            assert app.run_curator_if_due() == {}
        assert not curator.archive_path("default").exists()
