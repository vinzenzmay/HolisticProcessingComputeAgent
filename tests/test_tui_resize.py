"""Resizing the terminal must never take the app down.

A resize to ~50 columns used to crash it: the sessions and watchers columns
hold their min-widths, the chat column had none and was squeezed to nothing,
and a bordered widget with zero content width crashes Rich's text wrapping
("range() arg 3 must not be zero") while wrapping the entry's placeholder.
The columns are laid out as if the terminal were at least 74 wide, and the
terminal clips what does not fit.
"""

import json

import pytest

from hpca.config import LLMBackend, Settings
from hpca.llm import ChatResponse
from hpca.tui import manage_llms as manage_module
from hpca.tui.app import ChatInput, HpcaApp

# Widths either side of the old crash (~50), down to the absurd.
WIDTHS = (120, 80, 60, 55, 50, 46, 40, 30, 20, 12, 4, 1)
HEIGHTS = (30, 8, 3, 1)


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "a test session"}))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_discovery(monkeypatch):
    async def fake_scan(*args, **kwargs):
        return []

    async def fake_reachable(base_url, **kwargs):
        return True

    monkeypatch.setattr(manage_module, "scan_local_ports", fake_scan)
    monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)


async def squeeze(pilot, heights=(30,)):
    """Walk the terminal down to nothing, pausing so each layout renders."""
    for width in WIDTHS:
        for height in heights:
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            await pilot.pause()


class TestNarrowTerminal:
    async def test_empty_app_survives(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await squeeze(pilot, HEIGHTS)
            assert app.is_running

    async def test_open_session_survives(self, hpca_home):
        """The entry, and its placeholder, are what used to crash."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert app.query_one("#chat-input", ChatInput).display
            await squeeze(pilot, HEIGHTS)
            assert app.is_running

    async def test_a_conversation_survives(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("Four BAMs match the cohort.")]))
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "which BAMs are in the cohort dir?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await squeeze(pilot, HEIGHTS)
            assert app.is_running

    async def test_manage_llms_survives(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [
            LLMBackend(model="m", base_url="http://localhost:20001/v1")
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            await squeeze(pilot)
            assert app.is_running

    async def test_the_columns_never_collapse(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            for width in WIDTHS:
                await pilot.resize_terminal(width, 30)
                await pilot.pause()
                for column in ("sessions", "chat", "watchers"):
                    assert app.query_one(f"#{column}").size.width > 0, (
                        f"{column} collapsed at width {width}"
                    )

    async def test_wide_layout_is_unchanged(self, hpca_home):
        """The floor must not disturb terminals that have room to spare."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            widths = {
                column: app.query_one(f"#{column}").size.width
                for column in ("sessions", "chat", "watchers")
            }
            assert widths["chat"] == 58  # 2fr of the room left by its neighbours
            assert widths["sessions"] == widths["watchers"] == 28
