"""The per-session model line at the top of the chat column (decision 11).

The model in use moved off the app-top ``TopBar`` and onto a dedicated line
directly above the context meter, so it reads as a property of the session on
screen rather than a global app setting. It is session-specific and repaints
on session switch.
"""

import json

import pytest

from hpca.config import LLMBackend, Settings
from hpca.llm import ChatResponse
from hpca.tui.app import HpcaApp, TopBar
from hpca.tui.context_bar import ModelLine


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "t"}))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def with_backends():
    settings = Settings()
    settings.backends = [
        LLMBackend(model="qwen-a", base_url="http://a/v1", max_model_len=1000),
        LLMBackend(model="qwen-b", base_url="http://b/v1", max_model_len=2000),
    ]
    settings.save()


def _blob(model: str, base_url: str) -> str:
    return LLMBackend(model=model, base_url=base_url).model_dump_json()


async def test_model_line_shows_the_active_sessions_model(hpca_home):
    with_backends()
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        session = app.session_store.create(
            profile="default", backend=_blob("qwen-a", "http://a/v1")
        )
        await app.open_session(session)
        await pilot.pause()
        line = app.query_one(ModelLine)
        assert line.display is True
        assert line.text == "model: qwen-a"


async def test_model_line_updates_when_switching_sessions(hpca_home):
    with_backends()
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a = app.session_store.create(
            profile="default", backend=_blob("qwen-a", "http://a/v1")
        )
        b = app.session_store.create(
            profile="default", backend=_blob("qwen-b", "http://b/v1")
        )
        await app.open_session(a)
        await pilot.pause()
        assert app.query_one(ModelLine).text == "model: qwen-a"
        await app.open_session(b)
        await pilot.pause()
        assert app.query_one(ModelLine).text == "model: qwen-b"


async def test_model_line_hidden_without_a_session(hpca_home):
    with_backends()
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        # No session opened yet on mount.
        assert app.active_session is None
        assert app.query_one(ModelLine).display is False


async def test_top_bar_drops_model_keeps_profile(hpca_home):
    with_backends()
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        session = app.session_store.create(
            profile="default", backend=_blob("qwen-a", "http://a/v1")
        )
        await app.open_session(session)
        await pilot.pause()
        top = app.query_one(TopBar).render_text()
        assert "model:" not in top
        assert "qwen-a" not in top
        assert "profile:" in top
