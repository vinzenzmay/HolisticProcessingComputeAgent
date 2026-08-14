"""Per-session LLM: pick the backend at creation, stored per session, used by
the turn; no global default any more."""

import json

import pytest

from hpca.config import LLMBackend, Settings
from hpca.llm import ChatResponse
from hpca.tui.app import HpcaApp
from hpca.tui.switch_llm import SwitchLLMScreen


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


class TestCreation:
    async def test_no_backends_creates_without_a_picker(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")  # (new session)
            await pilot.pause()
            await pilot.press("enter")  # profile picker: default
            await pilot.pause()
            await pilot.pause()
            # no configured backends: no LLM picker, session uses bootstrap
            assert not isinstance(app.screen, SwitchLLMScreen)
            assert app.active_session is not None
            assert app.active_session.backend == ""

    async def test_creation_picks_and_stores_the_backend(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")  # (new session)
            await pilot.pause()
            await pilot.press("enter")  # profile picker: default
            await pilot.pause()
            assert isinstance(app.screen, SwitchLLMScreen)  # now pick the LLM
            switch_list = app.screen.query_one("#switch-list")
            switch_list.index = 1  # qwen-b
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.active_session is not None
            assert "qwen-b" in app.active_session.backend

    async def test_cancelling_the_llm_pick_creates_no_session(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")  # profile chosen
            await pilot.pause()
            assert isinstance(app.screen, SwitchLLMScreen)
            await pilot.press("escape")  # back out of the LLM pick
            await pilot.pause()
            assert app.active_session is None


class TestClientSelection:
    async def test_client_for_uses_the_sessions_backend(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            store = app.session_store
            blob = LLMBackend(model="qwen-b", base_url="http://b/v1").model_dump_json()
            session = store.create(profile="default", backend=blob)
            client = app._client_for(session)
            assert client is not app._llm  # its own client, not the bootstrap
            assert client._settings.model == "qwen-b"
            # a session sharing that backend reuses the same client
            again = store.create(profile="default", backend=blob)
            assert app._client_for(again) is client

    async def test_client_for_falls_back_to_bootstrap(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            session = app.session_store.create(profile="default")  # no backend
            assert app._client_for(session) is app._llm

    async def test_context_window_follows_the_session_backend(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            blob = LLMBackend(
                model="qwen-b", base_url="http://b/v1", max_model_len=2000
            ).model_dump_json()
            app.active_session = app.session_store.create(
                profile="default", backend=blob
            )
            assert app._active_max_model_len() == 2000
            assert app._active_model() == "qwen-b"


def _unwrap(labelled):
    """The underlying client behind a possibly-LoggedLLM wrapper."""
    return getattr(labelled, "_llm", labelled)


class TestLabelledLLM:
    """The app's own sub-agent calls (/conclude, /memorize, titling) must run
    on the session's *own* backend — the live client chat uses — not the
    bootstrap client, which may point at a backend that is no longer running
    once the session has been switched away (regression: /conclude failed with
    "all connection attempts failed")."""

    async def test_labelled_llm_routes_to_the_sessions_backend(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            blob = LLMBackend(
                model="qwen-b", base_url="http://b/v1"
            ).model_dump_json()
            session = app.session_store.create(profile="default", backend=blob)
            client = _unwrap(app._labelled_llm("conclude", session=session))
            assert client is app._client_for(session)
            assert client is not app._llm  # not the (possibly dead) bootstrap
            assert client._settings.model == "qwen-b"

    async def test_labelled_llm_defaults_to_the_active_session(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            blob = LLMBackend(
                model="qwen-b", base_url="http://b/v1"
            ).model_dump_json()
            app.active_session = app.session_store.create(
                profile="default", backend=blob
            )
            client = _unwrap(app._labelled_llm("memorize"))
            assert client._settings.model == "qwen-b"

    async def test_labelled_llm_falls_back_to_bootstrap(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            session = app.session_store.create(profile="default")  # no backend
            client = _unwrap(app._labelled_llm("conclude", session=session))
            assert client is app._llm


class TestToolContextLLM:
    """``ctx.llm`` is the door a tool's own firewalled sub-loop goes through:
    ``ask_docs`` (the doc-researcher, §4.2) and ``explain_job_failure``. It has
    to be the session's own backend for the same reason ``_labelled_llm`` does
    — the bootstrap client is built from ``settings.llm``, which a session
    pinning its own backend never updates, so it points at a default nothing is
    serving (regression: ask_docs failed with "All connection attempts failed"
    in a session whose chat was answering fine)."""

    async def test_tool_ctx_llm_routes_to_the_sessions_backend(self, hpca_home):
        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            blob = LLMBackend(
                model="qwen-b", base_url="http://b/v1"
            ).model_dump_json()
            session = app.session_store.create(profile="default", backend=blob)
            ctx = app._make_tool_ctx(session, None)
            assert ctx.llm is app._client_for(session)
            assert ctx.llm is not app._llm  # not the (possibly dead) bootstrap
            assert ctx.llm._settings.model == "qwen-b"

    async def test_logged_tool_ctx_llm_wraps_the_sessions_backend(self, hpca_home):
        """The transcript-logging wrapper must not quietly reintroduce the
        bootstrap client underneath the label."""
        from hpca.logs import open_log

        with_backends()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            blob = LLMBackend(
                model="qwen-b", base_url="http://b/v1"
            ).model_dump_json()
            session = app.session_store.create(profile="default", backend=blob)
            log = open_log(app.settings, session)
            assert log is not None  # logging is on by default
            ctx = app._make_tool_ctx(session, log)
            assert _unwrap(ctx.llm) is app._client_for(session)
            assert _unwrap(ctx.llm)._settings.model == "qwen-b"

    async def test_tool_ctx_llm_falls_back_to_bootstrap(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            session = app.session_store.create(profile="default")  # no backend
            assert app._make_tool_ctx(session, None).llm is app._llm
