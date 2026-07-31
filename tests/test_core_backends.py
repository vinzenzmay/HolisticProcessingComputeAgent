"""Tests for hpca.core.backends: the catalog, the clients, and who they belong to.

Hermetic: no socket is opened. The LLM client is a factory the test owns, so
identity ("did these two sessions get the same object?") and lifetime ("was it
closed, once?") are directly observable; cluster discovery goes through an
`httpx.MockTransport`, the same way `test_cluster_endpoints` fakes it.

The other half of the point is what is asserted. In `HpcaApp` these behaviours
were readable only as widget state — the star in a modal, the fill of a meter —
so a test had to run a Textual app to see them. Here the observable is the list
of emitted events, and an event names the session it is about, which is exactly
the property that lets the core stop caring which session is on screen.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from hpca.agent.compact import CHARS_PER_TOKEN
from hpca.config import LLMBackend, Settings
from hpca.core.backends import BackendRegistry, autoconnect_logger
from hpca.core.deps import CoreDeps
from hpca.db import connect, init_db
from hpca.discover import DiscoveredBackend
from hpca.llm import LLMError
from hpca.protocol import ContextEstimate, Notify
from hpca.sessions import SessionStore
from hpca.slurm import SlurmError

IP = "172.16.33.208"


def backend_a(**overrides) -> LLMBackend:
    """A fresh catalog entry per call, never a module-level constant: settings
    mutate the entries they hold — the window probe writes its answer back —
    and a shared instance would carry that into the next test."""
    return LLMBackend(
        model="qwen-a", base_url="http://a/v1", max_model_len=1000, **overrides
    )


def backend_b(**overrides) -> LLMBackend:
    return LLMBackend(
        model="qwen-b", base_url="http://b/v1", max_model_len=2000, **overrides
    )


# --- doubles -----------------------------------------------------------------


class FakeClient:
    """An LLMClient's whole surface as far as this module is concerned.

    Counts its own closes rather than being asserted against a mock, because
    "exactly once" is the property under test and a call count is the only
    honest way to see it.
    """

    def __init__(self, settings=None, *, window=None, probe_error=None,
                 close_error=None) -> None:
        self.settings = settings
        self.closes = 0
        self._window = window
        self._probe_error = probe_error
        self._close_error = close_error

    async def context_window(self):
        if self._probe_error is not None:
            raise self._probe_error
        return self._window

    async def close(self) -> None:
        self.closes += 1
        if self._close_error is not None:
            raise self._close_error


class Factory:
    """Builds `FakeClient`s and keeps every one, in build order."""

    def __init__(self, *, window=None, probe_error=None) -> None:
        self.built: list[FakeClient] = []
        self._window = window
        self._probe_error = probe_error

    def __call__(self, settings) -> FakeClient:
        client = FakeClient(
            settings, window=self._window, probe_error=self._probe_error
        )
        self.built.append(client)
        return client


class FakeLog:
    """The whole of `SessionLog` that `LoggedLLM` reaches for."""

    def write(self, header, text):
        pass


class FakeSlurm:
    """Stand-in exposing only ``job_states``; returns a dict or raises."""

    def __init__(self, states=None, error=None) -> None:
        self._states = states or {}
        self._error = error

    async def job_states(self, job_ids):
        if self._error is not None:
            raise self._error
        return {j: self._states[j] for j in job_ids if j in self._states}


def write_manifest(dir_path, jobid, port, model, role="llm", **overrides):
    data = {
        "role": role,
        "model": model,
        "jobid": jobid,
        "user": "mayv_c",
        "node": "hpc-gpu-8",
        "ip": IP,
        "port": port,
        "ctx_len": 131072,
        "needs_key": False,
        "started": "2026-07-23T13:30:00Z",
    }
    data.update(overrides)
    (dir_path / f"{jobid}-{port}.json").write_text(json.dumps(data))


def serve(model_id, max_model_len=131072):
    body = {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "max_model_len": max_model_len}
        ],
    }
    return lambda request: httpx.Response(200, json=body)


def locked(request):
    return httpx.Response(401, text="unauthorized")


def make_transport(routes):
    """MockTransport routing on request port; absent ports look unreachable."""

    def handler(request):
        handle = routes.get(request.url.port)
        if handle is None:
            raise httpx.ConnectError("refused", request=request)
        return handle(request)

    return httpx.MockTransport(handler)


def cluster(tmp_path, manifests, routes=None, *, states=None, slurm_error=None):
    """A manifest dir and the endpoints it claims, as `Harness` keywords.

    ``manifests`` is a list of ``(jobid, port, model)``, optionally with a
    fourth element of manifest overrides. Ports absent from ``routes`` look
    unreachable, which is what a job that is alive but not yet serving looks
    like; ``states`` defaults to every declared job being RUNNING.
    """
    dir_path = tmp_path / "endpoints"
    dir_path.mkdir(exist_ok=True)
    for jobid, port, model, *rest in manifests:
        write_manifest(dir_path, jobid, port, model, **(rest[0] if rest else {}))
    if states is None:
        states = {jobid: "RUNNING" for jobid, *_ in manifests}
    return {
        "endpoints_dir": dir_path,
        "slurm": FakeSlurm(states, error=slurm_error),
        "probe_transport": make_transport(routes or {}),
    }


# --- harness -----------------------------------------------------------------


class Harness:
    """A registry with everything around it visible to the test."""

    def __init__(
        self,
        home,
        *,
        slurm=None,
        window=None,
        probe_error=None,
        endpoints_dir=None,
        **kwargs,
    ) -> None:
        self.home = home
        self.settings = Settings()
        if endpoints_dir is not None:
            self.settings.endpoints.endpoints_dir = str(endpoints_dir)
        self.conn = connect(home / "hpca.db")
        init_db(self.conn)
        self.sessions = SessionStore(self.conn)
        self.events: list = []
        self.factory = Factory(window=window, probe_error=probe_error)
        self.embedder = FakeClient()

        async def run_db(fn):
            return fn(self.conn)

        self.deps = CoreDeps(
            settings=self.settings,
            app_dir=home,
            db=run_db,
            emit=self.events.append,
            conn=self.conn,
            slurm=slurm,
        )
        self.registry = BackendRegistry(
            self.deps,
            embedder=self.embedder,
            client_factory=self.factory,
            # A logger of the test's own: the default attaches a FileHandler to
            # the shared "hpca.autoconnect" logger, which would outlive the
            # tmp_path it was pointed at.
            logger=logging.getLogger("hpca.test.autoconnect"),
            **kwargs,
        )

    def session(self, backend: LLMBackend | str = "") -> str:
        blob = (
            backend.model_dump_json()
            if isinstance(backend, LLMBackend)
            else backend
        )
        return self.sessions.create(profile="default", backend=blob).session_id

    @property
    def notices(self) -> list[Notify]:
        return [e for e in self.events if isinstance(e, Notify)]

    @property
    def estimates(self) -> list[ContextEstimate]:
        return [e for e in self.events if isinstance(e, ContextEstimate)]


@pytest.fixture
def home(monkeypatch, tmp_path):
    """`Settings.save()` writes into the app dir; keep it in the test's own."""
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


# --- clients -----------------------------------------------------------------


class TestClients:
    def test_two_sessions_on_one_backend_share_a_client(self, home):
        h = Harness(home)
        one, two = h.session(backend_a()), h.session(backend_a())
        assert h.registry.client_for(one) is h.registry.client_for(two)
        assert len(h.factory.built) == 2  # the bootstrap, and the shared one

    def test_a_session_without_a_backend_uses_the_bootstrap(self, home):
        h = Harness(home)
        assert h.registry.client_for(h.session()) is h.registry.bootstrap

    def test_an_unknown_session_uses_the_bootstrap(self, home):
        # A deleted session whose turn is still unwinding must still resolve.
        h = Harness(home)
        assert h.registry.client_for("gone") is h.registry.bootstrap
        assert h.registry.client_for(None) is h.registry.bootstrap

    def test_different_backends_get_different_clients(self, home):
        h = Harness(home)
        a, b = h.session(backend_a()), h.session(backend_b())
        assert h.registry.client_for(a) is not h.registry.client_for(b)

    def test_a_session_client_carries_its_backends_connection_fields(self, home):
        h = Harness(home)
        h.settings.llm.max_retries = 7  # client behaviour comes from the base
        backend = LLMBackend(
            model="qwen-b", base_url="http://b/v1", api_key="sekrit"
        )
        settings = h.registry.client_for(h.session(backend)).settings
        assert (settings.model, settings.base_url) == ("qwen-b", "http://b/v1")
        assert settings.api_key == "sekrit"
        assert settings.max_retries == 7

    def test_an_unparseable_backend_blob_falls_back_to_the_bootstrap(self, home):
        h = Harness(home)
        session_id = h.session("{not json")
        assert h.registry.backend_for(session_id) is None
        assert h.registry.client_for(session_id) is h.registry.bootstrap

    def test_labelled_client_routes_to_the_sessions_own_backend(self, home):
        # The bootstrap may point at a backend that is no longer running once
        # the session has been switched away from it.
        h = Harness(home)
        session_id = h.session(backend_b())
        client = h.registry.labelled_client("conclude", session_id=session_id)
        assert client is h.registry.client_for(session_id)
        assert client is not h.registry.bootstrap

    def test_labelled_client_wraps_the_client_when_given_a_log(self, home):
        h = Harness(home)
        client = h.registry.labelled_client(
            "title", session_id=None, log=FakeLog()
        )
        assert client is not h.registry.bootstrap
        assert client._llm is h.registry.bootstrap


# --- switching ---------------------------------------------------------------


class TestSwitching:
    def test_switch_persists_the_choice(self, home):
        h = Harness(home)
        session_id = h.session()
        assert h.registry.switch_backend(session_id, backend_b()) is True
        assert h.sessions.get(session_id).backend == backend_b().model_dump_json()
        assert h.registry.model_for(session_id) == "qwen-b"

    def test_switch_leaves_another_session_alone(self, home):
        h = Harness(home)
        stays, moves = h.session(backend_a()), h.session(backend_a())
        before = h.registry.client_for(stays)
        h.registry.switch_backend(moves, backend_b())
        assert h.registry.client_for(stays) is before
        assert h.registry.max_model_len_for(stays) == 1000
        assert h.registry.client_for(moves) is not before
        assert h.registry.max_model_len_for(moves) == 2000

    def test_switch_restates_that_sessions_window(self, home):
        h = Harness(home)
        session_id = h.session(backend_a())
        h.registry.switch_backend(session_id, backend_b())
        assert h.estimates[-1].session_id == session_id
        assert h.estimates[-1].window == 2000

    def test_switch_is_refused_while_that_session_is_mid_turn(self, home):
        h = Harness(home)
        session_id = h.session(backend_a())
        assert h.registry.switch_backend(session_id, backend_b(), busy=True) is False
        assert h.sessions.get(session_id).backend == backend_a().model_dump_json()
        assert h.notices[-1].severity == "warning"
        assert "mid-reply" in h.notices[-1].text

    def test_switch_announces_the_new_model(self, home):
        h = Harness(home)
        h.registry.switch_backend(h.session(), backend_b())
        assert h.notices[-1].text == "This session now uses qwen-b"

    def test_marks_the_sessions_own_backend(self, home):
        h = Harness(home)
        session_id = h.session(backend_b())
        mark = h.registry.marks_session_backend
        assert mark(backend_b(), session_id=session_id) is True
        assert mark(backend_a(), session_id=session_id) is False

    def test_marks_the_active_backend_for_a_session_with_none(self, home):
        h = Harness(home)
        h.settings.activate_backend(backend_a())
        session_id = h.session()
        mark = h.registry.marks_session_backend
        assert mark(backend_a(), session_id=session_id) is True
        assert mark(backend_b(), session_id=session_id) is False


class TestNewSessionBackend:
    def test_no_choice_means_the_bootstrap(self, home):
        h = Harness(home)
        assert h.registry.backend_for_new_session() == ""
        assert h.registry.backend_for_new_session("") == ""

    def test_a_chosen_entry_is_stored_as_json(self, home):
        h = Harness(home)
        blob = h.registry.backend_for_new_session(backend_b())
        assert json.loads(blob)["model"] == "qwen-b"

    def test_a_blob_round_trips(self, home):
        h = Harness(home)
        blob = backend_b().model_dump_json()
        assert h.registry.backend_for_new_session(blob) == blob

    def test_an_unusable_blob_falls_back_to_the_bootstrap(self, home):
        h = Harness(home)
        assert h.registry.backend_for_new_session("{not json") == ""

    def test_choices_are_the_configured_catalog(self, home):
        h = Harness(home)
        h.settings.backends = [backend_a(), backend_b()]
        assert [b.model for b in h.registry.choices()] == ["qwen-a", "qwen-b"]


# --- teardown ----------------------------------------------------------------


class TestTeardown:
    async def test_every_client_is_closed_exactly_once(self, home):
        h = Harness(home)
        h.registry.client_for(h.session(backend_a()))
        h.registry.client_for(h.session(backend_b()))
        assert len(h.factory.built) == 3  # bootstrap + two backends
        await h.registry.aclose()
        assert [c.closes for c in h.factory.built] == [1, 1, 1]
        assert h.embedder.closes == 1

    async def test_closing_twice_closes_nothing_twice(self, home):
        h = Harness(home)
        h.registry.client_for(h.session(backend_a()))
        await h.registry.aclose()
        await h.registry.aclose()
        assert [c.closes for c in h.factory.built] == [1, 1]
        assert h.embedder.closes == 1

    async def test_an_injected_bootstrap_is_left_to_its_owner(self, home):
        injected = FakeClient()
        h = Harness(home, llm=injected)
        h.registry.client_for(h.session(backend_a()))
        await h.registry.aclose()
        assert injected.closes == 0
        assert [c.closes for c in h.factory.built] == [1]

    async def test_a_client_that_fails_to_close_does_not_strand_the_rest(self, home):
        h = Harness(home)
        h.registry.client_for(h.session(backend_a()))
        h.factory.built[0]._close_error = RuntimeError("pool already gone")
        await h.registry.aclose()
        assert [c.closes for c in h.factory.built] == [1, 1]
        assert h.embedder.closes == 1


# --- the window probe --------------------------------------------------------


class TestContextWindowProbe:
    async def test_the_window_is_remembered_and_written_back(self, home):
        # Written back so compaction benefits too, and so the number survives a
        # backend that is offline next time.
        h = Harness(home, window=32768)
        h.settings.backends = [backend_a(), backend_b()]
        h.settings.activate_backend(backend_b())
        assert await h.registry.discover_context_window() == 32768
        assert h.registry.discovered_window == 32768
        assert h.settings.backends[1].max_model_len == 32768
        assert h.settings.backends[0].max_model_len == 1000  # untouched

    async def test_a_failed_probe_degrades_without_raising(self, home):
        h = Harness(home, probe_error=LLMError("all connection attempts failed"))
        assert await h.registry.discover_context_window() is None
        assert h.registry.discovered_window is None
        assert h.registry.max_model_len_for(h.session()) is None
        assert h.events == []

    async def test_a_backend_that_does_not_say_leaves_the_window_unknown(self, home):
        h = Harness(home, window=None)
        assert await h.registry.discover_context_window() is None
        assert h.registry.discovered_window is None

    async def test_the_probed_window_answers_for_bootstrap_sessions(self, home):
        h = Harness(home, window=32768)
        await h.registry.discover_context_window()
        assert h.registry.max_model_len_for(h.session()) == 32768
        # ...but never for a session pinned to a backend of its own.
        assert h.registry.max_model_len_for(h.session(backend_a())) == 1000

    def test_the_active_catalog_entry_is_the_last_resort(self, home):
        h = Harness(home)
        h.settings.backends = [backend_a(), backend_b()]
        h.settings.activate_backend(backend_b())
        assert h.registry.max_model_len_for(h.session()) == 2000

    def test_an_unknown_window_is_none_rather_than_a_guess(self, home):
        # With no window, compaction stays off; guessing one folds history for
        # no reason.
        h = Harness(home)
        assert h.registry.max_model_len_for(h.session()) is None


# --- context accounting ------------------------------------------------------


class TestContextAccounting:
    def test_usage_is_recorded_and_announced_per_session(self, home):
        h = Harness(home)
        a, b = h.session(backend_a()), h.session(backend_b())
        h.registry.note_usage(a, {"prompt_tokens": 700})
        h.registry.note_usage(b, {"prompt_tokens": 120})
        assert h.registry.measured_for(a) == 700
        assert h.registry.measured_for(b) == 120
        assert [(e.session_id, e.used, e.window) for e in h.estimates] == [
            (a, 700, 1000),
            (b, 120, 2000),
        ]

    def test_a_background_sessions_count_is_kept_not_dropped(self, home):
        # Whether it reaches a screen is the renderer's call: the event names
        # its session, so the core never has to know which one is open.
        h = Harness(home)
        a, b = h.session(), h.session()
        h.deps.focused_session_id = a
        h.registry.note_usage(b, {"prompt_tokens": 4242})
        assert h.registry.measured_for(b) == 4242
        assert h.estimates[-1].session_id == b

    def test_a_report_without_a_prompt_count_says_nothing(self, home):
        h = Harness(home)
        h.registry.note_usage(h.session(), {"completion_tokens": 9})
        assert h.events == []

    def test_the_estimate_does_not_leak_between_sessions(self, home):
        h = Harness(home)
        a, b = h.session(), h.session()
        h.registry.estimate_context(a, {"messages": [_msg("x" * 400)]})
        h.registry.estimate_context(b, {})
        assert h.estimates[0].session_id == a
        assert h.estimates[0].used == 400 // CHARS_PER_TOKEN
        assert h.estimates[1] == ContextEstimate(session_id=b, used=0, window=0)

    def test_a_measured_count_supersedes_the_estimate(self, home):
        h = Harness(home)
        session_id = h.session()
        h.registry.note_usage(session_id, {"prompt_tokens": 900})
        h.registry.estimate_context(session_id, {"messages": [_msg("x" * 40)]})
        assert h.estimates[-1].used == 900

    def test_the_estimate_measures_the_folded_view(self, home):
        # Compaction is what the model will actually receive; estimating the
        # whole transcript would over-count exactly the sessions that folded.
        h = Harness(home)
        values = {
            "messages": [_msg("old" * 100), _msg("old" * 100), _msg("kept")],
            "compacted": {"upto": 2, "summary": _msg("summary")},
        }
        h.registry.estimate_context(h.session(), values)
        folded = len("summary") + len("kept")
        assert h.estimates[-1].used == folded // CHARS_PER_TOKEN

    def test_forgetting_a_session_drops_only_its_own_number(self, home):
        h = Harness(home)
        a, b = h.session(), h.session()
        h.registry.note_usage(a, {"prompt_tokens": 700})
        h.registry.note_usage(b, {"prompt_tokens": 120})
        h.registry.forget_session(a)
        assert h.registry.measured_for(a) is None
        assert h.registry.measured_for(b) == 120

    def test_a_forgotten_session_re_estimates_from_its_history(self, home):
        h = Harness(home)
        session_id = h.session()
        h.registry.note_usage(session_id, {"prompt_tokens": 900})
        h.registry.forget_session(session_id)
        h.registry.estimate_context(session_id, {"messages": [_msg("x" * 40)]})
        assert h.estimates[-1].used == 10


def _msg(content: str) -> dict:
    return {"role": "user", "content": content}


# --- reload ------------------------------------------------------------------


class TestReload:
    async def test_the_bootstrap_is_replaced_and_the_old_one_closed(self, home):
        h = Harness(home)
        old = h.registry.bootstrap
        assert await h.registry.reload() is True
        assert h.registry.bootstrap is not old
        assert old.closes == 1

    async def test_every_measurement_is_dropped(self, home):
        # A different model means a different window, and counts measured
        # against the old one no longer describe it.
        h = Harness(home, window=1234)
        a, b = h.session(), h.session()
        h.registry.note_usage(a, {"prompt_tokens": 700})
        h.registry.note_usage(b, {"prompt_tokens": 120})
        await h.registry.discover_context_window()
        await h.registry.reload()
        assert h.registry.measured_for(a) is None
        assert h.registry.measured_for(b) is None

    async def test_the_rebuild_hook_fires_after_the_swap(self, home):
        seen = []
        h = Harness(home, on_reload=lambda: seen.append(True))
        await h.registry.reload()
        assert seen == [True]

    async def test_reload_is_refused_while_any_turn_is_in_flight(self, home):
        h = Harness(home)
        old = h.registry.bootstrap
        assert await h.registry.reload(busy=True) is False
        assert h.registry.bootstrap is old
        assert old.closes == 0
        assert h.notices[-1].severity == "warning"

    async def test_an_injected_client_is_not_ours_to_replace(self, home):
        injected = FakeClient()
        h = Harness(home, llm=injected)
        assert await h.registry.reload() is False
        assert h.registry.bootstrap is injected
        assert injected.closes == 0

    async def test_the_new_client_is_probed_for_its_window(self, home):
        h = Harness(home, window=8192)
        await h.registry.reload()
        assert h.registry.discovered_window == 8192


# --- auto-connect ------------------------------------------------------------


class TestAutoConnect:
    async def test_off_the_cluster_it_does_nothing(self, home):
        h = Harness(home)  # no slurm
        assert await h.registry.auto_connect() is None
        assert h.events == []

    async def test_one_live_llm_connects_and_announces_itself(self, home, tmp_path):
        model = "Qwen/Qwen3.6-27B"
        h = Harness(
            home,
            **cluster(tmp_path, [("111", 20001, model)], {20001: serve(model)}),
        )
        plan = await h.registry.auto_connect()
        assert plan.connect.model == model
        assert h.settings.llm.model == model
        assert [b.model for b in h.settings.backends] == [model]
        assert h.notices[-1].text == f"Auto-connected to {model}"

    async def test_several_llms_are_announced_for_the_picker(self, home, tmp_path):
        h = Harness(
            home,
            **cluster(
                tmp_path,
                [("111", 20001, "model-a"), ("222", 20003, "model-b")],
                {20001: serve("model-a"), 20003: serve("model-b")},
            ),
        )
        await h.registry.auto_connect()
        assert h.notices[-1].text == "2 cluster LLMs discovered — ctrl+l to pick one"
        assert h.settings.backends == []  # nothing joined the catalog

    async def test_a_locked_endpoint_asks_for_a_key(self, home, tmp_path):
        # The failure this covers used to be complete silence, which is how a
        # stale pool key went unnoticed.
        h = Harness(
            home,
            **cluster(
                tmp_path,
                [("111", 20001, "model-a", {"needs_key": True})],
                {20001: locked},
            ),
        )
        await h.registry.auto_connect()
        assert "API key" in h.notices[-1].text
        assert "model-a" in h.notices[-1].text

    async def test_the_embeddings_server_is_wired_to_rag(self, home, tmp_path):
        h = Harness(
            home,
            **cluster(
                tmp_path,
                [("333", 20000, "minilm", {"role": "embedding"})],
                {20000: serve("minilm")},
            ),
        )
        await h.registry.auto_connect()
        assert h.settings.rag.embedding_base_url == f"http://{IP}:20000/v1"
        assert h.embedder.closes == 1  # the client it replaced
        assert h.registry.embedder is not h.embedder

    async def test_discovery_failure_is_swallowed(self, home, tmp_path):
        # Best-effort: startup must not die because squeue misbehaved in a way
        # cluster_endpoints does not already handle.
        h = Harness(
            home,
            **cluster(
                tmp_path,
                [("111", 20001, "model-a")],
                slurm_error=RuntimeError("squeue exploded"),
            ),
        )
        assert await h.registry.auto_connect() is None
        assert h.events == []

    async def test_an_unreachable_controller_still_probes(self, home, tmp_path):
        # A SlurmError means liveness is unknown, not that everything is dead.
        h = Harness(
            home,
            **cluster(
                tmp_path,
                [("111", 20001, "model-a")],
                {20001: serve("model-a")},
                slurm_error=SlurmError("no controller"),
            ),
        )
        await h.registry.auto_connect()
        assert h.settings.llm.model == "model-a"

    async def test_nothing_found_says_nothing(self, home, tmp_path):
        h = Harness(home, **cluster(tmp_path, []))
        plan = await h.registry.auto_connect()
        assert plan.choices == []
        assert h.events == []

    async def test_a_job_slurm_no_longer_lists_is_reaped(self, home, tmp_path):
        h = Harness(home, **cluster(tmp_path, [("111", 20001, "model-a")], states={}))
        plan = await h.registry.auto_connect()
        assert plan.choices == []
        assert list((tmp_path / "endpoints").glob("*.json")) == []

    async def test_an_unchanged_endpoint_causes_no_churn(self, home):
        h = Harness(home)
        discovered = DiscoveredBackend(base_url="http://a/v1", model="qwen-a")
        h.settings.activate_backend(backend_a())
        built_before = len(h.factory.built)
        assert await h.registry.auto_activate(discovered) is False
        assert len(h.factory.built) == built_before  # no client rebuilt
        assert h.events == []


class TestCatalog:
    def test_a_discovered_endpoint_joins_the_catalog(self, home):
        h = Harness(home)
        discovered = DiscoveredBackend(
            base_url=f"http://{IP}:20001/v1",
            model="model-a",
            max_model_len=131072,
            api_key="pool-key",
        )
        entry = h.registry.ensure_catalog(discovered)
        assert entry.model == "model-a"
        assert entry.max_model_len == 131072
        # Learned for next time: the port is scanned first, and the key is
        # tried against other locked endpoints.
        assert h.settings.known_llm_ports == [20001]
        assert h.settings.llm_api_keys == ["pool-key"]
        assert Settings.load().backends[0].model == "model-a"

    def test_a_known_endpoint_is_returned_not_duplicated(self, home):
        h = Harness(home)
        h.settings.backends = [backend_a()]
        discovered = DiscoveredBackend(base_url="http://a/v1", model="qwen-a")
        assert h.registry.ensure_catalog(discovered) is h.settings.backends[0]
        assert len(h.settings.backends) == 1


class TestWireEmbedding:
    async def test_an_unchanged_url_rebuilds_nothing(self, home):
        h = Harness(home)
        url = h.settings.rag.embedding_base_url
        assert await h.registry.wire_embedding(url) is False
        assert h.registry.embedder is h.embedder
        assert h.embedder.closes == 0


class TestAutoconnectLogger:
    def test_it_writes_only_to_the_app_dir(self, tmp_path):
        logger = autoconnect_logger(tmp_path)
        handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        try:
            assert logger.propagate is False  # a TUI owns the terminal
            assert len(handlers) == 1
            logger.info("hello")
            handlers[0].flush()
            assert "hello" in (tmp_path / "autoconnect.log").read_text()
        finally:
            # The logger is process-wide; leaving a handler on it would point
            # every later test's discovery at this tmp_path.
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
            logger.propagate = True
