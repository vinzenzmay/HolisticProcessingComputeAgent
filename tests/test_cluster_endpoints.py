"""Tests for hpca.cluster_endpoints: manifest-driven cluster discovery (§4).

These exercise the "auto-connect" reconciliation between manifest files
written by launch scripts, Slurm liveness, and a live ``/v1/models`` probe.
The MockTransport routes by request host:port so distinct endpoints answer
differently; the fake slurm just returns a preset state dict or raises.
"""

import getpass
import json
import logging

import httpx
import pytest

from hpca.cluster_endpoints import (
    ClusterEndpoints,
    Manifest,
    discover_cluster_endpoints,
    read_manifests,
)
from hpca.discover import KEY_REQUIRED
from hpca.slurm import SlurmError

IP = "172.16.33.208"


# --- helpers -----------------------------------------------------------------


def write_manifest(dir_path, jobid, port, model, role="llm", **overrides):
    """Write a manifest JSON at ``<jobid>-<port>.json`` and return its path.

    Owned by whoever runs the tests unless a test overrides ``user``: reaping
    is now restricted to our own manifests, so "mine" is the default case.
    """
    data = {
        "role": role,
        "model": model,
        "jobid": jobid,
        "user": getpass.getuser(),
        "node": "hpc-gpu-8",
        "ip": IP,
        "port": port,
        "ctx_len": 131072,
        "needs_key": role == "llm",
        "started": "2026-07-23T13:30:00Z",
    }
    data.update(overrides)
    path = dir_path / f"{jobid}-{port}.json"
    path.write_text(json.dumps(data))
    return path


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@pytest.fixture
def discovery_log():
    """Capture ``hpca.autoconnect`` records straight off the logger.

    Not caplog: the TUI attaches a file handler to this logger and turns
    propagation off — process-wide and for good — so whether a record ever
    reaches the root handler caplog listens on depends on which tests ran
    first in this worker.
    """
    logger = logging.getLogger("hpca.autoconnect")
    handler = _Capture()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def models_body(model_id, max_model_len=131072):
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "max_model_len": max_model_len}
        ],
    }


def make_transport(routes):
    """MockTransport routing on request port. ``routes`` maps port -> handler.

    Each handler takes the request and returns an ``httpx.Response``. Ports
    absent from the map answer as unreachable (connection error).
    """

    def handler(request):
        handle = routes.get(request.url.port)
        if handle is None:
            raise httpx.ConnectError("refused", request=request)
        return handle(request)

    return httpx.MockTransport(handler)


def serve(model_id, max_model_len=131072):
    return lambda request: httpx.Response(
        200, json=models_body(model_id, max_model_len)
    )


def locked(good_keys=()):
    """401 unless one of ``good_keys`` is presented as a bearer token."""

    def handle(request):
        auth = request.headers.get("Authorization", "")
        token = auth[len("Bearer ") :] if auth.startswith("Bearer ") else ""
        if token in good_keys:
            return httpx.Response(200, json=models_body(_LOCKED_MODEL))
        return httpx.Response(401, text="unauthorized")

    return handle


def refuse(request):
    raise httpx.ConnectError("refused", request=request)


_LOCKED_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


class FakeSlurm:
    """Stand-in exposing only ``job_states``; returns a dict or raises."""

    def __init__(self, states=None, error=False):
        self._states = states or {}
        self._error = error

    async def job_states(self, job_ids):
        if self._error:
            raise SlurmError("squeue: Unable to contact slurm controller")
        return {j: self._states[j] for j in job_ids if j in self._states}


# --- read_manifests ----------------------------------------------------------


class TestReadManifests:
    def test_valid_file_parsed(self, tmp_path):
        write_manifest(tmp_path, "1234567", 20001, "Model/M")
        manifests = read_manifests(tmp_path)
        assert len(manifests) == 1
        m = manifests[0]
        assert isinstance(m, Manifest)
        assert m.jobid == "1234567"
        assert m.port == 20001
        assert m.model == "Model/M"
        assert m.base_url == f"http://{IP}:20001/v1"
        assert m.path.exists()

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "bad-1.json").write_text("{not json")
        write_manifest(tmp_path, "1234567", 20001, "Model/M")
        manifests = read_manifests(tmp_path)
        assert [m.model for m in manifests] == ["Model/M"]

    def test_missing_required_key_skipped(self, tmp_path):
        # No "model" key -> skipped, not an error.
        (tmp_path / "incomplete-2.json").write_text(
            json.dumps({"role": "llm", "jobid": "9", "ip": IP, "port": 1})
        )
        write_manifest(tmp_path, "1234567", 20001, "Model/M")
        manifests = read_manifests(tmp_path)
        assert [m.model for m in manifests] == ["Model/M"]

    def test_noninteger_port_skipped(self, tmp_path):
        write_manifest(tmp_path, "1234567", "notaport", "Model/M")
        assert read_manifests(tmp_path) == []

    def test_nonexistent_dir_is_empty(self, tmp_path):
        assert read_manifests(tmp_path / "nope") == []

    def test_a_rejected_file_names_itself_and_the_field(
        self, tmp_path, discovery_log
    ):
        """Three servers running and an empty left panel looked exactly like
        "nothing launched" — the log has to separate "no files" from "files I
        threw away", and say which field lost them."""
        (tmp_path / "incomplete-2.json").write_text(
            json.dumps({"role": "llm", "jobid": "9", "ip": IP, "port": 1})
        )
        assert read_manifests(tmp_path) == []
        assert "incomplete-2.json" in discovery_log.text
        assert "missing model" in discovery_log.text

    def test_an_empty_dir_says_so_rather_than_blaming_a_file(
        self, tmp_path, discovery_log
    ):
        assert read_manifests(tmp_path) == []
        assert "holds no manifests" in discovery_log.text
        assert "rejected" not in discovery_log.text

    def test_a_missing_dir_is_not_an_empty_one(self, tmp_path, discovery_log):
        assert read_manifests(tmp_path / "nope") == []
        assert "no endpoints dir" in discovery_log.text

    def test_optional_defaults(self, tmp_path):
        (tmp_path / "min-3.json").write_text(
            json.dumps(
                {"role": "llm", "model": "M", "jobid": "5", "ip": IP, "port": 7}
            )
        )
        (m,) = read_manifests(tmp_path)
        assert m.user == ""
        assert m.node == ""
        assert m.ctx_len is None
        assert m.needs_key is False
        assert m.started == ""


# --- discover_cluster_endpoints ----------------------------------------------


class TestDiscover:
    async def test_empty_dir(self, tmp_path):
        result = await discover_cluster_endpoints(tmp_path, FakeSlurm())
        assert result == ClusterEndpoints([], None)

    async def test_happy_path_llm_and_embedding(self, tmp_path):
        write_manifest(tmp_path, "J1", 20001, "Model/M", role="llm")
        write_manifest(
            tmp_path, "J1", 20000, "sentence-transformers/MiniLM",
            role="embedding", needs_key=False,
        )
        transport = make_transport(
            {
                20001: serve("Model/M"),
                20000: serve("sentence-transformers/MiniLM"),
            }
        )
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
        )
        assert len(result.llms) == 1
        assert result.llms[0].model == "Model/M"
        assert result.llms[0].base_url == f"http://{IP}:20001/v1"
        assert result.embedding is not None
        assert result.embedding.model == "sentence-transformers/MiniLM"

    async def test_reaps_dead_job(self, tmp_path):
        path = write_manifest(tmp_path, "GONE", 20001, "Model/M")
        # Slurm knows nothing about GONE -> reap.
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({}), transport=make_transport({})
        )
        assert result.llms == []
        assert not path.exists()

    async def test_another_users_manifest_is_never_deleted(
        self, tmp_path, discovery_log
    ):
        """The endpoints dir is shared. Reaping is only ever an optimisation —
        an endpoint that does not answer is skipped anyway — so one bad squeue
        reading must not be able to delete the group's discovery."""
        path = write_manifest(
            tmp_path, "GONE", 20001, "Model/M", user="someone_else"
        )
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({}), transport=make_transport({})
        )
        assert result.llms == []  # still not surfaced: the job is gone
        assert path.exists()
        assert "skipping" in discovery_log.text

    async def test_an_unowned_manifest_is_still_reapable(self, tmp_path):
        # Written before manifests carried a user; nobody can claim it, and
        # leaving it forever would be the old bug in reverse.
        path = write_manifest(tmp_path, "GONE", 20001, "Model/M", user="")
        await discover_cluster_endpoints(
            tmp_path, FakeSlurm({}), transport=make_transport({})
        )
        assert not path.exists()

    async def test_no_reap_keeps_file(self, tmp_path):
        path = write_manifest(tmp_path, "GONE", 20001, "Model/M")
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({}), reap=False, transport=make_transport({})
        )
        assert result.llms == []
        assert path.exists()

    async def test_slurm_unreachable_no_reap_but_probes(self, tmp_path):
        path = write_manifest(tmp_path, "J1", 20001, "Model/M")
        transport = make_transport({20001: serve("Model/M")})
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm(error=True), transport=transport
        )
        # File kept (no reaping when SLURM is down) and endpoint still surfaced.
        assert path.exists()
        assert [b.model for b in result.llms] == ["Model/M"]

    async def test_stolen_port_not_surfaced_not_deleted(self, tmp_path):
        path = write_manifest(tmp_path, "J1", 20001, "Model/M")
        # Job alive, but the port now serves a DIFFERENT model.
        transport = make_transport({20001: serve("Other/Model")})
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
        )
        assert result.llms == []
        assert path.exists()

    async def test_dead_vllm_in_live_job_not_surfaced(self, tmp_path):
        path = write_manifest(tmp_path, "J1", 20001, "Model/M")
        # Job alive, but the endpoint refuses (vLLM starting up / crashed).
        transport = make_transport({20001: refuse})
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
        )
        assert result.llms == []
        assert path.exists()

    async def test_locked_llm_surfaced_named(self, tmp_path):
        write_manifest(tmp_path, "J1", 20001, "Model/M")
        transport = make_transport({20001: locked(good_keys=())})
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
        )
        assert len(result.llms) == 1
        backend = result.llms[0]
        assert backend.model == "Model/M"  # named from the manifest
        assert backend.needs_key is True
        assert backend.model != KEY_REQUIRED

    async def test_pool_key_unlocks(self, tmp_path):
        write_manifest(tmp_path, "J1", 20001, _LOCKED_MODEL)
        transport = make_transport({20001: locked(good_keys=("good",))})
        result = await discover_cluster_endpoints(
            tmp_path,
            FakeSlurm({"J1": "RUNNING"}),
            api_keys=["good"],
            transport=transport,
        )
        assert len(result.llms) == 1
        backend = result.llms[0]
        assert backend.model == _LOCKED_MODEL
        assert backend.needs_key is True
        assert backend.api_key == "good"

    async def test_first_embedding_wins(self, tmp_path):
        write_manifest(
            tmp_path, "J1", 20000, "Embed/One", role="embedding",
            needs_key=False,
        )
        write_manifest(
            tmp_path, "J1", 20002, "Embed/Two", role="embedding",
            needs_key=False,
        )
        transport = make_transport(
            {20000: serve("Embed/One"), 20002: serve("Embed/Two")}
        )
        result = await discover_cluster_endpoints(
            tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
        )
        assert result.embedding is not None
        assert result.embedding.model == "Embed/One"


# --- discovery log trail ------------------------------------------------------


class TestDiscoveryLog:
    """Discovery outcomes must leave a trail on the ``hpca.autoconnect``
    logger — silent skips are undebuggable in the TUI (which swallows them)."""

    async def test_key_locked_outcome_logged(self, tmp_path, caplog):
        write_manifest(tmp_path, "J1", 20001, _LOCKED_MODEL, needs_key=True)
        transport = make_transport({20001: locked(good_keys=("other",))})
        with caplog.at_level("INFO", logger="hpca.autoconnect"):
            await discover_cluster_endpoints(
                tmp_path,
                FakeSlurm({"J1": "RUNNING"}),
                api_keys=["stale-key"],
                transport=transport,
            )
        trail = caplog.text
        assert "key-locked" in trail
        assert _LOCKED_MODEL in trail
        assert "1 pool key(s)" in trail

    async def test_reap_and_no_answer_logged(self, tmp_path, caplog):
        write_manifest(tmp_path, "GONE", 20001, "Dead/Model")
        write_manifest(
            tmp_path, "J1", 20003, "Starting/Model", needs_key=False
        )
        transport = make_transport({})  # nothing answers
        with caplog.at_level("INFO", logger="hpca.autoconnect"):
            await discover_cluster_endpoints(
                tmp_path, FakeSlurm({"J1": "RUNNING"}), transport=transport
            )
        assert "gone from squeue" in caplog.text
        assert "not answering" in caplog.text

    async def test_squeue_failure_and_mismatch_logged(self, tmp_path, caplog):
        write_manifest(tmp_path, "J1", 20001, "Claimed/Model", needs_key=False)
        transport = make_transport({20001: serve("Other/Model")})
        with caplog.at_level("INFO", logger="hpca.autoconnect"):
            await discover_cluster_endpoints(
                tmp_path, FakeSlurm(error=True), transport=transport
            )
        assert "probe-only mode" in caplog.text
        assert "reused port" in caplog.text
