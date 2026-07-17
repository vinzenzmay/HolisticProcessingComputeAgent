"""Tests for hpca.discover: finding OpenAI-compatible endpoints on localhost."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from hpca.discover import (
    DiscoveredBackend,
    is_reachable,
    probe_endpoint,
    scan_local_ports,
)

VLLM_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "Qwen/Qwen3.6-27B-FP8",
            "object": "model",
            "max_model_len": 192000,
        }
    ],
}


def make_probe(payload=None, status=200):
    def handler(request):
        if payload is None:
            return httpx.Response(status, text="nope")
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


class TestProbeEndpoint:
    async def test_vllm_shape_parsed(self):
        backends = await probe_endpoint(
            "http://localhost:51941/v1", transport=make_probe(VLLM_MODELS)
        )
        assert len(backends) == 1
        b = backends[0]
        assert b.model == "Qwen/Qwen3.6-27B-FP8"
        assert b.base_url == "http://localhost:51941/v1"
        assert b.max_model_len == 192000
        assert b.needs_key is False

    async def test_auth_required_marks_needs_key(self):
        backends = await probe_endpoint(
            "http://localhost:1/v1",
            transport=make_probe({"error": "unauthorized"}, status=401),
        )
        assert len(backends) == 1
        assert backends[0].needs_key is True
        assert backends[0].model == "(api key required)"

    async def test_non_llm_service_ignored(self):
        assert await probe_endpoint(
            "http://localhost:1/v1", transport=make_probe(None)
        ) == []

    async def test_unreachable_ignored(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        assert await probe_endpoint(
            "http://localhost:1/v1", transport=httpx.MockTransport(handler)
        ) == []

    async def test_multiple_models_on_one_endpoint(self):
        payload = {
            "object": "list",
            "data": [{"id": "a"}, {"id": "b", "max_model_len": 256}],
        }
        backends = await probe_endpoint(
            "http://localhost:1/v1", transport=make_probe(payload)
        )
        assert [b.model for b in backends] == ["a", "b"]
        assert backends[1].max_model_len == 256


@pytest.fixture
def live_stub():
    """A real OpenAI-shaped /v1/models endpoint on an ephemeral port."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps(VLLM_MODELS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


class TestScanLocalPorts:
    async def test_finds_stub_among_closed_ports(self, live_stub):
        ports = range(live_stub - 3, live_stub + 4)
        backends = await scan_local_ports(ports)
        assert any(
            b.model == "Qwen/Qwen3.6-27B-FP8" and str(live_stub) in b.base_url
            for b in backends
        )

    async def test_all_closed_is_empty(self):
        # ports 47-53 in the reserved low range are extremely unlikely bound
        assert await scan_local_ports(range(47, 53)) == []


class TestIsReachable:
    async def test_reachable(self, live_stub):
        assert await is_reachable(f"http://127.0.0.1:{live_stub}/v1") is True

    async def test_unreachable(self):
        assert await is_reachable("http://127.0.0.1:47/v1") is False


class TestBackendDisplay:
    def test_describe(self):
        b = DiscoveredBackend(
            base_url="http://localhost:51941/v1",
            model="Qwen/Qwen3.6-27B-FP8",
            max_model_len=192000,
            needs_key=False,
        )
        text = b.describe()
        assert "Qwen/Qwen3.6-27B-FP8" in text
        assert "192k" in text
        assert "localhost:51941" in text
        assert "key: no" in text

    def test_describe_unknown_context(self):
        b = DiscoveredBackend(
            base_url="http://localhost:9/v1", model="m", max_model_len=None,
            needs_key=True,
        )
        text = b.describe()
        assert "ctx ?" in text
        assert "key: yes" in text


class TestScanProgress:
    async def test_progress_reported_in_chunks(self, live_stub):
        calls = []
        await scan_local_ports(
            range(1024, 1024 + 3000),
            progress=lambda done, total: calls.append((done, total)),
        )
        assert calls[-1] == (3000, 3000)
        assert len(calls) == 3  # 1024-port chunks
        assert calls[0][0] <= 1024


class TestIncrementalDiscovery:
    async def test_on_found_fires_before_scan_completes(self, live_stub):
        progress_calls = []
        found_at_progress: list[int] = []

        await scan_local_ports(
            # stub lands in the first of three 1024-port chunks
            range(live_stub - 100, live_stub - 100 + 3000),
            progress=lambda done, total: progress_calls.append(done),
            on_found=lambda b: found_at_progress.append(len(progress_calls)),
        )
        assert len(found_at_progress) == 1
        # found during the first chunk — before any later progress ticks
        assert found_at_progress[0] == 0


class TestOrderedPorts:
    def test_priority_then_likely_then_rest(self):
        from hpca.discover import LIKELY_PORTS, ordered_ports

        order = ordered_ports([51941])
        assert order[0] == 51941
        assert order[1 : 1 + len(LIKELY_PORTS) - 1]  # likely ports follow
        assert order.index(8000) < order.index(1024)
        assert len(order) == len(set(order))  # no duplicates
        assert len(order) == 64512  # full coverage preserved

    def test_invalid_priority_ports_dropped(self):
        from hpca.discover import ordered_ports

        order = ordered_ports([0, 70000, 8000])
        assert order[0] == 8000
        assert 70000 not in order
