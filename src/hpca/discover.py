"""Discovery of OpenAI-compatible LLM endpoints on localhost.

On this HPC setup every backend is a vLLM server on some GPU node reached
through an SSH-tunneled local port, so "find available LLMs" reduces to
scanning localhost ports and probing ``/v1/models``. Closed ports on
localhost fail instantly (ECONNREFUSED), so even the full range scan takes
seconds. A 401 marks an endpoint that exists but needs an API key.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import httpx

DEFAULT_PORT_RANGE = range(1024, 65536)
# Scanned first so known/likely endpoints appear within the first chunks:
# common serving ports (vLLM/llama.cpp defaults 8000-8100, 5000s) and this
# site's tunnel conventions (419xx/519xx).
LIKELY_PORTS = (
    list(range(8000, 8101))
    + [5000, 5001, 11434]
    + list(range(41900, 42001))
    + list(range(51900, 52001))
)
TCP_TIMEOUT_S = 0.25
PROBE_TIMEOUT_S = 2.0
SCAN_CHUNK = 1024  # ports probed concurrently per batch


def ordered_ports(priority: list[int] = ()) -> list[int]:
    """Full scan order: caller-known ports, likely ports, then everything."""
    seen: set[int] = set()
    ordered: list[int] = []
    for port in [*priority, *LIKELY_PORTS, *DEFAULT_PORT_RANGE]:
        if 0 < port < 65536 and port not in seen:
            seen.add(port)
            ordered.append(port)
    return ordered


@dataclass
class DiscoveredBackend:
    base_url: str
    model: str
    max_model_len: int | None = None
    needs_key: bool = False

    def details(self) -> str:
        """Everything but the name: context size, key requirement, endpoint."""
        if self.max_model_len:
            ctx = (
                f"{self.max_model_len // 1000}k"
                if self.max_model_len >= 1000
                else str(self.max_model_len)
            )
        else:
            ctx = "?"
        location = urlparse(self.base_url).netloc
        key = "yes" if self.needs_key else "no"
        return f"ctx {ctx} │ key: {key} │ {location}"

    def describe(self) -> str:
        """One-line display: name, context size, key requirement, endpoint."""
        return f"{self.model} │ {self.details()}"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def probe_endpoint(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = PROBE_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[DiscoveredBackend]:
    """Models served at one endpoint; [] if it is not an OpenAI-style API.

    Pass ``api_key`` to authenticate the probe: the scan leaves it unset (a
    key-locked endpoint just surfaces as ``needs_key``), but validating a key
    the user just typed sends it and reads a lingering 401 as "key rejected".
    """
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers=_auth_headers(api_key),
        timeout=timeout,
        transport=transport,
    ) as client:
        try:
            response = await client.get("models")
        except (httpx.HTTPError, Exception):
            return []
    if response.status_code in (401, 403):
        return [
            DiscoveredBackend(
                base_url=base_url, model="(api key required)", needs_key=True
            )
        ]
    if response.status_code != 200:
        return []
    try:
        entries = response.json().get("data", [])
    except ValueError:
        return []
    if not isinstance(entries, list):
        return []
    backends = []
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        backends.append(
            DiscoveredBackend(
                base_url=base_url,
                model=str(entry["id"]),
                max_model_len=entry.get("max_model_len"),
            )
        )
    return backends


async def _port_open(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=TCP_TIMEOUT_S
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


async def scan_local_ports(
    ports=DEFAULT_PORT_RANGE,
    *,
    host: str = "127.0.0.1",
    progress: Callable[[int, int], None] | None = None,
    on_found: Callable[[DiscoveredBackend], None] | None = None,
) -> list[DiscoveredBackend]:
    """Find OpenAI-compatible endpoints on the given localhost ports.

    Ports are probed in bounded chunks (never tens of thousands of pending
    coroutines at once). Open ports are probed for models as soon as their
    chunk finishes and each hit is reported via ``on_found`` immediately, so
    UIs can show results while the sweep continues; ``progress(done, total)``
    fires after every chunk. Callers embedding a UI should run this scan in a
    separate thread with its own event loop so their loop stays responsive.
    """
    port_list = list(ports)
    total = len(port_list)
    backends: list[DiscoveredBackend] = []
    for start in range(0, total, SCAN_CHUNK):
        chunk = port_list[start : start + SCAN_CHUNK]
        results = await asyncio.gather(*(_port_open(host, p) for p in chunk))
        open_ports = [p for p, is_open in zip(chunk, results) if is_open]
        for probe_results in await asyncio.gather(
            *(probe_endpoint(f"http://{host}:{port}/v1") for port in open_ports)
        ):
            for backend in probe_results:
                backends.append(backend)
                if on_found is not None:
                    on_found(backend)
        if progress is not None:
            progress(min(start + SCAN_CHUNK, total), total)
    return backends


async def is_reachable(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = PROBE_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """Whether a configured backend currently answers /v1/models.

    With ``api_key`` the request is authenticated and only a 200 counts as
    reachable — a 401/403 then means the stored key is bad, which should read
    as disconnected rather than the misleading "connected" an unauthenticated
    probe would show. Without a key, a 401/403 still counts as reachable (the
    endpoint is up, it just needs a key we are not supplying here).
    """
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers=_auth_headers(api_key),
        timeout=timeout,
        transport=transport,
    ) as client:
        try:
            response = await client.get("models")
        except httpx.HTTPError:
            return False
    if api_key:
        return response.status_code == 200
    return response.status_code in (200, 401, 403)
