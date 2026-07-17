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
from urllib.parse import urlparse

import httpx

DEFAULT_PORT_RANGE = range(1024, 65536)
TCP_TIMEOUT_S = 0.25
PROBE_TIMEOUT_S = 2.0
SCAN_CONCURRENCY = 500


@dataclass
class DiscoveredBackend:
    base_url: str
    model: str
    max_model_len: int | None = None
    needs_key: bool = False

    def describe(self) -> str:
        """One-line display: name, context size, key requirement, endpoint."""
        if self.max_model_len:
            ctx = f"{self.max_model_len // 1000}k" if self.max_model_len >= 1000 else str(self.max_model_len)
        else:
            ctx = "?"
        location = urlparse(self.base_url).netloc
        key = "yes" if self.needs_key else "no"
        return f"{self.model} │ ctx {ctx} │ key: {key} │ {location}"


async def probe_endpoint(
    base_url: str,
    *,
    timeout: float = PROBE_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[DiscoveredBackend]:
    """Models served at one endpoint; [] if it is not an OpenAI-style API."""
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport
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
    ports=DEFAULT_PORT_RANGE, *, host: str = "127.0.0.1"
) -> list[DiscoveredBackend]:
    """Find OpenAI-compatible endpoints on the given localhost ports."""
    semaphore = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def check(port: int) -> int | None:
        async with semaphore:
            return port if await _port_open(host, port) else None

    open_ports = [
        p for p in await asyncio.gather(*(check(p) for p in ports)) if p is not None
    ]
    backends: list[DiscoveredBackend] = []
    for results in await asyncio.gather(
        *(probe_endpoint(f"http://{host}:{port}/v1") for port in open_ports)
    ):
        backends.extend(results)
    return backends


async def is_reachable(base_url: str, *, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Whether a configured backend currently answers /v1/models."""
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/", timeout=timeout
    ) as client:
        try:
            response = await client.get("models")
        except httpx.HTTPError:
            return False
    return response.status_code in (200, 401, 403)
