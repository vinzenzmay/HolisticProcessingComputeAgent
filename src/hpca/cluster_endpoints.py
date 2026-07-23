"""Manifest-driven discovery of this session's cluster endpoints (§4).

Launch scripts drop one JSON manifest per vLLM server into the endpoints dir
(``<jobid>-<port>.json``). Discovery reconciles those manifests with reality
in three layers, cheapest first, because a manifest is only a *claim*: the job
may have ended, the server may still be starting, or the port may have been
reused by a different job serving a different model.

1. **Slurm liveness** — a job Slurm no longer lists is gone; its manifest is
   stale and gets reaped (unless SLURM itself is unreachable, in which case we
   trust nothing about liveness and fall back to probe-only).
2. **Live probe** — a live allocation whose ``/v1/models`` does not answer is
   not surfaced but also not reaped (vLLM may still be starting inside a live
   job).
3. **Model-id match** — the probe's reported model id must equal the manifest's
   ``model``; a mismatch means a reused port serving something else, so we
   neither surface nor delete it (the real job may be alive elsewhere).

The one exception to "verify the model id" is a key-locked endpoint no pool key
unlocked: we cannot read its model list, but the manifest names what it should
be, so we surface a NAMED ``needs_key`` backend for the existing add-a-key flow
rather than the anonymous ``(api key required)`` sentinel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

from hpca.discover import KEY_REQUIRED, DiscoveredBackend, probe_endpoint
from hpca.slurm import SlurmClient, SlurmError

# Manifest keys that must be present and coercible; anything else is skipped
# rather than raised, since these files are written by external scripts.
_REQUIRED = ("role", "model", "jobid", "ip", "port")


@dataclass
class Manifest:
    role: str
    model: str
    jobid: str
    user: str
    node: str
    ip: str
    port: int
    ctx_len: int | None
    needs_key: bool
    started: str
    path: Path  # source file, kept so a stale manifest can be reaped

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.port}/v1"


def _parse_manifest(data: dict, path: Path) -> Manifest | None:
    """One JSON dict -> Manifest, or None if a required field is bad."""
    if not all(k in data for k in _REQUIRED):
        return None
    try:
        port = int(data["port"])
    except (TypeError, ValueError):
        return None
    # Required strings must actually be strings (a numeric jobid is fine to
    # coerce, but a list/dict is not a model name).
    for key in ("role", "model", "jobid", "ip"):
        if not isinstance(data[key], (str, int)):
            return None
    return Manifest(
        role=str(data["role"]),
        model=str(data["model"]),
        jobid=str(data["jobid"]),
        user=str(data.get("user", "")),
        node=str(data.get("node", "")),
        ip=str(data["ip"]),
        port=port,
        ctx_len=data.get("ctx_len"),
        needs_key=bool(data.get("needs_key", False)),
        started=str(data.get("started", "")),
        path=path,
    )


def read_manifests(endpoints_dir: Path) -> list[Manifest]:
    """Parse every ``*.json`` manifest in the dir; malformed ones are skipped.

    A missing directory yields ``[]`` (no cluster endpoints declared yet).
    Files are returned in sorted filename order for a deterministic result.
    """
    if not endpoints_dir.is_dir():
        return []
    manifests: list[Manifest] = []
    for path in sorted(endpoints_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        manifest = _parse_manifest(data, path)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


@dataclass
class ClusterEndpoints:
    llms: list[DiscoveredBackend]
    embedding: DiscoveredBackend | None = None


async def discover_cluster_endpoints(
    endpoints_dir: Path,
    slurm: SlurmClient,
    *,
    api_keys: Sequence[str] = (),
    reap: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ClusterEndpoints:
    """Reconcile manifests with Slurm liveness and a live probe (see module doc)."""
    manifests = read_manifests(endpoints_dir)
    if not manifests:
        return ClusterEndpoints([], None)

    # Liveness in one squeue call. A SlurmError means the controller is
    # unreachable — we must NOT read that as "every job dead" and reap the lot,
    # so drop to probe-only mode by leaving states as None.
    job_ids = sorted({m.jobid for m in manifests})
    try:
        states: dict[str, str] | None = await slurm.job_states(job_ids)
    except SlurmError:
        states = None

    llms: list[DiscoveredBackend] = []
    embedding: DiscoveredBackend | None = None

    for m in manifests:
        # Liveness gate: only when Slurm answered. A job Slurm no longer lists
        # is gone; reap its stale manifest (idempotent — another HPCA instance
        # may have deleted it already) and move on.
        if states is not None and m.jobid not in states:
            if reap:
                try:
                    m.path.unlink()
                except OSError:
                    pass
            continue

        rows = await probe_endpoint(
            m.base_url, api_keys=api_keys, transport=transport
        )

        # Empty: alive allocation but vLLM not answering (starting up, crashed
        # inside a live job, or a non-LLM on a reused port). Don't surface,
        # don't delete — the job may still be coming up.
        if not rows:
            continue

        # Key-locked and no pool key unlocked it: we can't verify the model id,
        # but the manifest names it. Surface a NAMED needs_key backend for the
        # add-a-key flow; never auto-connect it.
        if len(rows) == 1 and rows[0].model == KEY_REQUIRED:
            backend = DiscoveredBackend(
                base_url=m.base_url,
                model=m.model,
                max_model_len=m.ctx_len,
                needs_key=True,
            )
        else:
            # Correctness guard: keep only rows whose model id matches the
            # manifest. A mismatch is a reused/stolen port serving something
            # else — don't surface, don't delete (the real job may be alive).
            matches = [r for r in rows if r.model == m.model]
            if not matches:
                continue
            # Prefer the probe's row: it carries the server's real
            # max_model_len and any pool api_key that unlocked it.
            backend = matches[0]

        if m.role == "embedding":
            # Effectively one embeddings server; keep the first live match.
            if embedding is None:
                embedding = backend
        else:
            llms.append(backend)

    return ClusterEndpoints(llms=llms, embedding=embedding)
