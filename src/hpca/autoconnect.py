"""Pre-agent connection decision for discovered cluster endpoints (spec §4.4/§4.6).

This is the pure logic behind auto-connect: given what
``cluster_endpoints.discover_cluster_endpoints`` found, decide which LLM (if
any) to connect to without asking, which embeddings endpoint to wire, and — when
nothing was found — what tunnel help to show. It runs before any LLM is
involved and never touches I/O, so it is trivially testable; the TUI applies its
result.

Selection is rule (c): "auto if one, list if many", with an optional
``preferred_models`` override for full auto-connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from hpca.cluster_endpoints import ClusterEndpoints
from hpca.discover import DiscoveredBackend


def is_connectable(backend: DiscoveredBackend) -> bool:
    """Whether HPCA may connect silently: an open endpoint, or a locked one a
    pool key already unlocked. A locked endpoint with no key must not
    auto-connect — it surfaces for the user to add a key (existing flow)."""
    return (not backend.needs_key) or (backend.api_key is not None)


@dataclass
class AutoConnectPlan:
    # The single LLM to activate now (one live one, or the top preferred match);
    # None when the choice is ambiguous or nothing is connectable.
    connect: DiscoveredBackend | None = None
    # Every live LLM discovered (including locked ones), for the picker / list.
    choices: list[DiscoveredBackend] = field(default_factory=list)
    # The embeddings endpoint to wire RAG to, if one is live.
    embedding_base_url: str | None = None


def _first_preferred(
    connectable: list[DiscoveredBackend], preferred_models: Sequence[str]
) -> DiscoveredBackend | None:
    """First connectable backend matching the earliest preferred pattern.

    Patterns are matched as case-insensitive substrings of the model id, tried
    in the caller's order, so ``["27B", "35B"]`` means "27B if it's up, else
    35B".
    """
    for pattern in preferred_models:
        needle = pattern.lower()
        for backend in connectable:
            if needle in backend.model.lower():
                return backend
    return None


def plan_auto_connect(
    endpoints: ClusterEndpoints,
    *,
    preferred_models: Sequence[str] = (),
) -> AutoConnectPlan:
    """Decide what to connect to from the discovered endpoints (rule (c))."""
    choices = list(endpoints.llms)
    connectable = [b for b in choices if is_connectable(b)]

    connect: DiscoveredBackend | None = None
    if preferred_models:
        connect = _first_preferred(connectable, preferred_models)
    if connect is None and len(connectable) == 1:
        connect = connectable[0]

    embedding_base_url = (
        endpoints.embedding.base_url if endpoints.embedding is not None else None
    )
    return AutoConnectPlan(
        connect=connect,
        choices=choices,
        embedding_base_url=embedding_base_url,
    )


def offcluster_help(login_target: str) -> str:
    """The generic tunnel template (§4.6) shown when nothing connects directly.

    ``login_target`` is ``user@host`` (or bare ``host``). node/port stay
    placeholders the user fills from the manifests — HPCA reads no remote files
    and spawns no SSH.
    """
    return (
        "Couldn't connect to a backend directly.\n"
        "On your workstation? Create a tunnel, then rescan.\n"
        "\n"
        "  1. On the cluster, list running endpoints:\n"
        "       cat ~/.hpca/endpoints/*.json\n"
        "  2. From your workstation, forward BOTH the LLM and the embeddings\n"
        "     server (they may be on different nodes) — one ssh, two -L forwards:\n"
        "\n"
        "       ssh -fN \\\n"
        "         -L <llm_port>:<llm_node>:<llm_port> \\\n"
        "         -L 20000:<embed_node>:20000 \\\n"
        f"         {login_target}\n"
        "\n"
        "  local port == remote port, so your app config just works."
    )
