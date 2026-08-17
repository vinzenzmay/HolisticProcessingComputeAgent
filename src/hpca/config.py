"""Application settings (§7 of the project plan).

Settings live in ``~/.HolisticProcessingComputeAgent/settings.json`` and are
edited both by the in-app settings menu and by hand, so loading is lenient:
missing keys fall back to defaults and unknown keys are ignored. Only malformed
JSON or values of the wrong type/range are reported as errors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hpca.thinking import ThinkingEffort

APP_DIR_NAME = ".HolisticProcessingComputeAgent"


def app_dir() -> Path:
    """Application data directory; override with $HPCA_HOME (used by tests)."""
    override = os.environ.get("HPCA_HOME")
    if override:
        return Path(override)
    return Path.home() / APP_DIR_NAME


def settings_path() -> Path:
    return app_dir() / "settings.json"


class SettingsError(Exception):
    """Raised when the settings file cannot be parsed or validated."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class LLMSettings(_Section):
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    model: str = "qwen3-6b"
    constrained_decoding: Literal["auto", "on", "off"] = "auto"
    # How a tool call travels. "native" uses the backend's own tool-calling
    # channel — the shape agent-trained models were post-trained on, and it
    # moves the tool listing out of the prompt into the chat template.
    # "envelope" is the hand-rolled JSON decision object under a grammar
    # (§4.3), which works on any OpenAI-compatible backend, including one too
    # old to have a tool-call parser.
    #
    # Envelope stays the default, and v0.22.0 is where that stopped being a
    # portability argument and became a measured one. specs-edit-eval.md §7.3:
    # on Qwen3.8-27B the shape error that made native lose on Qwen3.6 is gone
    # and native is now correct — but over n=72 it costs 24.5% more completion
    # tokens and 20.3% more wall for the same success rate, and §7.4 is the
    # bigger reason: a cut-off long write cannot be salvaged on that channel,
    # because the parser drops the fragment instead of returning it.
    #
    # Native is a supported, tested choice — set it here, or per backend
    # below. It needs the server started for it (vLLM:
    # --enable-auto-tool-choice plus a --tool-call-parser matching the model,
    # which is qwen3_coder for the Qwen3.8 the cluster serves, not the hermes
    # the Qwen3.6 notes named); a backend without it rejects the request with
    # a 400 rather than quietly degrading, so the failure is loud and the fix
    # is this setting (or the per-backend override on LLMBackend below).
    #
    # The parser is not a detail: it decides what a *cut-off* call looks like.
    # qwen3_coder drops the argument the generation died inside and reports
    # finish_reason "tool_calls" anyway — see middleware._hit_token_cap, which
    # is what keeps that from costing the turn.
    #
    # Deliberately a setting and not a probe: the protocol shapes the whole
    # conversation — the system prompt's respond-vs-tool guidance moves with
    # it — so a verdict that arrives mid-session would leave the prompt
    # describing a format the model can no longer emit. That mismatch is the
    # most expensive bug this area has had (specs-edit-eval.md §7).
    tool_protocol: Literal["envelope", "native"] = "envelope"
    max_retries: int = 3
    request_timeout_s: int = 120
    # Reasoning models think in a separate channel, shown in the chat window's
    # thinking box. This is only the fallback for a caller that expresses no
    # opinion (the sub-agents that hard-code False, a bare client in a test):
    # a chat turn's thinking comes from the *session's* effort level
    # (`agent.default_thinking`, `/thinking`, hpca.thinking), which reaches the
    # wire per request. It has to be per request and not a client property,
    # because one client is shared by every session on the same backend
    # (core.backends.client_for_backend keys them by base_url||model).
    #
    # Off by default, and worth keeping off: thinking is not generally better —
    # small models often route tools worse with it on — and it is slow.
    # Measured on Qwen3.6-27B: a turn took 133s thinking against 2.3s without.
    enable_thinking: bool = False
    # Tool calls allowed in one turn before the agent must stop and summarise.
    # Each run_bash call is one "look at the system" step; a real check needs
    # several, and debugging a script costs more. The
    # default is -1: no cap, the agent works until it is done. Set a positive
    # number to keep turns snappy or to bound autonomous work.
    max_tool_rounds: int = -1


class AgentSettings(_Section):
    """What a session starts with, and falls back to if it never chose.

    ``default_mode`` is the interaction mode (§3.5): ``manual`` shows every
    script to the user before it runs, ``auto`` works until the task is done,
    and ``full-auto`` additionally waives the destructive-op approvals.

    ``default_thinking`` is the reasoning-effort dial (hpca.thinking), which
    has the same lifecycle: stored per session, empty there meaning "whatever
    this says", changeable mid-session with ``/thinking``. It defaults to
    ``off`` so an upgrade changes nothing about how turns run — every level
    above off costs a thinking pass before *every* decision in a turn, and the
    server's own default once thinking is on is the slowest level (xhigh), so
    "on" is not a defensible default for a dial the user has not touched.
    """

    default_mode: Literal["manual", "auto", "full-auto"] = "manual"
    default_thinking: ThinkingEffort = "off"


class ClusterSettings(_Section):
    submit_host: str | None = None
    job_poll_seconds: int = 30


class SafetySettings(_Section):
    backup_limit_gb: float = 1
    trash_ttl_days: int = 7


class MemorySettings(_Section):
    # Token budget for the injected system-prompt memory scope. Hard for writes
    # — a full scope rejects new memories until the user condenses it — but
    # injection never truncates what is in the file. RAG has no budget.
    system_prompt_token_cap: int = 2400
    # Whether session_search may recall sessions of OTHER profiles. Off by
    # default: profiles exist to isolate what each context learns and sees.
    cross_profile_search: bool = False
    # RAG (retrieved memory): not injected wholesale, only the entries matching
    # the current request, within this per-turn prefetch budget.
    rag_prefetch_chars: int = 800
    rag_prefetch_count: int = 3
    # Curator: ages RAG entries out so retrieval does not decay as notes
    # accumulate. Archived entries are moved to <profile>.archive.md, never
    # deleted. 0 disables the pass.
    curator_interval_days: int = 7
    curator_stale_days: int = 30
    curator_archive_days: int = 90
    # Whether /conclude self-review may propose entirely NEW skills, as opposed
    # to patches to existing ones. On so the quality can be judged in practice;
    # set false if a small model's skill drafts prove not worth reviewing.
    propose_new_skills: bool = True


ClipboardMode = Literal["auto", "tmux", "screen", "zellij", "osc52", "command", "file"]


class ClipboardSettings(_Section):
    mode: ClipboardMode = "auto"
    command: str | None = None
    osc52_limit_kb: int = 74


class LoggingSettings(_Section):
    """Plain-text session transcripts for later analysis (see hpca.logs)."""

    enabled: bool = True
    # Default: <app dir>/chatlogs. Set to collect them somewhere else, e.g. on
    # a project share; ~ is expanded.
    dir: str | None = None


class RagSettings(_Section):
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_base_url: str = "http://localhost:20000/v1"


class EndpointsSettings(_Section):
    """Auto-connect: cluster endpoint discovery (spec §6).

    On the cluster, HPCA reads vLLM manifest files (written by the launch
    scripts) from ``endpoints_dir`` and connects directly. Off the cluster
    that dir is empty, so it falls through to the localhost scan and prints an
    SSH-tunnel template built from ``login_host``/``login_user``.

    The default is the shared group directory: manifests written by anyone on
    the team are visible to everyone's HPCA, so a user connects to a server
    they did not launch without configuring anything. Job ids are unique
    cluster-wide, so manifests from different users cannot collide. Point this
    somewhere private (and set ``HPCA_ENDPOINTS_DIR`` for the launch scripts)
    to opt out.
    """

    endpoints_dir: str = "/data/cephfs-1/work/groups/cubi/tools/hpca_connections"
    # Optional ordered model-id substrings for full auto-connect to the top
    # available match; empty => the "auto if one, list if many" default.
    preferred_models: list[str] = []
    login_host: str = "hpc-login-2.cubi.bihealth.org"
    login_user: str | None = None

    def dir_path(self) -> Path:
        return Path(os.path.expanduser(self.endpoints_dir))

    def login_target(self) -> str:
        """``user@host`` for the tunnel template, or just ``host`` if unset."""
        if self.login_user:
            return f"{self.login_user}@{self.login_host}"
        return self.login_host


class DatabaseSettings(_Section):
    """Where the sqlite databases actually run (see specs-db-local-cache.md).

    On a cluster node ``$HOME`` is NFS, where every sqlite call is network
    round-trips plus a remote fsync, and the TUI lags whenever the agent
    works. With ``local_cache`` on, the databases are copied to node-local
    storage at startup, used from there, and synced back to home periodically
    and on exit.
    """

    local_cache: bool = True
    # Where the working copies go. None => $TMPDIR (the per-job dir Slurm
    # sets, which is what makes them node-local), else the system temp dir.
    local_dir: str | None = None
    # How often the working copies are written back, and so the upper bound on
    # what a hard kill (walltime, node failure, SIGKILL) can lose. 0 syncs only
    # on exit.
    sync_interval_s: int = Field(default=60, ge=0)


class LLMBackend(_Section):
    """One entry of the configured backend catalog (manage-LLMs screen).

    There is no ``enable_thinking`` here any more. It was a per-backend flag on
    the reasoning that thinking is a property of the model — true, but the
    thing a user actually adjusts is how hard *this conversation* should think,
    and answering that by editing the catalog changed every session on that
    backend at once. It is now a per-session level (hpca.thinking, ``/thinking``).
    Entries written before v0.22.0 still carry the key; ``extra="ignore"`` on
    ``_Section`` drops it on load, so an old settings file needs no migration.
    """

    model: str
    base_url: str
    api_key: str | None = None
    max_model_len: int | None = None
    # Whether this server can carry a call on its own tool channel is a
    # property of how it was launched, not of the client — one entry may be a
    # vLLM started with --enable-auto-tool-choice while the next is an older
    # one that 400s on `tools`. None means "whatever llm.tool_protocol says",
    # which is what every entry written before v0.22.0 has.
    tool_protocol: Literal["envelope", "native"] | None = None


def llm_settings_for(backend: LLMBackend, base: LLMSettings) -> LLMSettings:
    """The LLMSettings for a client talking to ``backend``: the backend's
    connection fields over the base's client behavior (retries, timeout,
    constrained decoding, tool budget). Used for per-session clients."""
    settings = base.model_copy(deep=True)
    settings.base_url = backend.base_url
    settings.model = backend.model
    settings.api_key = backend.api_key
    if backend.tool_protocol is not None:
        settings.tool_protocol = backend.tool_protocol
    return settings


class Settings(_Section):
    llm: LLMSettings = LLMSettings()
    agent: AgentSettings = AgentSettings()
    backends: list[LLMBackend] = []
    known_llm_ports: list[int] = []
    llm_api_keys: list[str] = []
    cluster: ClusterSettings = ClusterSettings()
    safety: SafetySettings = SafetySettings()
    memory: MemorySettings = MemorySettings()
    clipboard: ClipboardSettings = ClipboardSettings()
    editor: str | None = None
    rag: RagSettings = RagSettings()
    logging: LoggingSettings = LoggingSettings()
    endpoints: EndpointsSettings = EndpointsSettings()
    database: DatabaseSettings = DatabaseSettings()

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or settings_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise SettingsError(f"Malformed JSON in {path}: {e}") from e
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise SettingsError(f"Invalid settings in {path}: {e}") from e

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    def activate_backend(self, backend: LLMBackend) -> None:
        """Make a catalog entry the active/default backend (used on startup).

        Only connection fields change; client behavior (retries, timeout,
        constrained decoding) is kept.
        """
        self.llm.base_url = backend.base_url
        self.llm.model = backend.model
        self.llm.api_key = backend.api_key
        if backend.tool_protocol is not None:
            self.llm.tool_protocol = backend.tool_protocol

    def remember_llm_ports(self, base_urls: Iterable[str]) -> bool:
        """Record ports that have served an LLM, so scans probe them first.

        Which GPU node hosts a model changes often, the tunneled local port
        rarely does. Returns whether anything new was learned, i.e. whether
        the settings need saving.
        """
        learned = False
        for base_url in base_urls:
            port = urlparse(base_url).port
            if port is not None and port not in self.known_llm_ports:
                self.known_llm_ports.append(port)
                learned = True
        return learned

    def remember_llm_key(self, api_key: str | None) -> bool:
        """Add an API key to the pool if new. Returns whether it was newly
        learned (i.e. whether settings need saving).

        Keys already known to the app are tried against key-locked endpoints,
        so any key typed for one backend can unlock the others. Empty or
        missing keys are ignored; dedup is by exact string.
        """
        if not api_key:
            return False
        if api_key in self.llm_api_keys:
            return False
        self.llm_api_keys.append(api_key)
        return True

    def is_active(self, backend: LLMBackend) -> bool:
        return (
            self.llm.base_url == backend.base_url
            and self.llm.model == backend.model
        )
