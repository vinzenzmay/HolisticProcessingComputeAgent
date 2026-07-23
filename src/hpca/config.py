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

from pydantic import BaseModel, ConfigDict, ValidationError

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
    max_retries: int = 3
    request_timeout_s: int = 120
    # Reasoning models think in a separate channel, shown in the chat window's
    # thinking box. Off by default: it is not generally better — small models
    # often route tools worse with it on — and it is slow. Measured on
    # Qwen3.6-27B: a turn took 133s thinking against 2.3s without. Per backend
    # in the catalog below; this is whichever one is active.
    enable_thinking: bool = False
    # Tool calls allowed in one turn before the agent must stop and summarise.
    # Each create_script+run_script (or run_bash) is one "look at the system"
    # step; a real check needs several, and debugging a script costs more. The
    # default is -1: no cap, the agent works until it is done. Set a positive
    # number to keep turns snappy or to bound autonomous work.
    max_tool_rounds: int = -1


class AgentSettings(_Section):
    """Interaction modes (§3.5): how much the agent may do unsupervised.

    ``default_mode`` is what a new session starts in (and what sessions
    that were never explicitly switched use): ``manual`` shows every script
    to the user before it runs, ``auto`` works until the task is done,
    ``full-auto`` additionally waives the destructive-op approvals, and
    ``plan`` drafts a checklist and executes nothing.
    """

    default_mode: Literal["manual", "auto", "full-auto", "plan"] = "manual"


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
    """

    endpoints_dir: str = "~/.hpca/endpoints"
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


class LLMBackend(_Section):
    """One entry of the configured backend catalog (manage-LLMs screen)."""

    model: str
    base_url: str
    api_key: str | None = None
    max_model_len: int | None = None
    # Thinking is a property of the model, not of the session: a reasoning
    # model may earn it while the next backend in the list does not.
    enable_thinking: bool = False


def llm_settings_for(backend: LLMBackend, base: LLMSettings) -> LLMSettings:
    """The LLMSettings for a client talking to ``backend``: the backend's
    connection fields over the base's client behavior (retries, timeout,
    constrained decoding, tool budget). Used for per-session clients."""
    settings = base.model_copy(deep=True)
    settings.base_url = backend.base_url
    settings.model = backend.model
    settings.api_key = backend.api_key
    settings.enable_thinking = backend.enable_thinking
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
        self.llm.enable_thinking = backend.enable_thinking

    def set_thinking(self, backend: LLMBackend, enabled: bool) -> None:
        """Toggle a catalog entry's thinking mode — and the live client's, if
        that entry is the one in use."""
        backend.enable_thinking = enabled
        if self.is_active(backend):
            self.llm.enable_thinking = enabled

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
