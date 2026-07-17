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
from typing import Literal

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


class ClusterSettings(_Section):
    submit_host: str | None = None
    job_poll_seconds: int = 30


class SafetySettings(_Section):
    backup_limit_gb: float = 1
    trash_ttl_days: int = 7


class MemorySettings(_Section):
    tier1_token_cap: int = 300
    tier2_token_cap: int = 800


ClipboardMode = Literal["auto", "tmux", "screen", "zellij", "osc52", "command", "file"]


class ClipboardSettings(_Section):
    mode: ClipboardMode = "auto"
    command: str | None = None
    osc52_limit_kb: int = 74


class RagSettings(_Section):
    store: Literal["sqlite-vec", "chromadb"] = "sqlite-vec"
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_base_url: str = "http://localhost:51943/v1"


class LLMBackend(_Section):
    """One entry of the configured backend catalog (manage-LLMs screen)."""

    model: str
    base_url: str
    api_key: str | None = None
    max_model_len: int | None = None


class Settings(_Section):
    llm: LLMSettings = LLMSettings()
    backends: list[LLMBackend] = []
    cluster: ClusterSettings = ClusterSettings()
    safety: SafetySettings = SafetySettings()
    memory: MemorySettings = MemorySettings()
    clipboard: ClipboardSettings = ClipboardSettings()
    editor: str | None = None
    rag: RagSettings = RagSettings()

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

    def is_active(self, backend: LLMBackend) -> bool:
        return (
            self.llm.base_url == backend.base_url
            and self.llm.model == backend.model
        )
