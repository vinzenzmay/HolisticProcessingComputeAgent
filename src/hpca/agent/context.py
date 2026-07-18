"""Per-session context handed to every tool handler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hpca.config import Settings
from hpca.embeddings import EmbeddingClient
from hpca.jobs import JobStore
from hpca.rag import RagStore
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.skills import Skill
from hpca.slurm import SlurmClient
from hpca.symbols import SymbolIndex
from hpca.trash import TrashManager


@dataclass
class ToolContext:
    registry: PathRegistry
    runner: ProcessRunner
    settings: Settings
    scripts_dir: Path
    session_id: str = ""
    profile: str = "default"
    slurm: SlurmClient | None = None
    jobs: JobStore | None = None
    job_log_dir: Path | None = None
    llm: object | None = None  # for tools that run their own firewalled LLM call
    current_tool: str = ""  # set by the graph; names sub-agent calls in the log
    trash: TrashManager | None = None
    tier1_text: str = ""  # standing notes for subagent-style tool calls (§6.1)
    symbols: SymbolIndex | None = None
    rag: RagStore | None = None
    embedder: EmbeddingClient | None = None
    skills: list[Skill] = field(default_factory=list)
    # Commands whose docs could not be fetched; probed at most once per session
    doc_probe_failed: set[str] = field(default_factory=set)
