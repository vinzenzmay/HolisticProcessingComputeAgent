"""Per-session context handed to every tool handler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hpca.config import Settings
from hpca.embeddings import EmbeddingClient
from hpca.episodic import EpisodicStore
from hpca.jobs import JobStore
from hpca.rag import RagStore
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.skills import Skill
from hpca.slurm import SlurmClient
from hpca.symbols import SymbolIndex
from hpca.trash import TrashManager
from hpca.watches import WatchStore


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
    # Logs and jobs pinned to the right column by the watch tools (§3.3).
    watches: WatchStore | None = None
    llm: object | None = None  # for tools that run their own firewalled LLM call
    current_tool: str = ""  # set by the graph; names sub-agent calls in the log
    trash: TrashManager | None = None
    symbols: SymbolIndex | None = None
    rag: RagStore | None = None
    episodic: EpisodicStore | None = None  # past-session recall
    # Queues a proposed memory batch for review at the next /conclude. The
    # agent may flag facts mid-conversation, but nothing is written until the
    # user concludes the session; returns a confirmation for the tool result.
    queue_memory_edits: Callable[[list], str] | None = None
    embedder: EmbeddingClient | None = None
    skills: list[Skill] = field(default_factory=list)
    # Commands whose docs could not be fetched; probed at most once per session
    doc_probe_failed: set[str] = field(default_factory=set)
    # Files the user has already approved an edit_file on, as resolved paths.
    # In auto mode the consent covers the file, not the single diff: further
    # edit_file calls to the same path run without re-gating (§3.5), so a
    # section-by-section fill of a long document costs one approval, not ten.
    approved_edit_paths: set[Path] = field(default_factory=set)
