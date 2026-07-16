"""Per-session context handed to every tool handler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hpca.config import Settings
from hpca.jobs import JobStore
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.slurm import SlurmClient


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
