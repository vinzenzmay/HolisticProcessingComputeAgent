"""Per-session context handed to every tool handler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hpca.config import Settings
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner


@dataclass
class ToolContext:
    registry: PathRegistry
    runner: ProcessRunner
    settings: Settings
    scripts_dir: Path
