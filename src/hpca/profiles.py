"""Agent profiles: two-tier memory in human-editable markdown (§6).

Format: YAML front matter, then ``## [tier1]`` / ``## [tier2]`` headings with
one memory per block, each optionally preceded by an inline metadata comment
``<!-- backend: qwen3-6b, created: 2026-07-16, kind: struggle -->``. Users
edit these files in vim/nano (§6.4), so the parser is lenient: anything it
cannot interpret becomes a ``problems`` entry, never an exception. Tier 3 is
accepted by the format but not injected anywhere yet (deferred, §6.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from hpca.config import app_dir

TIER_HEADING_RE = re.compile(r"^##\s*\[tier([123])\]\s*$")
HEADING_RE = re.compile(r"^##\s+")
META_RE = re.compile(r"^<!--\s*(.*?)\s*-->\s*$")


def profiles_dir() -> Path:
    return app_dir() / "profiles"


def estimate_tokens(text: str) -> int:
    """Soft token estimate (§6.4): tiktoken when available, else chars/4."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class Memory:
    text: str
    tier: int
    backend: str = ""
    created: str = ""
    kind: str = ""


@dataclass
class Profile:
    name: str
    created: str = ""
    default_backend: str = ""
    memories: list[Memory] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ io

    @staticmethod
    def path_for(name: str) -> Path:
        return profiles_dir() / f"{name}.md"

    @classmethod
    def load(cls, name: str) -> "Profile":
        path = cls.path_for(name)
        if not path.exists():
            return cls(name=name, created=date.today().isoformat())
        return cls.parse(path.read_text(), name=name)

    @staticmethod
    def list_profiles() -> list[str]:
        if not profiles_dir().exists():
            return []
        return sorted(p.stem for p in profiles_dir().glob("*.md"))

    def save(self) -> None:
        profiles_dir().mkdir(parents=True, exist_ok=True)
        self.path_for(self.name).write_text(self.render())

    # -------------------------------------------------------------- parsing

    @classmethod
    def parse(cls, text: str, *, name: str) -> "Profile":
        profile = cls(name=name)
        lines = text.splitlines()
        index = 0

        if lines and lines[0].strip() == "---":
            try:
                end = lines.index("---", 1)
                meta = yaml.safe_load("\n".join(lines[1:end])) or {}
                profile.created = str(meta.get("created", ""))
                profile.default_backend = str(meta.get("default_backend", ""))
                index = end + 1
            except (ValueError, yaml.YAMLError) as e:
                profile.problems.append(f"Unreadable front matter: {e}")
        elif text.strip():
            profile.problems.append("Missing YAML front matter")

        tier: int | None = None
        block_meta: dict[str, str] = {}
        block_lines: list[str] = []

        def flush() -> None:
            nonlocal block_meta, block_lines
            body = "\n".join(block_lines).strip()
            if body and tier is not None:
                profile.memories.append(
                    Memory(
                        text=body,
                        tier=tier,
                        backend=block_meta.get("backend", ""),
                        created=block_meta.get("created", ""),
                        kind=block_meta.get("kind", ""),
                    )
                )
            block_meta, block_lines = {}, []

        for line in lines[index:]:
            tier_match = TIER_HEADING_RE.match(line)
            if tier_match:
                flush()
                tier = int(tier_match.group(1))
                continue
            if HEADING_RE.match(line):
                flush()
                tier = None
                profile.problems.append(
                    f"Unknown heading ignored (use ## [tier1|tier2]): {line.strip()}"
                )
                continue
            meta_match = META_RE.match(line)
            if meta_match:
                flush()
                block_meta = _parse_meta(meta_match.group(1))
                continue
            block_lines.append(line)
        flush()
        return profile

    # ------------------------------------------------------------ rendering

    def render(self) -> str:
        front = yaml.safe_dump(
            {
                "name": self.name,
                "created": self.created or date.today().isoformat(),
                "default_backend": self.default_backend,
            },
            sort_keys=False,
        ).strip()
        parts = ["---", front, "---"]
        for tier in (1, 2, 3):
            memories = [m for m in self.memories if m.tier == tier]
            if tier == 3 and not memories:
                continue  # deferred tier: only written if something is in it
            parts += ["", f"## [tier{tier}]"]
            for memory in memories:
                meta_bits = [
                    f"{key}: {value}"
                    for key, value in (
                        ("backend", memory.backend),
                        ("created", memory.created),
                        ("kind", memory.kind),
                    )
                    if value
                ]
                parts.append("")
                if meta_bits:
                    parts.append(f"<!-- {', '.join(meta_bits)} -->")
                parts.append(memory.text)
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------- memories

    def add_memory(
        self, text: str, *, tier: int, backend: str = "", kind: str = ""
    ) -> Memory:
        memory = Memory(
            text=text.strip(),
            tier=tier,
            backend=backend,
            created=date.today().isoformat(),
            kind=kind,
        )
        self.memories.append(memory)
        return memory

    def tier_text(self, tier: int) -> str:
        return "\n\n".join(m.text for m in self.memories if m.tier == tier)

    def tier_tokens(self, tier: int) -> int:
        return estimate_tokens(self.tier_text(tier))

    def over_cap_tiers(self, *, tier1_cap: int, tier2_cap: int) -> list[int]:
        over = []
        if self.tier_tokens(1) > tier1_cap:
            over.append(1)
        if self.tier_tokens(2) > tier2_cap:
            over.append(2)
        return over


def _parse_meta(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta
