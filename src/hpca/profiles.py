"""Agent profiles: two-scope memory in human-editable markdown (§6).

Format: YAML front matter, then ``## [system-prompt]`` / ``## [rag]`` headings
with one memory per block, each optionally preceded by an inline metadata comment
``<!-- backend: qwen3-6b, created: 2026-07-16, kind: struggle -->``. Users edit
these files in vim/nano (§6.4), so the parser is lenient: anything it cannot
interpret becomes a ``problems`` entry, never an exception.

Two scopes, and only two:

* **system-prompt** — injected into the orchestrator prompt every turn, so it is
  held to a hard token budget (a full scope rejects new writes until condensed).
* **rag** — retrieved only when a request matches it (see :mod:`hpca.memory_index`);
  unbounded, because it costs nothing until recalled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path

import yaml

from hpca.config import app_dir

DEFAULT_PROFILE = "default"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$")


class MemoryScope(str, Enum):
    """Where a memory lives, and therefore how it reaches the model."""

    SYSTEM_PROMPT = "system-prompt"
    RAG = "rag"


SCOPE_HEADING_RE = re.compile(r"^##\s*\[(system-prompt|rag)\]\s*$")
HEADING_RE = re.compile(r"^##\s+")
META_RE = re.compile(r"^<!--\s*(.*?)\s*-->\s*$")


def profiles_dir() -> Path:
    return app_dir() / "profiles"


def estimate_tokens(text: str) -> int:
    """Token count for the memory budget: tiktoken when available, else chars/4.

    tiktoken's ``cl100k_base`` is not the active backend's exact tokenizer, so
    this is a close estimate rather than an exact count — deterministic and
    dependency-light, which is what a write-time budget check needs.
    """
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
    scope: MemoryScope
    backend: str = ""
    created: str = ""
    kind: str = ""


@dataclass
class Profile:
    name: str
    created: str = ""
    default_backend: str = ""
    # Which profile this one was copied from, and when. Two profiles that
    # share a base diverge from the moment of the copy, and months later the
    # only way to know why they overlap is if the copy said so.
    copied_from: str = ""
    copied_on: str = ""
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
        """Every profile that exists, with the default first and always
        present — it is the fallback, so it cannot be missing."""
        names = set()
        if profiles_dir().exists():
            names = {p.stem for p in profiles_dir().glob("*.md")}
        names.add(DEFAULT_PROFILE)
        return [DEFAULT_PROFILE] + sorted(names - {DEFAULT_PROFILE})

    @staticmethod
    def validate_name(name: str) -> str:
        """The name is a filename: reject what would escape the directory or
        collide. Returns the cleaned name; raises ValueError with the reason."""
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("A profile needs a name.")
        if not NAME_RE.match(cleaned):
            raise ValueError(
                "Use letters, digits, spaces, dots, dashes or underscores."
            )
        if cleaned in Profile.list_profiles():
            raise ValueError(f"A profile called “{cleaned}” already exists.")
        return cleaned

    @classmethod
    def create(cls, name: str) -> "Profile":
        """A new, empty profile on disk (its name already validated)."""
        profile = cls(name=name, created=date.today().isoformat())
        profile.save()
        return profile

    @classmethod
    def duplicate(cls, source: str, new_name: str) -> "Profile":
        """A copy of ``source`` under ``new_name``, free to diverge from it.

        Everything the profile has *learned* comes along — both scopes, struggle
        notes, the lot — because the point of a copy is to start specialised work
        from an established base rather than from nothing. Sessions do not: they
        belong to the conversations that happened, not to the knowledge that came
        out of them. Skills are copied by the caller
        (``skills.copy_profile_skills``), which owns that directory.

        The copy records where it came from, so two profiles that share a
        base can still be told apart from two that merely resemble each other.
        """
        original = cls.load(source)
        today = date.today().isoformat()
        copy = cls(
            name=new_name,
            created=today,
            default_backend=original.default_backend,
            copied_from=source,
            copied_on=today,
            memories=[Memory(**vars(memory)) for memory in original.memories],
        )
        copy.save()
        return copy

    @staticmethod
    def delete(name: str) -> None:
        """Remove a profile. The default is the fallback for sessions whose
        profile is deleted, so it is not removable."""
        if name == DEFAULT_PROFILE:
            raise ValueError("The default profile cannot be deleted.")
        Profile.path_for(name).unlink(missing_ok=True)

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
                profile.copied_from = str(meta.get("copied_from", ""))
                profile.copied_on = str(meta.get("copied_on", ""))
                index = end + 1
            except (ValueError, yaml.YAMLError) as e:
                profile.problems.append(f"Unreadable front matter: {e}")
        elif text.strip():
            profile.problems.append("Missing YAML front matter")

        scope: MemoryScope | None = None
        block_meta: dict[str, str] = {}
        block_lines: list[str] = []

        def flush() -> None:
            nonlocal block_meta, block_lines
            body = "\n".join(block_lines).strip()
            if body and scope is not None:
                profile.memories.append(
                    Memory(
                        text=body,
                        scope=scope,
                        backend=block_meta.get("backend", ""),
                        created=block_meta.get("created", ""),
                        kind=block_meta.get("kind", ""),
                    )
                )
            block_meta, block_lines = {}, []

        for line in lines[index:]:
            scope_match = SCOPE_HEADING_RE.match(line)
            if scope_match:
                flush()
                scope = MemoryScope(scope_match.group(1))
                continue
            if HEADING_RE.match(line):
                flush()
                scope = None
                profile.problems.append(
                    "Unknown heading ignored (use ## [system-prompt|rag]): "
                    f"{line.strip()}"
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
        meta = {
            "name": self.name,
            "created": self.created or date.today().isoformat(),
            "default_backend": self.default_backend,
        }
        if self.copied_from:  # only on copies, so ordinary files stay clean
            meta["copied_from"] = self.copied_from
            meta["copied_on"] = self.copied_on
        front = yaml.safe_dump(meta, sort_keys=False).strip()
        parts = ["---", front, "---"]
        for scope in (MemoryScope.SYSTEM_PROMPT, MemoryScope.RAG):
            memories = [m for m in self.memories if m.scope == scope]
            if scope is MemoryScope.RAG and not memories:
                continue  # retrieved scope: only written if something is in it
            parts += ["", f"## [{scope.value}]"]
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
        self,
        text: str,
        *,
        scope: MemoryScope,
        backend: str = "",
        kind: str = "",
    ) -> Memory:
        memory = Memory(
            text=text.strip(),
            scope=scope,
            backend=backend,
            created=date.today().isoformat(),
            kind=kind,
        )
        self.memories.append(memory)
        return memory

    def scope_text(self, scope: MemoryScope) -> str:
        return "\n\n".join(m.text for m in self.memories if m.scope == scope)

    def system_prompt_text(self, *, active_backend: str = "") -> str:
        """The system-prompt scope as injected: memories learned on a different
        backend are annotated, not dropped — a workaround for one small model
        often transfers, and the annotation lets the model weigh it."""
        parts = []
        for memory in self.memories:
            if memory.scope is not MemoryScope.SYSTEM_PROMPT:
                continue
            text = memory.text
            if memory.backend and active_backend and memory.backend != active_backend:
                text = f"(learned on {memory.backend}) {text}"
            parts.append(text)
        return "\n\n".join(parts)

    def system_prompt_tokens(self) -> int:
        return estimate_tokens(self.scope_text(MemoryScope.SYSTEM_PROMPT))

    def usage_meter(self, cap: int) -> str:
        """Hermes-style usage meter for the budgeted scope,
        e.g. ``58% — 1392/2400 tokens``."""
        used = self.system_prompt_tokens()
        percent = round(100 * used / cap) if cap else 0
        return f"{percent}% — {used}/{cap} tokens"

    def would_exceed(self, text: str, *, cap: int) -> bool:
        """Whether adding ``text`` to the system-prompt scope breaks its budget.

        The budget is hard for *writes* (§6.4 redesign): a full scope rejects
        new memories until the user condenses it. Injection never truncates —
        what is in the file is what the model sees. RAG has no budget, so this
        only concerns the system-prompt scope.
        """
        current = self.scope_text(MemoryScope.SYSTEM_PROMPT)
        addition = text.strip()
        combined = f"{current}\n\n{addition}" if current else addition
        return estimate_tokens(combined) > cap

    def over_budget(self, cap: int) -> bool:
        """Whether the system-prompt scope is over its token budget (only
        reachable through a hand edit; in-app writes are rejected at the cap)."""
        return self.system_prompt_tokens() > cap


def _parse_meta(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta
