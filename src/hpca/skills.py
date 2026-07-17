"""User-defined skills (§5.1): procedure files loaded per profile.

Markdown or YAML files in ``<app_dir>/skills/`` describing how the agent
should handle specific tasks. Like profiles (§6.2), these are hand-edited, so
parsing is lenient: a file without front matter is still a skill (its filename
is the name, its text the body), and a broken header reports a problem rather
than vanishing.

Skills are *surfaced* to the model as a short list in the system prompt; the
full body is fetched on demand via ``read_skill``, keeping the prompt small.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hpca.config import app_dir

SKILL_SUFFIXES = {".md", ".yaml", ".yml"}


def skills_dir() -> Path:
    return app_dir() / "skills"


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    body: str
    source: str = ""
    problems: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """Whole-word match of the name or any trigger against free text."""
        lowered = text.lower()
        for term in [self.name, *self.triggers]:
            if term and re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered):
                return True
        return False


def _split_front_matter(text: str) -> tuple[dict, str, list[str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, []
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text, ["Unterminated front matter"]
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
        if not isinstance(meta, dict):
            raise yaml.YAMLError("front matter is not a mapping")
    except yaml.YAMLError as e:
        return {}, "\n".join(lines[end + 1 :]), [f"Unreadable front matter: {e}"]
    return meta, "\n".join(lines[end + 1 :]), []


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_skill(text: str, *, filename: str) -> Skill | None:
    if not text.strip():
        return None
    stem = Path(filename).stem
    if Path(filename).suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            return Skill(
                name=stem, description="", triggers=[], body=text,
                source=filename, problems=[f"Unreadable YAML: {e}"],
            )
        return Skill(
            name=str(data.get("name", stem)),
            description=str(data.get("description", "")),
            triggers=_as_list(data.get("triggers")),
            body=str(data.get("body", "")).strip(),
            source=filename,
        )
    meta, body, problems = _split_front_matter(text)
    return Skill(
        name=str(meta.get("name", stem)),
        description=str(meta.get("description", "")),
        triggers=_as_list(meta.get("triggers")),
        body=body.strip(),
        source=filename,
        problems=problems,
    )


def load_skills(directory: Path | None = None) -> list[Skill]:
    directory = directory or skills_dir()
    if not directory.exists():
        return []
    skills = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SKILL_SUFFIXES or not path.is_file():
            continue
        skill = parse_skill(path.read_text(errors="replace"), filename=path.name)
        if skill is not None:
            skills.append(skill)
    return sorted(skills, key=lambda s: s.name)


def summarize_skills(skills: list[Skill]) -> str:
    """One line per skill for the system prompt (names + descriptions only)."""
    return "\n".join(
        f"- {s.name}: {s.description}" if s.description else f"- {s.name}"
        for s in skills
    )
