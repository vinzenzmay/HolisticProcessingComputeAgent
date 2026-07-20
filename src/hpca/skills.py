"""User-defined skills (§5.1): procedure files loaded per profile.

Markdown or YAML files under ``<app_dir>/skills/`` describing how the agent
should handle specific tasks. Like profiles (§6.2), these are hand-edited, so
parsing is lenient: a file without front matter is still a skill (its filename
is the name, its text the body), and a broken header reports a problem rather
than vanishing.

Layout (redesign Phase 1): ``skills/_shared/`` holds skills for every
profile, ``skills/<profile>/`` the profile's own; files directly in
``skills/`` are the legacy flat layout and count as shared. On a name
collision the profile's own skill wins.

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
SHARED_SKILLS_DIR = "_shared"


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


def _load_dir(directory: Path) -> list[Skill]:
    if not directory.exists():
        return []
    skills = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SKILL_SUFFIXES or not path.is_file():
            continue
        skill = parse_skill(path.read_text(errors="replace"), filename=path.name)
        if skill is not None:
            skills.append(skill)
    return skills


def load_skills(profile: str | None = None, *, root: Path | None = None) -> list[Skill]:
    """Skills visible to one profile: legacy flat files, ``_shared/``, then
    the profile's own directory — later sources win on a name collision."""
    root = root or skills_dir()
    directories = [root, root / SHARED_SKILLS_DIR]
    if profile:
        directories.append(root / profile)
    by_name: dict[str, Skill] = {}
    for directory in directories:
        for skill in _load_dir(directory):
            by_name[skill.name] = skill
    return sorted(by_name.values(), key=lambda s: s.name)


def load_own_skills(profile: str, *, root: Path | None = None) -> list[Skill]:
    """Only the skills in the profile's OWN directory — not ``_shared/`` and
    not the legacy flat files. This is the set the user may remove: deleting a
    shared procedure from one profile would silently change every other."""
    root = root or skills_dir()
    return sorted(_load_dir(root / profile), key=lambda s: s.name)


def delete_own_skill(skill: Skill, profile: str, *, root: Path | None = None) -> bool:
    """Remove one of a profile's own skill files. Returns whether it existed.

    Deletes by the file the skill was loaded from (``skill.source``), so it
    works even when a hand-edited file's name differs from its front-matter
    name. Never touches ``_shared/`` or legacy files (they live elsewhere).
    """
    root = root or skills_dir()
    if not skill.source:
        return False
    path = root / profile / skill.source
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def any_skills(root: Path | None = None) -> bool:
    """Whether any profile has any skill at all — decides if the skill tools
    are registered, since the active profile can change per session."""
    root = root or skills_dir()
    if not root.exists():
        return False
    return any(
        path.suffix.lower() in SKILL_SUFFIXES and path.is_file()
        for path in root.rglob("*")
    )


def skill_path(name: str, profile: str, *, root: Path | None = None) -> Path:
    """Where a profile's own copy of a skill lives. Patches always write
    here, never into ``_shared/``: one profile's correction must not silently
    change another profile's procedure."""
    root = root or skills_dir()
    # Separators become dashes so a name can never escape the profile dir,
    # and leading dots are stripped so it cannot become a hidden file either.
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-.") or "skill"
    return root / profile / f"{safe}.md"


def write_skill(
    skill: Skill, profile: str, *, root: Path | None = None
) -> Path:
    """Persist a skill under a profile, front matter included."""
    path = skill_path(skill.name, profile, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(
        {
            "name": skill.name,
            "description": skill.description,
            "triggers": skill.triggers,
        },
        sort_keys=False,
    ).strip()
    path.write_text(f"---\n{front}\n---\n\n{skill.body.strip()}\n")
    return path


def copy_profile_skills(
    source: str, target: str, *, root: Path | None = None
) -> int:
    """Copy one profile's own skills to another; returns how many.

    Only the profile's own directory is copied — ``_shared/`` is already
    visible to both, and duplicating it would turn one shared procedure into
    two that drift apart silently.
    """
    root = root or skills_dir()
    source_dir, target_dir = root / source, root / target
    if not source_dir.exists():
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() not in SKILL_SUFFIXES or not path.is_file():
            continue
        (target_dir / path.name).write_text(path.read_text(errors="replace"))
        copied += 1
    return copied


def delete_profile_skills(name: str, *, root: Path | None = None) -> None:
    """Remove a profile's own skills. ``_shared/`` is never touched."""
    import shutil

    root = root or skills_dir()
    directory = root / name
    if directory.exists() and directory.is_dir():
        shutil.rmtree(directory)


def patched_body(skill: Skill, correction: str) -> str:
    """A skill body with a correction appended under a stable heading.

    Appending rather than rewriting: a small model asked to restate a whole
    procedure will quietly drop steps it did not think about, and the user
    approving the patch can only reasonably review what changed.
    """
    heading = "## Corrections"
    body = skill.body.rstrip()
    if heading in body:
        return f"{body}\n\n- {correction.strip()}\n"
    return f"{body}\n\n{heading}\n\n- {correction.strip()}\n"


def summarize_skills(skills: list[Skill]) -> str:
    """One line per skill for the system prompt (names + descriptions only)."""
    return "\n".join(
        f"- {s.name}: {s.description}" if s.description else f"- {s.name}"
        for s in skills
    )
