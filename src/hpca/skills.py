"""Skills (§5.1): procedure files loaded per profile, plus the few HPCA ships.

Markdown or YAML files describing how the agent should handle specific tasks.
Like profiles (§6.2), these are hand-edited, so parsing is lenient: a file
without front matter is still a skill (its filename is the name, its text the
body), and a broken header reports a problem rather than vanishing.

Four *levels* decide where a skill lives and who sees it:

- **builtin** — shipped with HPCA itself, in this package's ``data/skills/``.
  Visible to every profile with no setup, so a fresh install already knows a
  few procedures. Ranked *below* every user level: a user file of the same name
  shadows the shipped one instead of colliding with it, and the shipped file is
  never written to or deleted through the app.
- **global** — every profile. Stored under ``<app_dir>/skills/_shared/``.
  This is the original "shared" concept, surfaced to the user as "global".
  (Files directly in ``<app_dir>/skills/`` are the legacy flat layout and are
  treated as global too.)
- **profile** — the current profile only. Stored under
  ``<app_dir>/skills/<profile>/``.
- **project** — tied to the directory the agent is run from. Stored under a
  hidden ``<cwd>/.hpca/skills/`` and only visible while running there. The
  project root is threaded explicitly (``project_root``) so it stays testable.

On a name collision the most specific level wins: **project > profile > global
> builtin** (legacy flat files rank with global).

Skills are *surfaced* to the model as a short list in the system prompt; the
full body is fetched on demand via ``read_skill``, keeping the prompt small.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from hpca.config import app_dir

SKILL_SUFFIXES = {".md", ".yaml", ".yml"}
SHARED_SKILLS_DIR = "_shared"
# Project skills hide under this dir in the working directory, so a repo can
# carry its own procedures without them leaking into other projects.
PROJECT_SKILLS_SUBDIR = ".hpca/skills"
# Skills shipped with HPCA, as package data next to the other bundled files.
BUILTIN_SKILLS_DIR = Path(__file__).parent / "data" / "skills"

# Where a newly-created (or written-back) skill is stored. "global" maps to the
# existing ``_shared/`` location for backward compatibility. The shipped
# ``builtin`` level is deliberately absent: it is read-only.
SkillLevel = Literal["global", "profile", "project"]


def skills_dir() -> Path:
    return app_dir() / "skills"


def project_skills_dir(project_root: Path | None = None) -> Path:
    """Skills tied to the working directory, under ``<cwd>/.hpca/skills``.

    ``project_root`` defaults to the current working directory so it can be
    pinned to a ``tmp_path`` in tests, matching how ``root`` is injected.
    """
    return (project_root or Path.cwd()) / ".hpca" / "skills"


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    body: str
    source: str = ""
    problems: list[str] = field(default_factory=list)
    # Which level the skill was loaded from ("builtin", "global", "profile",
    # "project"). Set by the loader, not by the file: the level is the
    # location. Empty for skills built in memory (tests, the creator form).
    level: str = ""

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


def _load_dir(directory: Path, level: str = "") -> list[Skill]:
    if not directory.exists():
        return []
    skills = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SKILL_SUFFIXES or not path.is_file():
            continue
        skill = parse_skill(path.read_text(errors="replace"), filename=path.name)
        if skill is not None:
            skill.level = level
            skills.append(skill)
    return skills


def load_builtin_skills() -> list[Skill]:
    """The skills shipped with HPCA. Always visible, never removable."""
    return sorted(_load_dir(BUILTIN_SKILLS_DIR, "builtin"), key=lambda s: s.name)


def load_skills(
    profile: str | None = None,
    *,
    root: Path | None = None,
    project_root: Path | None = None,
) -> list[Skill]:
    """Skills visible to one profile: the shipped ones, legacy flat files,
    ``_shared/`` (global), the profile's own directory, then the project's
    ``.hpca/skills`` — later sources win on a name collision, so precedence is
    project > profile > global > builtin."""
    root = root or skills_dir()
    sources = [
        (BUILTIN_SKILLS_DIR, "builtin"),
        (root, "global"),  # legacy flat files rank with global
        (root / SHARED_SKILLS_DIR, "global"),
    ]
    if profile:
        sources.append((root / profile, "profile"))
    sources.append((project_skills_dir(project_root), "project"))
    by_name: dict[str, Skill] = {}
    for directory, level in sources:
        for skill in _load_dir(directory, level):
            by_name[skill.name] = skill
    return sorted(by_name.values(), key=lambda s: s.name)


def load_own_skills(profile: str, *, root: Path | None = None) -> list[Skill]:
    """Only the skills in the profile's OWN directory — not ``_shared/`` and
    not the legacy flat files. This is the set the user may remove: deleting a
    shared procedure from one profile would silently change every other."""
    root = root or skills_dir()
    return sorted(_load_dir(root / profile, "profile"), key=lambda s: s.name)


def load_project_skills(*, project_root: Path | None = None) -> list[Skill]:
    """The project-level skills under ``<cwd>/.hpca/skills``. Removable like a
    profile's own (they belong to this directory, not to other profiles)."""
    return sorted(
        _load_dir(project_skills_dir(project_root), "project"),
        key=lambda s: s.name,
    )


def delete_own_skill(
    skill: Skill,
    profile: str,
    *,
    root: Path | None = None,
    project_root: Path | None = None,
) -> bool:
    """Remove one profile- or project-level skill file. Returns whether it
    existed.

    Deletes by the file the skill was loaded from (``skill.source``), so it
    works even when a hand-edited file's name differs from its front-matter
    name. Never touches ``_shared/`` (global) or legacy flat files: removing
    one would silently change every other profile that sees it.
    """
    root = root or skills_dir()
    if not skill.source or skill.level == "builtin":
        return False  # shipped skills belong to the package, not the user
    # Profile dir first, then the project dir — the two removable levels.
    for path in (
        root / profile / skill.source,
        project_skills_dir(project_root) / skill.source,
    ):
        if path.exists() and path.is_file():
            path.unlink()
            return True
    return False


def skill_path(
    name: str,
    profile: str,
    *,
    root: Path | None = None,
    level: SkillLevel = "profile",
    project_root: Path | None = None,
) -> Path:
    """Where a skill file lives for the chosen level. Defaults to the profile's
    own directory (where self-review patches always land, so one profile's
    correction never changes another's procedure)."""
    root = root or skills_dir()
    # Separators become dashes so a name can never escape its dir, and leading
    # dots are stripped so it cannot become a hidden file either.
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-.") or "skill"
    if level == "global":
        return root / SHARED_SKILLS_DIR / f"{safe}.md"
    if level == "project":
        return project_skills_dir(project_root) / f"{safe}.md"
    return root / profile / f"{safe}.md"


def write_skill(
    skill: Skill,
    profile: str,
    *,
    root: Path | None = None,
    level: SkillLevel = "profile",
    project_root: Path | None = None,
) -> Path:
    """Persist a skill at the chosen level, front matter included."""
    path = skill_path(
        skill.name, profile, root=root, level=level, project_root=project_root
    )
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
