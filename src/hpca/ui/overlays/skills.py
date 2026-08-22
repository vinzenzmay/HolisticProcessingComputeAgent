"""The skills a profile may edit (`s` on the profiles screen, §4.3 items 27
and 33).

Its own and this project's — the two levels a user owns, which is what
`SkillInfo.removable` says and what the core answers `skill.get` and
`skill.delete` for. Shared (`_shared/`) and shipped skills are neither listed
nor editable here: removing or forking one would change every other profile
that sees it, and the core refuses them on the way in as well.

Enter opens one skill's raw file in the same editor the memories use, and `d`
deletes one after asking. Both are sent as intents from here rather than
handed up to the profiles list, because a skill belongs to the profile this
screen was opened about and that profile is what it carries.
"""

from __future__ import annotations

from hpca.ui.overlays.base import ListOverlay
from hpca.ui.overlays.textedit import SKILL, TextEditOverlay
from hpca.ui.pane import Item
from hpca.ui.state import (
    DeleteSkill,
    FetchSkills,
    ProfileInfo,
    SaveSkill,
    SkillInfo,
)

EMPTY = "(no skills for this profile)"
DELETE_QUESTION = "Delete skill “{name}”?"


def skill_item(skill: SkillInfo) -> Item:
    """One row: the name, and what it is for.

    No body: the file is a `skill.get` away and is fetched when the editor
    opens, so there is nothing to preview here that would not be a second,
    staler copy of it.
    """
    head = skill.name
    if skill.description:
        head = f"{skill.name:<24}{skill.description}"
    return Item(head=head, kind="skill", text=skill.name, key=skill.name)


class SkillsOverlay(ListOverlay):
    """The skills one profile can call its own, editable one file at a time."""

    name = "skills"

    def __init__(self, profile: ProfileInfo) -> None:
        self.profile = profile
        super().__init__(self._rows())
        self.subject = ""

    def opened(self) -> None:
        """Ask for the list (`skill.list`) as the screen goes up.

        The list on `ProfileInfo` is whatever the last answer was, which may be
        nothing at all — a profile nobody has opened has never been asked about
        — so the screen draws what it has and corrects itself when the rows
        land.
        """
        self.send(FetchSkills(self.profile.name))

    def fill_list(self, key, rows) -> bool:
        if key != ("skills", self.profile.name):
            return False
        self.profile.skills = list(rows)
        self.replace(self._rows())
        return True

    def _rows(self) -> list[Item]:
        if not self.profile.skills:
            # A row rather than an empty pane: an empty list and a list that
            # failed to load look identical, and one of them is worth saying.
            return [Item(head=EMPTY, kind="empty")]
        return [skill_item(x) for x in self.profile.skills]

    def heading(self) -> str:
        return f"skills · {self.profile.name}"

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "move"), ("enter", "edit"), ("d", "delete"), ("esc", "back")]

    def skill(self, name: str) -> SkillInfo | None:
        return next((x for x in self.profile.skills if x.name == name), None)

    def chose(self, item: Item | None) -> bool:
        if item is None or item.kind != "skill":
            return True
        self.subject = item.text
        # Fetched as the editor opens (`skill.get`), never carried by the row:
        # `protocol.SkillGet` spells out why — a body shipped with a listing is
        # stale by the time the editor is over it, and `skill.save` writes
        # what it is given.
        return self.open(
            TextEditOverlay(
                profile=self.profile.name,
                name=item.text,
                kind=SKILL,
                awaiting=("skill", (self.profile.name, item.text)),
            ),
            SKILL,
        )

    def other(self, key: str, item: Item | None) -> bool:
        if key == "d" and item is not None and item.kind == "skill":
            self.subject = item.text
            self.ask(DELETE_QUESTION.format(name=item.text))
        return True

    def answered(self, question: str, yes: bool) -> bool:
        if not yes:
            self.note = "kept"
            return True
        self.send(DeleteSkill(self.profile.name, self.subject))
        self.profile.skills = [
            x for x in self.profile.skills if x.name != self.subject
        ]
        self.replace(self._rows())
        self.note = f"deleted “{self.subject}”"
        return True

    def child_closed(self, child) -> None:
        if child.tag != SKILL or not getattr(child, "saved", False):
            return
        self.send(SaveSkill(self.profile.name, self.subject, child.text))
        skill = self.skill(self.subject)
        if skill is not None:
            skill.text = child.text
        self.replace(self._rows())
        self.note = f"kept “{self.subject}”"
