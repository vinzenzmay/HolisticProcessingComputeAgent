"""A profile's own skills (`s` on the profiles screen, §4.3 items 27 and 33).

The profile's own copy only — shared and project skills are edited where they
live, not from one profile's screen, which is the same rule the core enforces
on the way in (`skill.save` writes into the profile's directory and nowhere
else).

Enter opens one skill's raw file in the same editor the memories use, and `d`
deletes one after asking. Both are sent as intents from here rather than
handed up to the profiles list, because a skill belongs to the profile this
screen was opened about and that profile is what it carries.
"""

from __future__ import annotations

from hpca.ui.overlays.base import ListOverlay
from hpca.ui.overlays.textedit import SKILL, TextEditOverlay
from hpca.ui.pane import Item
from hpca.ui.state import DeleteSkill, ProfileInfo, SaveSkill, SkillInfo

EMPTY = "(no skills for this profile)"
DELETE_QUESTION = "Delete skill “{name}”?"


def skill_item(skill: SkillInfo) -> Item:
    head = skill.name
    if skill.description:
        head = f"{skill.name:<24}{skill.description}"
    return Item(
        head=head,
        body=skill.text.split("\n")[:12] if skill.text else [],
        kind="skill",
        text=skill.name,
        key=skill.name,
    )


class SkillsOverlay(ListOverlay):
    """The skills one profile can call its own, editable one file at a time."""

    name = "skills"

    def __init__(self, profile: ProfileInfo) -> None:
        self.profile = profile
        super().__init__(self._rows())
        self.subject = ""

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
        skill = self.skill(item.text)
        return self.open(
            TextEditOverlay(
                skill.text if skill else "",
                profile=self.profile.name,
                name=item.text,
                kind=SKILL,
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
