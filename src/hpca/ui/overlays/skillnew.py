"""The two skill screens a slash command opens (§4.3 item 33).

`/skill-creator` is the one built-in the core has no shape for, and says so:
"a form the front-end owns; the finished skill arrives as skill.save". So the
form is here — name, description, body — and what leaves it is one
`state.SaveSkill` carrying the file as it will be written, front matter and all.

`/skill-remove` is a picker over the profile's own skills. The core also
answers `/skill-remove <name>` directly, and that path is left alone; this is
what a bare `/skill-remove` opens, so that "the chosen skill" is a row the user
points at rather than a name they have to remember and retype.

Neither screen can offer the *level* a skill is written at ("global", "project"
or the profile's own). `protocol.SkillSave` carries a profile and nothing else,
so every skill saved from here is a profile skill; the picker likewise only
ever lists what `skill.list` calls the profile's own, which is also the only
thing `skill.delete` can remove.
"""

from __future__ import annotations

import json

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS
from hpca.ui.overlays.base import BACK_KEYS, ListOverlay, Overlay, framed
from hpca.ui.overlays.skills import skill_item
from hpca.ui.pane import Item
from hpca.ui.state import DeleteSkill, SkillInfo

# The fields, in the order they are filled and in the order a skill file
# carries them. ``body`` is last and takes the rest of the screen: it is the
# procedure, and the other two are a line each.
NAME, DESCRIPTION, BODY = "name", "description", "body"
FIELDS = (
    (NAME, "one word — this is what /<skill> types"),
    (DESCRIPTION, "when the model should reach for it"),
    (BODY, "the procedure itself"),
)

EMPTY_NAME = "a name is needed"
TAKEN = "“{name}” already exists in this profile"
SPACED = "a skill name cannot contain spaces — /<skill> splits on the first one"
EMPTY_LIST = "(this profile has no skills of its own)"
REMOVE_QUESTION = "Remove skill “{name}”?"
KEEP_CHANGES = "Save this skill?"


def skill_file(name: str, description: str, body: str) -> str:
    """The file `skill.save` writes, verbatim — front matter and body.

    Written the way `hpca.skills.write_skill` writes one, with the two scalars
    JSON-quoted: a description is a sentence and sentences contain colons,
    which is exactly the character that turns an unquoted YAML scalar into a
    mapping the loader then rejects. JSON's quoting is a subset of YAML's
    double-quoted style, so this needs no yaml dependency to be safe.
    """
    front = "\n".join(
        (
            "---",
            f"name: {json.dumps(name)}",
            f"description: {json.dumps(description)}",
            "triggers: []",
            "---",
        )
    )
    return f"{front}\n\n{body.strip()}\n"


class SkillCreatorOverlay(Overlay):
    """`/skill-creator`: a skill, written by hand, saved to this profile.

    Three editors and a cursor between them, rather than three screens in a
    row: a skill is one thing, and a name typed on a screen that has already
    gone is a name that cannot be corrected once the description makes it
    obvious it was wrong.

    **What is missing, and it is a channel and not an omission.** The Textual
    command took an argument — `/skill-creator <what it should do>` — and had
    the model draft the three fields before the form opened. There is no
    command that asks the core for a draft and no event that could carry one
    back (`core.service._run_slash` answers `/skill-creator` with a warning),
    so a bare form is what the argument gets too; the request is kept on the
    rule so nothing is silently swallowed.
    """

    title = "new skill"

    def __init__(
        self,
        profile: str,
        *,
        taken: tuple[str, ...] = (),
        request: str = "",
        name: str = "",
        description: str = "",
        body: str = "",
    ) -> None:
        super().__init__()
        self.profile = profile
        self.taken = tuple(taken)
        self.request = request
        self.editors = {
            NAME: Editor(),
            DESCRIPTION: Editor(),
            BODY: Editor(wrap=True),
        }
        for field, value in ((NAME, name), (DESCRIPTION, description), (BODY, body)):
            if value:
                self.editors[field].set_text(value)
        self.at = 0
        # What was decided, read by `RowUI` after the screen closes.
        self.name = ""
        self.text = ""
        self.saved = False
        if request:
            self.note = "drafting is not on the wire — write it here"

    # ------------------------------------------------------------ the fields

    @property
    def field(self) -> str:
        return FIELDS[self.at][0]

    @property
    def editor(self) -> Editor:
        return self.editors[self.field]

    def value(self, field: str) -> str:
        return self.editors[field].text().strip()

    @property
    def description(self) -> str:
        """What the menu will show next to the name, once this is saved."""
        return self.value(DESCRIPTION)

    @property
    def dirty(self) -> bool:
        return any(self.value(f) for f, _ in FIELDS)

    def heading(self) -> str:
        return f"{self.title} · {self.profile}"

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("⇥", "next field"),
            ("⇧⇥", "previous"),
            ("^s", "save"),
            ("esc", "cancel"),
        ]

    # ----------------------------------------------------------- the drawing

    def body_rows(self, width: int, height: int) -> list[str]:
        out: list[str] = []
        for at, (field, hint) in enumerate(FIELDS):
            here = at == self.at
            label = f" {field}" + (f"  ({hint})" if here else "")
            out.append((BOLD + CYAN if here else DIM) + pad(label, width) + RESET)
            if field != BODY:
                out += self.editors[field].render(width, 1, focused=here)
                continue
            rest = max(1, height - len(out))
            out += self.editors[field].render(width, rest, focused=here)
        return out[:height]

    def render(self, width: int, height: int) -> list[str]:
        tail = self.question_rows(width)
        return framed(
            self.heading(),
            self.body_rows(width, max(1, height - 1 - len(tail))),
            width,
            height,
            note=self.note,
            tail=tail,
        )

    def paste(self, text: str) -> None:
        self.editor.insert_text(text)

    # -------------------------------------------------------------- the keys

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            if not self.dirty:
                return False  # nothing written, so nothing is asked
            self.ask(KEEP_CHANGES)
            return True
        if key == "tab":
            self.at = (self.at + 1) % len(FIELDS)
        elif key == "shift-tab":
            self.at = (self.at - 1) % len(FIELDS)
        elif key == "ctrl-s":
            return self.save()
        elif key == "enter" or key in NEWLINE_KEYS:
            # Enter moves on from the one-line fields and breaks the line in
            # the body, which is the only field a newline means anything in.
            if self.field == BODY:
                self.editor.newline()
            else:
                self.at = min(self.at + 1, len(FIELDS) - 1)
        else:
            self.editor.handle(key)
            self.note = ""
        return True

    def save(self) -> bool:
        """Refuse on screen rather than close and quietly change nothing."""
        name = self.value(NAME)
        if not name:
            self.at, self.note = 0, EMPTY_NAME
            return True
        if any(ch.isspace() for ch in name):
            self.at, self.note = 0, SPACED
            return True
        if name in self.taken:
            self.at, self.note = 0, TAKEN.format(name=name)
            return True
        self.name = name
        self.text = skill_file(name, self.value(DESCRIPTION), self.value(BODY))
        self.saved = True
        return False

    def answered(self, question: str, yes: bool) -> bool:
        if not yes:
            return False  # abandoned, with whatever was typed
        return self.save()


class SkillRemoveOverlay(ListOverlay):
    """`/skill-remove`: pick one of this profile's own skills, and confirm.

    Its own class rather than `SkillsOverlay` with a different Enter, because
    the two screens answer different questions: that one is "edit my skills",
    reached from the profiles list, and this one is a command with exactly one
    thing to do that closes when it is done.
    """

    title = "remove a skill"
    name = "skills"

    def __init__(self, profile: str, skills: tuple[SkillInfo, ...] = ()) -> None:
        self.profile = profile
        self.skills = list(skills)
        rows = [skill_item(x) for x in self.skills] or [
            # A row rather than an empty pane: an empty list and a list that
            # failed to load look identical, and one of them is worth saying.
            Item(head=EMPTY_LIST, kind="empty")
        ]
        super().__init__(rows)
        self.removed = ""
        self.subject = ""

    def heading(self) -> str:
        return f"{self.title} · {self.profile}"

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "move"), ("enter", "remove"), ("esc", "keep them all")]

    def chose(self, item: Item | None) -> bool:
        if item is None or item.kind != "skill":
            return True
        self.subject = item.text
        self.ask(REMOVE_QUESTION.format(name=self.subject))
        return True

    def answered(self, question: str, yes: bool) -> bool:
        if not yes:
            self.note = "kept"
            return True
        self.send(DeleteSkill(self.profile, self.subject))
        self.removed = self.subject
        return False
