"""The two skill screens a slash command opens (§4.3 item 33).

`/skill-creator` is a form the core does not own and says so: "the finished
skill arrives as skill.save". So the form is here — name, description, level,
body — and what leaves it is one `state.SaveSkill` carrying the file as it
will be written, front matter and all, plus the level it is written at.

The *draft* is the half that is not this side's. `/skill-creator <what it
should do>` is a model call, and only the core makes those: the request goes
out as `skill.draft` and comes back as three fields this same form opens over
(`RowUI.skill_drafted`). One form either way — a draft is a head start, not an
author, and it is edited and confirmed exactly like a hand-typed skill.

`/skill-remove` is a picker over the skills the profile may remove — its own
and this project's. The core also answers `/skill-remove <name>` directly, and
that path is left alone; this is what a bare `/skill-remove` opens, so that
"the chosen skill" is a row the user points at rather than a name they have to
remember and retype.
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
# carries them. ``level`` sits before the body because it is a choice and the
# body is a paragraph; ``body`` is last and takes the rest of the screen: it is
# the procedure, and the other two are a line each.
NAME, DESCRIPTION, LEVEL, BODY = "name", "description", "level", "body"
FIELDS = (
    (NAME, "one word — this is what /<skill> types"),
    (DESCRIPTION, "when the model should reach for it"),
    (LEVEL, "← → where it lives"),
    (BODY, "the procedure itself"),
)

# The three levels a user may write at, in the order they are offered, with
# what choosing one means. The shipped level is absent because it is package
# data — `protocol.SkillLevel` does not admit it either.
LEVELS = (
    ("profile", "this profile only"),
    ("global", "every profile"),
    ("project", "this directory only"),
)
WHERE = dict(LEVELS)

EMPTY_NAME = "a name is needed"
TAKEN = "“{name}” already exists at the {level} level"
SPACED = "a skill name cannot contain spaces — /<skill> splits on the first one"
EMPTY_LIST = "(no skills of this profile's or this project's to remove)"
REMOVE_QUESTION = "Remove skill “{name}”?"
KEEP_CHANGES = "Save this skill?"
DRAFTED = "the model's draft — edit it, then ^s to save"


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
    """`/skill-creator`: a skill, written or drafted, saved at a chosen level.

    Three editors, a level, and a cursor between them, rather than four
    screens in a row: a skill is one thing, and a name typed on a screen that
    has already gone is a name that cannot be corrected once the description
    makes it obvious it was wrong.

    Opened empty by a bare command, or over the model's draft when the command
    carried a request (`RowUI.skill_drafted`). The same screen either way, on
    purpose: what the user does to a draft — read it, fix the name, cut two
    steps, save — is what they were going to do to their own typing.
    """

    title = "new skill"

    def __init__(
        self,
        profile: str,
        *,
        taken: tuple[tuple[str, str], ...] = (),
        request: str = "",
        name: str = "",
        description: str = "",
        body: str = "",
        level: str = LEVELS[0][0],
    ) -> None:
        super().__init__()
        self.profile = profile
        # (name, level) pairs: a name is only taken at the level it was
        # written at. The same name at two levels is not a collision, it is
        # how a project overrides a profile's procedure (`hpca.skills`).
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
        self.level = level
        self.at = 0
        # What was decided, read by `RowUI` after the screen closes.
        self.name = ""
        self.text = ""
        self.saved = False
        self.drafted = bool(name or description or body)
        if self.drafted:
            self.note = DRAFTED

    # ------------------------------------------------------------ the fields

    @property
    def field(self) -> str:
        return FIELDS[self.at][0]

    @property
    def editor(self) -> Editor:
        """The editor the keys go to. The level is a choice, not a field
        anyone types into, so it borrows the body's — a stray character while
        the cursor is on it lands where the procedure is being written rather
        than nowhere."""
        return self.editors[BODY if self.field == LEVEL else self.field]

    def value(self, field: str) -> str:
        return self.editors[field].text().strip()

    @property
    def where(self) -> str:
        """What choosing this level means, in the words the form offers."""
        return WHERE.get(self.level, self.level)

    @property
    def description(self) -> str:
        """What the menu will show next to the name, once this is saved."""
        return self.value(DESCRIPTION)

    @property
    def dirty(self) -> bool:
        """Whether anything would be lost by closing.

        The level is not counted: it has a value from the moment the screen
        opens, and a form nobody has typed into is one escape closes.
        """
        return any(self.value(f) for f in (NAME, DESCRIPTION, BODY))

    def heading(self) -> str:
        return f"{self.title} · {self.profile}"

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("⇥", "next field"),
            ("⇧⇥", "previous"),
            ("←→", "level"),
            ("^s", "save"),
            ("esc", "cancel"),
        ]

    # ----------------------------------------------------------- the drawing

    def level_row(self, width: int, focused: bool) -> str:
        """The three levels on one row, the chosen one marked.

        A row of choices rather than a list: there are three, they fit, and a
        list would put the body another screenful down from the name.
        """
        parts = []
        for name, what in LEVELS:
            mark = "●" if name == self.level else "○"
            parts.append(f"{mark} {name} ({what})")
        row = pad("  " + "   ".join(parts), width)
        return (BOLD + CYAN + row + RESET) if focused else (DIM + row + RESET)

    def body_rows(self, width: int, height: int) -> list[str]:
        out: list[str] = []
        for at, (field, hint) in enumerate(FIELDS):
            here = at == self.at
            label = f" {field}" + (f"  ({hint})" if here else "")
            out.append((BOLD + CYAN if here else DIM) + pad(label, width) + RESET)
            if field == LEVEL:
                out.append(self.level_row(width, here))
                continue
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
        elif self.field == LEVEL and key in ("left", "right"):
            step = 1 if key == "right" else -1
            names = [x for x, _ in LEVELS]
            self.level = names[(names.index(self.level) + step) % len(names)]
            self.note = ""
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
        if (name, self.level) in self.taken:
            # Only at *this* level: the same name at another one is not a
            # collision but an override, which is how a project skill replaces
            # a profile's for as long as you work in that directory.
            self.at = 0
            self.note = TAKEN.format(name=name, level=self.level)
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
    """`/skill-remove`: pick one of the removable skills, and confirm.

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
