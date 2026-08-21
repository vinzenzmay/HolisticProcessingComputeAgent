"""Profiles & learnings (`a`, §4.3 item 27): the list, and what opens off it.

Profiles are entirely the user's: there are no built-in ones beyond the
default, and each accumulates its own memories and skills over time. This
screen is the list of them plus five keys, and each of those five opens a
child screen rather than a second top-level overlay — the stack is genuinely
three deep (list → a profile's skills → one skill's file), and escaping the
file has to land back on the skills.

What is *not* here, deliberately: the guards. Whether a profile can be deleted
depends on a reply being in flight and on live sub-processes, neither of which
the UI can see (rule 2 of §4.2), so `profile.delete` is sent and the core
refuses it with a `notify` (`core/service.py:_delete_profile`). The one guard
kept on this side is the default profile, because it is a fact about the name
and the alternative is a key that offers a round trip whose only answer is no.
"""

from __future__ import annotations

from collections.abc import Iterable

from hpca.ui.overlays.base import ListOverlay, PromptOverlay
from hpca.ui.overlays.skills import SkillsOverlay
from hpca.ui.overlays.textedit import ARCHIVE, MEMORIES, TextEditOverlay
from hpca.ui.pane import Item
from hpca.ui.state import (
    CopyProfile,
    CreateProfile,
    DeleteProfile,
    ProfileInfo,
    SaveProfile,
)

NEW_PROFILE = "(new profile)"

# Why the default is not deletable, in the words the core uses when it refuses
# the same thing — so a user who reaches it either way reads one explanation.
DEFAULT_REFUSAL = (
    "the default profile cannot be deleted — it is where other "
    "profiles' sessions go"
)
DELETE_QUESTION = (
    "Delete profile “{name}”? Its sessions move to the default profile, "
    "and its skills go with it."
)
# Why an editor will not open. Refused rather than opened over an empty box,
# because `profile.save` writes what it is given verbatim and a save from a box
# that could not be filled would truncate the file it failed to read.
UNREADABLE = "its files could not be read — nothing here would be safe to save"

# The three tags a child screen comes back under. One editor serves memories,
# the archive and (through the skills screen) a skill file, so the tag is what
# says where the text goes.
NAME_NEW, NAME_COPY = "new", "copy"


def profile_item(info: ProfileInfo) -> Item:
    """One row: `name ★ · 12 memories · copied from hpc`.

    The same line `tui/profiles_screen.py` built, including the provenance —
    which profiles share a base is the thing you need to know when they start
    disagreeing with each other.
    """
    star = " ★" if info.default else ""
    said = [f"{info.memories} " + ("memory" if info.memories == 1 else "memories")]
    if info.working:
        # A different fact from the star: this is the one the core is running
        # under, and it is usually not the default.
        said.append("in use")
    if info.skills:
        said.append(f"{len(info.skills)} " + ("skill" if len(info.skills) == 1 else "skills"))
    if info.sessions:
        said.append(
            f"{info.sessions} session" + ("s" if info.sessions != 1 else "")
        )
    if info.copied_from:
        said.append(f"copied from {info.copied_from}")
    body = []
    if info.default:
        body.append("the fallback; cannot be deleted")
    if info.working:
        body.append("the profile the core is working under")
    if info.copied_from:
        body.append(f"started as a copy of {info.copied_from}")
    return Item(
        head=f"{info.name + star:<22}{'  ·  '.join(said)}",
        body=body,
        kind="profile",
        text=info.name,
        key=info.name or "?",
    )


def profile_rows(profiles: Iterable[ProfileInfo]) -> list[Item]:
    """The list, with `(new profile)` last — where the Textual screen's
    cousin, the picker, also put it."""
    rows = [profile_item(x) for x in profiles]
    rows.append(Item(head=NEW_PROFILE, kind="new", key="#new"))
    return rows


class ProfilesOverlay(ListOverlay):
    """`a` from the sessions column: every profile, and the five keys on one.

    Enter edits a profile's memories, `s` its skills, `r` the RAG archive, `c`
    copies it under a new name, `d` deletes it. On the `(new profile)` row
    Enter names and creates one and the other four are inert — a key list that
    lies is worse than a short one.
    """

    title = "profiles & learnings"
    name = "profiles"

    def __init__(self, profiles: Iterable[ProfileInfo] | None = None) -> None:
        self.profiles = list(profiles or [])
        super().__init__(profile_rows(self.profiles))
        # The name the last child screen was opened about, since the child
        # itself is content-agnostic and a `d` in between could have moved the
        # cursor. Carried the way every deciding screen here carries its
        # subject.
        self.subject = ""

    def keymap(self) -> list[tuple[str, str]]:
        if self.on_new:
            return [("↑↓", "move"), ("enter", "create a profile"), ("esc", "back")]
        return [
            ("↑↓", "move"),
            ("enter", "memories"),
            ("s", "skills"),
            ("r", "archive"),
            ("c", "copy"),
            ("d", "delete"),
            ("esc", "back"),
        ]

    # ------------------------------------------------------------ the cursor

    # The footer is drawn from the cursor and the keys act on it, so both ask
    # the same question. 120 is only the width the flattening is measured at;
    # a row's identity does not depend on it.
    @property
    def on_new(self) -> bool:
        item = self.item(120)
        return item is None or item.kind == "new"

    def info(self, name: str) -> ProfileInfo | None:
        return next((x for x in self.profiles if x.name == name), None)

    def refresh(self) -> None:
        self.replace(profile_rows(self.profiles))

    # ------------------------------------------------------------- the keys

    def chose(self, item: Item | None) -> bool:
        if item is None:
            return True
        if item.kind == "new":
            return self.open(
                PromptOverlay(
                    title="new profile",
                    hint="a short name; it becomes a directory",
                ),
                NAME_NEW,
            )
        return self._edit(item.text, MEMORIES)

    def _edit(self, name: str, kind: str) -> bool:
        """Open one of a profile's two files, or say why it cannot be."""
        self.subject = name
        info = self.info(name)
        if info is None or not info.loaded:
            self.note = UNREADABLE
            return True
        return self.open(
            TextEditOverlay(
                info.text if kind == MEMORIES else info.archive,
                profile=name,
                kind=kind,
            ),
            kind,
        )

    def other(self, key: str, item: Item | None) -> bool:
        if item is None or item.kind == "new":
            return True  # the four profile keys are inert on `(new profile)`
        self.subject = item.text
        info = self.info(item.text)
        if key == "r":
            return self._edit(item.text, ARCHIVE)
        if key == "s":
            if info is None or not info.loaded:
                self.note = UNREADABLE
                return True
            return self.open(SkillsOverlay(info))
        if key == "c":
            return self.open(
                PromptOverlay(
                    f"{item.text} copy",
                    title=f"copy “{item.text}” to",
                    hint="it starts from everything the original learned",
                ),
                NAME_COPY,
            )
        if key == "d":
            if info is not None and info.default:
                self.note = DEFAULT_REFUSAL
                return True
            self.ask(DELETE_QUESTION.format(name=item.text))
        return True

    def answered(self, question: str, yes: bool) -> bool:
        if not yes:
            self.note = "kept"
            return True
        name = self.subject
        self.send(DeleteProfile(name))
        # Taken off here rather than waited for, like the sidebar's delete: the
        # core refuses some deletions and says so in a toast, and the next list
        # it sends is what puts the row back.
        self.profiles = [x for x in self.profiles if x.name != name]
        self.refresh()
        self.note = f"deleted “{name}”"
        return True

    def child_closed(self, child) -> None:
        if child.tag in (MEMORIES, ARCHIVE) and getattr(child, "saved", False):
            self.send(SaveProfile(self.subject, child.tag, child.text))
            info = self.info(self.subject)
            if info is not None:
                if child.tag == MEMORIES:
                    info.text = child.text
                    info.memories = sum(
                        1 for line in child.text.splitlines() if line.strip()
                    )
                else:
                    info.archive = child.text
                self.refresh()
            self.note = f"{child.tag} kept"
        elif child.tag == NAME_NEW and getattr(child, "name", ""):
            self.send(CreateProfile(child.name))
            # Not added to the list here: whether the name is usable is the
            # profile store's rule and its wording (`_create_profile`), so a
            # row invented now could be a row for a profile that was refused.
            self.note = f"creating “{child.name}”"
        elif child.tag == NAME_COPY and getattr(child, "name", ""):
            self.send(CopyProfile(self.subject, child.name))
            self.note = f"copying “{self.subject}” to “{child.name}”"
        elif isinstance(child, SkillsOverlay):
            # The skills screen sent its own saves; what comes back is only
            # what the list now has to say about how many there are.
            info = self.info(child.profile.name)
            if info is not None:
                info.skills = list(child.profile.skills)
                self.refresh()
