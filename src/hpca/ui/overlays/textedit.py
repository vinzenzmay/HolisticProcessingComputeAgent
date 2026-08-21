"""The raw text editor (§4.3 item 27): memories, the archive, a skill file.

One screen for three things, because it is the same act: open a file the agent
also writes, edit it as text, and decide on the way out whether the edit
stands. `tui/profiles_screen.py`'s `MemoryEditorScreen` said this in words —
"content-agnostic: it returns the edited text and the caller decides where it
lands" — and that is exactly the split kept here: the screen knows the title
and the text, the profiles screen knows which file it came from.

Everything that makes it an editor is `EditorOverlay`: escape asks to keep
changes, and no edits means no question.
"""

from __future__ import annotations

from hpca.ui.ansi import DIM, RESET, pad
from hpca.ui.overlays.base import EditorOverlay

# What the three uses are called on the rule, and what the parent does with the
# text afterwards. Strings rather than an enum because two of them are also
# `protocol.ProfileSave.kind` values and would only be translated again.
MEMORIES, ARCHIVE, SKILL = "memories", "archive", "skill"

# What each is *about*, under the rule, since "archive" alone does not say that
# these are the entries the curator aged out and that moving a block back up is
# the reason to be here.
ABOUT = {
    MEMORIES: "what this profile has learned — the agent reads this every turn",
    ARCHIVE: "aged out of RAG by the curator; move a block back to keep it",
    SKILL: "the procedure, front matter and all",
}


class TextEditOverlay(EditorOverlay):
    """A profile's memories, its archive, or one skill's file, as plain text.

    Prose rather than code, so no line numbers: the memory file is read by the
    model as sentences, and a gutter of numbers is width taken from the
    sentences to say something nobody needs here. The config editor keeps its
    numbers, because a JSON error names a line.
    """

    numbers = False

    def __init__(
        self,
        text: str = "",
        *,
        profile: str = "",
        name: str = "",
        kind: str = MEMORIES,
        awaiting=None,
    ) -> None:
        super().__init__(text, awaiting=awaiting)
        self.profile = profile
        # The skill's name for a skill file, and empty for the other two: what
        # the parent needs to know where the text goes back to.
        self.name = name
        self.kind = kind

    def heading(self) -> str:
        subject = self.name or self.kind
        return f"{subject} · {self.profile}" if self.profile else subject

    def body(self, width: int, height: int) -> list[str]:
        about = ABOUT.get(self.kind, "")
        rows = [DIM + pad(f"  {about}", width) + RESET] if about else []
        return rows + super().body(width, max(1, height - len(rows)))
