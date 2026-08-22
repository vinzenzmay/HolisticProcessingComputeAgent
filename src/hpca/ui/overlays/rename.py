"""Name something small: rename a session (§4.3 item 11).

One line of `Editor`, prefilled and with the cursor at the end — which is
`PromptOverlay`, the shape this screen turned out to share with naming a new
profile and naming the copy of one (`base.PromptOverlay`). What is left here
is the two things that are a *session's*: the id the rename belongs to, and
the old name under the box.

An empty name is refused rather than sent: a nameless row is a row the user
cannot find again, and the core refuses it too (`_rename_session`). It is
refused by keeping the screen open with a word about why, because the other
way to spell "refused" — closing and quietly changing nothing — looks exactly
like the save having worked.
"""

from __future__ import annotations

from hpca.ui.overlays.base import PromptOverlay

EMPTY_REFUSAL = "a session needs a name"


class RenameOverlay(PromptOverlay):
    """`r` on a session row: the name, editable, in a box.

    Carries the session it was opened on, for the reason every screen in this
    package that decides something does: a rename that closes after the user
    has switched sessions must not rename the new one.
    """

    title = "rename session"
    refusal = EMPTY_REFUSAL

    def __init__(self, title: str = "", session_id: str = "") -> None:
        super().__init__(title, hint=f"was “{title}”")
        self.session_id = session_id

    def keymap(self) -> list[tuple[str, str]]:
        return [("enter", "save"), ("esc", "keep the old name")]
