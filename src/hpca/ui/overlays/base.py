"""What every screen drawn over the rows has in common.

Nine screens landed in one milestone, and the thing worth protecting is that
the ninth is short. Four shapes came out of the five that already existed
(`newsession`, `rename`, `queued`, `rewind`, `help`), and every screen in the
package is one of them:

* ``Overlay`` — a titled frame with a note on its rule, a footer of key hints,
  escape closes, and the one yes/no a screen ever has to ask *itself*
  ("Keep changes?", "delete this?"). That question is drawn by the screen and
  not by `RowUI.ask`, because it is about the screen's own unsaved state: a
  confirmation that outlived the screen would be answering for a dialog that
  is no longer there.
* ``ListOverlay`` — one `Pane` in that frame, with the motion keys every list
  in this UI answers the same way, and ``chose``/``other`` for the two
  decisions a list actually makes.
* ``EditorOverlay`` — one `Editor` in that frame. Escape asks to keep changes
  **only when there are changes** — "no edits means no question", which the
  Textual `MemoryEditorScreen` and `SettingsScreen` both spelled out by
  comparing against the text they opened with — and a subclass can refuse to
  close at all by answering ``refuse()``.
* ``PromptOverlay`` — one line of `Editor` for naming something, prefilled,
  with an empty answer refused on screen rather than sent.

The frame itself is `framed()`: a rule, a body, whatever trails it, padded to
exactly ``height``. Overlays are where the width discipline breaks, because
they draw frames inside frames, so there is one function that does the padding
and every screen goes through it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, YELLOW, pad, rule
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS
from hpca.ui.pane import Item, Pane

# What closes a screen everywhere. `quit` is here because ctrl+c on a modal
# means "get me out of this", not "kill the app underneath it".
BACK_KEYS = ("esc", "quit")

# The question an editor asks on the way out, in the words the Textual
# `MemoryEditorScreen` used, so the reflex transfers.
KEEP_CHANGES = "Keep changes?"

# How a yes/no is answered, on every screen that asks one. Escape is an answer
# here rather than a way past — the same rule the app's own confirm follows.
VERDICTS = {
    "y": True,
    "Y": True,
    "n": False,
    "N": False,
    "esc": False,
    "quit": False,
}


def framed(
    title: str,
    body: Sequence[str],
    width: int,
    height: int,
    *,
    note: str = "",
    tail: Sequence[str] = (),
) -> list[str]:
    """A titled rule, a body, and a tail — exactly ``height`` rows.

    ``body`` and ``tail`` are expected to be width-exact already (`Pane.render`
    and `Editor.render` both are), because padding a styled line a second time
    is how a row ends up one cell short: `pad` counts the SGR escapes.
    """
    out = [BOLD + CYAN + rule(title, width, note) + RESET]
    tail = list(tail)
    room = max(0, height - 1 - len(tail))
    body = list(body)
    out += body[:room]
    out += [" " * width] * max(0, room - len(body))
    out += tail
    return out[:height]


def keyed(text: str, width: int, key_cells: int = 12, at: int = 6) -> str:
    """One `      key         what it does` row, coloured by column.

    Padded plain and coloured afterwards, never by adding the escape lengths
    to the width — that arithmetic is exactly the kind that leaves a row one
    cell short. Lifted out of `help.py`, where three screens had copied it.
    """
    row = pad(text, width)
    return row[:at] + CYAN + row[at : at + key_cells] + RESET + row[at + key_cells :]


def options(pairs: Iterable[tuple[str, str]], width: int) -> list[str]:
    """The key/label block the dialogs draw under their subject."""
    return [keyed(f"      {key:<12}{label}", width) for key, label in pairs]


class Overlay:
    """A screen drawn over the rows. ``handle`` returning False closes it."""

    title = ""
    # A word on the right of the rule: what just happened, or what is wrong.
    note = ""
    # The yes/no this screen is asking itself, or "". Class-level so a screen
    # that never asks one need not call ``super().__init__``.
    asking = ""
    # A screen this one wants opened over it, once. `RowUI` takes it, wires its
    # ``send``, and pushes it; when it closes, ``child_closed`` is called here.
    #
    # Nested rather than a second top-level overlay because the profiles screen
    # is genuinely three deep — the list, a profile's skills, one skill's file
    # — and escaping the file has to land back on the skills, not on the rows.
    child: "Overlay | None" = None
    # What the parent opened this screen *for*, since one editor serves the
    # memories, the archive and a skill body. Set by whoever opens the child.
    tag = ""

    def __init__(self) -> None:
        self.note = ""
        self.asking = ""
        self.child = None
        self.tag = ""
        # What this screen asked the core to do. Replaced by `RowUI` with its
        # own sender when the screen is pushed; kept as a list otherwise, so a
        # test can construct the screen alone and read what it decided.
        self.sent: list = []
        self.send = self.sent.append

    def open(self, child: "Overlay", tag: str = "") -> bool:
        """Put a screen over this one. Always returns True — the caller is a
        ``chose``/``other`` answer, and opening a child never closes us."""
        child.tag = tag
        self.child = child
        return True

    def child_closed(self, child: "Overlay") -> None:
        """A screen opened over this one has gone. Read what it decided."""

    # ------------------------------------------------ what a subclass fills in

    def heading(self) -> str:
        """The rule's label. Overridden where a screen's title has stages."""
        return self.title

    def body(self, width: int, height: int) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def keys(self, key: str, width: int, height: int) -> bool:
        """A key, with no question in the way. False closes the screen."""
        return key not in BACK_KEYS

    def answered(self, question: str, yes: bool) -> bool:
        """The verdict on the question this screen asked. False closes it."""
        return True

    def keymap(self) -> list[tuple[str, str]]:
        return [("esc", "back")]

    # ------------------------------------------------------------- the frame

    def render(self, width: int, height: int) -> list[str]:
        tail = self.question_rows(width)
        return framed(
            self.heading(),
            self.body(width, max(1, height - 1 - len(tail))),
            width,
            height,
            note=self.note,
            tail=tail,
        )

    def question_rows(self, width: int) -> list[str]:
        if not self.asking:
            return []
        return [
            YELLOW + pad(f"  {self.asking}", width) + RESET,
            DIM + pad("  (y) yes · (n) no · (esc) no", width) + RESET,
        ]

    def ask(self, question: str) -> None:
        """Put a yes/no on this screen. The answer comes back to ``answered``."""
        self.asking = question

    # -------------------------------------------------------------- the keys

    def handle(self, key: str, width: int, height: int) -> bool:
        if self.asking:
            # A question is answered before anything else can be done, the same
            # bargain the app's own confirm makes: it is a question with two
            # answers and no third thing to be doing meanwhile.
            verdict = VERDICTS.get(key)
            if verdict is None:
                return True
            question, self.asking = self.asking, ""
            return self.answered(question, verdict)
        return self.keys(key, width, height)

    def paste(self, text: str) -> None:
        """A block the terminal handed over whole. Ignored unless the screen
        has somewhere to put text — the key list has not."""

    def footer(self) -> list[tuple[str, str]]:
        if self.asking:
            return [("y", "yes"), ("n", "no"), ("esc", "no")]
        return self.keymap()


class ListOverlay(Overlay):
    """One list in the frame, and the two decisions a list makes.

    The motion keys are here rather than in each screen for the reason the
    footer is: a list that scrolled differently from the sessions column would
    be a second set of habits to learn, and there are nine of these.
    """

    name = "list"

    def __init__(self, items: Iterable[Item] = (), *, name: str = "") -> None:
        super().__init__()
        self.pane = Pane(name or self.name, list(items))

    def body(self, width: int, height: int) -> list[str]:
        return self.pane.render(width, max(1, height), focused=True)

    # The pane's own rule eats one row of the frame's body, and the question —
    # when one is up — eats two more. What scrolls has to agree with what was
    # drawn, or page-down moves by a different amount than it showed.
    def inner(self, width: int) -> int:
        return max(8, width - 2)

    def view(self, height: int) -> int:
        return max(1, height - 2 - len(self.question_rows(1)))

    def index(self, width: int) -> int:
        return self.pane.current(self.inner(width))

    def item(self, width: int) -> Item | None:
        at = self.index(width)
        return self.pane.items[at] if 0 <= at < len(self.pane.items) else None

    def replace(self, items: Iterable[Item], width: int = 120) -> None:
        """A new list, with the cursor kept on the row it was on."""
        self.pane.replace(list(items), self.inner(width))

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return False
        if self.move(key, width, height):
            return True
        if key == "enter":
            return self.chose(self.item(width))
        return self.other(key, self.item(width))

    def move(self, key: str, width: int, height: int) -> bool:
        pane, inner, view = self.pane, self.inner(width), self.view(height)
        steps = {
            "up": -1,
            "down": 1,
            "pgup": -view,
            "pgdn": view,
            "home": -(10**9),
            "end": 10**9,
        }
        if key in steps:
            pane.move(steps[key], view, inner)
        elif key == "right":
            # Open it; on one already open, step into what it opened, the way
            # the rows do.
            if not pane.expand(inner) and pane.is_open(pane.current(inner)):
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        else:
            return False
        return True

    def chose(self, item: Item | None) -> bool:
        """Enter on the row under the cursor. False closes the screen."""
        return True

    def other(self, key: str, item: Item | None) -> bool:
        """A key the list itself has no answer for. False closes the screen."""
        return True


class EditorOverlay(Overlay):
    """Raw text in the frame; escape asks to keep changes, if there are any.

    "No edits means no question" is the claim (specs-ui-acceptance.md,
    Profiles), and it is one comparison against the text the screen opened
    with — which is also what makes an accidental open-and-escape free.
    """

    numbers = True
    question = KEEP_CHANGES

    def __init__(self, text: str = "", *, title: str = "") -> None:
        super().__init__()
        self.editor = Editor(text)
        self.was = text
        # What was kept. ``saved`` rather than ``bool(self.text)``, because
        # keeping an emptied file is a decision and it looks like "".
        self.text = ""
        self.saved = False
        if title:
            self.title = title

    @property
    def dirty(self) -> bool:
        return self.editor.text() != self.was

    def body(self, width: int, height: int) -> list[str]:
        return self.editor.render(
            width, max(1, height), focused=True, numbers=self.numbers
        )

    def paste(self, text: str) -> None:
        self.editor.insert_text(text)

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("↑↓←→", "move"),
            ("^u", "clear"),
            ("esc", "back — asks to keep changes"),
        ]

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return self.close()
        if key == "enter" or key in NEWLINE_KEYS:
            self.editor.newline()
        else:
            self.editor.handle(key)
        self.note = ""
        return True

    def close(self) -> bool:
        if not self.dirty:
            return False  # nothing changed, so nothing is asked
        blocker = self.refuse()
        if blocker:
            # Refused rather than dismissed: the work stays on screen, which is
            # the whole reason the config editor validates on the way *out*.
            self.note = blocker
            return True
        self.ask(self.question)
        return True

    def refuse(self) -> str:
        """Why this text cannot be kept, or "". The config editor's validator."""
        return ""

    def answered(self, question: str, yes: bool) -> bool:
        if yes:
            self.text = self.editor.text()
            self.saved = True
        return False


class PromptOverlay(Overlay):
    """Name something small: one line of `Editor`, prefilled.

    The point of the screen is that a name is *edited* rather than retyped,
    which is what `tui/rename_screen.py`'s ``cursor_position = len(value)``
    was for. An empty name is refused here rather than sent, by keeping the
    screen open with a word about why — because the other way to spell
    "refused", closing and quietly changing nothing, looks exactly like the
    save having worked.
    """

    title = "name"
    refusal = "a name is needed"

    def __init__(self, value: str = "", *, title: str = "", hint: str = "") -> None:
        super().__init__()
        self.editor = Editor()
        self.editor.set_text(value)  # cursor left at the end of it
        self.was = value
        self.hint = hint
        self.name = ""
        if title:
            self.title = title

    def keymap(self) -> list[tuple[str, str]]:
        return [("enter", "save"), ("esc", "cancel")]

    def paste(self, text: str) -> None:
        self.editor.insert_text(text)

    def render(self, width: int, height: int) -> list[str]:
        tail = list(self.question_rows(width))
        if self.hint:
            tail = [DIM + pad(f"  {self.hint}", width) + RESET] + tail
        return framed(
            self.heading(),
            self.editor.render(width, max(1, height - 1 - len(tail)), focused=True),
            width,
            height,
            note=self.note,
            tail=tail,
        )

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return False
        if key == "enter":
            name = self.editor.text().strip()
            if not name:
                self.note = self.refusal
                return True
            self.name = name
            return False
        self.editor.handle(key)
        return True
