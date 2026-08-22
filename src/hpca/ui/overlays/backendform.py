"""The backend form (§4.3 item 28): URL, model, key, context length.

The port scan lists a key-locked endpoint only as "(api key required)" — it
cannot read the model name or the context length through the 401 — so this
form is where a key is supplied and, with it, the details that could not be
read. Four fields, tab between them, and the whole thing is one `Overlay`
rather than four widgets, because a form is a list of one-line editors and the
list is what makes tab mean something.

**Enter probes; ctrl+s saves anyway.** `tui/backend_form.py` made an
authenticated `/v1/models` call on enter, and the answer does three jobs at
once: it says whether anything OpenAI-shaped is there, it validates the key
that was typed, and it names the models — one auto-fills the blank model
field, several open a picker. The call itself belongs to the core (rule 2 of
§4.2: this UI opens no sockets), and it is `backend.probe`, which landed with
the scan. What survives from before it did is the escape hatch: ctrl+s saves
without asking anything, which is what you want for an endpoint that is
momentarily down but whose URL and model id you already know — and the way
past a key the endpoint rejected but you know is right.
"""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad
from hpca.ui.editor import Editor
from hpca.ui.overlays.backends import backend_item
from hpca.ui.overlays.base import BACK_KEYS, ListOverlay, Overlay
from hpca.ui.state import BackendInfo, ProbeBackend, SetBackend

URL, MODEL, KEY, CONTEXT = "base_url", "model", "api_key", "max_model_len"

# The four fields, in the order they are filled, with what each is for. The
# last two say "auto" because a blank one is not an omission — the endpoint
# knows both, and typing them is only necessary when it cannot be asked.
FIELDS = [
    (URL, "endpoint url", "http://localhost:20001/v1"),
    (MODEL, "model", "blank: enter asks the endpoint what it serves"),
    (KEY, "api key", "blank: none needed"),
    (CONTEXT, "context length", "blank: whatever the endpoint reports"),
]

NEED_URL = "an endpoint url is needed — e.g. http://localhost:20001/v1"
NEED_NUMBER = "context length must be a whole number"
CHECKING = "asking the endpoint what it serves…"
# The three things a probe can come back with. Each one ends in the way past
# it, because a form that reports a refusal without saying what to do next is
# a form the user escapes out of.
KEY_REFUSED = (
    "the endpoint is there and refused that key — fix it, "
    "or press ctrl+s to save anyway"
)
NO_ANSWER = "nothing answered at {url} — press ctrl+s to save it anyway"
ANSWERED = "{model} — enter saves it"
PICK_MODEL = "that endpoint serves several — pick one"
# What a key looks like on screen. Shown as dots for the reason the Textual
# form used `password=True`: this window is over a terminal somebody may be
# screen-sharing, and the key is the one field on it that is a secret.
MASK = "•"
PICKER_TAG = "models"


def same_endpoint(one: str, other: str) -> bool:
    """Whether two endpoint URLs name the same thing, trailing slash aside.

    A probe is answered by the URL it was asked about (`BackendProbed`), and
    the core echoes back what it was given — so the comparison is this side's,
    and the one difference that shows up in practice is a `/v1` against a
    `/v1/`.
    """
    return one.rstrip("/") == other.rstrip("/")


class ModelPickerOverlay(ListOverlay):
    """One endpoint, several models: which of them this backend is.

    `protocol.BackendProbed.models` is a list of `LLMEntry`, drawn by the same
    row renderer the catalog uses, because a probed model carries the same
    four facts as a catalogued one.
    """

    title = "which model"
    name = "models"

    def __init__(self, models: list[BackendInfo]) -> None:
        super().__init__(backend_item(x, star=False) for x in models)
        self.models = list(models)
        self.chosen: BackendInfo | None = None

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "move"), ("enter", "use this one"), ("esc", "cancel")]

    def chose(self, item) -> bool:
        if item is None:
            return True
        self.chosen = next(
            (x for x in self.models if x.label == item.key), None
        )
        return False


class BackendFormOverlay(Overlay):
    """Collect a backend and hand it back as the blob `backend.set` takes.

    ``locked_url`` is the endpoint-was-discovered case: the URL is a fact
    about the row this was opened on, and letting it be edited would make the
    form silently describe a different server than the one that was picked.
    """

    title = "add llm backend"

    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "",
        locked_url: bool = False,
        context: int = 0,
    ) -> None:
        super().__init__()
        self.locked_url = locked_url and bool(base_url)
        self.values = {
            URL: Editor(),
            MODEL: Editor(),
            KEY: Editor(),
            CONTEXT: Editor(),
        }
        self.values[URL].set_text(base_url)
        self.values[MODEL].set_text(model)
        self.values[CONTEXT].set_text(str(context) if context else "")
        # Land on the first field the user actually has to fill: the URL when
        # it is blank and editable, otherwise the model.
        self.at = 1 if self.locked_url or base_url else 0
        self.backend: dict | None = None
        # Whether a probe has come back happy for what is in the fields *now*.
        # Cleared by every edit, because a key typed after an endpoint said yes
        # is a key nothing has checked, and enter would otherwise save it as
        # though it had been.
        self.checked = False
        # Whether one is out. Only the note reads it; nothing here blocks.
        self.probing = False

    # ------------------------------------------------------------- the frame

    def field(self) -> str:
        return FIELDS[self.at][0]

    def value(self, name: str) -> str:
        return self.values[name].text().strip()

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("tab", "next field"),
            ("enter", "save" if self.checked else "check the endpoint"),
            ("^s", "save without checking"),
            ("esc", "cancel"),
        ]

    def body(self, width: int, height: int) -> list[str]:
        rows: list[str] = []
        for index, (name, label, hint) in enumerate(FIELDS):
            here = index == self.at
            locked = name == URL and self.locked_url
            prefix = f"  {'›' if here else ' '} {label:<16}"
            if here and not locked and name != KEY:
                # The value is drawn by the editor itself, so that word motion
                # and selection behave in the form exactly as they do in the
                # message box — and so the cursor is where it says it is.
                drawn = self.values[name].render(
                    max(8, width - len(prefix)), 1, focused=True
                )[0]
                rows.append(CYAN + prefix + RESET + drawn)
                continue
            text = self.value(name)
            if name == KEY:
                # Dots for the reason the Textual form used `password=True`:
                # this window is over a terminal somebody may be sharing.
                text = MASK * len(text) + ("▏" if here else "")
            shown = text or ("" if here else hint)
            style = BOLD + CYAN if here else (DIM if not text else "")
            rows.append(style + pad(prefix + shown, width) + RESET)
            if here and locked:
                rows.append(
                    DIM
                    + pad("    the endpoint this was opened on — not editable", width)
                    + RESET
                )
        rows.append(" " * width)
        rows.append(
            DIM + pad(f"  {FIELDS[self.at][2]}", width) + RESET
        )
        return rows

    # -------------------------------------------------------------- the keys

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return False
        if key in ("tab", "down", "ctrl-down"):
            self.at = (self.at + 1) % len(FIELDS)
            return True
        if key in ("shift-tab", "up", "ctrl-up"):
            self.at = (self.at - 1) % len(FIELDS)
            return True
        if key == "enter":
            return self.save() if self.checked else self.probe()
        if key == "ctrl-s":
            # The deliberate escape hatch: an endpoint that is momentarily
            # down, or a key a proxy rejects but the user knows is right, is
            # still a backend worth having in the catalog.
            return self.save()
        if self.field() == URL and self.locked_url:
            return True
        self.values[self.field()].handle(key)
        self.note = ""
        self.checked = False
        return True

    def paste(self, text: str) -> None:
        if self.field() == URL and self.locked_url:
            return
        self.values[self.field()].insert_text(text.replace("\n", " ").strip())
        self.checked = False

    # ------------------------------------------------------------- the probe

    def probe(self) -> bool:
        """Ask the core what this endpoint serves (`backend.probe`).

        Nothing is waited for: the answer comes back through `probed`, which
        may be a whole timeout later, and the form stays exactly as usable as
        it was — escape still closes it and ctrl+s still saves it.
        """
        if not self.value(URL):
            self.note = NEED_URL
            self.at = 0
            return True
        self.probing = True
        self.note = CHECKING
        self.send(ProbeBackend(self.value(URL), self.value(KEY)))
        return True

    def probed(
        self,
        base_url: str,
        models: list[BackendInfo],
        needs_key: bool = False,
    ) -> None:
        """`backend.probed`, if it is about the endpoint this form is on.

        Three outcomes in two fields (`protocol.BackendProbed`) and each is
        answered here rather than by closing: what the user typed stays on
        screen in every one of them, which is the whole difference between a
        rejected key that can be fixed and one that has to be retyped.
        """
        if not same_endpoint(base_url, self.value(URL)):
            return  # answered after the URL moved on; not ours
        self.probing = False
        if not models:
            self.note = (
                KEY_REFUSED if needs_key else NO_ANSWER.format(url=base_url)
            )
            if needs_key:
                self.at = [x[0] for x in FIELDS].index(KEY)
            return
        if len(models) > 1:
            self.note = PICK_MODEL
            self.open(ModelPickerOverlay(models), PICKER_TAG)
            return
        self.fill_from(models[0])

    def fill_from(self, entry: BackendInfo) -> None:
        """What the endpoint said, into the two fields it can answer for.

        The model and the context length, and neither is overwritten blindly:
        a user who typed a model id meant it, and the probe is confirming the
        endpoint rather than correcting them.
        """
        if not self.value(MODEL):
            self.values[MODEL].set_text(entry.model)
        if not self.value(CONTEXT) and entry.context:
            self.values[CONTEXT].set_text(str(entry.context))
        self.checked = True
        self.note = ANSWERED.format(model=self.value(MODEL) or entry.model)

    def child_closed(self, child) -> None:
        if child.tag != PICKER_TAG or child.chosen is None:
            return
        # A picked model is the answer the single-model case got for free.
        self.values[MODEL].set_text(child.chosen.model)
        self.values[CONTEXT].set_text(
            str(child.chosen.context) if child.chosen.context else ""
        )
        self.checked = True
        self.note = ANSWERED.format(model=child.chosen.model)

    # -------------------------------------------------------------- the save

    def save(self) -> bool:
        """Everything that can be checked without a socket, then send it."""
        if not self.value(URL):
            self.note = NEED_URL
            self.at = 0
            return True
        raw = self.value(CONTEXT)
        if raw and not raw.isdigit():
            self.note = NEED_NUMBER
            self.at = [x[0] for x in FIELDS].index(CONTEXT)
            return True
        backend = {URL: self.value(URL)}
        if self.value(MODEL):
            backend[MODEL] = self.value(MODEL)
        if self.value(KEY):
            backend[KEY] = self.value(KEY)
        if raw:
            backend[CONTEXT] = int(raw)
        self.backend = backend
        # No session: this is the global half of `backend.set`, which is what
        # "add it to the catalog" means — the core adds and persists it, and
        # refuses a blob the settings model will not accept.
        self.send(SetBackend(backend))
        return False

    def info(self) -> BackendInfo:
        """What was entered, as a row the catalog can draw before the core
        answers — the same shown-before-it-is-true bargain the mode bar makes."""
        return BackendInfo(
            # The label the *core* mints is what a command may name an entry
            # by, and it will arrive with the next `llm.catalog`. This one is
            # only for the row drawn in the meantime, which is why nothing
            # sends it anywhere.
            label=self.value(MODEL) or self.value(URL),
            base_url=self.value(URL),
            model=self.value(MODEL),
            context=int(self.value(CONTEXT)) if self.value(CONTEXT).isdigit() else 0,
            needs_key=bool(self.value(KEY)),
        )
