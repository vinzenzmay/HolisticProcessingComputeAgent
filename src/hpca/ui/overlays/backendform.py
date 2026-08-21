"""The backend form (§4.3 item 28): URL, model, key, context length.

The port scan lists a key-locked endpoint only as "(api key required)" — it
cannot read the model name or the context length through the 401 — so this
form is where a key is supplied and, with it, the details that could not be
read. Four fields, tab between them, and the whole thing is one `Overlay`
rather than four widgets, because a form is a list of one-line editors and the
list is what makes tab mean something.

**The probe is not here.** `tui/backend_form.py` made an authenticated
`/v1/models` call on enter, both to validate the key and to auto-fill the
model and the context from the answer. That call belongs to the core (rule 2
of §4.2: the UI opens no sockets of its own) and there is no command on the
wire for it yet. Until there is, enter *saves* — the core validates the blob
against the real settings model and refuses an unusable one with a `notify`,
which is the same verdict arriving one round trip later.
"""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad
from hpca.ui.editor import Editor
from hpca.ui.overlays.base import BACK_KEYS, Overlay
from hpca.ui.state import BackendInfo, SetBackend

URL, MODEL, KEY, CONTEXT = "base_url", "model", "api_key", "max_model_len"

# The four fields, in the order they are filled, with what each is for. The
# last two say "auto" because a blank one is not an omission — the endpoint
# knows both, and typing them is only necessary when it cannot be asked.
FIELDS = [
    (URL, "endpoint url", "http://localhost:20001/v1"),
    (MODEL, "model", "blank: whatever the endpoint serves"),
    (KEY, "api key", "blank: none needed"),
    (CONTEXT, "context length", "blank: whatever the endpoint reports"),
]

NEED_URL = "an endpoint url is needed — e.g. http://localhost:20001/v1"
NEED_NUMBER = "context length must be a whole number"
# What a key looks like on screen. Shown as dots for the reason the Textual
# form used `password=True`: this window is over a terminal somebody may be
# screen-sharing, and the key is the one field on it that is a secret.
MASK = "•"


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

    # ------------------------------------------------------------- the frame

    def field(self) -> str:
        return FIELDS[self.at][0]

    def value(self, name: str) -> str:
        return self.values[name].text().strip()

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("tab", "next field"),
            ("enter", "save"),
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
            return self.save()
        if self.field() == URL and self.locked_url:
            return True
        self.values[self.field()].handle(key)
        self.note = ""
        return True

    def paste(self, text: str) -> None:
        if self.field() == URL and self.locked_url:
            return
        self.values[self.field()].insert_text(text.replace("\n", " ").strip())

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
