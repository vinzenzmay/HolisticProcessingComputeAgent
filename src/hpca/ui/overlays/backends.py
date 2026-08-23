"""How a backend is drawn, wherever it is drawn.

Three lists show LLMs — manage-LLMs' two panels and `ctrl+l`'s switcher — and
they show the same things: the two marks, the label, the endpoint, the model
and the context size. One renderer, because a row that meant `●` in one panel
and something else in another would be the marks stopping being worth reading.
"""

from __future__ import annotations

from hpca.ui import theme
from hpca.ui.pane import Item
from hpca.ui.state import BackendInfo

# ● answered the last probe, ○ did not, · nobody has asked. The third is not
# decoration: the catalog arrives before the probes do, and "not asked yet"
# drawn as "not answering" would libel every backend for as long as the probes
# take (`protocol.LLMEntry.reachable`, `protocol.LLMCatalog.probed`).
MARKS: dict[bool | None, str] = {True: "●", False: "○", None: "·"}
# Reachable is the only one that gets a colour of its own; unknown and
# unreachable are both "nothing to say yet". Roles by name, resolved when the
# row is drawn (`ui.theme`).
MARK_ROLES: dict[bool | None, str] = {True: "ok", False: "faint", None: "faint"}


def mark_colour(reachable: bool | None) -> str:
    """The colour a backend's ● is drawn in."""
    return getattr(theme, MARK_ROLES[reachable])

# What `★` means, and it is only ever this: the backend a session gets by not
# choosing one. Not "the one this screen would pick".
ACTIVE = "★"

# What a scan puts in the model field of an endpoint that answered its probe
# with a 401: it is up, and the model name is behind the key. `hpca.discover`
# mints it; the string is copied rather than imported for the reason
# `app.DEFAULT_PROFILE` is — nothing under `hpca.ui` imports the rest of the
# package to read one constant — and `tests/test_ui_overlays.py` holds the two
# to each other.
KEY_REQUIRED = "(api key required)"


def context_label(size: int) -> str:
    """`112k` — a context length as the width of a column allows."""
    if not size:
        return ""
    if size >= 1000:
        return f"{round(size / 1000)}k"
    return str(size)


def backend_head(info: BackendInfo, *, star: bool = True) -> str:
    """One line: the marks, the label, and the three details behind it."""
    mark = MARKS[info.reachable]
    lead = f"{ACTIVE if info.active else ' '} {mark} " if star else f"{mark} "
    details = [
        x for x in (info.base_url, info.model, context_label(info.context)) if x
    ]
    if info.needs_key:
        details.append("api key required")
    return f"{lead}{info.label or info.model:<22}{'   '.join(details)}"


def row_key(info: BackendInfo) -> str:
    """What a backend row *is*, for a list repainted under a cursor.

    The label, because that is the identity the core mints and the one field a
    command may name an entry by — two vLLMs serving one model on two nodes
    get two labels precisely so a user can tell them apart.
    """
    return info.label


def backend_item(info: BackendInfo, *, star: bool = True) -> Item:
    """The row, and what it is, under it."""
    body = []
    if info.discovered:
        body.append("found by a scan; not configured")
    if info.active:
        body.append("the active default — what an unpinned session talks to")
    if info.reachable is False:
        body.append("did not answer the last probe")
    elif info.reachable is None:
        body.append("not probed")
    return Item(
        head=backend_head(info, star=star),
        body=body,
        accent=mark_colour(info.reachable),
        kind="backend",
        text=info.label,
        key=row_key(info),
    )
