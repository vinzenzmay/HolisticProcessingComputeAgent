"""The palette, as the settings file last described it.

Every colour this UI draws with used to be a module constant in `ui.ansi`, and
constants are what a themeable palette cannot be: twenty-one modules say
``from hpca.ui.ansi import CYAN``, which binds the *string* at import time, so
rebinding the name later reaches none of them. A palette that changes while the
app runs has to be looked up when the row is drawn, not when the module is
loaded. That is the whole reason this module exists.

So there is one `Theme` instance, `theme`, and call sites read
``theme.chrome`` at paint time. Its fields are replaced together by `apply`,
never edited one at a time — a half-applied palette would draw one frame in two
themes, and the frame after it would look correct, which is the kind of
divergence nobody reports because it is gone before it can be described. The
same argument `state.Display` makes for being frozen, made here about the swap
rather than about the object.

A global, then, and deliberately: the palette is a property of the *terminal*,
of which this process drives exactly one. `Display` is per-session data and is
threaded; this is not.

Colours are written in the settings file the way a terminal thinks of them:
either an xterm-256 index (``"215"``) or a hex triple (``"#ffaf5f"``). Both are
kept rather than one being normalised into the other, because they are not
interchangeable in what they cost: an index is four bytes on the wire and works
on every terminal ever built, and a hex triple needs truecolor and eleven. The
index is the default for everything, and hex is what a hand-built theme reaches
for when the 256 palette has no colour close enough — which is most of the time
when the theme is a light one, since the cube's dark end is where its resolution
went.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hpca.ui.ansi import ESC

# What a colour may look like in settings.json. Anchored, because a half-match
# is a string that is *nearly* a colour, and the useful answer to that is the
# refusal that names the field, not a row drawn in whatever the prefix meant.
INDEX = re.compile(r"^(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# The six values an xterm-256 cube channel can take. Unevenly spaced, which is
# the fact the pulse ramp below is built around.
LEVELS = (0, 95, 135, 175, 215, 255)


def valid(spec: str) -> bool:
    """Whether `spec` is a colour this module can draw."""
    return bool(INDEX.match(spec) or HEX.match(spec))


def sgr(spec: str, *, background: bool = False) -> str:
    """The escape sequence that selects `spec`.

    ``background`` picks layer 48 over 38, and is used by two callers — the
    focus flash, which was the first thing in this UI ever to set a background
    colour, and the ground every toast is drawn on.
    Everything else is foreground over whatever ground the user's terminal
    paints.
    """
    if not valid(spec):
        raise ValueError(f"{spec!r} is not a colour")
    layer = 48 if background else 38
    if spec.startswith("#"):
        r, g, b = (int(spec[i : i + 2], 16) for i in (1, 3, 5))
        return f"{ESC}[{layer};2;{r};{g};{b}m"
    return f"{ESC}[{layer};5;{spec}m"


def rgb(spec: str) -> tuple[int, int, int]:
    """`spec` as three 0-255 channels, whichever way it was written."""
    if spec.startswith("#"):
        return tuple(int(spec[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    n = int(spec)
    if n < 16:  # the sixteen the terminal's own theme names; approximated
        base = 0xC0 if n < 8 else 0xFF
        return (base * (n & 1), base * ((n >> 1) & 1), base * ((n >> 2) & 1))
    if n < 232:  # the 6x6x6 cube
        n -= 16
        return (LEVELS[n // 36], LEVELS[(n // 6) % 6], LEVELS[n % 6])
    return ((n - 232) * 10 + 8,) * 3  # the 24-step grey ramp


def _levels(spec: str) -> tuple[int, int, int]:
    """`spec` as cube levels 0-5, by nearest channel."""
    return tuple(  # type: ignore[return-value]
        min(range(6), key=lambda i: abs(LEVELS[i] - c)) for c in rgb(spec)
    )


def ramp(start: str, end: str, steps: int) -> tuple[str, ...]:
    """`steps` colours walking from `start` to `end`, inclusive of both.

    Two walks, chosen by how the ends were written, and the distinction is the
    one `ansi` documented when this was the decision prompt's private ramp:
    interpolating two cube indices in *RGB* and quantising back rounds the three
    channels at different points, which drops a grey step into the middle of a
    walk that never meant to leave its hue. So indices are walked in level
    space, where every step lands on a colour that exists.

    Hex ends have no such constraint — truecolor has all the colours between —
    so those are walked in RGB, which is the more faithful path of the two and
    is available exactly when it can be used.
    """
    if start.startswith("#") or end.startswith("#"):
        a, z = rgb(start), rgb(end)
        out = [
            "#%02x%02x%02x"
            % tuple(round(p + (q - p) * (i / max(1, steps - 1))) for p, q in zip(a, z))
            for i in range(steps)
        ]
    else:
        a, z = _levels(start), _levels(end)
        out = []
        for i in range(steps):
            share = i / max(1, steps - 1)
            r, g, b = (round(p + (q - p) * share) for p, q in zip(a, z))
            out.append(str(16 + 36 * r + 6 * g + b))
    # The two ends are what was asked for, exactly, whatever the walk between
    # them had to round to. It matters because the ends are the colours the
    # sweep *rests* on — a breath lingers at its extremes and crosses the
    # middle fastest — so a pale end that came back as the cube's nearest white
    # rather than the palette's own would be the one part of the ramp anybody
    # could see was wrong.
    out[0], out[-1] = start, end
    return tuple(out)


# How many colours the decision prompt's breath is made of. Seven is what the
# 256-colour cube actually has along the longest walk between two of its
# corners; asking for more only repeats colours. `ui.ansi.pulse` indexes this.
PULSE_STEPS = 7


@dataclass(frozen=True)
class Theme:
    """The palette in the form the renderer uses it: finished escape sequences.

    Built once per settings change rather than per row. Every field is the
    whole sequence, ready to concatenate, because the alternative is `sgr` in
    the repaint's way for an answer that cannot have changed since the last
    time the settings did.
    """

    # The two that carry prose somebody sits and reads.
    agent: str
    user: str
    # Rules, focused titles, key hints — the structure, not the content.
    chrome: str
    # The signal three. Nothing decorative is ever drawn in these: the whole
    # value of `ok` is that a green on screen means something is well.
    ok: str
    warn: str
    danger: str
    # Secondary and tertiary text. These replace the `DIM` attribute, whose
    # trouble was never that it looked wrong but that what it looked like was
    # the terminal theme's opinion — barely visible on one, nearly full
    # brightness on the next, and the same screen either way.
    muted: str
    faint: str
    # The working row's drop, head first. As many cells as the trail is long;
    # `rain.spinner` lights min(len(this), SPINNER_WIDTH) of them.
    spinner: tuple[str, ...]
    # The focus flash: a background, and how long it is held. Held rather than
    # decayed — a fade needs frames to fade over, and the flash is over before
    # the second one would arrive.
    flash: str
    flash_hold: float
    # The ground every toast is drawn on — the second background, after the
    # flash.
    overlay: str
    # The decision prompt's breath, palest first.
    pulse: tuple[str, ...]


def build(
    *,
    agent: str,
    user: str,
    chrome: str,
    ok: str,
    warn: str,
    danger: str,
    muted: str,
    faint: str,
    spinner: tuple[str, ...] | list[str],
    flash: str,
    flash_hold: float,
    overlay: str,
) -> Theme:
    """A `Theme` from the colour *specs* a settings file holds."""
    return Theme(
        agent=sgr(agent),
        user=sgr(user),
        chrome=sgr(chrome),
        ok=sgr(ok),
        warn=sgr(warn),
        danger=sgr(danger),
        muted=sgr(muted),
        faint=sgr(faint),
        spinner=tuple(sgr(s) for s in spinner),
        flash=sgr(flash, background=True),
        flash_hold=flash_hold,
        overlay=sgr(overlay, background=True),
        # Between the colour prose is written in and the colour the chrome is,
        # which is what the ramp has always walked — it was written as two
        # hard-coded triples when both ends were constants, and it is the same
        # two ends now that they are settings.
        pulse=tuple(sgr(c) for c in ramp(agent, chrome, PULSE_STEPS)),
    )


# What the app draws with until `hello` lands, and what a settings file that
# names no colours gets. The values are `config.PaletteSettings`' defaults; the
# duplication is the same one `state.Display` carries, for the same reason —
# this module is reachable with no core in the process to have been told by.
# How long the focus flash is held, in seconds, when nothing has said
# otherwise. Named here as well as in `config` because a renderer is reachable
# with no core in the process to have been told by — the same standing-in
# `state.Display` does for the rain and the pulse.
FLASH_HOLD = 0.1

DEFAULTS = dict(
    agent="255",
    user="215",
    chrome="73",
    ok="71",
    warn="172",
    danger="167",
    muted="250",
    faint="245",
    spinner=("73", "66", "23", "236"),
    flash="23",
    flash_hold=FLASH_HOLD,
    overlay="236",
)

_current = build(**DEFAULTS)  # type: ignore[arg-type]


def __getattr__(name: str) -> object:
    """Read a colour off the palette in force *now*.

    The indirection is the feature. A caller writes ``from hpca.ui import
    theme`` and then ``theme.chrome``, so what it holds is this module — one
    object that never changes — and the colour is fetched when the row is
    drawn. Had it imported the palette itself it would hold whichever one was
    current at import, which for a themeable UI is the wrong one from the first
    save onwards.

    PEP 562, and it fires on every access because these are deliberately not
    module attributes: making them real names is exactly the staleness this
    exists to prevent.
    """
    try:
        return getattr(_current, name)
    except AttributeError:
        raise AttributeError(f"no colour named {name!r}") from None


def current() -> Theme:
    """The whole palette at once, for a caller that wants it consistent.

    A render pass that reads six colours through `__getattr__` could in
    principle straddle a settings save. Nothing in this UI does — the swap
    happens between frames, in `RowUI.set_display` — but a caller that wants
    the guarantee in its own hands takes it here rather than by luck.
    """
    return _current


def _sound(specs: dict[str, object]) -> dict[str, object]:
    """`specs` with anything undrawable dropped, so a bad key costs one colour.

    The settings model refuses a malformed colour where the user can read the
    refusal (`config.PaletteSettings`), and this is the second line rather than
    the first: what arrives here has crossed a wire, and the process it arrives
    in is holding a terminal in raw mode. A palette is not worth a lost screen,
    and neither is it worth an all-or-nothing verdict — one unreadable entry
    falls back to the built-in for *that role* and the rest of the theme the
    user wrote still applies.
    """
    out: dict[str, object] = {}
    for name, value in specs.items():
        if name == "flash_hold":
            out[name] = value if isinstance(value, (int, float)) and value >= 0 else FLASH_HOLD
        elif name == "spinner":
            trail = [s for s in (value or ()) if isinstance(s, str) and valid(s)]
            if trail:
                out[name] = tuple(trail)
        elif isinstance(value, str) and valid(value):
            out[name] = value
    return out


def apply(**specs: object) -> None:
    """Swap the whole palette for one built from `specs`.

    Whole, and that is the point: one rebinding of one name to a new frozen
    `Theme`, so no frame can be drawn against half a palette. Callers pass the
    fields they know and inherit the rest, which is what lets a settings file
    name three colours and mean it.
    """
    global _current
    _current = build(**{**DEFAULTS, **_sound(specs)})  # type: ignore[arg-type]


def reset() -> None:
    """Back to the built-in palette. For tests, which share this module."""
    apply()
