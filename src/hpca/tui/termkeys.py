"""Key-decoding fixes that have to happen below Textual's event layer.

By the time a key reaches a widget, Textual has already named it; a modifier
the parser dropped cannot be recovered there. What is patched here is that
naming step.
"""

from __future__ import annotations

from textual import events
from textual._xterm_parser import XTermParser

_patched = False


def patch_alt_enter() -> None:
    """Make an ESC-prefixed Return arrive as ``alt+enter``, not a bare ``enter``.

    Plain terminals cannot report shift+enter at all — Return is one byte with
    no room for the modifier — so the usual workaround is a terminal binding
    that turns shift+enter into ESC CR, the escape prefix meaning "alt". That
    is what Alacritty's ``chars = "\\u001B\\r"`` binding sends, and what
    Claude Code's own terminal setup writes.

    Textual sees the ESC prefix and carries an ``alt=True`` flag down, but only
    prefixes ``alt+`` onto *single character* key names. "enter" is five, so
    the flag is dropped and the key surfaces as an ordinary enter. In the chat
    box that means shift+enter sends the draft instead of starting a new line.

    Terminals that speak the Kitty keyboard protocol (which Textual turns on)
    never take this path: they report shift+enter as ``CSI 13;2u`` and Textual
    names it correctly. This only rescues the escape-prefix workaround, and
    only for Return — leaving ESC ESC alone, which must stay ``escape``.
    """
    global _patched
    if _patched:
        return
    _patched = True

    decode = XTermParser._sequence_to_key_events

    def _sequence_to_key_events(self, sequence: str, alt: bool = False):
        for key in decode(self, sequence, alt):
            if alt and key.key == "enter":
                # No character: a modified key inserts nothing, matching what
                # the Kitty protocol path yields for alt+enter.
                yield events.Key("alt+enter", None)
            else:
                yield key

    XTermParser._sequence_to_key_events = _sequence_to_key_events
