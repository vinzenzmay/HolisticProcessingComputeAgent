"""One read off the wire into a list of key names."""

from __future__ import annotations

from hpca.ui.ansi import ESC

KEYS = {
    f"{ESC}[A": "up",
    f"{ESC}OA": "up",
    f"{ESC}[B": "down",
    f"{ESC}OB": "down",
    f"{ESC}[C": "right",
    f"{ESC}OC": "right",
    f"{ESC}[D": "left",
    f"{ESC}OD": "left",
    f"{ESC}[1;5A": "ctrl-up",
    f"{ESC}[1;5B": "ctrl-down",
    f"{ESC}[1;5C": "ctrl-right",
    f"{ESC}[1;5D": "ctrl-left",
    f"{ESC}[1;3A": "alt-up",
    f"{ESC}[1;3B": "alt-down",
    f"{ESC}[1;3C": "ctrl-right",
    f"{ESC}[1;3D": "ctrl-left",
    f"{ESC}b": "ctrl-left",
    f"{ESC}f": "ctrl-right",
    # Shift is selection everywhere: plain, by word, and to the ends.
    f"{ESC}[1;2A": "shift-up",
    f"{ESC}[1;2B": "shift-down",
    f"{ESC}[1;2C": "shift-right",
    f"{ESC}[1;2D": "shift-left",
    f"{ESC}[1;6C": "shift-ctrl-right",
    f"{ESC}[1;6D": "shift-ctrl-left",
    f"{ESC}[1;4C": "shift-ctrl-right",
    f"{ESC}[1;4D": "shift-ctrl-left",
    f"{ESC}[1;2H": "shift-home",
    f"{ESC}[1;2F": "shift-end",
    f"{ESC}[1;2~": "shift-home",
    f"{ESC}[4;2~": "shift-end",
    f"{ESC}[5~": "pgup",
    f"{ESC}[6~": "pgdn",
    f"{ESC}[H": "home",
    f"{ESC}[F": "end",
    f"{ESC}OH": "home",
    f"{ESC}OF": "end",
    f"{ESC}[1~": "home",
    f"{ESC}[4~": "end",
    f"{ESC}[3~": "delete",
    f"{ESC}[3;5~": "ctrl-delete",
    f"{ESC}[3;3~": "ctrl-delete",
    f"{ESC}\x7f": "ctrl-backspace",
    f"{ESC}[Z": "shift-tab",
    f"{ESC}\r": "alt-enter",
    ESC: "esc",
    "\t": "tab",
    "\r": "enter",
    "\n": "enter",
    "\x7f": "backspace",
    "\x08": "ctrl-backspace",  # what most terminals send for ctrl+backspace
    "\x17": "ctrl-backspace",  # and ctrl-w, for the terminals that do not
    "\x13": "ctrl-s",
    "\x15": "ctrl-u",
    "\x03": "quit",
    "\x04": "quit",
}


def escape_len(text: str, at: int) -> int:
    """How many characters the escape sequence starting at ``at`` occupies.

    Measured by shape rather than looked up, so a sequence this table does not
    know — a shift+alt+arrow from some other terminal, a bracketed paste
    marker — is still consumed whole. Matching by table alone left the leading
    ESC to be read as the esc key and the rest as typed letters, which is why
    shift+arrow used to throw you out of the message box and into the chat.
    """
    n = len(text)
    if at + 1 >= n:
        return 1
    nxt = text[at + 1]
    if nxt == "[":
        i = at + 2
        while i < n and (text[i].isdigit() or text[i] in ";?<"):
            i += 1
        return min(n, i + 1) - at
    if nxt == "O":
        return min(n, at + 3) - at
    if nxt == ESC:
        return 1
    return 2  # alt-<char>


def decode(data: bytes) -> list[str]:
    """One read into a list of key names. Unknown escapes are dropped."""
    text = data.decode("utf-8", "replace")
    keys: list[str] = []
    at = 0
    while at < len(text):
        if text[at] == ESC:
            size = escape_len(text, at)
            chunk = text[at : at + size]
            if chunk in KEYS:
                keys.append(KEYS[chunk])
            elif size == 1:
                keys.append("esc")
            at += size
            continue
        keys.append(KEYS.get(text[at], text[at]))
        at += 1
    return keys
