"""One read off the wire into a list of key names, plus what was held back.

A read is not a key boundary. An escape sequence can straddle two of them, and
a bracketed paste routinely spans dozens, so ``decode`` returns the tail it
could not name yet along with the names it could, and the read loop hands that
tail back at the front of the next read.
"""

from __future__ import annotations

from hpca.ui.ansi import ESC

# Bracketed paste: the terminal wraps anything pasted in these two markers, so
# a pasted newline can be told from a pressed Return. screen.py asks for it
# with ?2004h.
PASTE_START = f"{ESC}[200~"
PASTE_END = f"{ESC}[201~"

# A paste is one key whose name carries its payload, so that the whole block
# travels through the same ``handle(key)`` path as everything else rather than
# needing a second channel through four layers.
PASTE = "paste:"


def is_paste(key: str) -> bool:
    return key.startswith(PASTE)


def paste_text(key: str) -> str:
    return key[len(PASTE) :]


# Plain Return sends, so a newline in the message box needs a modifier — and a
# terminal has three different ways of reporting one on a key with no room for
# a modifier byte. The table below resolves all three onto these names; all
# three have to work, because which one a given terminal sends is not the
# user's choice.
NEWLINE_KEYS = ("alt-enter", "shift-enter", "ctrl-j")

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
    # Deliberately asymmetric, and it is not a mistake to be tidied up: alt+←/→
    # fold onto ctrl-left/ctrl-right because on macOS the word-motion key *is*
    # alt, so both spellings must mean word motion. alt+↑/↓ keep their own
    # names because there they mean something else entirely — reordering the
    # watcher under the cursor — and folding them onto ctrl-up/ctrl-down would
    # make moving a watcher jump between rows instead.
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
    # ---- newline in the message box, which plain Return cannot mean.
    # Return is one byte with no room for a modifier, so a terminal reports
    # shift+enter in one of three ways and the box has to accept all of them.
    # (1) The escape-prefix workaround — a terminal binding that sends ESC CR,
    # which is what Alacritty's ESC-CR key binding and Claude Code's own
    # terminal setup write. This is the one the Textual UI needed a parser
    # monkeypatch for (hpca/tui/termkeys.py); measuring escapes ourselves makes
    # it a table entry.
    f"{ESC}\r": "alt-enter",
    f"{ESC}\n": "alt-enter",
    # (2) The kitty keyboard protocol, which reports Return as keycode 13 with
    # a modifier. Not requested by screen.py, so it only arrives from a
    # terminal that has it on for its own reasons — harmless to understand.
    f"{ESC}[13;2u": "shift-enter",
    f"{ESC}[13;3u": "alt-enter",
    f"{ESC}[13;5u": "ctrl-j",  # ctrl+enter: the same intent as ^J
    # (3) xterm's modifyOtherKeys, the same keycode in the older CSI ~ shape.
    f"{ESC}[27;2;13~": "shift-enter",
    f"{ESC}[27;3;13~": "alt-enter",
    f"{ESC}[27;5;13~": "ctrl-j",
    "\t": "tab",
    "\r": "enter",
    # ^J is LF. Return in raw mode is CR — ICRNL is off — so a bare LF is the
    # control key rather than the Return key, and it is the third way to ask
    # for a newline in the box.
    "\n": "ctrl-j",
    "\x7f": "backspace",
    "\x08": "ctrl-backspace",  # what most terminals send for ctrl+backspace
    "\x17": "ctrl-backspace",  # and ctrl-w, for the terminals that do not
    "\x0c": "ctrl-l",  # switch this session's LLM (§5)
    "\x13": "ctrl-s",
    "\x15": "ctrl-u",
    "\x03": "quit",
    "\x04": "quit",
    ESC: "esc",
}


def escape_span(text: str, at: int) -> int | None:
    """How many characters the escape sequence at ``at`` occupies, or None if
    it has not all arrived yet.

    Measured by shape rather than looked up, so a sequence this table does not
    know — a shift+alt+arrow from some other terminal, a mouse report — is
    still consumed whole. Matching by table alone left the leading ESC to be
    read as the esc key and the rest as typed letters, which is why shift+arrow
    used to throw you out of the message box and into the chat.

    None is the other half of that: a CSI with no final byte yet, an SS3 with
    two of its three characters, or a bare trailing ESC are *not* short
    sequences, they are the beginnings of long ones, and naming them from what
    has arrived so far is how half an arrow key used to arm the stop gesture.
    """
    n = len(text)
    if at + 1 >= n:
        return None  # a bare trailing ESC: an escape key, or the start of one
    nxt = text[at + 1]
    if nxt == ESC:
        return 1  # the second ESC begins its own sequence, and may be a key
    if nxt == "[":
        i = at + 2
        while i < n and (text[i].isdigit() or text[i] in ";?<>"):
            i += 1
        return None if i >= n else i + 1 - at
    if nxt == "O":
        return None if at + 2 >= n else 3
    return 2  # alt-<char>


def escape_len(text: str, at: int) -> int:
    """``escape_span``, resolved: an unfinished sequence is however much of it
    is actually here. Only correct once nothing more is coming, which is why
    ``decode`` consults ``escape_span`` first and only falls back to this."""
    size = escape_span(text, at)
    return len(text) - at if size is None else size


def clean_paste(text: str) -> str:
    """A pasted payload as text this UI can draw and measure.

    Line endings are normalised because a pasted CR is a line break in the
    source, not a send. Tabs become spaces because how wide a tab is is the
    terminal's decision, and a row whose width the terminal decides is a row
    that cannot be padded to an exact number of cells. Everything else in the
    control range goes: it would be counted as zero cells and drawn as
    something, which is the one combination that corrupts a differential
    repaint.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    return "".join(ch for ch in text if ch == "\n" or (ch >= " " and ch != "\x7f"))


def decode(data: bytes | str, *, final: bool = False) -> tuple[list[str], str]:
    """One read into key names, plus the tail that is not decodable yet.

    The caller keeps the tail and puts it in front of the next read. ``final``
    says the read timed out with nothing following, so what is held is all
    there will ever be: a lone ESC becomes the escape key, and an escape
    sequence that never finished is dropped rather than typed.

    That timeout is the whole answer to the one ambiguity here. A pressed
    escape key and the first byte of an arrow key are the same byte; what
    separates them is that the rest of the arrow is already in the terminal's
    buffer and arrives within microseconds, while a person's next keystroke
    does not. So neither is named on arrival — the byte is held, and the poll
    in run.py names it when nothing follows it inside the escape timeout. The
    stop gesture keeps working; half an arrow key never arms it.

    An unfinished *paste* is held even when ``final`` is set: a paste that
    stalls mid-transfer has to stay one unit, and flushing it as keystrokes is
    precisely the bug bracketing exists to prevent — one of those keystrokes is
    the newline that would send half the message.
    """
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    keys: list[str] = []
    at, n = 0, len(text)
    while at < n:
        if text[at] != ESC:
            keys.append(KEYS.get(text[at], text[at]))
            at += 1
            continue
        if text.startswith(PASTE_START, at):
            end = text.find(PASTE_END, at + len(PASTE_START))
            if end < 0:
                return keys, text[at:]
            keys.append(PASTE + clean_paste(text[at + len(PASTE_START) : end]))
            at = end + len(PASTE_END)
            continue
        size = escape_span(text, at)
        if size is None:
            if not final:
                return keys, text[at:]
            size = n - at
        chunk = text[at : at + size]
        if chunk in KEYS:
            keys.append(KEYS[chunk])
        elif size == 1:
            keys.append("esc")
        at += size
    return keys, ""
