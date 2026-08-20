"""Escape sequences, and the three string helpers built on them.

Everything that draws goes through ``pad``: every line is built plain, padded
to an exact width, and only then wrapped in SGR. Keeping that order in one
place is what stops an escape sequence from ever being counted as width or cut
in half by a truncation.
"""

from __future__ import annotations

ESC = "\x1b"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
REVERSE = f"{ESC}[7m"
CYAN = f"{ESC}[38;5;44m"
GREEN = f"{ESC}[38;5;71m"
YELLOW = f"{ESC}[38;5;179m"
RED = f"{ESC}[38;5;167m"
BLUE = f"{ESC}[38;5;68m"


def pad(text: str, width: int) -> str:
    """Exactly ``width`` visible characters — truncated with an ellipsis, or
    padded out.

    Every line is built plain and padded *before* any SGR is wrapped around it,
    so a highlight covers the full row and no escape sequence is ever cut in
    half by the truncation.
    """
    if width <= 0:
        return ""
    if len(text) > width:
        return text[: width - 1] + "…" if width > 1 else text[:width]
    return text + " " * (width - len(text))


def rule(label: str, width: int, right: str = "") -> str:
    left = f"── {label} "
    tail = f"{right} ──" if right else "──"
    gap = max(1, width - len(left) - len(tail))
    return pad(f"{left}{'─' * gap}{tail}", width)


def reverse(text: str, ranges: list[tuple[int, int]]) -> str:
    """``text`` with those column ranges highlighted, and nothing else moved."""
    spans = sorted((a, b) for a, b in ranges if b > a)
    if not spans:
        return text
    out: list[str] = []
    at = 0
    for start, end in spans:
        start, end = max(start, at), min(end, len(text))
        if end <= start:
            continue
        out.append(text[at:start])
        out.append(REVERSE + text[start:end] + RESET)
        at = end
    out.append(text[at:])
    return "".join(out)


def footer_line(
    pairs: list[tuple[str, str]], width: int, note: str = "", style: str = YELLOW
) -> str:
    """As many ``key label`` pairs as fit, keys bright and labels dim.

    Truncation is by whole pairs rather than by characters: half a hint is
    worse than one hint fewer, and ``?`` opens the full list anyway — which is
    the honest answer to "show *all* the hotkeys" on an 80-column terminal.
    """
    plain: list[str] = []
    styled: list[str] = []
    used = 1
    if note:
        used += len(note) + 2
    for key, label in pairs:
        piece = f"{key} {label}"
        extra = len(piece) + (2 if plain else 0)
        if used + extra > width - 1:
            break
        plain.append(piece)
        styled.append(f"{CYAN}{key}{RESET} {DIM}{label}{RESET}")
        used += extra
    head = f"{style}{note}{RESET}  " if note else ""
    return " " + head + "  ".join(styled) + " " * max(0, width - used)
