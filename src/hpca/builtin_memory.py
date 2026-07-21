"""Built-in tier-3 memories shipped with HPCA.

These are indexed under a reserved profile (:data:`BUILTIN_PROFILE`) and
searched alongside whatever profile is active, so *every* agent can recall
them regardless of which profile the user is on — including profiles that
existed before the memory was added. Like all tier-3 memories they are
retrieved (BM25), never injected wholesale, so they cost context only on the
turns whose wording matches — e.g. a user asking how to copy text out of the
app.

The text is written for the *agent* to relay to a user who is not expected to
know terminal internals: it names the "Ubuntu standard terminal" (GNOME
Terminal), and gives concrete, copy-pasteable setup steps.
"""

from __future__ import annotations

from hpca.profiles import Memory

BUILTIN_PROFILE = "__builtin__"


# Keyword-rich so BM25 fires on the many ways a user phrases this: copy, paste,
# clipboard, select, highlight, terminal, GNOME, Ubuntu, tmux, screen.
TERMINAL_COPY_PASTE = Memory(
    text=(
        "Terminal copy/paste help (how to get text OUT of HPCA onto the "
        "system clipboard).\n"
        "\n"
        "Why copy can fail while paste works: HPCA runs in a terminal. The "
        "Ubuntu standard terminal is GNOME Terminal, which does NOT support "
        "the OSC 52 clipboard escape — so HPCA cannot push text to the "
        "clipboard directly. Pasting still works because Ctrl+Shift+V is the "
        "terminal's own paste, which just types text into HPCA.\n"
        "\n"
        "If the user does not know which terminal they have: on standard "
        "Ubuntu/GNOME it is GNOME Terminal. The fixes below apply to it.\n"
        "\n"
        "Plain GNOME Terminal (no tmux/screen):\n"
        "- Best fix: install a clipboard helper, then HPCA's copy (Ctrl+C on a "
        "selection, or a chat entry's copy) works automatically. Wayland (the "
        "modern default): `sudo apt install wl-clipboard`. X11: "
        "`sudo apt install xclip`. Restart HPCA afterwards.\n"
        "- No-install fallback: hold SHIFT and drag to select text, then press "
        "Ctrl+Shift+C. Holding Shift bypasses HPCA's mouse capture and lets "
        "the terminal select natively. Paste is Ctrl+Shift+V.\n"
        "\n"
        "GNOME Terminal + tmux:\n"
        "- Install wl-clipboard or xclip as above (locally, this is the "
        "reliable path). To also let OSC 52 pass through tmux to terminals "
        "that support it, add to ~/.tmux.conf: `set -g allow-passthrough on` "
        "and `set -g set-clipboard on`, then reload tmux.\n"
        "- tmux also has its own paste buffer; HPCA writes to it, and "
        "Ctrl+b ] pastes it within tmux.\n"
        "- SHIFT+drag then Ctrl+Shift+C still works (it selects across the "
        "whole terminal window, ignoring tmux panes).\n"
        "\n"
        "GNOME Terminal + GNU screen:\n"
        "- Install wl-clipboard or xclip as above; locally that is the "
        "reliable path. HPCA also writes screen's paste register (paste with "
        "Ctrl+a ]).\n"
        "- SHIFT+drag then Ctrl+Shift+C works as the terminal-native fallback.\n"
        "\n"
        "Over SSH: install nothing on the remote — a local helper would copy "
        "to the remote machine. HPCA sends OSC 52 back to the user's own "
        "terminal instead; this works if that terminal supports OSC 52 (most "
        "do; GNOME Terminal does not, so SHIFT+drag is the fallback there).\n"
        "\n"
        "If nothing reaches the clipboard, HPCA writes the copied text to "
        "clipboard.txt in its app directory and says so in the notification.\n"
        "\n"
        "Forcing a method in settings (clipboard section): set mode = "
        '"command" and command = "wl-copy" (Wayland) or '
        '"xclip -selection clipboard -i" (X11) to always route copy through '
        "that tool."
    ),
    tier=3,
    kind="reference",
)


BUILTIN_MEMORIES: list[Memory] = [TERMINAL_COPY_PASTE]
