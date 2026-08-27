"""Reading one end of a file without reading the file.

Four places in hpca want the last few lines of something a process wrote: the
answer run_bash hands back, the completion event for a background script, the
log scan behind job triage, and the window Enter opens over a watch box. All
four kept a bounded amount and every one of them but the last got there by
``path.read_text()`` — pulling a whole log into memory to keep four kilobytes
of it. On a cluster that log is routinely gigabytes of progress bars, it lives
on NFS, and the read happens on the event loop the TUI draws from, so the cost
lands as a freeze the user sees. Measured on a 104 MB file: 170ms of blocked
loop and 269 MB of resident memory, to keep 4000 characters.

So: seek, read a window, decode. Both functions are deliberately dumb about
what they return — a byte window is not a line window, the decode is lossy at
the seam, and the front of a tail is very likely half a line. Callers keep far
less than they ask for here, which is what makes that harmless; the margin
between the window and what is kept is the caller's to choose, and
:func:`read_tail` reports whether its window reached the start of the file so
a caller can tell "this is all of it" from "this is the end of it".

Errors are not caught. Each caller already had its own answer for an
unreadable log — an empty string, or an exception the model is shown — and
this module is not the place to overrule it.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_tail(path: Path | str, max_bytes: int) -> tuple[str, bool]:
    """``(text, whole)`` — the last ``max_bytes`` of ``path``, decoded.

    ``whole`` says the window reached the start of the file, i.e. the text is
    the entire file and not merely its end. That is the only thing a caller
    can learn about what came before: counting the lines it did not read means
    reading them.

    The window is taken from a size read off the open handle and a fixed
    offset, so a file still being appended to yields a consistent snapshot
    ending where the file ended when it was measured, rather than a read that
    chases a moving end.
    """
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - max(max_bytes, 0))
        handle.seek(start)
        raw = handle.read(max(max_bytes, 0))
    return raw.decode("utf-8", errors="replace"), start == 0


def read_head(path: Path | str, max_bytes: int) -> str:
    """The first ``max_bytes`` of ``path``, decoded.

    The counterpart for the thing a tail cannot see: the first error a script
    that kept running printed.
    """
    with Path(path).open("rb") as handle:
        raw = handle.read(max(max_bytes, 0))
    return raw.decode("utf-8", errors="replace")
