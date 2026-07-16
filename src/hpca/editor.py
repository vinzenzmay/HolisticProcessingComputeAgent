"""External editor resolution (§6.4): settings → $VISUAL → $EDITOR → nano."""

from __future__ import annotations

import shlex
from typing import Mapping


def resolve_editor(
    settings_editor: str | None, env: Mapping[str, str]
) -> list[str]:
    for candidate in (settings_editor, env.get("VISUAL"), env.get("EDITOR")):
        if candidate and candidate.strip():
            return shlex.split(candidate)
    return ["nano"]
