"""The screens that draw over the rows, one module each.

An overlay is an independent ``render``/``handle`` pair: it is handed the width
and height it may use, it answers keys, and returning False from ``handle`` is
how it closes. ``app`` reads whatever it decided out of the object afterwards,
so nothing here reaches back into the rows.

What they share is in `base`: a titled frame, a list with a cursor, an editor
that asks before discarding, a one-line prompt, and the yes/no a screen asks
about its own unsaved state. A screen that needs a screen of its own — the
profiles list needs three — sets ``child`` and `RowUI` stacks it.
"""

from __future__ import annotations

from hpca.ui.overlays.backendform import BackendFormOverlay
from hpca.ui.overlays.backends import backend_head, backend_item, row_key
from hpca.ui.overlays.base import (
    BACK_KEYS,
    KEEP_CHANGES,
    EditorOverlay,
    ListOverlay,
    Overlay,
    PromptOverlay,
    framed,
    keyed,
    options,
)
from hpca.ui.overlays.config import ConfigOverlay
from hpca.ui.overlays.help import HelpOverlay
from hpca.ui.overlays.inspect import InspectOverlay
from hpca.ui.overlays.llm import LlmOverlay
from hpca.ui.overlays.memory import MemoryReviewOverlay
from hpca.ui.overlays.newsession import (
    BACKEND,
    PROFILE,
    NewSessionOverlay,
    choice,
)
from hpca.ui.overlays.profiles import (
    DEFAULT_REFUSAL,
    NEW_PROFILE,
    ProfilesOverlay,
    profile_rows,
)
from hpca.ui.overlays.queued import UNQUEUE, QueuedOverlay
from hpca.ui.overlays.rename import EMPTY_REFUSAL, RenameOverlay
from hpca.ui.overlays.rewind import (
    COPY,
    FORK,
    PREVIEW_CHARS,
    PREVIEW_LINES,
    ROLLBACK,
    ChoiceDialog,
    RewindOverlay,
    quoted,
)
from hpca.ui.overlays.skillnew import (
    EMPTY_LIST,
    REMOVE_QUESTION,
    SkillCreatorOverlay,
    SkillRemoveOverlay,
    skill_file,
)
from hpca.ui.overlays.skills import SkillsOverlay
from hpca.ui.overlays.switchllm import SwitchLlmOverlay
from hpca.ui.overlays.textedit import ARCHIVE, MEMORIES, SKILL, TextEditOverlay
from hpca.ui.overlays.thinking import NO_SESSION, ThinkingOverlay

__all__ = [
    "ARCHIVE",
    "BACKEND",
    "BACK_KEYS",
    "COPY",
    "DEFAULT_REFUSAL",
    "EMPTY_LIST",
    "EMPTY_REFUSAL",
    "FORK",
    "KEEP_CHANGES",
    "MEMORIES",
    "NEW_PROFILE",
    "NO_SESSION",
    "PREVIEW_CHARS",
    "PREVIEW_LINES",
    "PROFILE",
    "REMOVE_QUESTION",
    "ROLLBACK",
    "SKILL",
    "UNQUEUE",
    "BackendFormOverlay",
    "ChoiceDialog",
    "ConfigOverlay",
    "EditorOverlay",
    "HelpOverlay",
    "InspectOverlay",
    "ListOverlay",
    "LlmOverlay",
    "MemoryReviewOverlay",
    "NewSessionOverlay",
    "Overlay",
    "ProfilesOverlay",
    "PromptOverlay",
    "QueuedOverlay",
    "RenameOverlay",
    "RewindOverlay",
    "SkillCreatorOverlay",
    "SkillRemoveOverlay",
    "SkillsOverlay",
    "SwitchLlmOverlay",
    "TextEditOverlay",
    "ThinkingOverlay",
    "backend_head",
    "backend_item",
    "choice",
    "framed",
    "keyed",
    "options",
    "profile_rows",
    "quoted",
    "row_key",
    "skill_file",
]
