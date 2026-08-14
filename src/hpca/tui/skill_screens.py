"""Modals for the skill slash commands (§5.1): create one, or pick one to
remove. Skills are per-profile procedure files; the app owns writing/deleting.

``SkillCreatorScreen`` collects a new skill in one form — name, description,
level, then body, top to bottom — and dismisses a ``(Skill, level)`` pair (or
``None``). The level chooses where the skill is stored: global (every
profile), profile (this one), or project (this directory). The fields can be
seeded with a model-written draft (``/skill-creator <what it should do>``);
the form is the same either way, so a draft is edited and confirmed exactly
like a hand-typed skill. ``SkillPickerScreen`` lists the removable skills and
dismisses the chosen one for removal.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import (
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    Static,
    TextArea,
)

from hpca.skills import Skill, SkillLevel

# RadioButton id -> level, so the selection maps back to a stored location.
_LEVEL_BY_ID: dict[str, SkillLevel] = {
    "level-global": "global",
    "level-profile": "profile",
    "level-project": "project",
}


class SkillCreatorScreen(ModalScreen["tuple[Skill, SkillLevel] | None"]):
    """Create a skill: name, description, level, body. escape saves (asks
    first), escape on an empty form cancels.

    A single form rather than several pop-ups — the fields are filled top to
    bottom (name → description → level → body), tab moves between them.

    ``draft`` pre-fills those three fields with a model-written first draft.
    A pre-filled form is not "already saved": escape still asks, and the
    empty-form cancel only applies once the user has cleared every field.
    """

    # Save is resolved on escape ("Save skill? y/n") — no ctrl+s (reserved
    # hotkey: terminal XOFF). esc first so the footer shows it leftmost.
    BINDINGS = [Binding("escape", "close", "back", priority=True)]

    DEFAULT_CSS = """
    SkillCreatorScreen { align: center middle; }
    #skill-dialog {
        width: 76;
        height: auto;
        max-height: 90%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #skill-title { text-style: bold; color: $accent; }
    #skill-body { height: auto; max-height: 18; margin: 1 0 0 0; }
    #skill-level { height: auto; border: none; background: $surface; }
    .skill-field-label { color: $text-muted; margin: 1 0 0 0; }
    #skill-hint { color: $text-muted; margin: 1 0 0 0; }
    """

    def __init__(
        self,
        *,
        name: str = "",
        description: str = "",
        body: str = "",
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._body = body
        self._drafted = bool(name or description or body)

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-dialog"):
            title = "New skill — draft, edit it" if self._drafted else "New skill"
            yield Static(title, id="skill-title")
            yield Static("name", classes="skill-field-label")
            yield Input(
                self._name, placeholder="short-kebab-name", id="skill-name"
            )
            yield Static("description", classes="skill-field-label")
            yield Input(
                self._description,
                placeholder="one line — when should the agent use this?",
                id="skill-description",
            )
            yield Static("level (where it lives)", classes="skill-field-label")
            with RadioSet(id="skill-level"):
                yield RadioButton("global — every profile", id="level-global")
                # Default to the historical behaviour: this profile only.
                yield RadioButton(
                    "profile — this profile only", id="level-profile", value=True
                )
                yield RadioButton(
                    "project — this directory only", id="level-project"
                )
            yield Static("body (the procedure)", classes="skill-field-label")
            yield TextArea(self._body, id="skill-body")
            yield Static(
                "(esc) save & close (asks first) · (tab) next field",
                id="skill-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#skill-name", Input).focus()

    def _selected_level(self) -> SkillLevel:
        pressed = self.query_one("#skill-level", RadioSet).pressed_button
        if pressed is None:
            return "profile"
        return _LEVEL_BY_ID.get(pressed.id or "", "profile")

    def action_close(self) -> None:
        name = self.query_one("#skill-name", Input).value.strip()
        description = self.query_one("#skill-description", Input).value.strip()
        body = self.query_one("#skill-body", TextArea).text.strip()
        if not name and not description and not body:
            self.dismiss(None)  # nothing entered; nothing to save
            return
        if not name:
            self.notify("A skill needs a name.", severity="warning")
            self.query_one("#skill-name", Input).focus()
            return
        if not body:
            self.notify("A skill needs a body (the procedure).", severity="warning")
            self.query_one("#skill-body", TextArea).focus()
            return
        from hpca.tui.confirm_screen import ConfirmScreen

        skill = Skill(name=name, description=description, triggers=[], body=body)
        level = self._selected_level()

        def verdict(keep: bool | None) -> None:
            self.dismiss((skill, level) if keep else None)

        self.app.push_screen(ConfirmScreen(f"Save skill “{name}”?"), verdict)


class SkillPickerScreen(ModalScreen[Skill | None]):
    """Pick one of a profile's own skills to remove. Dismisses the chosen
    ``Skill`` or ``None``."""

    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    SkillPickerScreen { align: center middle; }
    #skill-picker-dialog {
        width: 72;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #skill-picker-title { height: 1; text-style: bold; }
    #skill-picker-hint { color: $text-muted; }
    """

    def __init__(self, skills: list[Skill]) -> None:
        super().__init__()
        self._skills = skills

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-picker-dialog"):
            yield Static("Remove which skill?", id="skill-picker-title")
            with VerticalScroll():
                yield ListView(id="skill-picker-list")
            yield Static("(esc) cancel · (enter) remove", id="skill-picker-hint")

    def on_mount(self) -> None:
        picker = self.query_one("#skill-picker-list", ListView)
        for index, skill in enumerate(self._skills):
            label = skill.name
            if skill.description:
                label = f"{skill.name} — {skill.description}"
            item = ListItem(Label(Content(label)))
            item.data_index = index
            picker.append(item)
        picker.index = 0
        picker.focus()

    @on(ListView.Selected, "#skill-picker-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        index = getattr(event.item, "data_index", None)
        if index is None:
            return
        self.dismiss(self._skills[index])

    def action_cancel(self) -> None:
        self.dismiss(None)
