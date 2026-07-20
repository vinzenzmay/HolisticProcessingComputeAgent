"""Modals for the skill slash commands (§5.1): create one, or pick one to
remove. Skills are per-profile procedure files; the app owns writing/deleting.

``SkillCreatorScreen`` collects a new skill in one form — name, description,
then body, top to bottom — and dismisses a built ``Skill`` (or ``None``).
``SkillPickerScreen`` lists a profile's own skills and dismisses the chosen
one for removal.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static, TextArea

from hpca.skills import Skill


class SkillCreatorScreen(ModalScreen[Skill | None]):
    """Create a skill: name, description, body. ctrl+s saves, escape cancels.

    A single form rather than three pop-ups — the fields are filled top to
    bottom (name → description → body), tab moves between them.
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
    .skill-field-label { color: $text-muted; margin: 1 0 0 0; }
    #skill-hint { color: $text-muted; margin: 1 0 0 0; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-dialog"):
            yield Static("New skill", id="skill-title")
            yield Static("name", classes="skill-field-label")
            yield Input(placeholder="short-kebab-name", id="skill-name")
            yield Static("description", classes="skill-field-label")
            yield Input(
                placeholder="one line — when should the agent use this?",
                id="skill-description",
            )
            yield Static("body (the procedure)", classes="skill-field-label")
            yield TextArea(id="skill-body")
            yield Static(
                "(esc) save & close (asks first) · (tab) next field",
                id="skill-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#skill-name", Input).focus()

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

        def verdict(keep: bool | None) -> None:
            self.dismiss(skill if keep else None)

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
