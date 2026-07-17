"""Profiles & learnings screen (a): manage profiles and their tier-1 memories.

A list like the sessions column: ↑/↓ move, enter opens the highlighted
profile's memories in a plain-text editor (copy/paste works), escape asks
whether to keep the edits. Enter on the "(new profile)" row names and creates
one; (d) deletes — never the default, and never a profile a session is
actively using (a turn in flight, or live sub-processes). Sessions on a
deleted profile fall back to the default so nothing points at a gone file.

``ProfilePickerScreen`` is the modal cousin: every new session starts by
choosing its profile there (or creating one on the spot).

The app owns the guards and the reassignment; this screen calls back into it
so the checks live in one place.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static, TextArea

from hpca.profiles import DEFAULT_PROFILE, Profile

NEW_PROFILE_LABEL = "(new profile)"


class ProfilesList(ListView):
    """The profile list; enter opens a profile's memories — or, on the
    "(new profile)" row, creates one. (d) only on a removable profile."""

    BINDINGS = [
        Binding("enter", "select_cursor", "edit / create", show=True),
        Binding("d", "delete_profile", "delete profile", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        name = getattr(self.highlighted_child, "data_profile", None)
        if action == "delete_profile":
            # "(new profile)" is not a profile; the default is the fallback
            return name is not None and name != DEFAULT_PROFILE
        return True

    def action_delete_profile(self) -> None:
        self.screen.delete_selected()


class MemoryEditorScreen(ModalScreen[str | None]):
    """A profile's memories as raw editable text; esc asks to keep changes."""

    BINDINGS = [Binding("escape", "close", "back", priority=True)]

    DEFAULT_CSS = """
    MemoryEditorScreen { align: center middle; }
    #memory-dialog {
        width: 88;
        height: 84%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #memory-title { height: 1; text-style: bold; }
    #memory-editor { height: 1fr; }
    #memory-hint { color: $text-muted; }
    """

    def __init__(self, profile_name: str, text: str) -> None:
        super().__init__()
        self._profile_name = profile_name
        self._original = text

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-dialog"):
            yield Static(
                f"Memories — {self._profile_name}", id="memory-title"
            )
            yield TextArea(self._original, id="memory-editor")
            yield Static("esc — back (asks to keep changes)", id="memory-hint")

    def on_mount(self) -> None:
        self.query_one("#memory-editor", TextArea).focus()

    def action_close(self) -> None:
        text = self.query_one("#memory-editor", TextArea).text
        if text == self._original:
            self.dismiss(None)  # nothing changed; no need to ask
            return
        from hpca.tui.confirm_screen import ConfirmScreen

        def verdict(keep: bool | None) -> None:
            self.dismiss(text if keep else None)

        self.app.push_screen(ConfirmScreen("Keep changes?"), verdict)


class ProfilesScreen(Screen):
    BINDINGS = [Binding("escape", "close", "back", priority=True)]

    DEFAULT_CSS = """
    ProfilesScreen #profiles-panel {
        border: solid $panel;
        height: 1fr;
    }
    ProfilesScreen #profiles-panel:focus-within { border: heavy $accent; }
    .profiles-title {
        height: 1;
        text-style: bold;
        text-align: center;
        background: $boost;
    }
    #profiles-status { height: auto; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Profiles & learnings", id="profiles-header", classes="profiles-title"
        )
        with Vertical(id="profiles-panel"):
            yield Static("Profiles", classes="profiles-title")
            yield ProfilesList(id="profiles-list")
        yield Static("", id="profiles-status")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_profiles()
        self.query_one("#profiles-list", ListView).focus()

    async def refresh_profiles(self) -> None:
        profiles_list = self.query_one("#profiles-list", ListView)
        previous = profiles_list.index
        await profiles_list.clear()
        new_item = ListItem(Label(NEW_PROFILE_LABEL))
        new_item.data_profile = None
        items = [new_item]
        for name in Profile.list_profiles():
            profile = Profile.load(name)
            tier1 = sum(1 for m in profile.memories if m.tier == 1)
            star = " ★" if name == DEFAULT_PROFILE else ""  # the default profile
            label = f"{name}{star}  ·  {tier1} tier-1 " + (
                "memory" if tier1 == 1 else "memories"
            )
            item = ListItem(Label(Content(label)))
            item.data_profile = name
            items.append(item)
        profiles_list.extend(items)
        profiles_list.index = min(previous or 1, len(items) - 1)

    # ---------------------------------------------------------------- actions

    def action_close(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#profiles-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "data_profile", None)
        if name is None:
            self.add_profile()
        else:
            self.edit_memories(name)

    def edit_memories(self, name: str) -> None:
        profile = Profile.load(name)

        def apply(edited: str | None) -> None:
            if edited is not None:
                self.app.save_profile_memories(name, edited)
                self.run_worker(self.refresh_profiles(), group="profiles")

        self.app.push_screen(
            MemoryEditorScreen(name, profile.render()), apply
        )

    def add_profile(self) -> None:
        from hpca.tui.rename_screen import RenameScreen

        def apply(name: str | None) -> None:
            if not name:
                return
            error = self.app.create_profile(name)
            if error:
                self.notify(error, severity="error")
            else:
                self.run_worker(self.refresh_profiles(), group="profiles")

        self.app.push_screen(RenameScreen("", label="New profile name"), apply)

    def delete_selected(self) -> None:
        highlighted = self.query_one("#profiles-list", ListView).highlighted_child
        name = getattr(highlighted, "data_profile", None)
        if name is None or name == DEFAULT_PROFILE:
            return
        blocker = self.app.profile_delete_blocker(name)
        if blocker is not None:
            self.notify(blocker, severity="warning")
            return
        from hpca.tui.confirm_screen import ConfirmScreen

        def verdict(confirmed: bool | None) -> None:
            if confirmed:
                self.app.delete_profile(name)
                self.run_worker(self.refresh_profiles(), group="profiles")

        self.app.push_screen(
            ConfirmScreen(
                f"Delete profile “{name}”?\n"
                "Its sessions move to the default profile."
            ),
            verdict,
        )


class ProfilePickerScreen(ModalScreen[str | None]):
    """Choose the profile a new session runs under — or create one first.

    The cursor starts on the current profile, so the common case is just
    enter; escape means no session. Choosing "(new profile)" names one, and
    a successful creation is the choice.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    ProfilePickerScreen { align: center middle; }
    #picker-dialog {
        width: 64;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #picker-title { height: 1; text-style: bold; }
    #picker-hint { color: $text-muted; }
    """

    def __init__(self, current: str = DEFAULT_PROFILE) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Static("Profile for the new session", id="picker-title")
            yield ListView(id="picker-list")
            yield Static("(enter) choose · (esc) cancel", id="picker-hint")

    def on_mount(self) -> None:
        picker = self.query_one("#picker-list", ListView)
        names = Profile.list_profiles()
        for name in names:
            star = " ★" if name == DEFAULT_PROFILE else ""
            item = ListItem(Label(Content(f"{name}{star}")))
            item.data_profile = name
            picker.append(item)
        new_item = ListItem(Label(NEW_PROFILE_LABEL))
        new_item.data_profile = None
        picker.append(new_item)
        picker.index = names.index(self._current) if self._current in names else 0
        picker.focus()

    @on(ListView.Selected, "#picker-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "data_profile", None)
        if name is not None:
            self.dismiss(name)
            return
        from hpca.tui.rename_screen import RenameScreen

        def apply(new_name: str | None) -> None:
            if not new_name:
                return  # back to the picker
            error = self.app.create_profile(new_name)
            if error:
                self.notify(error, severity="error")
            else:
                self.dismiss(new_name)

        self.app.push_screen(RenameScreen("", label="New profile name"), apply)

    def action_cancel(self) -> None:
        self.dismiss(None)
