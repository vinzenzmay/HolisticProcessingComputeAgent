"""Profiles & learnings screen (a): manage profiles and their memories.

Profiles are entirely the user's: there are no built-in ones beyond the
default, and each accumulates its own memories and skills over time.

A list like the sessions column: ↑/↓ move, enter opens the highlighted
profile's memories in a plain-text editor (copy/paste works), escape asks
whether to keep the edits. Enter on the "(new profile)" row names and creates
a blank one; (c) copies the highlighted profile under a new name — the copy
starts from everything the original has learned and diverges from there,
which is how a general base profile becomes several specialised ones; (d)
deletes — never the default, and never a profile a session is actively using
(a turn in flight, or live sub-processes). Sessions on a deleted profile fall
back to the default so nothing points at a gone file.

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

from hpca.curator import archive_path
from hpca.profiles import DEFAULT_PROFILE, Profile
from hpca.skills import load_own_skills, skill_path

NEW_PROFILE_LABEL = "(new profile)"


class ProfilesList(ListView):
    """The profile list; enter opens a profile's memories — or, on the
    "(new profile)" row, creates one. (c) copies the highlighted profile,
    (d) deletes it — neither applies to "(new profile)", and the default
    cannot be deleted."""

    BINDINGS = [
        # esc first so the footer shows "back" leftmost (§ esc/quit ordering);
        # the screen's own escape binding handles it, this orders the display.
        Binding("escape", "close", "back", show=True),
        Binding("enter", "select_cursor", "edit memories", show=True),
        Binding("s", "edit_skills", "skills", show=True),
        Binding("r", "edit_archive", "archive", show=True),
        Binding("c", "copy_profile", "copy profile", show=True),
        Binding("d", "delete_profile", "delete profile", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        name = getattr(self.highlighted_child, "data_profile", None)
        if action == "delete_profile":
            # "(new profile)" is not a profile; the default is the fallback
            return name is not None and name != DEFAULT_PROFILE
        if action in ("copy_profile", "edit_skills", "edit_archive"):
            return name is not None  # not on the "(new profile)" row
        return True

    def action_copy_profile(self) -> None:
        self.screen.copy_selected()

    def action_delete_profile(self) -> None:
        self.screen.delete_selected()

    def action_edit_skills(self) -> None:
        self.screen.edit_skills_selected()

    def action_edit_archive(self) -> None:
        self.screen.edit_archive_selected()


class MemoryEditorScreen(ModalScreen[str | None]):
    """Raw editable text — a profile's memories, its archive, or a skill body;
    esc asks to keep changes. Content-agnostic: it returns the edited text and
    the caller decides where it lands."""

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

    def __init__(self, profile_name: str, text: str, *, title: str = "") -> None:
        super().__init__()
        self._profile_name = profile_name
        self._original = text
        self._title = title or f"Memories — {profile_name}"

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-dialog"):
            yield Static(self._title, id="memory-title")
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


class ProfileSkillsScreen(ModalScreen[None]):
    """A profile's own skills: enter edits a skill's raw file, (d) deletes one.

    The profile's own copy only — shared and project skills are edited where
    they live, not from one profile's screen."""

    BINDINGS = [
        Binding("escape", "close", "back", priority=True),
        Binding("d", "delete_skill", "delete", show=True),
    ]

    DEFAULT_CSS = """
    ProfileSkillsScreen { align: center middle; }
    #pskills-dialog {
        width: 72;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #pskills-title { height: 1; text-style: bold; }
    #pskills-hint { color: $text-muted; }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name

    def compose(self) -> ComposeResult:
        with Vertical(id="pskills-dialog"):
            yield Static(f"Skills — {self._profile_name}", id="pskills-title")
            yield ListView(id="pskills-list")
            yield Static("(enter) edit · (d) delete · (esc) back", id="pskills-hint")

    async def on_mount(self) -> None:
        await self._refresh()
        self.query_one("#pskills-list", ListView).focus()

    async def _refresh(self) -> None:
        listview = self.query_one("#pskills-list", ListView)
        await listview.clear()
        skills = load_own_skills(self._profile_name)
        if not skills:
            empty = ListItem(Label("(no skills for this profile)"))
            empty.data_skill = None
            listview.append(empty)
            return
        for skill in skills:
            item = ListItem(Label(Content(f"{skill.name}  ·  {skill.description}")))
            item.data_skill = skill.name
            listview.append(item)

    def action_close(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#pskills-list")
    def _on_selected(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "data_skill", None)
        if name is None:
            return
        path = skill_path(name, self._profile_name)
        text = path.read_text() if path.exists() else ""

        def apply(edited: str | None) -> None:
            if edited is not None:
                self.app.save_skill_file(self._profile_name, name, edited)
                self.run_worker(self._refresh(), group="pskills")

        self.app.push_screen(
            MemoryEditorScreen(
                self._profile_name,
                text,
                title=f"Skill “{name}” — {self._profile_name}",
            ),
            apply,
        )

    def action_delete_skill(self) -> None:
        listview = self.query_one("#pskills-list", ListView)
        name = getattr(listview.highlighted_child, "data_skill", None)
        if name is None:
            return
        from hpca.tui.confirm_screen import ConfirmScreen

        def verdict(confirmed: bool | None) -> None:
            if confirmed:
                self.app.delete_profile_skill(self._profile_name, name)
                self.run_worker(self._refresh(), group="pskills")

        self.app.push_screen(ConfirmScreen(f"Delete skill “{name}”?"), verdict)


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
            total = len(profile.memories)
            star = " ★" if name == DEFAULT_PROFILE else ""  # the default profile
            label = f"{name}{star}  ·  {total} " + (
                "memory" if total == 1 else "memories"
            )
            if profile.copied_from:
                # Which profiles share a base is the thing you need to know
                # when they start disagreeing with each other.
                label += f"  ·  copied from {profile.copied_from}"
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

    def _highlighted_profile(self) -> str | None:
        highlighted = self.query_one("#profiles-list", ListView).highlighted_child
        return getattr(highlighted, "data_profile", None)

    def edit_archive_selected(self) -> None:
        """The RAG archive (curator-aged entries) as raw editable text — the
        one place to see what was aged out and move a block back into ## [rag]."""
        name = self._highlighted_profile()
        if name is None:
            return
        path = archive_path(name)
        text = path.read_text() if path.exists() else ""

        def apply(edited: str | None) -> None:
            if edited is not None:
                self.app.save_profile_archive(name, edited)

        self.app.push_screen(
            MemoryEditorScreen(name, text, title=f"Archive — {name}"), apply
        )

    def edit_skills_selected(self) -> None:
        name = self._highlighted_profile()
        if name is None:
            return
        self.app.push_screen(ProfileSkillsScreen(name))

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

    def copy_selected(self) -> None:
        """Fork the highlighted profile under a new name (§ profiles).

        The workflow this exists for: work under a base profile until it
        knows the site, then copy it per specialism so each one accumulates
        its own learnings from that common starting point.
        """
        highlighted = self.query_one("#profiles-list", ListView).highlighted_child
        source = getattr(highlighted, "data_profile", None)
        if source is None:
            return
        from hpca.tui.rename_screen import RenameScreen

        def apply(name: str | None) -> None:
            if not name:
                return
            error = self.app.duplicate_profile(source, name)
            if error:
                self.notify(error, severity="error")
            else:
                self.run_worker(self.refresh_profiles(), group="profiles")

        self.app.push_screen(
            RenameScreen(f"{source} copy", label=f"Copy “{source}” to"), apply
        )

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
            yield Static("(esc) cancel · (enter) choose", id="picker-hint")

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
