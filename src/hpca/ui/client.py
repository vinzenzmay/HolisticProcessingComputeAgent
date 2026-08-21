"""The protocol client: events become state, intents become commands.

The only module in `hpca.ui` that imports `hpca.protocol`, and the reason
`app.py` can be tested by comparing strings (specs-ui-replacement.md §3.1). It
draws nothing and it reads no terminal: one direction takes a frame off the
wire and mutates `ui/state.py` objects, the other takes an intent from a
keypress and puts a command on the wire.

Two properties from §3.2 are the whole point of the file and are asserted by
`tests/test_ui_client.py`:

1. **Events are addressed, and most of them are not for the visible session.**
   Every handler starts by asking which session the event names, and updates
   that session whether or not it is on screen. The sidebar marker is the only
   thing a background session may change about the frame.
2. **The chat is append-only between resets.** `chat.reset` is the only path
   that replaces rows; `chat.append` is the only path that adds one; and
   `chat.update` revises a row it can already name. Nothing here rebuilds a
   transcript from a snapshot, which is what the Textual app did every turn and
   what its queued-message bugs were made of.

Anything not in §3.2's table is dropped, counted in `dropped` so a test can say
so out loud rather than a frame quietly missing something.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

from hpca import protocol
from hpca.agent.modes import next_mode
from hpca.transport import Connection
from hpca.ui import state
from hpca.ui.ansi import GREEN, RED
from hpca.ui.app import RowUI
from hpca.ui.pane import Item

logger = logging.getLogger("hpca.ui.client")

# The `PanelRow.classes` the core sends, as colours. Open-ended on the wire on
# purpose (`watches.watch_class`), so a class this table has never heard of
# draws plain rather than raising.
WATCH_ACCENTS = {"watch-live": GREEN, "watch-dead": RED}

# What the box's border title is padded to, so the state columns line up.
TITLE_COLUMN = 16


def _entry(wire: protocol.Entry) -> state.ChatEntry:
    """A wire entry as the plain thing the renderer takes."""
    return state.ChatEntry(
        kind=wire.kind,
        text=wire.text,
        seq=wire.seq,
        index=wire.index,
        steps=wire.steps,
        reasoning_chars=wire.reasoning_chars,
        parts=[
            state.ChatPart(
                kind=part.kind,
                text=part.text,
                tool=part.tool,
                target=part.target,
                result=part.result,
                done=part.done,
                failed=part.failed,
            )
            for part in wire.parts
        ],
    )


def _settings_error(text: str) -> str:
    """Why this text is not JSON, in one line, or "".

    Deliberately only half the check, and the split is `protocol.SettingsSave`'s:
    whether the text is *JSON* is answerable here with the standard library and
    no idea what a setting is, which is exactly what the editor needs
    synchronously in order to refuse to close (§4.3 item 26). Whether it is a
    valid `Settings` — which fields exist, what they may hold — is the core's
    model to know, and comes back as `SettingsBody.error` after the save is
    refused. A front-end carrying a copy of the schema in order to be trusted
    with it is the thing this arrangement avoids.
    """
    import json

    try:
        json.loads(text)
    except ValueError as e:
        return f"invalid json: {e}"
    return ""


def _panel_item(row: protocol.PanelRow) -> Item:
    """One watch box, keyed by the name the core gave it.

    `panel.update` repaints the column whole, so the row's `key` is what keeps
    the cursor and the open boxes where the user left them (§3.2). ``ref`` is
    carried in ``text`` because that is what a keypress on the row sends back.
    """
    lines = row.text.split("\n") if row.text else [""]
    accent = ""
    for name in row.classes.split():
        accent = WATCH_ACCENTS.get(name, accent)
    head = f"{row.title:<{TITLE_COLUMN}}{lines[0]}" if row.title else lines[0]
    return Item(
        head=head,
        body=lines[1:],
        accent=accent,
        kind=row.kind,
        text=row.ref,
        key=row.key,
    )



class UIClient:
    """One `RowUI`, one connection, and the translation between them.

    Constructed with either a `Connection` (commands queue in `outbox` and
    `flush` puts them on the wire) or a plain `send` callable (they are handed
    over in place, which is what the `--demo` loopback core wants and what
    keeps `run.py --demo` free of an event loop it does not yet need).
    """

    def __init__(
        self,
        ui: RowUI,
        conn: Connection | None = None,
        *,
        send: Callable[[protocol.Command], None] | None = None,
    ) -> None:
        self.ui = ui
        self._conn = conn
        self._send = send
        self.outbox: list[protocol.Command] = []
        self.dropped: Counter[str] = Counter()
        self.hello: protocol.Hello | None = None
        # Whether anything has been opened yet. Not "is a session active" —
        # the sidebar's first row is active from the moment the list arrives,
        # and it is still not the conversation the user is looking at until
        # something has asked the core for it.
        self._opened = False
        # Which profile a ctrl+e is out for, or "": the editor cannot open
        # until `profile.body` arrives, and there is one terminal to hand over.
        self._editing = ""
        ui.send = self.intent
        # Asked for on the first `/`, and answered a frame later by
        # `skill.rows` — the menu is a function of what has arrived, so it
        # fills itself the moment it does.
        ui.skills_loader = self.ask_skills
        ui.edit_profile = self.edit_profile
        # Syntax only, and eagerly: it is a pure function of the text, and the
        # verdict that needs the settings model comes back as
        # `SettingsBody.error` (see `_settings_error`).
        ui.validate_settings = _settings_error

    # ------------------------------------------------------------- outbound

    def intent(self, intent: state.Intent) -> None:
        """What a keypress meant, as the command that carries it.

        `app.py` never names a command; this is the one place the two
        vocabularies meet, and it is why a `RowUI` method can be read without
        knowing the protocol exists.
        """
        session = getattr(intent, "session_id", "")
        if isinstance(intent, state.NewSession):
            # No session id in either direction: there is nothing to name yet,
            # and what comes back is a `session.created` the UI opens. The
            # backend string is passed through untouched — None rather than ""
            # when nothing was chosen, because `protocol.SessionNew` spells
            # "the core decides" as null.
            self.command(
                protocol.SessionNew(
                    profile=intent.profile, backend=intent.backend or None
                )
            )
        elif isinstance(intent, state.Rename):
            self.command(
                protocol.SessionRename(session_id=session, title=intent.title)
            )
        elif isinstance(intent, state.Retitle):
            self.command(protocol.SessionRetitle(session_id=session))
        elif isinstance(intent, state.DeleteSession):
            self.command(protocol.SessionDelete(session_id=session))
        elif isinstance(intent, state.OpenSession):
            self._opened = True
            # Two commands for one intent, and deliberately: the core has to be
            # told what is on screen every time (`session.focus` decides where a
            # background completion is delivered), but the transcript is asked
            # for once. A session already held is kept current by appends, so
            # re-opening it would cost a whole `chat.reset` and throw away the
            # cursor line and the open entries the user left in it.
            if not self.ui.session_for(session).loaded:
                self.command(protocol.SessionOpen(session_id=session))
            self.command(protocol.SessionFocus(session_id=session))
        elif isinstance(intent, state.Submit):
            self.command(
                protocol.TurnSubmit(
                    session_id=session,
                    text=intent.text,
                    # None rather than "" when no skill was named: the wire
                    # spells "no forced skill" as null, and the core looks the
                    # name up in the *session's* profile (`_skill_named`).
                    forced_skill=intent.forced_skill or None,
                )
            )
        elif isinstance(intent, state.Interrupt):
            self.command(protocol.TurnInterrupt(session_id=session))
        elif isinstance(intent, state.Decide):
            # The half-typed reason was a UI draft until this moment; only the
            # finished string crosses (specs-core-process.md §4.4).
            self.command(
                protocol.DecisionResolve(
                    session_id=session,
                    approved=intent.approved,
                    reason=intent.reason,
                )
            )
        elif isinstance(intent, state.Unqueue):
            self.command(
                protocol.TurnUnqueue(session_id=session, seq=intent.seq)
            )
        elif isinstance(intent, state.Answer):
            # By id, not by session: a `confirm.requested` need not belong to
            # a conversation at all — a triage offer comes from a poll.
            self.command(
                protocol.ConfirmResolve(id=intent.id, confirmed=intent.confirmed)
            )
        elif isinstance(intent, state.CycleMode):
            self._cycle_mode(session)
        elif isinstance(intent, state.Fork | state.Rollback):
            self._rewind(intent)
        elif isinstance(intent, state.Peek | state.Drop):
            self._watch(intent)
        elif isinstance(intent, state.SetThinking):
            self.command(
                protocol.ThinkingSet(
                    session_id=session, effort=intent.effort
                )
            )
        elif isinstance(intent, state.SetBackend):
            # None, not "": `protocol.BackendSet` spells "the default backend
            # rather than one session's" as a missing session, and an empty
            # string would name a conversation that does not exist.
            self.command(
                protocol.BackendSet(
                    backend=intent.backend,
                    label=intent.label,
                    session_id=intent.session_id or None,
                )
            )
        elif isinstance(intent, state.SetProfile):
            self.command(protocol.ProfileSet(name=intent.name))
        elif isinstance(intent, state.SaveProfile):
            self.command(
                protocol.ProfileSave(
                    name=intent.name, kind=intent.kind, text=intent.text
                )
            )
        elif isinstance(intent, state.CreateProfile):
            self.command(protocol.ProfileCreate(name=intent.name))
            self._reread_profiles()
        elif isinstance(intent, state.CopyProfile):
            self.command(
                protocol.ProfileDuplicate(
                    name=intent.name, source=intent.source or None
                )
            )
            self._reread_profiles()
        elif isinstance(intent, state.DeleteProfile):
            self.command(protocol.ProfileDelete(name=intent.name))
            self._reread_profiles()
        elif isinstance(intent, state.SaveSkill):
            self.command(
                protocol.SkillSave(
                    profile=intent.profile,
                    name=intent.name,
                    text=intent.text,
                    # Where it lands. The creator asks; every other path is
                    # editing a file that already exists and leaves it alone.
                    level=intent.level,
                )
            )
        elif isinstance(intent, state.DraftSkill):
            # The half of `/skill-creator` that is not this side's: a draft is
            # a model call. None rather than "" for a request made with
            # nothing open — the wire spells "no session" as null.
            self.command(
                protocol.SkillDraft(
                    profile=intent.profile,
                    request=intent.request,
                    session_id=intent.session_id or None,
                )
            )
        elif isinstance(intent, state.DeleteSkill):
            self.command(
                protocol.SkillDelete(profile=intent.profile, name=intent.name)
            )
        elif isinstance(intent, state.ResolveMemory):
            self.command(
                protocol.MemoryResolve(
                    session_id=session, approved=list(intent.approved)
                )
            )
        elif isinstance(intent, state.RunCommand):
            self.command(
                protocol.CommandRun(
                    name=intent.name,
                    args=intent.args,
                    session_id=intent.session_id or None,
                )
            )
        elif isinstance(intent, state.Fetch):
            # The three read paths, one intent (`state.Fetch`): a screen names
            # what it wants and the answer comes back addressed by the same
            # key, so a body that outlived the screen that asked for it fills
            # nothing (`RowUI.body_arrived`).
            if intent.what == "profile":
                name, kind = intent.key
                self.command(protocol.ProfileGet(name=name, kind=kind))
            elif intent.what == "skill":
                profile, name = intent.key
                self.command(protocol.SkillGet(profile=profile, name=name))
            elif intent.what == "settings":
                self.command(protocol.SettingsGet())
            else:  # pragma: no cover - a screen asking for something new
                logger.warning("no read path for %r", intent.what)
        elif isinstance(intent, state.FetchSkills):
            self.command(
                protocol.SkillList(profile=intent.profile, scope=intent.scope)
            )
        elif isinstance(intent, state.SaveSettings):
            # Straight out, with only the JSON check already done on screen.
            # What is *in* the file is the core's model to judge, and it is the
            # core that has to rebuild the clients a change affects — which is
            # the half the old front-end could do nothing about and answered
            # with "applies on next start".
            self.command(protocol.SettingsSave(text=intent.text))
        else:  # pragma: no cover - every Intent member is handled above
            logger.warning("no command for %r", intent)

    def _reread_profiles(self) -> None:
        """Ask for the list again after something changed it.

        Nothing announces a created, copied or deleted profile: the core says
        so in a `notify` and restates the *sessions* (their profile moved), but
        `profile.rows` is only ever sent when it is asked for. The screen shows
        what it believes in the meantime; this is what makes it true, or takes
        it back when the core refused the name.
        """
        self.command(protocol.ProfileList())

    # ------------------------------------------- the skills, and $EDITOR

    def ask_skills(self, profile: str) -> None:
        """`skill.list`: what the "/" menu offers besides the seven built-ins.

        The **visible** scope, which is everything the profile can call: the
        skills HPCA ships, the shared ones, its own and the project's, each
        row saying which level it came from. The narrower `own` scope is what
        an editor asks for, and a menu built from it reports `/plan` unknown
        on a fresh install — the profile has no skills of its own yet and the
        shipped ones are what a fresh install *does* have.

        The rows are not a permission: `SkillInfo.removable` is what decides
        which of them the remove picker may offer, because a shipped or shared
        skill is not one profile's to delete.
        """
        self.command(protocol.SkillList(profile=profile, scope="visible"))

    # --------------------------------------------------------------- $EDITOR

    def edit_profile(self, profile: str) -> None:
        """ctrl+e: this profile's memories in `$EDITOR`, in three steps.

        The steps are `profile.get`, the editor, `profile.save`, and they are
        in that order for the reason every editor here fetches: the file is
        also written by the agent, and `profile.save` writes verbatim — an
        editor opened over a stale copy silently reverts whatever the agent
        learned in between.

        That makes the whole thing asynchronous, which is why this returns
        nothing: the request goes out now and `_profile_body` picks it up when
        the answer lands, hands the terminal over (`RowUI.suspend`, wired by
        `ui/run.py`) and takes it back. A second ctrl+e while one is out is
        ignored rather than queued — there is one terminal.
        """
        if self._editing:
            return
        self._editing = profile
        self.command(protocol.ProfileGet(name=profile, kind="memories"))

    def _run_editor(self, profile: str, text: str) -> None:
        """The middle step: a scratch file, the user's editor, and the save.

        A temporary file rather than the profile's own path, because a path is
        not on the wire and asking for one would be asking the core to hand
        out filesystem access — the thing this protocol is careful not to be.
        What comes back out is sent as `profile.save`, which is the same
        command the memory editor uses and the one that invalidates the core's
        loaded copy (`core.memory_service.invalidate`).

        Editor resolution is `hpca.editor.resolve_editor` — settings, then
        `$VISUAL`, then `$EDITOR`, then nano — and is not reimplemented; the
        settings half comes out of the JSON this UI already holds, so no
        config module is imported to read one field.
        """
        import json
        import os
        import subprocess
        import tempfile

        from hpca.editor import resolve_editor

        chosen = None
        try:
            chosen = json.loads(self.ui.settings_json or "{}").get("editor")
        except ValueError:  # a settings file too broken to parse: still edit
            pass
        argv = resolve_editor(chosen if isinstance(chosen, str) else None, os.environ)
        with tempfile.TemporaryDirectory(prefix="hpca-edit-") as scratch:
            path = os.path.join(scratch, f"{profile or 'profile'}.md")
            try:
                with open(path, "w") as handle:
                    handle.write(text)
                code = subprocess.call([*argv, path])
                edited = open(path).read()
            except OSError as e:
                self.ui.toast(f"could not run {argv[0]}: {e}", "error")
                return
        if code != 0:
            self.ui.toast(
                f"{argv[0]} exited with {code} — nothing was saved", "warning"
            )
            return
        if edited == text:
            self.ui.toast(f"“{profile}” unchanged")
            return
        self.command(protocol.ProfileSave(name=profile, kind="memories", text=edited))
        self._reread_profiles()
        self.ui.toast(f"saved and reloaded “{profile}”")

    def _cycle_mode(self, session_id: str) -> None:
        """The next mode, worked out here and shown before the core answers.

        Here because the cycle is `hpca.agent`'s (`next_mode`), and a key
        dispatcher holding its own copy of the mode list would be a second
        place for it to be wrong. Shown immediately because the bar is the
        feedback for the keypress: the core persists it and the next
        `session.rows` says so, and if the core refuses, that same repaint is
        what puts the bar back.
        """
        session = self._session(session_id)
        session.mode = next_mode(session.mode)
        self.ui.refresh_sidebar()
        self.command(
            protocol.ModeSet(session_id=session_id, mode=session.mode)
        )

    def _rewind(self, intent: state.Fork | state.Rollback) -> None:
        """A row the user pointed at, as the message index the wire wants.

        The UI knows its rows by `seq` and nothing else; `_Rewind.index` is the
        thread message the cut is made at. Resolving one into the other is this
        side's job — and a row that is not a message (a fold, an error the UI
        wrote itself) has no index, so the answer is to say so rather than to
        send a cut aimed at the wrong place.
        """
        entry = self.ui.session_for(intent.session_id).entry_of(intent.seq)
        if entry is None or entry.index < 0:
            self.ui.toast("there is nothing to rewind to on that row", "warning")
            return
        if isinstance(intent, state.Fork):
            self.command(
                protocol.SessionFork(
                    session_id=intent.session_id, index=entry.index
                )
            )
        else:
            self.command(
                protocol.SessionRollback(
                    session_id=intent.session_id, index=entry.index
                )
            )

    def _watch(self, intent: state.Peek | state.Drop) -> None:
        """`PanelRow.ref` is a string for every kind of row; a watch is an int.

        The conversion has to happen somewhere, and `protocol.PanelRow` says
        explicitly that it is the sender's errand.
        """
        try:
            watch_id = int(intent.ref)
        except ValueError:
            self.ui.toast(f"that row is not a watch ({intent.ref!r})", "warning")
            return
        if isinstance(intent, state.Peek):
            self.command(protocol.WatchPeek(watch_id=watch_id))
        else:
            self.command(protocol.WatchDrop(watch_id=watch_id))

    def command(self, cmd: protocol.Command) -> None:
        if self._send is not None:
            self._send(cmd)
        else:
            self.outbox.append(cmd)

    async def flush(self) -> None:
        """Put what the keys asked for on the wire, in the order they asked."""
        pending, self.outbox = self.outbox, []
        for cmd in pending:
            if self._conn is not None:
                await self._conn.send(cmd)

    async def run(self) -> None:
        """Apply every frame the core sends, until it stops sending.

        The whole of the client's asyncio: one loop, no threads, and the state
        it writes into is read synchronously by the next `render`.
        """
        if self._conn is None:  # pragma: no cover - a demo client has no wire
            return
        async for env in self._conn:
            self.apply(env)

    # -------------------------------------------------------------- inbound

    def apply(self, frame: protocol.Envelope | protocol.Message) -> None:
        """One frame, into state. A frame nobody can draw is dropped.

        Bad frames are counted rather than raised: a reader loop that dies of
        one malformed event takes the session with it, and the same bargain is
        already made one layer down in `transport`.
        """
        if isinstance(frame, protocol.Envelope):
            try:
                frame = protocol.parse(frame)
            except protocol.ProtocolError as e:
                logger.warning("dropping a frame the UI cannot read: %s", e)
                self.dropped["unparseable"] += 1
                return
        handler = self._HANDLERS.get(type(frame).__name__)
        if handler is None:
            # §3.2: a client ignores what it cannot draw. Counted so a test can
            # assert that an unhandled event is a decision and not an oversight.
            self.dropped[frame.TYPE] += 1
            return
        handler(self, frame)

    # Each handler starts by naming its session; that is property 1 of §3.2
    # made into a habit rather than a rule someone has to remember.
    def _session(self, session_id: str) -> state.SessionState:
        return self.ui.session_for(session_id)

    def _hello(self, msg: protocol.Hello) -> None:
        self.hello = msg
        if msg.version != protocol.PROTOCOL_VERSION:
            # Not fatal here: the handshake in `coreproc` is what refuses a
            # stale core. On this side it is something to say out loud.
            self.ui.toast(
                f"the core speaks protocol {msg.version}, "
                f"this UI speaks {protocol.PROTOCOL_VERSION}",
                "error",
            )
        self.ui.core_profile = msg.profile
        self.command(protocol.SessionList())
        # The two lists the screens draw and the UI cannot work out for itself
        # (rule 2 of §4.2). Asked for on connect rather than when a screen
        # opens, because `m` and `a` and the new-session picker must not each
        # begin with a round trip — and because the catalog is what decides
        # whether the new-session flow has a second stage at all.
        #
        # With probes: this client draws ● / ○, and the first frame is
        # identical either way (`protocol.LLMList`), so asking for them costs
        # nothing before the catalog can be drawn.
        self.command(protocol.LLMList(probe=True))
        self.command(protocol.ProfileList())
        # And the third: how often each command has been run, which is what
        # sorts the "/" menu. Asked for here rather than carried by `hello`
        # because the numbers change as commands are run and the greeting is
        # stated once — the core restates this one (`protocol.CommandCounts`).
        self.command(protocol.CommandList())

    def _rows(self, msg: protocol.SessionRows) -> None:
        self.ui.sync_sessions(
            [
                state.SidebarRow(
                    session_id=row.session_id,
                    title=row.title,
                    profile=row.profile,
                    mode=row.mode,
                    model=row.model,
                    thinking=row.thinking,
                    flags=tuple(row.flags),
                )
                for row in msg.rows
            ]
        )
        if not self._opened and self.ui.sessions:
            # Something has to be open for the chat row to have anything in it,
            # and the core does not decide what the user is looking at — so the
            # first list to arrive is answered by opening its first row, once.
            self.ui.open_session(self.ui.sessions[0].session_id, announce=False)

    def _created(self, msg: protocol.SessionCreated) -> None:
        """A session the core made because the user asked for it — so open it.

        The reason this event exists rather than the UI diffing two sidebars:
        two forks of one conversation have the same title.
        """
        session = self._session(msg.row.session_id)
        session.title = msg.row.title
        session.profile = msg.row.profile
        session.mode = msg.row.mode
        session.model = msg.row.model
        session.thinking = msg.row.thinking
        session.context.effort = msg.row.thinking
        session.flags = list(msg.row.flags)
        self.ui.adopt(session)
        # Straight into the message box: a conversation that exists because
        # the user asked for one — `session.new` or a fork — exists in order
        # to be typed in, which is what `start_new_session` ended with too.
        self.ui.open_session(session.session_id, land_in_box=True)

    def _catalog(self, msg: protocol.LLMCatalog) -> None:
        """Every LLM the core knows about, whole (`protocol.LLMCatalog`).

        Replaced rather than merged, because the event is a statement of what
        the catalog *is* — and the second frame, the one with the probes in
        it, is the same catalog with `reachable` filled in.

        An open screen is restated too: `m` can be sitting on the panel while
        the probes land, and a list that only refreshed on the next open would
        show `·` for the whole time the answer was already in.
        """
        self.ui.catalog = [
            state.BackendInfo(
                label=entry.label,
                model=entry.model,
                base_url=entry.base_url,
                context=entry.max_model_len or 0,
                needs_key=entry.needs_key,
                active=entry.active,
                reachable=entry.reachable,
                discovered=entry.discovered,
            )
            for entry in msg.entries
        ]
        for screen in self.ui.overlays:
            catalog = getattr(screen, "catalog_changed", None)
            if catalog is not None:
                catalog(self.ui.catalog)

    def _profiles(self, msg: protocol.ProfileRows) -> None:
        """The profiles, whole — the answer to `profile.list`.

        The bodies the editors open are deliberately absent: the row carries a
        *count* (`protocol.ProfileRow`), and each file is a `profile.get` away,
        fetched when its editor opens rather than shipped to every client on
        connect. The skills a profile already listed are kept across the
        refresh — this event says nothing about them, and dropping them would
        empty a screen that is open over them.
        """
        known = {x.name: x.skills for x in self.ui.profiles}
        self.ui.profiles = [
            state.ProfileInfo(
                name=row.name,
                memories=row.memories,
                copied_from=row.copied_from,
                default=row.is_default,
                working=row.working,
                skills=known.get(row.name, []),
            )
            for row in msg.rows
        ]

    def _reset(self, msg: protocol.ChatReset) -> None:
        self._session(msg.session_id).reset([_entry(e) for e in msg.entries])

    def _append(self, msg: protocol.ChatAppend) -> None:
        self._session(msg.session_id).append(_entry(msg.entry))

    def _update(self, msg: protocol.ChatUpdate) -> None:
        if not self._session(msg.session_id).update(_entry(msg.entry)):
            # An update for a row we do not have means we are out of step with
            # a reset. Inventing the row would hide that; the next reset fixes
            # it (`protocol.ChatUpdate`).
            self.dropped["chat.update"] += 1

    def _tick(self, session: state.SessionState) -> None:
        """Put the working row where the turn's state now says it goes.

        Done here as well as at render time because the row is *navigable*: a
        pane whose live row only appeared when a frame was drawn would answer
        "end" with the wrong row for one keypress, and Enter on that row is
        the interrupt.
        """
        session.tick(self.ui.wall())

    def _started(self, msg: protocol.TurnStarted) -> None:
        session = self._session(msg.session_id)
        # The stamp comes with the event because the clock the user reads is
        # "how long since I sent it", and a turn is silent for as long as the
        # backend takes to answer the first time — a spinner that only started
        # counting at the first `turn.activity` would show nothing for that
        # whole wait (`protocol.TurnStarted`).
        session.start_turn(msg.started_at)
        self._tick(session)
        self.ui.refresh_sidebar()

    def _activity(self, msg: protocol.TurnActivity) -> None:
        session = self._session(msg.session_id)
        was = session.turn.busy
        session.turn.activity_is(msg.activity, msg.started_at)
        self._tick(session)
        if session.turn.busy != was:
            # A backend call that is not a turn — the titler, a compaction, a
            # silent `/conclude` — starts and ends on this event alone, and
            # the sidebar's "⟳" is how a user working in another conversation
            # sees it (§4.3 item 15). Only on the edges: `turn.activity`
            # arrives once per tool call, and repainting fourteen rows per
            # step to say the same thing is the poll this UI does not have.
            self.ui.refresh_sidebar()
        if not msg.activity and not session.turn.working:
            # A backend call that was never a turn, saying it is done. The
            # scheduler's own "" arrives just before `turn.finished` and is
            # answered there; this is the other emitter (`memory_service`),
            # which has no turn to finish.
            session.end_turn()

    def _finished(self, msg: protocol.TurnFinished) -> None:
        self._session(msg.session_id).end_turn()
        self.ui.refresh_sidebar()

    def _failed(self, msg: protocol.TurnFailed) -> None:
        session = self._session(msg.session_id)
        session.end_turn()
        # A failure the user has to be able to find again afterwards, so it goes
        # in the transcript rather than only into a toast that expires. seq 0:
        # the core did not number this row and nothing may revise it.
        session.append(state.ChatEntry(kind="error", text=msg.error))
        self.ui.refresh_sidebar()

    def _usage(self, msg: protocol.TurnUsage) -> None:
        self._session(msg.session_id).context.measure(
            msg.prompt_tokens, msg.max_model_len
        )

    def _estimate(self, msg: protocol.ContextEstimate) -> None:
        # A measured fill beats a guess at the same thread; `Context.estimate`
        # is where that precedence lives, since `context.estimate` carries no
        # flag saying which of the two it is.
        self._session(msg.session_id).context.estimate(msg.used, msg.window)

    def _decision(self, msg: protocol.DecisionRequested) -> None:
        # The same payload twice is the same question — the core re-emits a
        # parked decision on subscribe (§4.4) — so a refusal half-written when
        # the socket dropped survives the reconnect.
        self._session(msg.session_id).request_decision(dict(msg.payload))
        self.ui.decision_arrived(msg.session_id)

    def _decision_cleared(self, msg: protocol.DecisionCleared) -> None:
        self._session(msg.session_id).clear_decision()
        self.ui.decision_cleared(msg.session_id)

    def _confirm(self, msg: protocol.ConfirmRequested) -> None:
        self.ui.confirm_requested(msg.id, msg.question)

    def _unqueued(self, msg: protocol.TurnUnqueued) -> None:
        """A typed-ahead message taken back: its row goes, its text stays.

        The one event that removes a chat row, and the protocol is what makes
        it an exception to §3.2 rather than the UI deciding: the message was
        never in the graph, so no `chat.reset` could take it off the screen.
        The text goes to the draft of the session it was typed in — which need
        not be the one on screen, since the answer can arrive after a switch.
        """
        session = self._session(msg.session_id)
        if session.remove(msg.seq) is None:
            self.dropped["turn.unqueued"] += 1
        self.ui.hand_back(msg.session_id, msg.text)

    def _interrupted(self, msg: protocol.TurnInterrupted) -> None:
        """A stopped turn's message, back to the session it was typed in.

        `turn.unqueued`'s sibling and deliberately not the same event: this one
        names no row, because the `chat.reset` that precedes it has already
        un-drawn the abandoned attempt (`protocol.TurnInterrupted`). All that
        is left to do with it is the half both share — park the text as that
        session's draft, which is why they share the routine.
        """
        self.ui.hand_back(msg.session_id, msg.text)

    def _panel(self, msg: protocol.PanelUpdate) -> None:
        # A panel without a session belongs to the profile rather than to a
        # conversation; there is one column, so it goes to what is on screen.
        session_id = msg.session_id or self.ui.active_id
        if not session_id:
            self.dropped["panel.update"] += 1
            return
        self._session(session_id).set_watchers([_panel_item(r) for r in msg.rows])

    def _proposals(self, msg: protocol.MemoryProposals) -> None:
        """Memories the agent wants to keep, and the screen that answers them.

        Held on the session first, because the offer belongs to the
        conversation it came out of and answering it against another one would
        resolve the wrong batch. Put on screen straight away only when that
        conversation is the one on screen — a review that covered somebody
        else's chat would be the modal mistake §4.3 item 21 is careful about,
        one screen over.
        """
        session = self._session(msg.session_id)
        session.proposals = [
            state.Proposal(scope=p.scope, kind=p.kind, text=p.text)
            for p in msg.proposals
        ]
        if not session.proposals:
            return
        if msg.session_id == self.ui.active_id and self.ui.overlay is None:
            self.ui.review_memories(msg.session_id)
        else:
            self.ui.toast(
                f"{len(session.proposals)} memories to review in "
                f"“{session.title or msg.session_id}”"
            )

    # ------------------------------------------------------- the read paths

    def _profile_body(self, msg: protocol.ProfileBody) -> None:
        """`profile.body`: to the editor waiting for it, or into `$EDITOR`.

        Two callers, one event, told apart by whether a ctrl+e is out. The
        editor suspend happens here rather than at the keypress because this is
        the moment the text exists: `_run_editor` writes it, runs the program
        and sends what comes back.
        """
        if self._editing == msg.name and msg.kind == "memories":
            profile, self._editing = self._editing, ""
            if msg.error:
                self.ui.toast(msg.error, "error")
                return
            suspend = getattr(self.ui, "suspend", None)
            if suspend is None:
                self.ui.toast("no terminal to hand over", "warning")
                return
            try:
                suspend(lambda: self._run_editor(profile, msg.text))
            except Exception as e:
                self.ui.toast(f"cannot suspend for editing: {e}", "error")
            return
        self.ui.body_arrived(
            ("profile", (msg.name, msg.kind)), msg.text, msg.error
        )

    def _skill_rows(self, msg: protocol.SkillRows) -> None:
        """`skill.rows`: the list a screen is drawing, or the "/" menu offers.

        Routed by the scope it answers, and it has to be: the menu asked about
        everything callable and an editor asked about what it may overwrite,
        and filling either list with the other's answer is how a screen ends
        up offering to delete a skill that ships with HPCA.
        """
        skills = [
            state.SkillInfo(
                name=row.name, description=row.description, level=row.level
            )
            for row in msg.skills
        ]
        if msg.scope == "visible":
            self.ui.skills_listed(msg.profile, skills)
            return
        self.ui.own_skills_listed(msg.profile, skills)
        self.ui.list_arrived(("skills", msg.profile), skills)

    def _skill_drafted(self, msg: protocol.SkillDrafted) -> None:
        """`skill.drafted`: the form, opened over what the model wrote.

        A failed draft opens the empty form rather than losing the command —
        the core sends this event either way, which is what makes that
        possible (`protocol.SkillDrafted`).
        """
        drafted = (
            state.SkillInfo(
                name=msg.name, description=msg.description, text=msg.body
            )
            if msg.name
            else None
        )
        self.ui.skill_drafted(drafted, msg.request, msg.error)

    def _command_counts(self, msg: protocol.CommandCounts) -> None:
        """`command.counts`: how the "/" menu sorts. Replaced whole — it is a
        statement of the table, not a delta (`protocol.CommandCounts`)."""
        self.ui.command_counts = dict(msg.counts)

    def _skill_body(self, msg: protocol.SkillBody) -> None:
        self.ui.body_arrived(
            ("skill", (msg.profile, msg.name)), msg.text, msg.error
        )

    def _settings_body(self, msg: protocol.SettingsBody) -> None:
        """`settings.body`: the file as it is on disk now.

        Sent both in answer to `settings.get` and after every `settings.save`,
        including a save the core refused — in which case ``error`` says why
        and ``text`` still describes the file that is still there. So the
        error is a toast and the body is applied either way, and the editor is
        never reopened over rejected text: what the user typed is on their
        screen, not here.
        """
        self.ui.settings_json = msg.text
        self.ui.body_arrived(("settings", ()), msg.text)
        if msg.error:
            self.ui.toast(msg.error, "error")

    def _notify(self, msg: protocol.Notify) -> None:
        """A toast, heading and all — the whole of §3.2's `notify` row.

        `title` is passed through rather than glued onto the front of the text,
        which is exactly what the field exists for (`protocol.Notify.title`): a
        few of the core's answers are a heading plus a block — the skills a
        profile can see, the summary `/compact` just wrote, the xhigh warning
        — and a renderer handed one string can no longer tell which half is
        which.
        """
        self.ui.toast(msg.text, msg.severity, msg.timeout, title=msg.title)

    def _peeked(self, msg: protocol.WatchPeeked) -> None:
        """The answer to a keypress, drawn where a toast is drawn.

        Not in §3.2's table because that table is about state; this one is a
        reply, and the alternative to showing it is a key that does nothing.
        """
        head = f"{msg.title}: " if msg.title else ""
        self.ui.toast(head + " ".join(msg.text.split()))

    _HANDLERS: dict[str, Callable[[UIClient, protocol.Message], None]] = {}


UIClient._HANDLERS = {
    protocol.Hello.__name__: UIClient._hello,
    protocol.SessionRows.__name__: UIClient._rows,
    protocol.SessionCreated.__name__: UIClient._created,
    protocol.LLMCatalog.__name__: UIClient._catalog,
    protocol.ProfileRows.__name__: UIClient._profiles,
    protocol.ChatReset.__name__: UIClient._reset,
    protocol.ChatAppend.__name__: UIClient._append,
    protocol.ChatUpdate.__name__: UIClient._update,
    protocol.TurnStarted.__name__: UIClient._started,
    protocol.TurnActivity.__name__: UIClient._activity,
    protocol.TurnFinished.__name__: UIClient._finished,
    protocol.TurnFailed.__name__: UIClient._failed,
    protocol.TurnUnqueued.__name__: UIClient._unqueued,
    protocol.TurnInterrupted.__name__: UIClient._interrupted,
    protocol.TurnUsage.__name__: UIClient._usage,
    protocol.ContextEstimate.__name__: UIClient._estimate,
    protocol.DecisionRequested.__name__: UIClient._decision,
    protocol.DecisionCleared.__name__: UIClient._decision_cleared,
    protocol.ConfirmRequested.__name__: UIClient._confirm,
    protocol.PanelUpdate.__name__: UIClient._panel,
    protocol.MemoryProposals.__name__: UIClient._proposals,
    protocol.ProfileBody.__name__: UIClient._profile_body,
    protocol.SkillRows.__name__: UIClient._skill_rows,
    protocol.SkillDrafted.__name__: UIClient._skill_drafted,
    protocol.CommandCounts.__name__: UIClient._command_counts,
    protocol.SkillBody.__name__: UIClient._skill_body,
    protocol.SettingsBody.__name__: UIClient._settings_body,
    protocol.Notify.__name__: UIClient._notify,
    protocol.WatchPeeked.__name__: UIClient._peeked,
}
