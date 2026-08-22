"""Tests for hpca.protocol: the envelope, the NDJSON codec, the registries.

Not one socket here. The framing is a pure function of the bytes, and keeping
it testable without a transport is why it lives in its own module: a frame
that survives a round trip in memory survives one over AF_UNIX too.
"""

from __future__ import annotations

import dataclasses
import json
from typing import ClassVar

import pytest

from hpca import protocol, transcript
from hpca.protocol import (
    COMMANDS,
    EVENTS,
    CommandCounts,
    CommandList,
    PROTOCOL_VERSION,
    ChatAppend,
    ChatReset,
    ChatUpdate,
    Command,
    CommandRun,
    Entry,
    Envelope,
    Event,
    Hello,
    MemoryResolve,
    Message,
    Notify,
    PanelRow,
    PanelUpdate,
    Part,
    ProtocolError,
    SessionCreated,
    SessionFork,
    SessionRollback,
    BackendProbed,
    BackendScanned,
    BackendSet,
    LLMCatalog,
    LLMEntry,
    ProfileBody,
    ProfileGet,
    SettingsBody,
    SettingsSave,
    SkillBody,
    SkillDraft,
    SkillDrafted,
    SkillGet,
    SkillList,
    SkillRow,
    SkillRows,
    ProfileRow,
    ProfileRows,
    SessionNew,
    SessionRow,
    SessionRows,
    TurnInterrupted,
    TurnStarted,
    TurnSubmit,
    TurnUnqueue,
    TurnUnqueued,
    TurnUsage,
    WatchPeek,
    WatchPeeked,
    decode,
    encode,
    parse,
)

# The two tables of §4, written out so that deleting or renaming a message
# type has to be a deliberate edit in two places rather than a silent drop.
SPEC_COMMANDS = {
    "session.list",
    "session.new",
    "session.open",
    "session.close",
    "session.rename",
    "session.retitle",
    "session.delete",
    "session.fork",
    "session.rollback",
    "session.focus",
    "turn.submit",
    "turn.interrupt",
    "turn.unqueue",
    "decision.resolve",
    "command.run",
    "confirm.resolve",
    "memory.resolve",
    "mode.set",
    "thinking.set",
    "backend.set",
    # Not in the §4.1 table: the catalog and the profile listing the table
    # never had, added because a front-end cannot draw either by inference.
    "llm.list",
    "profile.list",
    # Nor these. §4.1 gave the four editable things a *write* command each and
    # no way to read what was about to be overwritten, which is not a missing
    # convenience — `profile.save` writes verbatim, so an editor opened over a
    # body it could not fetch truncates the file. See `TestTheReadPaths`.
    "profile.get",
    "skill.list",
    "skill.get",
    # And the two the "/" menu needs behind it: a draft is a model call, so it
    # cannot happen on the front-end's side of the socket, and the usage
    # counts are a table rule 2 puts out of a front-end's reach.
    "skill.draft",
    "command.list",
    "settings.get",
    "settings.save",
    # And the three the manage-LLMs screen needs behind it: an endpoint check,
    # a scan for endpoints nobody has configured, and the removal that §4.1's
    # add-only catalog left with no command at all.
    "backend.probe",
    "backend.scan",
    "backend.remove",
    "profile.set",
    "profile.save",
    "profile.create",
    "profile.delete",
    "profile.duplicate",
    "skill.save",
    "skill.delete",
    "process.kill",
    "job.cancel",
    "watch.peek",
    "watch.drop",
    "shutdown",
}

SPEC_EVENTS = {
    "hello",
    "session.rows",
    "session.created",
    "llm.catalog",
    "profile.rows",
    "profile.body",
    "skill.rows",
    "skill.body",
    "skill.drafted",
    "command.counts",
    "settings.body",
    "backend.probed",
    "backend.scanned",
    "chat.reset",
    "chat.append",
    "chat.update",
    "turn.started",
    "turn.activity",
    "turn.usage",
    "turn.finished",
    "turn.failed",
    "turn.unqueued",
    "turn.interrupted",
    "decision.requested",
    "decision.cleared",
    "panel.update",
    "watch.peeked",
    "memory.proposals",
    "confirm.requested",
    "context.estimate",
    "notify",
}


def submit() -> TurnSubmit:
    return TurnSubmit(session_id="s1", text="which BAMs are in the cohort?")


class TestEnvelope:
    def test_only_the_type_is_required(self):
        env = Envelope(type="shutdown")
        assert env.seq == 0
        assert env.id is None
        assert env.payload == {}

    def test_an_unknown_envelope_key_is_rejected(self):
        # extra="forbid": both ends ship from one install, so a stray key is a
        # bug on the sending side, not a peer being forward-compatible.
        with pytest.raises(ProtocolError):
            decode('{"type": "shutdown", "urgent": true}')


class TestCodec:
    def test_encode_is_exactly_one_line_ending_in_a_newline(self):
        raw = encode(submit().to_envelope())
        assert isinstance(raw, bytes)
        assert raw.endswith(b"\n")
        assert raw.count(b"\n") == 1

    def test_round_trip_preserves_every_envelope_field(self):
        env = Envelope(seq=12, id="c7", type="turn.submit", payload={"a": [1, 2]})
        assert decode(encode(env)) == env

    def test_decode_takes_bytes_or_str(self):
        raw = encode(Envelope(type="shutdown"))
        assert decode(raw) == decode(raw.decode()) == Envelope(type="shutdown")

    def test_decode_tolerates_the_trailing_newline_the_framing_adds(self):
        assert decode('{"type": "shutdown"}\n').type == "shutdown"

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "not json at all",
            '{"type": "shutdown"',
            '{"type": "shutdown"} {"type": "shutdown"}',
        ],
    )
    def test_malformed_json_is_a_protocol_error(self, line):
        with pytest.raises(ProtocolError):
            decode(line)

    @pytest.mark.parametrize("line", ["[1, 2]", '"shutdown"', "5", "null", "true"])
    def test_json_that_is_not_an_object_is_a_protocol_error(self, line):
        with pytest.raises(ProtocolError):
            decode(line)

    @pytest.mark.parametrize(
        "line",
        [
            "{}",
            '{"seq": 1}',
            '{"type": null}',
            '{"type": 7}',
            '{"type": ""}',
            '{"type": "shutdown", "payload": []}',
        ],
    )
    def test_a_missing_or_invalid_type_is_a_protocol_error(self, line):
        with pytest.raises(ProtocolError):
            decode(line)

    def test_bytes_that_are_not_utf8_are_a_protocol_error(self):
        with pytest.raises(ProtocolError):
            decode(b'{"type": "\xff\xfe"}')

    def test_a_payload_the_json_encoder_cannot_take_is_a_protocol_error(self):
        # A live object in a payload is a bug on the sending side; it must
        # still surface as the one exception a writer loop catches.
        env = Envelope(type="notify", payload={"text": object()})
        with pytest.raises(ProtocolError):
            encode(env)


class TestFraming:
    def test_a_newline_inside_a_string_field_still_round_trips(self):
        # A pasted traceback is the first thing that would break NDJSON if
        # JSON did not escape the delimiter for us.
        text = "line one\nline two\r\nline three"
        raw = encode(TurnSubmit(session_id="s1", text=text).to_envelope())
        assert raw.count(b"\n") == 1  # the frame delimiter, and nothing else
        assert parse(decode(raw)).text == text

    def test_a_multiline_chat_entry_survives_the_wire(self):
        entry = Entry(
            kind="thinking",
            text="— reasoning —\nfirst\n\n— step —\n[tool result] list_dir: 12",
            steps=1,
            reasoning_chars=5,
            parts=[Part(kind="reasoning", text="first\nsecond")],
        )
        event = ChatAppend(session_id="s1", entry=entry)
        raw = encode(event.to_envelope())
        assert raw.count(b"\n") == 1
        assert parse(decode(raw)) == event


class TestMessages:
    def test_a_command_round_trips_losslessly(self):
        message = TurnSubmit(session_id="s1", text="hello", forced_skill="triage")
        assert parse(decode(encode(message.to_envelope()))) == message

    def test_an_event_round_trips_losslessly(self):
        message = PanelUpdate(
            profile="default",
            session_id="s1",
            rows=[
                PanelRow(key="w:3", text="● RUNNING", classes="watch-live",
                         title="train.log", kind=protocol.PANEL_WATCH, ref="3"),
                PanelRow(key="w:4", text="○ not polled yet", classes="watch-idle",
                         title="run.log", kind=protocol.PANEL_WATCH, ref="4"),
            ],
        )
        assert parse(decode(encode(message.to_envelope()))) == message

    def test_to_envelope_stamps_the_type_and_carries_seq_and_id(self):
        env = submit().to_envelope(seq=3, id="c9")
        assert (env.type, env.seq, env.id) == ("turn.submit", 3, "c9")
        assert env.payload["session_id"] == "s1"

    def test_seq_and_id_default_to_the_unnumbered_case(self):
        env = submit().to_envelope()
        assert (env.seq, env.id) == (0, None)

    def test_optional_fields_survive_as_null(self):
        env = TurnSubmit(session_id="s1", text="x").to_envelope()
        assert env.payload["forced_skill"] is None
        assert parse(env).forced_skill is None

    def test_hello_announces_the_current_protocol_version(self):
        assert Hello(profile="default", settings_digest="abc").version == (
            PROTOCOL_VERSION
        )

    def test_an_event_can_answer_a_command(self):
        # §4: a reply echoes the command's correlation id in payload.reply_to.
        rows = SessionRows(rows=[], reply_to="c7")
        assert parse(decode(encode(rows.to_envelope()))).reply_to == "c7"

    def test_from_envelope_refuses_an_envelope_of_another_type(self):
        with pytest.raises(ProtocolError):
            TurnSubmit.from_envelope(Notify(text="hi").to_envelope())


class TestParse:
    def test_an_unknown_type_is_a_protocol_error(self):
        with pytest.raises(ProtocolError):
            parse(Envelope(type="turn.teleport"))

    def test_a_payload_that_does_not_fit_the_model_is_a_protocol_error(self):
        with pytest.raises(ProtocolError):
            parse(Envelope(type="turn.submit", payload={"session_id": "s1"}))

    def test_a_payload_field_of_the_wrong_type_is_a_protocol_error(self):
        with pytest.raises(ProtocolError):
            parse(Envelope(type="process.kill", payload={"pid": "not a pid"}))

    def test_an_unexpected_payload_key_is_a_protocol_error(self):
        env = Envelope(
            type="shutdown", payload={"and_delete_everything": True}
        )
        with pytest.raises(ProtocolError):
            parse(env)

    def test_a_pydantic_error_never_escapes_as_itself(self):
        # Callers get to catch exactly one exception type at the boundary.
        with pytest.raises(ProtocolError):
            parse(Envelope(type="notify", payload={"severity": "loud", "text": ""}))

    def test_parse_finds_both_commands_and_events(self):
        assert isinstance(parse(submit().to_envelope()), Command)
        assert isinstance(parse(Notify(text="hi").to_envelope()), Event)


class TestRegistries:
    def test_no_type_string_is_claimed_twice(self):
        assert set(COMMANDS) & set(EVENTS) == set()
        classes = list(COMMANDS.values()) + list(EVENTS.values())
        assert len(classes) == len(set(classes))

    def test_every_key_matches_the_class_it_maps_to(self):
        for type_name, model in {**COMMANDS, **EVENTS}.items():
            assert model.TYPE == type_name

    def test_the_registries_cover_the_spec_tables(self):
        assert set(COMMANDS) == SPEC_COMMANDS
        assert set(EVENTS) == SPEC_EVENTS

    def test_classes_land_in_the_registry_for_their_direction(self):
        assert all(issubclass(m, Command) for m in COMMANDS.values())
        assert all(issubclass(m, Event) for m in EVENTS.values())

    def test_a_duplicate_type_fails_at_class_definition(self):
        with pytest.raises(ProtocolError):

            class Clashing(Command):
                TYPE: ClassVar[str] = "turn.submit"

    def test_a_message_that_is_neither_command_nor_event_fails(self):
        with pytest.raises(ProtocolError):

            class Homeless(Message):
                TYPE: ClassVar[str] = "nowhere.at.all"

    def test_the_abstract_bases_are_not_registered(self):
        assert Message not in COMMANDS.values()
        assert Command not in COMMANDS.values()
        assert Event not in EVENTS.values()


class TestPayloadShapes:
    def test_entry_mirrors_the_transcript_dataclass(self):
        # protocol.Entry is deliberately a copy rather than an import: this
        # module must stay free of the agent side. The copy is only safe if
        # it stays in step, so the shape is asserted — minus the fields that
        # exist only on the wire, which are named here so that adding one is
        # as deliberate an edit as dropping a rendered field would be.
        assert set(Entry.model_fields) - {"seq"} == {
            field.name for field in dataclasses.fields(transcript.Entry)
        }

    def test_the_wire_only_entry_field_is_the_row_name(self):
        # `seq` has no transcript twin because the transcript has no concept
        # of a row being revised: it is rebuilt, which is exactly the cost
        # this protocol exists to delete.
        assert "seq" not in {f.name for f in dataclasses.fields(transcript.Entry)}

    def test_part_mirrors_a_transcript_step(self):
        assert set(Part.model_fields) == {
            field.name for field in dataclasses.fields(transcript.Step)
        }

    def test_a_panel_row_carries_what_the_ui_needs_to_act_on_it(self):
        assert set(PanelRow.model_fields) == {
            "key", "text", "classes", "title", "kind", "ref"
        }

    def test_a_session_row_says_which_model_the_session_talks_to(self):
        # The sidebar and the message row both drew it, and a front-end cannot
        # derive it: the backend is a JSON blob in the database, which §4.2
        # rule 2 puts out of its reach.
        row = SessionRow(session_id="s1", title="a session", model="gemma-3-27b")
        assert parse(decode(encode(SessionRows(rows=[row]).to_envelope()))).rows == [
            row
        ]

    def test_a_bootstrap_session_has_no_model_of_its_own(self):
        # Empty rather than the app's default model name: the row says what
        # this session is pinned to, and a session that pinned nothing is
        # exactly what the absence means.
        assert SessionRow(session_id="s1", title="a session").model == ""

    def test_a_session_row_carries_the_thinking_level_too(self):
        # The other per-session dial, on the same row as `mode` rather than in
        # an event of its own: the meter has to draw the level of whatever
        # session is opened next, and an event could only ever describe the one
        # that just changed (see `protocol.SessionRow.thinking`).
        row = SessionRow(session_id="s1", title="a session", thinking="medium")
        assert parse(decode(encode(SessionRows(rows=[row]).to_envelope()))).rows == [
            row
        ]

    def test_a_session_that_never_chose_a_level_says_nothing(self):
        # Empty, not the configured default: the row reports what the session
        # chose, and "it follows the setting" is what the absence means.
        assert SessionRow(session_id="s1", title="a session").thinking == ""

    def test_no_event_competes_with_the_row_for_the_thinking_level(self):
        # One source, or two clients disagree about which is authoritative.
        assert not [name for name in EVENTS if name.startswith("thinking.")]

    def test_an_entry_needs_only_a_kind_and_text(self):
        entry = Entry(kind="user", text="hi")
        assert (entry.steps, entry.reasoning_chars, entry.parts) == (0, 0, [])

    def test_a_session_scoped_slash_command_names_its_session(self):
        # /compact folds one conversation's history. Letting the core infer
        # which one from the last session.focus would aim a destructive fold
        # at whatever the user happened to switch to in the meantime.
        fold = CommandRun(name="compact", args="keep the QC findings", session_id="s1")
        assert parse(decode(encode(fold.to_envelope()))).session_id == "s1"

    def test_a_profile_scoped_slash_command_names_no_session(self):
        assert CommandRun(name="skills-list").session_id is None

    def test_answering_a_memory_offer_carries_no_memory_text(self):
        # The core keeps the proposal objects; only the yes/no crosses. If the
        # text came back in the answer, a front-end could approve a memory the
        # user never saw.
        answer = MemoryResolve(session_id="s1", approved=[True, False, True])
        assert set(answer.to_envelope().payload) == {"session_id", "approved"}
        assert parse(decode(encode(answer.to_envelope()))).approved == [
            True, False, True
        ]


class TestRewind:
    """The chat rewind: what the UI is allowed to say about a cut point."""

    def test_a_rewind_names_the_entry_it_cuts_at_not_a_message_count(self):
        # The graph functions take `keep`; the wire carries `index`. The core
        # owns the thread and issued that index in the first place, so it is
        # the side that can still tell whether it means what the user saw once
        # a turn has appended to the thread.
        for command in (
            SessionRollback(session_id="s1", index=4),
            SessionFork(session_id="s1", index=4),
        ):
            assert set(command.to_envelope().payload) == {"session_id", "index"}
            assert parse(decode(encode(command.to_envelope()))) == command

    def test_a_fork_does_not_name_the_session_it_is_about_to_make(self):
        # No title, no profile, no backend: the core copies all three off the
        # source, and a front-end that could name them could fork a
        # conversation into a profile the user never chose.
        assert set(SessionFork.model_fields) == {"session_id", "index"}

    def test_the_new_session_comes_back_whole(self):
        # A fork has to be opened, so a bare id would cost either a second
        # round trip or a diff of two sidebars before the UI could show it.
        created = SessionCreated(
            row=SessionRow(
                session_id="s2",
                title="a session (fork)",
                profile="default",
                mode="build",
            ),
            reply_to="c7",
        )
        back = parse(decode(encode(created.to_envelope())))
        assert back == created
        assert back.row.session_id == "s2"
        assert back.reply_to == "c7"


class TestPeek:
    def test_a_peek_asks_by_watch_id_like_a_drop_does(self):
        peek = WatchPeek(watch_id=3)
        assert set(peek.to_envelope().payload) == {"watch_id"}
        assert parse(decode(encode(peek.to_envelope()))) == peek

    def test_the_tail_answers_the_box_it_was_asked_of(self):
        # Not a `notify`: two boxes can be peeked in a row and a job peek costs
        # an squeue call, so the answer has to say which box it belongs to.
        peeked = WatchPeeked(
            watch_id=3,
            title="train.log",
            text="epoch 4/10\nloss 0.31\n…",
            reply_to="c7",
        )
        raw = encode(peeked.to_envelope())
        assert raw.count(b"\n") == 1  # a log tail is multi-line by nature
        assert parse(decode(raw)) == peeked

    def test_a_peek_carries_no_severity_and_no_timeout(self):
        # How long a tail stays on screen is the renderer's decision; a core
        # reading a file has no business setting it.
        assert set(WatchPeeked.model_fields) == {
            "watch_id", "title", "text", "reply_to"
        }


class TestQueuedMessages:
    def test_a_queued_message_is_an_ordinary_chat_entry(self):
        # No queue-specific event: a message typed ahead is a chat row like any
        # other, and `kind` is what makes it look like one that has not run.
        entry = Entry(kind="queued", text="and then plot it")
        event = ChatAppend(session_id="s1", entry=entry)
        assert parse(decode(encode(event.to_envelope()))) == event
        assert not [name for name in EVENTS if name.startswith("queue.")]

    def test_a_queued_entry_is_not_a_thread_message(self):
        # index stays -1: nothing has been written to the thread yet, so there
        # is no cut point and the rewind must not be offered on it. Taking it
        # back is `turn.unqueue`, which needs no thread surgery at all.
        assert Entry(kind="queued", text="x").index == -1

    def test_cancelling_names_the_message_by_its_row(self):
        # By the seq the core put on its queued row: not by text (the same
        # message queued twice must lose one copy rather than both) and not by
        # position (the turn ahead can finish while the dialog is open, and a
        # position then quietly names the neighbour instead).
        cancel = TurnUnqueue(session_id="s1", seq=7)
        assert set(cancel.to_envelope().payload) == {"session_id", "seq"}
        assert parse(decode(encode(cancel.to_envelope()))) == cancel

    def test_the_cancelled_text_comes_back_to_be_edited(self):
        # Cancelling lands where an interrupt lands: the text in the entry box.
        undone = TurnUnqueued(session_id="s1", seq=7, text="and then plot it")
        assert parse(decode(encode(undone.to_envelope()))) == undone


class TestTheMessageAnInterruptRecovers:
    """`turn.interrupted`: the stopped turn's message, handed back.

    The queue's twin, and separate from it for the reason written on the
    class — the two mean different things about the rows already on screen.
    """

    def test_it_carries_the_text_and_the_session_it_belongs_to(self):
        # Addressed, because the answer can arrive after the user has switched
        # away: the message waits as *that* session's draft, not as the
        # visible one's.
        handed_back = TurnInterrupted(session_id="s1", text="draft with a typo")
        assert handed_back.session_id == "s1"
        assert parse(decode(encode(handed_back.to_envelope()))) == handed_back

    def test_it_names_no_row(self):
        # A `chat.reset` has already taken the abandoned attempt's rows off the
        # screen; a seq here would be a field every client had to ignore.
        assert set(TurnInterrupted.model_fields) == {
            "session_id",
            "text",
            "reply_to",
        }

    def test_it_is_not_the_queues_event(self):
        # One handler for both would draw the queue's "drop that row" over a
        # chat that has just been reset.
        assert TurnInterrupted.TYPE != TurnUnqueued.TYPE


class TestTheTurnsClock:
    def test_the_start_says_when(self):
        # "How long since I sent it" starts here, not at the first activity
        # report: the wait for the backend's first answer is the longest
        # silence in a turn, and it belongs on the clock.
        started = TurnStarted(
            session_id="s1", started_at="2026-08-20T10:00:00+00:00"
        )
        assert parse(decode(encode(started.to_envelope()))) == started

    def test_a_start_with_no_stamp_still_parses(self):
        # Defaulted, so a core that cannot read a clock still announces turns.
        assert TurnStarted(session_id="s1").started_at == ""


class TestChatAddressing:
    """Naming a chat row, so a later frame can revise it in place.

    Without this the only way to change a row that is already drawn is to send
    the whole transcript again — the per-turn rebuild this protocol exists to
    delete (§4.2 property 1).
    """

    def test_a_row_carries_the_name_the_core_gave_it(self):
        entry = Entry(kind="assistant", text="done", seq=4)
        assert parse(decode(encode(ChatAppend(
            session_id="s1", entry=entry
        ).to_envelope()))).entry.seq == 4

    def test_an_unnamed_row_is_the_default(self):
        # 0, like `Envelope.seq`, means "not numbered": an entry a test or a
        # renderer built, never one the core sent.
        assert Entry(kind="user", text="hi").seq == 0

    def test_an_update_carries_the_whole_row_and_names_itself(self):
        # No separate id field on the event: the entry it carries is the row,
        # and `seq` inside it is which row. One place to get it wrong instead
        # of two that can disagree.
        update = ChatUpdate(
            session_id="s1",
            entry=Entry(
                kind="thinking",
                text="— step —\nread_file: 40 lines",
                seq=4,
                steps=1,
                parts=[Part(kind="call", text="read_file", tool="read_file",
                            result="40 lines", done=True)],
            ),
        )
        assert set(update.to_envelope().payload) == {
            "session_id", "entry", "reply_to"
        }
        assert parse(decode(encode(update.to_envelope()))) == update

    def test_a_call_row_is_filled_in_under_its_own_seq(self):
        # `Part.done` already documents this shape: the same part comes again
        # with its result rather than a second row appearing below.
        running = Entry(kind="thinking", text="read_file", seq=4,
                        parts=[Part(kind="call", text="read_file")])
        finished = running.model_copy(update={
            "parts": [Part(kind="call", text="read_file", result="40 lines",
                           done=True)]
        })
        assert finished.seq == running.seq
        assert parse(decode(encode(ChatUpdate(
            session_id="s1", entry=finished
        ).to_envelope()))).entry.parts[0].done

    def test_a_queued_row_keeps_its_seq_when_its_turn_starts(self):
        # The promotion is an update, not a second row: same seq, new kind.
        queued = Entry(kind="queued", text="and then plot it", seq=9)
        started = queued.model_copy(update={"kind": "user"})
        assert (started.seq, started.text) == (queued.seq, queued.text)

    def test_a_reset_carries_the_names_the_updates_will_use(self):
        # The snapshot has to agree with the deltas that follow, or a reopened
        # session cannot be updated at all. Every entry brings its own seq, so
        # nothing has to be derived from list position.
        reset = ChatReset(
            session_id="s1",
            entries=[
                Entry(kind="user", text="q1", seq=1, index=0),
                Entry(kind="assistant", text="a1", seq=2, index=1),
            ],
        )
        back = parse(decode(encode(reset.to_envelope())))
        assert [e.seq for e in back.entries] == [1, 2]

    def test_the_row_name_is_not_the_message_index(self):
        # Two different numbers on purpose: `index` says which *thread
        # message* an entry is (and is -1 for the many entries that are not
        # one), while `seq` names the *row on screen*. A thinking entry folds
        # several messages into one row; a queued entry is a row with no
        # message at all.
        folded = Entry(kind="thinking", text="…", seq=3)
        assert (folded.seq, folded.index) == (3, -1)


class TestTheMeter:
    """What `turn.usage` has to carry for the context meter to be drawable."""

    def test_it_carries_the_speed_as_two_measurements_not_a_rate(self):
        # Two different claims: the token count is the backend's, the wall
        # clock is ours (an OpenAI-style body carries no timing, so the client
        # times the request). Dividing them is the renderer's decision, and a
        # rate cannot be turned back into "3.4s for 210 tokens".
        usage = TurnUsage(
            session_id="s1",
            prompt_tokens=12_000,
            max_model_len=32_768,
            completion_tokens=210,
            request_seconds=3.4,
        )
        back = parse(decode(encode(usage.to_envelope())))
        assert back == usage
        assert back.completion_tokens / back.request_seconds == pytest.approx(61.76, abs=0.1)

    def test_a_prompt_size_with_no_generation_behind_it_still_parses(self):
        # A session restated after a backend switch has a window and a prompt
        # size and nothing generated since; the speed is unknown, not zero.
        usage = TurnUsage(session_id="s1", prompt_tokens=12_000)
        assert (usage.completion_tokens, usage.request_seconds) == (0, None)
        assert parse(decode(encode(usage.to_envelope()))) == usage


class TestNotifyTitle:
    def test_a_toast_can_carry_a_headline_over_its_body(self):
        # The core's longer answers are a heading plus a block — a profile's
        # skills, a summary /compact just wrote — and glueing the heading onto
        # the front of `text` loses the renderer's ability to tell them apart.
        toast = Notify(
            severity="warning",
            title="Thinking: xhigh — NOT USABLE",
            text="any turn that writes a file is likely to be lost…",
            timeout=25,
        )
        assert parse(decode(encode(toast.to_envelope()))) == toast

    def test_most_toasts_have_no_title(self):
        # A hint, like `timeout`: one line of news needs no heading, and a
        # front-end with nowhere to put one may ignore it.
        assert Notify(text="Deleted “notes”").title == ""


class TestWireFormat:
    def test_the_frame_is_a_flat_json_object(self):
        raw = encode(submit().to_envelope(seq=1, id="c1"))
        obj = json.loads(raw)
        assert set(obj) == {"seq", "id", "type", "payload"}
        assert obj["type"] == "turn.submit"
        assert obj["payload"]["text"] == "which BAMs are in the cohort?"

    def test_non_ascii_survives_the_utf8_round_trip(self):
        message = Notify(text="job ✗ failed · 3 nodes")
        assert parse(decode(encode(message.to_envelope()))).text == message.text


class TestTheLLMCatalog:
    """`llm.catalog`: the configured backends, drawable without the keys.

    What this closes is not a rough edge but an absence — nothing carried the
    catalog to a front-end, so the new-session picker and the manage-LLMs
    screen could only be populated by a demo, and §4.2 rule 2 (the UI never
    reads the core's state) leaves an event as the only way to fill them.
    """

    def test_an_entry_carries_what_a_row_draws_and_never_the_key(self):
        assert set(LLMEntry.model_fields) == {
            "label",
            "model",
            "base_url",
            "max_model_len",
            "needs_key",
            "active",
            "reachable",
            "discovered",
        }
        # The precedent is `SessionRow.model`: a name rather than the entry,
        # because shipping the entry would put an api_key on the wire to
        # render one line.
        assert "api_key" not in LLMEntry.model_fields

    def test_a_key_cannot_be_smuggled_in_as_an_extra_field(self):
        with pytest.raises(ProtocolError):
            parse(
                Envelope(
                    type="llm.catalog",
                    payload={
                        "entries": [
                            {
                                "label": "qwen",
                                "model": "qwen",
                                "api_key": "sk-secret",
                            }
                        ]
                    },
                )
            )

    def test_a_catalog_round_trips(self):
        entry = LLMEntry(
            label="qwen3-32b",
            model="qwen3-32b",
            base_url="http://node07:20001/v1",
            max_model_len=32768,
            needs_key=True,
            active=True,
            reachable=True,
        )
        catalog = LLMCatalog(entries=[entry], probed=True)
        assert parse(decode(encode(catalog.to_envelope()))).entries == [entry]

    def test_reachability_is_three_states_not_two(self):
        # None is "nobody has asked yet". A probe costs a round trip to a
        # cluster node, so the first frame answers with nothing known — and a
        # client that drew ○ for None would libel every backend until the
        # scan lands.
        assert LLMEntry(label="q", model="q").reachable is None
        assert LLMCatalog().probed is False

    def test_the_entry_identity_is_the_label_a_command_sends_back(self):
        # `session.new` names a backend by this string and by nothing else.
        entry = LLMEntry(label="qwen3-32b @ node07:20001", model="qwen3-32b")
        assert SessionNew(profile="hpc", backend=entry.label).backend == entry.label


class TestTheProfileListing:
    """`profile.list` / `profile.rows`: an event where inference used to be."""

    def test_a_row_carries_what_the_profiles_screen_draws(self):
        assert set(ProfileRow.model_fields) == {
            "name",
            "memories",
            "copied_from",
            "is_default",
            "working",
        }

    def test_the_rows_round_trip(self):
        row = ProfileRow(
            name="bioinformatics", memories=12, copied_from="default", working=True
        )
        assert parse(decode(encode(ProfileRows(rows=[row]).to_envelope()))).rows == [
            row
        ]

    def test_the_default_and_the_working_profile_are_different_questions(self):
        # The ★ marks the profile a deleted one's sessions fall back to; the
        # other marks what the core is running under right now. Usually not
        # the same profile, so one flag could not answer both.
        row = ProfileRow(name="hpc", working=True)
        assert (row.is_default, row.working) == (False, True)


class TestTheReadPaths:
    """The four editors' `get` half — the invariant, not a convenience.

    Every one of these has had a *write* command since §4.1 and no way to read
    what it was about to write over, so a front-end read the app dir itself,
    which is precisely the rule §4.2 spends its second paragraph on. The shape
    they all share is that the body is verbatim and that "empty" and
    "unreadable" are told apart, because the save behind them truncates.
    """

    def test_the_read_names_the_same_thing_the_write_does(self):
        # Same two words, so an editor cannot fetch one file and save another.
        assert set(ProfileGet.model_fields) == {"name", "kind"}

    def test_a_body_is_told_apart_from_a_body_that_could_not_be_read(self):
        empty = ProfileBody(name="default", kind="archive")
        broken = ProfileBody(
            name="default", kind="memories", error="permission denied"
        )
        assert (empty.text, empty.error) == ("", "")
        assert broken.text == "" and broken.error
        # And that is the whole difference an editor may open on: identical
        # text, opposite verdicts.

    def test_a_profile_body_round_trips(self):
        body = ProfileBody(
            name="hpc", kind="memories", text="## [rag]\n- lives in /data\n"
        )
        assert parse(decode(encode(body.to_envelope()))).text == body.text

    def test_only_the_two_editable_kinds_are_addressable(self):
        # Not a path: a front-end may name these two files and nothing else.
        with pytest.raises(ProtocolError):
            parse(
                Envelope(
                    type="profile.get",
                    payload={"name": "default", "kind": "/etc/passwd"},
                )
            )

    def test_a_skill_listing_draws_a_menu_and_carries_no_bodies(self):
        assert set(SkillRow.model_fields) == {"name", "description", "level"}
        rows = SkillRows(
            profile="hpc", skills=[SkillRow(name="qc", description="run QC")]
        )
        assert "text" not in rows.model_dump_json()

    def test_listing_and_writing_have_different_scopes(self):
        """A front-end may be told about a skill it may not write.

        `skill.list` reports four levels — the shipped ones included, or a
        fresh install's menu could not offer `/plan`. `skill.save` names three:
        `builtin` is package data, and a write that could name it would edit
        the install.
        """
        from hpca.protocol import SkillLevel, SkillOrigin
        from typing import get_args

        assert set(get_args(SkillOrigin)) == {
            "builtin",
            "global",
            "profile",
            "project",
        }
        assert set(get_args(SkillLevel)) == set(get_args(SkillOrigin)) - {
            "builtin"
        }
        with pytest.raises(ProtocolError):
            parse(
                Envelope(
                    type="skill.save",
                    payload={"profile": "hpc", "name": "qc", "level": "builtin"},
                )
            )

    def test_a_listing_says_which_scope_it_answers(self):
        # Two screens ask the same command different questions; an answer that
        # did not say which would fill an editor's list with a menu's.
        assert SkillList(profile="hpc").scope == "own"
        wide = SkillList(profile="hpc", scope="visible")
        assert parse(decode(encode(wide.to_envelope()))).scope == "visible"
        assert SkillRows(profile="hpc").scope == "own"

    def test_a_draft_is_asked_for_and_comes_back_unwritten(self):
        # A draft is a model call, so it happens on the core's side; nothing
        # is written until the user confirms, which arrives as `skill.save`.
        ask = SkillDraft(profile="hpc", request="watch a jupyter run")
        assert parse(decode(encode(ask.to_envelope()))).request == ask.request
        assert ask.session_id is None
        drafted = SkillDrafted(
            profile="hpc", request=ask.request, name="watch-run", body="1. go"
        )
        assert parse(decode(encode(drafted.to_envelope()))).name == "watch-run"
        assert drafted.error == "", "empty error is what makes it a draft"

    def test_a_failed_draft_is_still_an_answer(self):
        # The empty form opens behind it, so silence is the one thing this
        # event may not be.
        failed = SkillDrafted(profile="hpc", request="x", error="no backend")
        assert failed.name == "" and failed.error

    def test_the_command_counts_are_a_mapping(self):
        counts = CommandCounts(counts={"compact": 3, "memorize": 1})
        assert parse(decode(encode(counts.to_envelope()))).counts["compact"] == 3
        assert CommandCounts().counts == {}, "a core that has counted nothing"
        assert "counts" not in set(CommandList.model_fields), "it asks, it tells"

    def test_a_skill_body_is_fetched_one_at_a_time(self):
        body = SkillBody(profile="hpc", name="qc", text="---\nname: qc\n---\n")
        assert parse(decode(encode(body.to_envelope()))).text == body.text
        assert set(SkillGet.model_fields) == {"profile", "name"}

    def test_the_settings_cross_as_text_not_as_a_model(self):
        # Text both ways, because the file is what is edited: parsing it into
        # a model here would make the protocol carry a second copy of the
        # settings schema, and normalising it would rewrite what the user is
        # looking at.
        assert set(SettingsSave.model_fields) == {"text"}
        assert set(SettingsBody.model_fields) == {"text", "error", "reply_to"}

    def test_a_refused_save_says_why_and_still_describes_the_file(self):
        refused = SettingsBody(text="{}", error="invalid: llm.model — required")
        assert refused.error and refused.text == "{}"


class TestNamingABackend:
    """`backend.set` by label — what a picker can actually send back.

    The catalog carries no api_key by design, so a front-end handed one row
    cannot reconstruct the entry it names. With only the by-value form, ctrl+l
    could not switch a session to a key-locked backend at all.
    """

    def test_a_backend_can_be_named_by_its_catalog_label(self):
        entry = LLMEntry(label="qwen3-32b @ node07:20001", model="qwen3-32b")
        command = BackendSet(label=entry.label, session_id="s1")
        assert command.label == entry.label and command.backend == {}

    def test_the_whole_entry_form_survives(self):
        # A hand-filled connection form names a backend that is in no catalog
        # yet, so there is no label for it to be named by.
        command = BackendSet(
            backend={"model": "qwen3-32b", "base_url": "http://h/v1"}
        )
        assert command.label == "" and command.backend["model"] == "qwen3-32b"

    def test_removal_names_the_entry_the_same_way(self):
        assert set(protocol.BackendRemove.model_fields) == {"label"}


class TestProbingAndScanning:
    def test_a_probe_carries_the_key_out_and_never_back(self):
        assert set(protocol.BackendProbe.model_fields) == {"base_url", "api_key"}
        answered = BackendProbed(
            base_url="http://h/v1",
            models=[LLMEntry(label="qwen", model="qwen", needs_key=True)],
        )
        assert "api_key" not in answered.model_dump_json()

    def test_three_probe_outcomes_without_a_status_field(self):
        nothing = BackendProbed(base_url="http://h/v1")
        locked = BackendProbed(base_url="http://h/v1", needs_key=True)
        served = BackendProbed(
            base_url="http://h/v1", models=[LLMEntry(label="q", model="q")]
        )
        assert (nothing.models, nothing.needs_key) == ([], False)
        assert (locked.models, locked.needs_key) == ([], True)
        assert served.models and served.needs_key is False

    def test_an_empty_scan_carries_the_verdict_not_just_the_count(self):
        # "Nothing found" means three different things, and which one it is
        # depends on probes only the core runs.
        quiet = BackendScanned(found=0, cluster=1)
        nothing_new = BackendScanned(found=0, notice="Nothing new on localhost")
        offcluster = BackendScanned(found=0, help="ssh -fN -L ...")
        assert (quiet.notice, quiet.help) == ("", "")
        assert nothing_new.notice and not nothing_new.help
        assert offcluster.help and not offcluster.notice

