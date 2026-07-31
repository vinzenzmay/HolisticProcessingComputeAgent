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
    PROTOCOL_VERSION,
    ChatAppend,
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
    SessionRows,
    TurnSubmit,
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
    "session.focus",
    "turn.submit",
    "turn.interrupt",
    "decision.resolve",
    "command.run",
    "confirm.resolve",
    "memory.resolve",
    "mode.set",
    "backend.set",
    "profile.set",
    "profile.save",
    "profile.create",
    "profile.delete",
    "profile.duplicate",
    "skill.save",
    "skill.delete",
    "process.kill",
    "job.cancel",
    "watch.drop",
    "shutdown",
}

SPEC_EVENTS = {
    "hello",
    "session.rows",
    "chat.reset",
    "chat.append",
    "turn.started",
    "turn.activity",
    "turn.usage",
    "turn.finished",
    "turn.failed",
    "decision.requested",
    "decision.cleared",
    "panel.update",
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
        # it stays in step, so the shape is asserted.
        assert set(Entry.model_fields) == {
            field.name for field in dataclasses.fields(transcript.Entry)
        }

    def test_part_mirrors_a_transcript_step(self):
        assert set(Part.model_fields) == {
            field.name for field in dataclasses.fields(transcript.Step)
        }

    def test_a_panel_row_carries_what_the_ui_needs_to_act_on_it(self):
        assert set(PanelRow.model_fields) == {
            "key", "text", "classes", "title", "kind", "ref"
        }

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
