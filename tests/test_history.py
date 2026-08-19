"""Tests for hpca.agent.history: the two messages one tool call adds (§4.3).

The model has to see its own call in the shape it was trained on — an
assistant message carrying the call, then the result answering it — and it has
to see it *whole* while the job it belongs to is still running. What the window
cannot afford is carrying every payload it ever wrote forever, so the record is
stored intact and thinned by age, once the model has moved on to something
else.
"""

import json

from pydantic import BaseModel, Field

from hpca.agent.history import (
    ELISION_CLOSE,
    ELISION_SENTINEL,
    KEEP_RECENT_CALLS,
    MAX_LIST_CHARS,
    MAX_STRING_CHARS,
    call_json,
    carries_elision_marker,
    elide,
    elide_arguments,
    fold_old_payloads,
    is_tool_call_message,
    native_call_message,
    omitted_list,
    result_message,
    tool_call_message,
    tool_exchange,
)
from hpca.agent.middleware import decision_schema
from hpca.agent.tools import Tool, ToolRegistry


def _payload(tag: str, count: int = 120) -> list[str]:
    """A payload list well past ``MAX_LIST_CHARS``, tagged per call.

    The tag matters: when a fold is supposed to have kept a payload, or
    supposed to have kept no line of one, the assertion has to be able to say
    *which* call's lines it is looking at, and interchangeable "line 3"s
    cannot.
    """
    return [f"{tag} line {index:03d} " + "-" * 40 for index in range(count)]


def _arguments(message: dict) -> dict:
    """The arguments of a stored call, whichever protocol carries them.

    Both encodings hold the same record; a test about folding should not have
    to care which one it is reading.
    """
    if message.get("tool_calls"):
        return json.loads(message["tool_calls"][0]["function"]["arguments"])
    return json.loads(message["content"])["arguments"]


def _history(payloads: list[list[str]], *, call_id: str = "") -> list[dict]:
    """A conversation of one create_file call per payload, results included."""
    messages: list[dict] = [{"role": "user", "content": "write the files"}]
    for index, lines in enumerate(payloads):
        messages += tool_exchange(
            "create_file",
            {"path": f"f{index}.md", "content_lines": lines},
            "written",
            call_id=f"{call_id}{index}" if call_id else "",
        )
    return messages


class TestExchangeShape:
    def test_two_messages_call_then_result(self):
        messages = tool_exchange("read_file", {"registry_key": "cohort"}, "12 lines")
        assert [m["role"] for m in messages] == ["assistant", "user"]

    def test_the_assistant_message_is_the_decision_envelope(self):
        messages = tool_exchange("read_file", {"registry_key": "cohort"}, "ok")
        assert json.loads(messages[0]["content"]) == {
            "action": "tool_call",
            "tool": "read_file",
            "arguments": {"registry_key": "cohort"},
        }

    def test_the_envelope_matches_what_the_schema_constrains(self):
        # The point of the assistant copy is that it is what the model itself
        # would have written; if the envelope drifts from decision_schema the
        # history teaches the model a format it is not allowed to emit.
        class EchoParams(BaseModel):
            text: str = Field(description="Text to echo")

        async def echo(args, ctx):
            return "ok"

        tools = ToolRegistry()
        tools.register(
            Tool(name="echo", description="Echo", params=EchoParams, handler=echo)
        )
        branch = next(
            b
            for b in decision_schema(tools)["anyOf"]
            if b["properties"]["action"]["const"] == "tool_call"
        )
        envelope = json.loads(tool_exchange("echo", {"text": "hi"}, "ok")[0]["content"])
        assert set(envelope) == set(branch["required"])
        assert envelope["tool"] == branch["properties"]["tool"]["const"]

    def test_the_result_message_is_unchanged_from_before(self):
        # Byte-identical to what the graph has always written: the model has
        # learned this prefix, and the transcript keys tool steps off it.
        messages = tool_exchange("run_bash", {}, "exit 0\nhello")
        assert messages[1] == {
            "role": "user",
            "content": "[tool result] run_bash: exit 0\nhello",
        }

    def test_non_ascii_survives_the_serialization(self):
        messages = tool_exchange("echo", {"text": "µ-Ansatz — 5 °C"}, "ok")
        assert "µ-Ansatz — 5 °C" in messages[0]["content"]

    def test_the_call_is_recognisable_as_a_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok")
        assert is_tool_call_message(call)
        assert not is_tool_call_message(result)

    def test_a_real_answer_is_not_a_call(self):
        assert not is_tool_call_message(
            {"role": "assistant", "content": 'The file starts {"action": "tool_call"}'}
        )
        assert not is_tool_call_message({"role": "assistant", "content": "Four BAMs."})


class TestCallsAreStoredWhole:
    """The record of a call is written verbatim, however large the payload.

    This is the half of the design that used to be wrong, and it earns its own
    tests because nothing downstream can repair it. Elision happened here, at
    write time: by the time the model composed its next move, its record of the
    file it had *just* written was already a summary of that file. A model
    mid-job reaches for exactly that record — asked to rewrite what it had
    written a moment earlier, it reproduced the summary as the new
    ``content_lines``, so the marker landed on disk and the file collapsed to
    whatever the summary had kept. The next rewrite elided *that*. Nothing in
    the tools noticed, and the model concluded its own writer was truncating
    and burned a dozen rounds bisecting a bug that did not exist.

    So the writer no longer has an opinion about size. Thinning is a property
    of the view (:func:`fold_old_payloads`), which is what makes it reversible
    as the conversation moves on, and what lets the transcript the user reads
    and the checkpoint on disk keep every line."""

    def test_a_call_written_now_keeps_every_line(self):
        lines = _payload("notes")
        content = call_json("create_file", {"path": "notes.md", "content_lines": lines})
        assert json.loads(content)["arguments"]["content_lines"] == lines
        # Nothing stands in for anything here: had the writer folded, this is
        # the sentinel that would be sitting in the record instead.
        assert ELISION_SENTINEL not in content

    def test_the_message_the_graph_stores_carries_the_payload_verbatim(self):
        lines = _payload("script")
        message = tool_call_message("create_file", {"content_lines": lines})
        assert _arguments(message)["content_lines"] == lines
        assert lines[0] in message["content"]
        assert lines[-1] in message["content"]

    def test_the_native_copy_is_stored_whole_too(self):
        # Same record, other encoding. The protocol decides where arguments
        # sit, never whether they are worth keeping.
        lines = _payload("native")
        message = native_call_message("create_file", {"content_lines": lines}, "c1")
        assert _arguments(message)["content_lines"] == lines

    def test_the_exchange_the_graph_appends_is_whole(self):
        lines = _payload("exchange")
        call, _ = tool_exchange("create_file", {"content_lines": lines}, "written")
        assert _arguments(call)["content_lines"] == lines


class TestFoldingOldPayloads:
    """``fold_old_payloads`` is where the window gets paid for instead.

    Recency is the whole mechanism. A model that reproduces its own record does
    it *immediately* — on the next call, while it is still finishing the job
    the record belongs to — so the record it reaches for is never an old one.
    Keeping the most recent ``KEEP_RECENT_CALLS`` intact means there is nothing
    wrong to copy at the only moment copying happens, and the window still
    never carries more than a handful of payloads at once. Older calls are
    replaced by a description of what they wrote, because by then the model
    only needs to know that a file exists and roughly how large it is; the
    content is one ``read_file`` away, and the file, unlike the record, is not
    a copy.

    The fold is a *view*. The stored history is never rewritten — the same rule
    compaction follows — so a call folded on this turn is still whole in the
    transcript, still whole in the checkpoint, and whole again in the model's
    view if the conversation ever comes back to it."""

    def test_the_three_most_recent_calls_keep_their_payloads(self):
        payloads = [_payload(f"file{i}") for i in range(6)]
        folded = fold_old_payloads(_history(payloads))
        calls = [m for m in folded if is_tool_call_message(m)]
        assert len(calls) == 6  # nothing was dropped on the way through
        kept = [_arguments(m)["content_lines"] for m in calls[-KEEP_RECENT_CALLS:]]
        assert kept == payloads[-KEEP_RECENT_CALLS:]

    def test_everything_older_than_that_is_folded(self):
        payloads = [_payload(f"file{i}") for i in range(6)]
        folded = fold_old_payloads(_history(payloads))
        calls = [m for m in folded if is_tool_call_message(m)]
        for index, message in enumerate(calls[:-KEEP_RECENT_CALLS]):
            arguments = _arguments(message)
            assert arguments["path"] == f"f{index}.md"  # the target always survives
            assert isinstance(arguments["content_lines"], str)
            assert arguments["content_lines"].startswith(ELISION_SENTINEL)

    def test_a_folded_payload_carries_no_line_of_the_original(self):
        # The regression that cost a session's worth of files: any real line
        # left in the record is a line the model can paste back as content, so
        # there must be none of them, head or tail.
        payloads = [_payload(f"file{i}") for i in range(6)]
        oldest = next(
            m for m in fold_old_payloads(_history(payloads)) if is_tool_call_message(m)
        )
        descriptor = _arguments(oldest)["content_lines"]
        assert not any(line in descriptor for line in payloads[0])
        assert descriptor.endswith(ELISION_CLOSE)

    def test_a_folded_payload_says_how_much_it_stands_for(self):
        # What is left has to be enough to reason about the file without
        # re-reading it: how many lines went in, and how big they were.
        payloads = [
            _payload(f"file{i}", count=n) for i, n in enumerate([73, 120, 120, 120])
        ]
        oldest = next(
            m for m in fold_old_payloads(_history(payloads)) if is_tool_call_message(m)
        )
        descriptor = _arguments(oldest)["content_lines"]
        assert "73 lines" in descriptor
        assert f"{sum(len(line) for line in payloads[0])} chars" in descriptor

    def test_a_payload_under_the_budget_is_never_folded_however_old(self):
        # The threshold is real, not decorative. An ordinary edit — twenty
        # lines of code — costs a few hundred characters of window, and
        # mangling the record of it to save that would be a bad trade even
        # long after the call.
        small = [f"line {i}" for i in range(20)]
        assert sum(len(line) for line in small) < MAX_LIST_CHARS
        payloads = [small] + [_payload(f"file{i}") for i in range(5)]
        oldest = next(
            m for m in fold_old_payloads(_history(payloads)) if is_tool_call_message(m)
        )
        assert _arguments(oldest)["content_lines"] == small

    def test_the_native_protocol_folds_too(self):
        payloads = [_payload(f"file{i}") for i in range(5)]
        folded = fold_old_payloads(_history(payloads, call_id="c"))
        calls = [m for m in folded if is_tool_call_message(m)]
        oldest = json.loads(calls[0]["tool_calls"][0]["function"]["arguments"])
        assert oldest["content_lines"].startswith(ELISION_SENTINEL)
        # …and the id still ties the call to the result answering it, which is
        # the one thing this encoding cannot afford to lose in a rewrite.
        assert calls[0]["tool_calls"][0]["id"] == "c0"
        newest = json.loads(calls[-1]["tool_calls"][0]["function"]["arguments"])
        assert newest["content_lines"] == payloads[-1]

    def test_non_call_messages_pass_through_untouched(self):
        # A result is not a call, and neither is an answer: folding either
        # would be thinning what the model was *told*, not what it did.
        messages = [
            {"role": "user", "content": "write the file"},
            *tool_exchange("create_file", {"content_lines": _payload("f")}, "written"),
            {"role": "assistant", "content": "Written, 120 lines."},
            {"role": "user", "content": "thanks"},
        ]
        folded = fold_old_payloads(messages, keep_recent=0)
        assert [m for i, m in enumerate(folded) if i != 1] == [
            m for i, m in enumerate(messages) if i != 1
        ]

    def test_the_stored_history_is_not_mutated(self):
        # The list handed in is AgentState's own, and it is also what the
        # transcript and the checkpoint read. A fold that wrote through it
        # would destroy the record it is supposed to be a view of.
        payloads = [_payload(f"file{i}") for i in range(5)]
        envelopes = _history(payloads)
        natives = _history(payloads, call_id="c")
        fold_old_payloads(envelopes)
        fold_old_payloads(natives)
        assert _arguments(envelopes[1])["content_lines"] == payloads[0]
        assert _arguments(natives[1])["content_lines"] == payloads[0]

    def test_a_call_it_cannot_read_survives_unchanged(self):
        # A fold is an optimisation; losing a message to a parse error would
        # not be one, and even an unreadable call still tells the model it
        # acted rather than answered.
        unreadable = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "not json at all"},
                }
            ],
        }
        argumentless = {
            "role": "assistant",
            "content": json.dumps({"action": "tool_call", "tool": "echo"}),
        }
        messages = [
            unreadable,
            argumentless,
            *tool_exchange("echo", {"text": "x"}, "ok"),
        ]
        folded = fold_old_payloads(messages, keep_recent=0)
        assert len(folded) == len(messages)
        assert folded[0] == unreadable
        assert folded[1] == argumentless

    def test_keep_recent_zero_folds_everything(self):
        # The boundary: an empty "keep" has to mean keep nothing, not slice
        # from the end of the list and so keep everything.
        payloads = [_payload(f"file{i}") for i in range(3)]
        calls = [
            m
            for m in fold_old_payloads(_history(payloads), keep_recent=0)
            if is_tool_call_message(m)
        ]
        assert len(calls) == 3
        assert all(
            _arguments(m)["content_lines"].startswith(ELISION_SENTINEL) for m in calls
        )

    def test_a_history_without_calls_comes_back_as_it_went_in(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Four BAMs."},
        ]
        assert fold_old_payloads(messages) == messages


class TestWhatAnOmissionLooksLike:
    """What a folded payload is replaced *by* is the part that was got wrong
    once. The record used to keep a head of the real lines and close it with
    "... 97 more lines elided ...", which reads exactly like file content — and
    a model rewriting the file copied its own record back, marker and all. So
    an omission is named as one: a descriptor wrapped in a sentinel no real
    file line carries, holding no fragment anyone could paste back as content.

    Recency is what prevents the copy-back now, so the descriptor no longer
    argues with the model about it — the long "must never be sent back as
    content" warning is gone, and what is left is a statement of the shape of
    what was written."""

    def test_a_list_past_the_char_budget_becomes_a_descriptor_of_it(self):
        lines = _payload("notes")
        elided = elide_arguments({"path": "notes.md", "content_lines": lines})
        assert elided["path"] == "notes.md"  # the target is never touched
        descriptor = elided["content_lines"]
        # A string, not a shortened list: nothing about its shape invites the
        # model to treat it as the lines it stands in for.
        assert isinstance(descriptor, str)
        assert descriptor.startswith(ELISION_SENTINEL)
        assert descriptor.endswith(ELISION_CLOSE)
        assert f"{len(lines)} lines" in descriptor  # it says what it left out

    def test_the_descriptor_carries_no_line_of_the_payload(self):
        lines = _payload("notes")
        descriptor = elide_arguments({"content_lines": lines})["content_lines"]
        assert not any(line in descriptor for line in lines)

    def test_the_descriptor_is_recognisable_if_it_comes_back(self):
        # The file tools still refuse content carrying the marker — a cheap
        # backstop now rather than the mechanism — and that guard is only as
        # good as the marker being detectable in what the fold emits.
        assert carries_elision_marker(omitted_list(_payload("notes")))

    def test_a_list_at_the_budget_is_left_whole(self):
        lines = ["x" * 100] * (MAX_LIST_CHARS // 100)
        assert elide_arguments({"content_lines": lines})["content_lines"] == lines

    def test_one_character_over_the_budget_folds(self):
        lines = ["x" * 100] * (MAX_LIST_CHARS // 100) + ["y"]
        elided = elide_arguments({"content_lines": lines})["content_lines"]
        assert isinstance(elided, str)
        assert f"{len(lines)} lines" in elided

    def test_a_long_string_keeps_its_head_and_names_the_omission(self):
        # A string, unlike a list, keeps its head: a cut string still reads as
        # a fragment, where a head of plausible file lines reads as the file.
        text = "x" * (MAX_STRING_CHARS + 50)
        elided = elide_arguments({"script": text})["script"]
        assert elided.startswith("x" * MAX_STRING_CHARS)
        assert ELISION_SENTINEL in elided
        assert "50 more chars" in elided
        assert elided.endswith(ELISION_CLOSE)
        assert carries_elision_marker(elided)

    def test_a_string_at_the_budget_is_left_alone(self):
        text = "x" * MAX_STRING_CHARS
        assert elide_arguments({"script": text})["script"] == text

    def test_a_short_string_is_left_alone(self):
        assert elide_arguments({"path": "/data/x.bam"}) == {"path": "/data/x.bam"}

    def test_other_scalars_pass_through(self):
        arguments = {"timeout_s": 60, "recursive": True, "start": None}
        assert elide_arguments(arguments) == arguments

    def test_a_payload_nested_one_level_down_is_folded_too(self):
        # Burying the payload in a sub-object is not a way around the budget.
        lines = _payload("edit")
        elided = elide_arguments({"edit": {"new_lines": lines}})["edit"]["new_lines"]
        assert isinstance(elided, str)
        assert f"{len(lines)} lines" in elided
        assert lines[0] not in elided

    def test_a_long_line_inside_a_kept_list_is_still_cut(self):
        arguments = {"content_lines": ["y" * (MAX_STRING_CHARS + 10), "short"]}
        elided = elide_arguments(arguments)["content_lines"]
        assert elided[0].startswith("y" * MAX_STRING_CHARS)
        assert "10 more chars" in elided[0]
        assert elided[1] == "short"

    def test_the_originals_are_not_mutated(self):
        # The same dict is the record the chat shows (AgentState.calls); the
        # user's copy has to keep every line.
        lines = _payload("notes")
        arguments = {"content_lines": lines}
        elide_arguments(arguments)
        assert arguments["content_lines"] == lines
        assert len(lines) == 120


class TestElisionMarkerDetection:
    """``carries_elision_marker`` is what the file tools ask of every line they
    are about to write. It is a backstop rather than the defence now — recency
    is what keeps a model from quoting its own record back — but a cheap one,
    and the failure it catches is silent and destructive, so it stays. Its two
    jobs still pull against each other: catch anything the history could have
    handed the model, and never flag a line a person actually wrote, because a
    false positive refuses a legitimate write."""

    def test_the_sentinel_is_recognised(self):
        assert carries_elision_marker(omitted_list([f"line {i}" for i in range(30)]))
        assert carries_elision_marker(elide("z" * (MAX_STRING_CHARS + 1)))

    def test_the_sentinel_is_recognised_mid_line(self):
        # It comes back embedded, not alone: a model pasting its record back
        # indents it, or wraps it in the surrounding line it was rewriting.
        assert carries_elision_marker(f"  {ELISION_SENTINEL} payload of 9 lines>>")

    def test_the_legacy_wording_is_recognised(self):
        # The pre-0.23.3 marker still sits in checkpointed sessions, and in the
        # files already written from one, so it has to be caught on the way in
        # even though nothing emits it any more.
        assert carries_elision_marker("... 22 more lines elided ...")
        assert carries_elision_marker("... 97 more chars elided ...")

    def test_ordinary_file_lines_do_not_trip_it(self):
        assert not carries_elision_marker("sample_id\tcondition\treads")
        assert not carries_elision_marker("counts <- read.delim(path, sep = '\\t')")
        assert not carries_elision_marker("if a >> 2 and b < 3:")

    def test_prose_about_eliding_is_not_a_marker(self):
        # A methods paragraph is allowed to use the word; it is the shape of
        # the marker that counts, not its vocabulary.
        assert not carries_elision_marker(
            "Low-coverage samples were elided before the merge."
        )
        assert not carries_elision_marker("The elided rows are listed in Table 2.")


class TestNativeProtocol:
    """The same pair, encoded for the backend's own tool channel."""

    def test_the_call_rides_tool_calls_and_the_result_the_tool_role(self):
        call, result = tool_exchange(
            "read_file", {"registry_key": "cohort"}, "12 lines", call_id="call_1"
        )
        assert call["role"] == "assistant"
        assert call["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {
            "registry_key": "cohort"
        }
        assert result["role"] == "tool"
        assert result["content"] == "[tool result] read_file: 12 lines"

    def test_the_id_ties_the_result_to_its_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok", call_id="abc")
        assert call["tool_calls"][0]["id"] == "abc" == result["tool_call_id"]

    def test_payloads_ride_this_channel_whole_too(self):
        # Not a protocol matter either way: what a stored call keeps is decided
        # by its age, in the view, and identically for both encodings.
        lines = _payload("native")
        call = tool_exchange(
            "create_file", {"content_lines": lines}, "written", call_id="c1"
        )[0]
        arguments = call["tool_calls"][0]["function"]["arguments"]
        assert ELISION_SENTINEL not in arguments
        assert lines[-1] in arguments

    def test_a_native_call_is_recognised_as_a_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok", call_id="c1")
        assert is_tool_call_message(call)  # or the transcript renders it as a reply
        assert not is_tool_call_message(result)

    def test_content_is_passed_through_untouched(self):
        # What answers a call is also a denial or a tool error; relabelling one
        # as a result is how a refused call reads as a successful one.
        denial = result_message("delete_file", "[denied] the user said no", "c1")
        assert denial["content"] == "[denied] the user said no"
        assert denial["role"] == "tool"
        assert result_message("echo", "[tool error] boom")["role"] == "user"
