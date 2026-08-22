"""Tests for hpca.transcript: folding state into chat entries."""

from hpca.agent.hints import MODEL_HINTS
from hpca.agent.history import tool_call_message
from hpca.llm import STAMP_KEY
from hpca.transcript import (
    RESULT_RULE,
    Entry,
    build_entries,
    call_step,
    result_text,
)

USER_MSG = {"role": "user", "content": "which BAMs are in the cohort?"}
# The model's own copy of the call that produced STEP (hpca.agent.history).
CALL_MSG = tool_call_message("list_dir", {"path": "/data/cohort"})
STEP = {"role": "user", "content": "[tool result] list_dir: 12 entries"}
STEP2 = {"role": "user", "content": "[tool error] read_file: not found"}
ANSWER = {"role": "assistant", "content": "Four BAMs match."}


def kinds(entries):
    return [e.kind for e in entries]


class TestGrouping:
    def test_plain_exchange_has_no_thinking_box(self):
        entries = build_entries([USER_MSG, ANSWER], [])
        assert kinds(entries) == ["user", "assistant"]
        assert entries[0].text == "which BAMs are in the cohort?"
        assert entries[1].text == "Four BAMs match."

    def test_tool_steps_fold_into_one_box_before_the_answer(self):
        entries = build_entries([USER_MSG, STEP, STEP2, ANSWER], [])
        assert kinds(entries) == ["user", "thinking", "assistant"]
        box = entries[1]
        assert box.steps == 2
        # each result under a heading naming its tool, the "[tool result]"
        # framing dropped — the heading is what it said
        assert "— list_dir —\n12 entries" in box.text
        assert "— read_file (error) —\nnot found" in box.text

    def test_reasoning_and_steps_interleave_in_the_order_they_happened(self):
        thinking = [
            {"after": 1, "reasoning": "I should list the directory first."},
            {"after": 2, "reasoning": "Now I can answer."},
        ]
        entries = build_entries([USER_MSG, STEP, ANSWER], thinking)
        assert kinds(entries) == ["user", "thinking", "assistant"]
        box = entries[1]
        assert box.text.index("list the directory") < box.text.index("list_dir")
        assert box.text.index("list_dir") < box.text.index("Now I can answer")
        assert box.steps == 1
        assert box.reasoning_chars == len("I should list the directory first.") + len(
            "Now I can answer."
        )

    def test_reasoning_without_any_tool_step_still_boxes(self):
        thinking = [{"after": 1, "reasoning": "Simple arithmetic."}]
        entries = build_entries([USER_MSG, ANSWER], thinking)
        assert kinds(entries) == ["user", "thinking", "assistant"]
        assert entries[1].steps == 0
        assert "Simple arithmetic." in entries[1].text

    def test_each_turn_gets_its_own_box(self):
        messages = [USER_MSG, STEP, ANSWER, USER_MSG, STEP, ANSWER]
        entries = build_entries(messages, [])
        assert kinds(entries) == [
            "user", "thinking", "assistant", "user", "thinking", "assistant"
        ]

    def test_box_stays_open_when_a_turn_is_interrupted(self):
        # approval pending: steps so far, no answer yet
        entries = build_entries([USER_MSG, STEP], [])
        assert kinds(entries) == ["user", "thinking"]

    def test_empty_reasoning_is_dropped(self):
        entries = build_entries([USER_MSG, ANSWER], [{"after": 1, "reasoning": "  "}])
        assert kinds(entries) == ["user", "assistant"]

    def test_system_messages_never_shown(self):
        messages = [{"role": "system", "content": "prompt"}, USER_MSG, ANSWER]
        assert kinds(build_entries(messages, [])) == ["user", "assistant"]


class TestParts:
    """The thinking entry keeps its parts ordered and labelled, so an expanded
    box can show each as its own collapsible element."""

    def test_parts_preserve_order_and_kind(self):
        thinking = [
            {"after": 1, "reasoning": "I should list the directory first."},
            {"after": 2, "reasoning": "Now I can answer."},
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], thinking)[1]
        assert [p.kind for p in box.parts] == ["reasoning", "step", "reasoning"]
        assert box.parts[0].text == "I should list the directory first."
        # the "[tool result] list_dir:" framing is the row's header now
        assert box.parts[1].text == "12 entries"
        assert box.parts[2].text == "Now I can answer."

    def test_step_label_is_the_tool_name(self):
        box = build_entries([USER_MSG, STEP, STEP2, ANSWER], [])[1]
        assert box.parts[0].label() == "list_dir"
        # a tool error is labelled as such
        assert box.parts[1].label() == "read_file (error)"

    def test_reasoning_part_labels_as_reasoning(self):
        box = build_entries([USER_MSG, ANSWER], [{"after": 1, "reasoning": "hm"}])[1]
        assert box.parts[0].label() == "reasoning"

    def test_parts_and_folded_text_agree(self):
        # the log still writes one block; the parts are the same content, split
        box = build_entries([USER_MSG, STEP, ANSWER], [])[1]
        assert len(box.parts) == box.steps
        for part in box.parts:
            assert part.text.strip() in box.text


class TestToolCalls:
    """The call is shown next to its result: what the approval prompt shows
    for a gated call stays readable in the chat once the prompt is gone."""

    def test_call_part_precedes_its_result(self):
        calls = [
            {
                "after": 1,
                "tool": "list_dir",
                "arguments": {"path": "/data/cohort"},
            }
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        # one exchange, one part: the call on top and its result below it
        assert [p.kind for p in box.parts] == ["call"]
        assert box.parts[0].label() == "list_dir · cohort"
        assert "cohort" in box.parts[0].text
        assert box.parts[0].result == "12 entries"
        assert box.parts[0].done

    def test_call_comes_after_the_reasoning_that_produced_it(self):
        thinking = [{"after": 1, "reasoning": "List it first."}]
        calls = [{"after": 1, "tool": "list_dir", "arguments": {}}]
        box = build_entries([USER_MSG, STEP, ANSWER], thinking, calls)[1]
        assert [p.kind for p in box.parts] == ["reasoning", "call"]

    def test_script_is_carried_in_the_call_part(self):
        calls = [
            {
                "after": 1,
                "tool": "run_bash",
                "arguments": {"content_lines": ["ls /data"], "timeout_s": 30},
                "script": "ls /data",
            }
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        call = box.parts[0]
        assert "ls /data" in call.text
        # the script is the block below, not repeated as a JSON argument
        assert "content_lines" not in call.text
        # nor is the timeout: it is how the harness ran the command, not what
        # the agent is doing, and the command is the whole of the latter
        assert "timeout_s" not in call.text
        assert call.text.strip() == "ls /data"

    def test_edited_lines_are_shown_as_the_diff_not_as_json(self):
        calls = [
            {
                "after": 1,
                "tool": "edit_file",
                "arguments": {
                    "path": "/work/run.sh",
                    "old_lines": ["echo one"],
                    "new_lines": ["echo two"],
                },
                "script": "- echo one\n+ echo two",
            }
        ]
        call = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1].parts[0]
        assert "- echo one\n+ echo two" in call.text
        assert "old_lines" not in call.text and "new_lines" not in call.text
        assert "run.sh" in call.text  # which file is still an argument

    def test_details_are_shown_when_the_tool_resolved_the_call(self):
        calls = [
            {
                "after": 1,
                "tool": "delete_file",
                "arguments": {"path": "/work/scratch"},
                "details": "rm /scratch/old.bam\n(12 bytes; will be recoverable)",
            }
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert "rm /scratch/old.bam" in box.parts[0].text
        # the tool already resolved the key into that path; saying
        # "path: /work/scratch" underneath would be the same fact, worse
        assert "path:" not in box.parts[0].text

    def test_calls_do_not_inflate_the_step_count(self):
        calls = [{"after": 1, "tool": "list_dir", "arguments": {}}]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert box.steps == 1  # one tool step, shown as call + result

    def test_call_left_behind_by_a_rollback_is_passed_over(self):
        # An interrupted turn's messages are truncated; a call anchored past
        # the end has no result to sit before, and must not surface alone.
        calls = [{"after": 5, "tool": "run_bash", "script": "rm -rf /tmp/x"}]
        entries = build_entries([USER_MSG, ANSWER], [], calls)
        assert kinds(entries) == ["user", "assistant"]

    def test_call_is_written_into_the_folded_text(self):
        calls = [{"after": 1, "tool": "run_bash", "script": "ls /data"}]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert "ls /data" in box.text  # the session log writes this block

    def test_calls_are_optional(self):
        box = build_entries([USER_MSG, STEP, ANSWER], [])[1]
        assert [p.kind for p in box.parts] == ["step"]


class TestCallMessages:
    """The model's own copy of a call (hpca.agent.history) is in ``messages``
    so the history has the shape it was trained on. The user reads the call
    from its anchored record instead — unelided, with the script — so the
    message itself must never surface as a reply."""

    def test_the_models_copy_of_a_call_is_not_shown_as_an_answer(self):
        entries = build_entries([USER_MSG, CALL_MSG, STEP, ANSWER], [])
        assert kinds(entries) == ["user", "thinking", "assistant"]
        assert entries[2].text == "Four BAMs match."

    def test_it_does_not_split_the_thinking_box(self):
        # An assistant message closes the box that produced it; the call is
        # mid-working, so a box holding two calls must stay one box.
        messages = [USER_MSG, CALL_MSG, STEP, CALL_MSG, STEP2, ANSWER]
        entries = build_entries(messages, [])
        assert kinds(entries) == ["user", "thinking", "assistant"]
        assert entries[1].steps == 2

    def test_the_call_record_still_lands_before_its_result(self):
        # Anchored to the index of the model's copy, which is the message
        # right before the result it produced.
        calls = [{"after": 1, "tool": "list_dir", "arguments": {"path": "/data/cohort"}}]
        thinking = [{"after": 1, "reasoning": "List it first."}]
        box = build_entries([USER_MSG, CALL_MSG, STEP, ANSWER], thinking, calls)[1]
        assert [p.kind for p in box.parts] == ["reasoning", "call"]
        assert box.parts[1].result == "12 entries"

    def test_a_call_message_is_never_a_rewind_cut_point(self):
        # Only real user messages name one; indices stay absolute either way.
        entries = build_entries([USER_MSG, CALL_MSG, STEP, ANSWER], [])
        assert [(e.kind, e.index) for e in entries if e.index >= 0] == [
            ("user", 0),
            ("assistant", 3),
        ]

    def test_an_answer_that_merely_mentions_the_format_is_still_an_answer(self):
        quoted = {
            "role": "assistant",
            "content": 'Call it with {"action": "tool_call", "tool": "echo"} — but',
        }
        assert kinds(build_entries([USER_MSG, quoted], [])) == ["user", "assistant"]


class TestTail:
    def test_calls_outside_the_tail_are_left_out(self):
        messages = [USER_MSG, ANSWER, USER_MSG, STEP, ANSWER]
        calls = [
            {"after": 1, "tool": "first_call", "arguments": {}},
            {"after": 3, "tool": "second_call", "arguments": {}},
        ]
        entries = build_entries(messages, [], calls, start=2)
        assert "second_call" in entries[1].text
        assert "first_call" not in entries[1].text

    def test_start_selects_one_turn_keeping_absolute_anchors(self):
        messages = [USER_MSG, ANSWER, USER_MSG, STEP, ANSWER]
        thinking = [
            {"after": 1, "reasoning": "first turn"},
            {"after": 4, "reasoning": "second turn"},
        ]
        entries = build_entries(messages, thinking, start=2)
        assert kinds(entries) == ["user", "thinking", "assistant"]
        assert "second turn" in entries[1].text
        assert "first turn" not in entries[1].text

    def test_start_past_the_end_is_empty(self):
        assert build_entries([USER_MSG, ANSWER], [], start=2) == []


class TestIndex:
    """Entries that ARE a thread message carry its index, so activating one in
    the chat (rewind: fork / roll back) can name the exact cut point."""

    def test_user_and_assistant_entries_carry_their_message_index(self):
        entries = build_entries([USER_MSG, STEP, ANSWER, USER_MSG, ANSWER], [])
        by_kind = [(e.kind, e.index) for e in entries]
        assert ("user", 0) in by_kind
        assert ("assistant", 2) in by_kind
        assert ("user", 3) in by_kind
        assert ("assistant", 4) in by_kind

    def test_event_entries_carry_their_index_too(self):
        event = {"role": "user", "content": "[process 3 exited] ok"}
        entries = build_entries([USER_MSG, ANSWER, event], [])
        assert (entries[2].kind, entries[2].index) == ("event", 2)

    def test_synthetic_entries_carry_no_index(self):
        # The thinking box folds several messages; recall rides its user
        # message. Neither is one message, so neither names a cut point.
        messages = [
            {
                "role": "user",
                "content": "hi",
                "api_content": "hi\n\n<memory-context>\nnote\n</memory-context>",
            },
            STEP,
            ANSWER,
        ]
        entries = build_entries(messages, [])
        assert [e.kind for e in entries] == ["user", "recall", "thinking", "assistant"]
        assert entries[1].index == -1  # recall
        assert entries[2].index == -1  # thinking

    def test_a_tail_keeps_absolute_indices(self):
        entries = build_entries([USER_MSG, ANSWER, USER_MSG, ANSWER], [], start=2)
        assert [(e.kind, e.index) for e in entries] == [
            ("user", 2),
            ("assistant", 3),
        ]


class TestSummary:
    def test_summary_counts_steps_and_reasoning(self):
        entry = Entry(kind="thinking", text="", steps=2, reasoning_chars=1500)
        assert entry.summary() == "1,500 chars reasoning · 2 steps"

    def test_summary_without_reasoning(self):
        assert Entry(kind="thinking", text="", steps=1).summary() == "1 step"

    def test_summary_without_steps_does_not_advertise_zero(self):
        entry = Entry(kind="thinking", text="", steps=0, reasoning_chars=111)
        assert entry.summary() == "111 chars reasoning"


class TestRecalledMemory:
    """Redesign Phase 5: recall is visible in the transcript — silent
    injection would make the agent's behavior inexplicable."""

    def test_recall_entry_after_the_user_message(self):
        messages = [
            {
                "role": "user",
                "content": "run my snakemake workflow",
                "api_content": (
                    "run my snakemake workflow\n\n<memory-context>\n"
                    "[System note: recalled memory, NOT new user input.]\n"
                    "Past struggle (2026-06-02): dry-runs fail here.\n"
                    "</memory-context>"
                ),
            },
            {"role": "assistant", "content": "ok"},
        ]
        entries = build_entries(messages)
        assert [e.kind for e in entries] == ["user", "recall", "assistant"]
        assert entries[0].text == "run my snakemake workflow"
        assert entries[1].text == "Past struggle (2026-06-02): dry-runs fail here."
        # the system-note scaffolding is not shown to the user
        assert "System note" not in entries[1].text

    def test_no_recall_entry_without_a_sidecar(self):
        entries = build_entries([{"role": "user", "content": "hello"}])
        assert [e.kind for e in entries] == ["user"]

    def test_sidecar_without_a_fence_is_ignored(self):
        entries = build_entries(
            [{"role": "user", "content": "hi", "api_content": "hi"}]
        )
        assert [e.kind for e in entries] == ["user"]


class TestOneExchangeIsOnePart:
    """A call and the result it returned are one row, not two.

    They were two for as long as the record was written that way, which put the
    tool's name on the screen twice and left the user scrolling between a
    question and its answer.
    """

    def test_the_result_lands_on_the_call_that_produced_it(self):
        calls = [{"after": 1, "tool": "list_dir", "arguments": {"path": "/data/cohort"}}]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert len(box.parts) == 1
        part = box.parts[0]
        assert part.kind == "call" and part.done
        assert "cohort" in part.text  # the call, on top
        assert part.result == "12 entries"  # the result, below it

    def test_the_body_puts_the_call_above_the_result(self):
        calls = [
            {
                "after": 1,
                "tool": "run_bash",
                "arguments": {"content_lines": ["ls /data"]},
                "script": "ls /data",
            }
        ]
        body = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1].parts[0].body()
        assert body.index("ls /data") < body.index("12 entries")
        assert RESULT_RULE in body

    def test_a_call_with_no_result_yet_shows_only_the_call(self):
        # What the chat renders the moment the announcement arrives: the turn
        # is parked on an approval, or the tool is simply still running.
        step = call_step({"tool": "run_bash", "script": "sleep 30"})
        assert not step.done
        assert step.body() == "sleep 30"
        assert RESULT_RULE not in step.body()
        assert step.label().endswith("…")

    def test_the_result_fills_the_same_part_in_place(self):
        step = call_step({"tool": "run_bash", "script": "sleep 30"})
        step.attach("[tool result] run_bash: ran (exit 0).")
        assert step.done and not step.failed
        assert step.label() == "run_bash"
        assert step.body() == f"sleep 30\n\n{RESULT_RULE}\nran (exit 0)."

    def test_a_failed_result_marks_the_row_it_lands_on(self):
        step = call_step({"tool": "read_file", "arguments": {"path": "/no/such"}})
        step.attach("[tool error] read_file: FileNotFoundError: /no/such")
        assert step.failed
        assert step.label() == "read_file · such (error)"
        assert "FileNotFoundError: /no/such" in step.body()

    def test_a_refusal_reads_as_a_failure_even_though_nothing_raised(self):
        step = call_step({"tool": "delete_file", "arguments": {}})
        step.attach("[tool result] delete_file: DENIED by the user — not executed.")
        assert step.failed
        assert step.label() == "delete_file (error)"

    def test_two_calls_before_either_answers_pair_up_in_order(self):
        calls = [
            {"after": 1, "tool": "list_dir", "arguments": {}},
            {"after": 1, "tool": "read_file", "arguments": {}},
        ]
        box = build_entries([USER_MSG, STEP, STEP2, ANSWER], [], calls)[1]
        assert [p.tool for p in box.parts] == ["list_dir", "read_file"]
        assert box.parts[0].result == "12 entries"
        assert box.parts[1].result == "not found"

    def test_a_result_whose_call_is_out_of_view_stands_on_its_own(self):
        # A log tail that begins between the two halves has no call to land on.
        box = build_entries([USER_MSG, STEP, ANSWER], [])[1]
        assert [p.kind for p in box.parts] == ["step"]
        assert box.parts[0].label() == "list_dir"
        assert box.parts[0].text == "12 entries"

    def test_the_folded_block_writes_one_heading_per_exchange(self):
        # The session log is this text; it must not say "call" then "step".
        calls = [{"after": 1, "tool": "list_dir", "arguments": {"path": "/data/cohort"}}]
        text = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1].text
        assert text.count("— list_dir · cohort —") == 1
        assert "— call —" not in text and "— step —" not in text
        assert text.index("cohort") < text.index("12 entries")


class TestModelFacingFramingIsNotShown:
    """A tool result says both what happened and what the model should do about
    it. Only the first is news for the user; the second is prompt."""

    def test_the_result_prefix_comes_off(self):
        text, failed = result_text("[tool result] list_dir: 12 entries")
        assert (text, failed) == ("12 entries", False)

    def test_an_error_is_flagged_and_unwrapped(self):
        text, failed = result_text("[tool error] read_file: OSError: nope")
        assert (text, failed) == ("OSError: nope", True)

    def test_every_hint_the_tools_carry_is_stripped(self):
        for hint in MODEL_HINTS:
            text, _ = result_text(f"[tool result] create_file: Created x.tsv. {hint}")
            assert hint not in text
            assert text == "Created x.tsv."

    def test_a_hint_taken_out_of_a_line_leaves_no_gap(self):
        hint = MODEL_HINTS[0]
        text, _ = result_text(f"[tool result] t: before {hint} after")
        assert "  " not in text
        assert text == "before after"

    def test_a_hint_that_finishes_a_sentence_takes_its_connective_with_it(self):
        # Half the hints are the back half of a sentence. Lifting one out on
        # its own would leave the line ending in a dash or a semicolon.
        for joiner in ("; ", " — ", ", ", ": "):
            text, _ = result_text(
                f"[tool result] t: The document index is empty{joiner}{MODEL_HINTS[0]}"
            )
            assert text == "The document index is empty"

    def test_a_full_stop_before_a_hint_is_not_eaten_with_it(self):
        # There the hint was its own sentence, and the stop closes the one
        # before it — which is news, and keeps its punctuation.
        text, _ = result_text(f"[tool result] t: Created x.tsv. {MODEL_HINTS[0]}")
        assert text == "Created x.tsv."

    def test_a_result_that_is_nothing_but_guidance_is_still_shown(self):
        # Some tools answer a malformed call with advice and no news at all. A
        # row showing a call and then a blank where the result goes reads as a
        # tool that hung, so the advice stands rather than emptying the row.
        text, _ = result_text(f"[tool result] session_search: {MODEL_HINTS[0]}")
        assert text == MODEL_HINTS[0]

    def test_output_that_merely_looks_like_framing_is_left_alone(self):
        # run_bash printing a bracketed line of its own is still output.
        text, failed = result_text(
            "[tool result] run_bash: ran (exit 0).\n\nstdout:\n[tool result] echoed"
        )
        assert text.endswith("[tool result] echoed")
        assert not failed

    def test_columns_in_command_output_keep_their_alignment(self):
        # Whitespace is only touched around a hint that was removed; a table
        # the user is reading must not be reflowed.
        table = "a     1\nbb    2"
        text, _ = result_text(f"[tool result] run_bash: ran (exit 0).\n\n{table}")
        assert table in text


class TestEntriesCarryTheirInstant:
    """`Entry.at`: when the message this entry reads was added.

    Straight off `llm.STAMP_KEY`, which `graph._append_messages` puts on every
    message as it arrives. Read here rather than stamped here, because a
    transcript is a *reading* of stored messages — one built twice must come
    out the same, and a clock read at render time would not.
    """

    AT = "2026-08-21T12:34:56+00:00"

    def said(self, **extra):
        return {"role": "user", "content": "what is the coverage?", STAMP_KEY: self.AT}

    def test_a_message_gives_its_entry_the_stamp(self):
        [entry] = build_entries([self.said()])
        assert entry.at == self.AT

    def test_and_so_does_an_answer(self):
        entries = build_entries(
            [self.said(), {"role": "assistant", "content": "31x", STAMP_KEY: self.AT}]
        )
        assert entries[-1].at == self.AT

    def test_a_message_with_no_stamp_says_nothing(self):
        # A thread written before there were stamps. "" means "not known",
        # where a zero would mean the epoch.
        [entry] = build_entries([{"role": "user", "content": "hi"}])
        assert entry.at == ""

    def test_a_recall_rides_the_message_it_came_with(self):
        # The one entry with no index of its own that can still name an
        # instant: it happened when the message carrying it did.
        message = self.said()
        message["api_content"] = (
            "what is the coverage?\n\n<memory-context>\n"
            "Past struggle (2026-06-02): dry-runs fail here.\n</memory-context>"
        )
        entries = build_entries([message])
        assert [e.kind for e in entries] == ["user", "recall"]
        assert [e.at for e in entries] == [self.AT, self.AT]

    def test_a_thinking_box_names_no_instant(self):
        # It folds several messages, so there is no one moment it happened at.
        entries = build_entries(
            [
                self.said(),
                {"role": "assistant", "content": "31x", STAMP_KEY: self.AT},
            ],
            calls=[{"after": 1, "tool": "read_file", "arguments": {}}],
        )
        assert [e.at for e in entries if e.kind == "thinking"] == [""]
