"""Tests for hpca.transcript: folding state into chat entries."""

from hpca.transcript import Entry, build_entries

USER_MSG = {"role": "user", "content": "which BAMs are in the cohort?"}
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
        assert "list_dir: 12 entries" in box.text
        assert "read_file: not found" in box.text

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
        assert box.parts[1].text == STEP["content"]
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
                "arguments": {"registry_key": "cohort"},
            }
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert [p.kind for p in box.parts] == ["call", "step"]
        assert box.parts[0].label() == "list_dir (call)"
        assert "cohort" in box.parts[0].text

    def test_call_comes_after_the_reasoning_that_produced_it(self):
        thinking = [{"after": 1, "reasoning": "List it first."}]
        calls = [{"after": 1, "tool": "list_dir", "arguments": {}}]
        box = build_entries([USER_MSG, STEP, ANSWER], thinking, calls)[1]
        assert [p.kind for p in box.parts] == ["reasoning", "call", "step"]

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
        assert "timeout_s" in call.text  # the other arguments still say how

    def test_details_are_shown_when_the_tool_resolved_the_call(self):
        calls = [
            {
                "after": 1,
                "tool": "delete_file",
                "arguments": {"registry_key": "scratch"},
                "details": "rm /scratch/old.bam\n(12 bytes; will be recoverable)",
            }
        ]
        box = build_entries([USER_MSG, STEP, ANSWER], [], calls)[1]
        assert "rm /scratch/old.bam" in box.parts[0].text

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
