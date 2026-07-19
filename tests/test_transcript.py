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


class TestTail:
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
