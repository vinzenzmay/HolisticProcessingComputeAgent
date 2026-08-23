"""Tests for hpca.agent.struggle: struggle detection and matching (§4.4).

Proposing the note itself lives in hpca.agent.reflect (redesign Phase 4).
"""

from hpca.agent.struggle import (
    STRUGGLE_KIND,
    matching_struggles,
    note_keywords,
    turn_struggled,
)
from hpca.profiles import Memory, MemoryScope


class TestTurnStruggled:
    def test_tool_error_detected(self):
        messages = [
            {"role": "user", "content": "do it"},
            {"role": "user", "content": "[tool error] create_script: boom"},
        ]
        assert turn_struggled(messages)

    def test_exhausted_retries_detected(self):
        messages = [
            {"role": "assistant", "content": "I failed to produce a valid action: x"}
        ]
        assert turn_struggled(messages)

    def test_tool_budget_detected(self):
        messages = [
            {"role": "assistant", "content": "I stopped after 8 tool calls in one "
             "turn (tool budget exhausted). Tell me how to proceed."}
        ]
        assert turn_struggled(messages)

    def test_user_abort_flag(self):
        assert turn_struggled([{"role": "user", "content": "hi"}], aborted=True)

    def test_clean_turn_not_a_struggle(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "[tool result] echo: hi"},
            {"role": "assistant", "content": "done"},
        ]
        assert not turn_struggled(messages)


class TestKeywordMatching:
    def make_note(self, text: str) -> Memory:
        return Memory(text=text, scope=MemoryScope.RAG, kind=STRUGGLE_KIND)

    def test_keywords_parsed(self):
        memory = self.make_note("Sniffles DAG errors are hard.\nkeywords: sniffles, dag")
        assert note_keywords(memory) == ["sniffles", "dag"]

    def test_note_without_keywords(self):
        assert note_keywords(self.make_note("no keyword line here")) == []

    def test_matching_struggle_found(self):
        memories = [
            self.make_note("Sniffles DAGs are fiddly.\nkeywords: sniffles, dag")
        ]
        assert matching_struggles(memories, "run my sniffles workflow")

    def test_non_struggle_memories_ignored(self):
        memories = [
            Memory(text="keywords: sniffles", scope=MemoryScope.RAG, kind="learning")
        ]
        assert matching_struggles(memories, "sniffles please") == []

    def test_unrelated_request_no_match(self):
        memories = [self.make_note("x\nkeywords: sniffles, dag")]
        assert matching_struggles(memories, "align a bam file") == []

    def test_word_boundary(self):
        memories = [self.make_note("x\nkeywords: qc")]
        assert matching_struggles(memories, "run qc")
        assert matching_struggles(memories, "qcircuit stuff") == []
