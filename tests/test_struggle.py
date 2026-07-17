"""Tests for hpca.agent.struggle: self-reflection struggle notes (§4.4)."""

import json

import pytest

from hpca.agent.struggle import (
    STRUGGLE_KIND,
    StruggleError,
    matching_struggles,
    note_keywords,
    propose_struggle_note,
    turn_struggled,
)
from hpca.llm import ChatResponse
from hpca.profiles import Memory


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
        return Memory(text=text, tier=2, kind=STRUGGLE_KIND)

    def test_keywords_parsed(self):
        memory = self.make_note("Snakemake DAG errors are hard.\nkeywords: snakemake, dag")
        assert note_keywords(memory) == ["snakemake", "dag"]

    def test_note_without_keywords(self):
        assert note_keywords(self.make_note("no keyword line here")) == []

    def test_matching_struggle_found(self):
        memories = [
            self.make_note("Snakemake DAGs are fiddly.\nkeywords: snakemake, dag")
        ]
        assert matching_struggles(memories, "run my snakemake workflow")

    def test_non_struggle_memories_ignored(self):
        memories = [Memory(text="keywords: snakemake", tier=2, kind="learning")]
        assert matching_struggles(memories, "snakemake please") == []

    def test_unrelated_request_no_match(self):
        memories = [self.make_note("x\nkeywords: snakemake, dag")]
        assert matching_struggles(memories, "align a bam file") == []

    def test_word_boundary(self):
        memories = [self.make_note("x\nkeywords: qc")]
        assert matching_struggles(memories, "run qc")
        assert matching_struggles(memories, "qcircuit stuff") == []


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


VALID = json.dumps(
    {
        "note": "Snakemake dry-runs fail here when profiles are involved.",
        "keywords": ["snakemake", "profile", "dry-run"],
    }
)

MESSAGES = [
    {"role": "user", "content": "run my snakemake workflow"},
    {"role": "user", "content": "[tool error] create_script: snakemake -n failed"},
]


class TestProposeStruggleNote:
    async def test_returns_note_with_keywords(self):
        llm = FakeLLM([VALID])
        note = await propose_struggle_note(llm, MESSAGES)
        assert "Snakemake" in note.note
        assert "snakemake" in note.keywords

    async def test_render_includes_keyword_line(self):
        llm = FakeLLM([VALID])
        note = await propose_struggle_note(llm, MESSAGES)
        rendered = note.render()
        assert "keywords: snakemake, profile, dry-run" in rendered
        # round-trips through the matcher
        memory = Memory(text=rendered, tier=2, kind=STRUGGLE_KIND)
        assert matching_struggles([memory], "another snakemake job")

    async def test_transcript_reaches_model(self):
        llm = FakeLLM([VALID])
        await propose_struggle_note(llm, MESSAGES)
        user_text = llm.calls[0]["messages"][-1]["content"]
        assert "snakemake -n failed" in user_text

    async def test_invalid_json_retried(self):
        llm = FakeLLM(["junk", VALID])
        note = await propose_struggle_note(llm, MESSAGES)
        assert note.keywords
        assert len(llm.calls) == 2

    async def test_exhausted_retries_raise(self):
        llm = FakeLLM(["junk"] * 5)
        with pytest.raises(StruggleError):
            await propose_struggle_note(llm, MESSAGES, max_retries=1)
