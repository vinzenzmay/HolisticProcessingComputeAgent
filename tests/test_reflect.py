"""Tests for the self-review loop (redesign Phase 4)."""

import json

import pytest

from hpca.agent.reflect import (
    ReflectError,
    Reflection,
    digest,
    propose_reflections,
)
from hpca.llm import ChatResponse
from hpca.profiles import MemoryScope


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return ChatResponse(content=self._outputs.pop(0))


def reply(*proposals):
    return json.dumps({"proposals": list(proposals)})


MESSAGES = [
    {"role": "system", "content": "you are hpca"},
    {"role": "user", "content": "use R for the plots from now on"},
    {"role": "assistant", "content": "understood"},
]


class TestDigest:
    def test_system_messages_dropped(self):
        assert "you are hpca" not in digest(MESSAGES)

    def test_recent_turns_verbatim(self):
        assert "use R for the plots from now on" in digest(MESSAGES)

    def test_older_turns_compressed_into_one_line(self):
        messages = [
            {"role": "user", "content": f"question number {i}"} for i in range(40)
        ]
        text = digest(messages)
        assert text.startswith("[earlier in this session:")
        assert "question number 39" in text  # the tail is verbatim
        # older turns appear only inside the digest line, not as their own
        assert "user: question number 0" not in text

    def test_long_messages_clipped(self):
        messages = [{"role": "user", "content": "x" * 5000}]
        assert len(digest(messages)) < 1000


class TestProposals:
    async def test_memory_proposal(self):
        llm = FakeLLM(
            [
                reply(
                    {
                        "kind": "memory",
                        "scope": "system-prompt",
                        "text": "The user prefers R.",
                    }
                )
            ]
        )
        proposals = await propose_reflections(llm, MESSAGES)
        assert len(proposals) == 1
        assert proposals[0].kind == "memory"
        assert proposals[0].scope == MemoryScope.SYSTEM_PROMPT
        assert proposals[0].describe() == "system-prompt memory"

    async def test_struggle_proposal_renders_keywords(self):
        llm = FakeLLM(
            [
                reply(
                    {
                        "kind": "struggle",
                        "text": "Snakemake dry-runs fail with site profiles.",
                        "keywords": ["snakemake", "dry-run"],
                    }
                )
            ]
        )
        proposal = (await propose_reflections(llm, MESSAGES))[0]
        assert proposal.memory_text().endswith("keywords: snakemake, dry-run")

    async def test_nothing_to_save_is_valid(self):
        llm = FakeLLM([reply()])
        assert await propose_reflections(llm, MESSAGES) == []

    async def test_empty_text_dropped(self):
        llm = FakeLLM(
            [reply({"kind": "memory", "scope": "system-prompt", "text": "   "})]
        )
        assert await propose_reflections(llm, MESSAGES) == []

    async def test_skill_proposal_without_name_dropped(self):
        llm = FakeLLM([reply({"kind": "skill_patch", "text": "add a step"})])
        assert await propose_reflections(llm, MESSAGES) == []

    async def test_skill_patch_kept_with_name(self):
        llm = FakeLLM(
            [
                reply(
                    {
                        "kind": "skill_patch",
                        "text": "index the BAM first",
                        "skill_name": "bam-subset",
                    }
                )
            ]
        )
        proposal = (await propose_reflections(llm, MESSAGES))[0]
        assert proposal.describe() == "patch to skill “bam-subset”"

    async def test_retries_on_invalid_json(self):
        llm = FakeLLM(
            [
                "not json",
                reply(
                    {"kind": "memory", "scope": "system-prompt", "text": "cubi"}
                ),
            ]
        )
        proposals = await propose_reflections(llm, MESSAGES)
        assert proposals[0].text == "cubi"
        assert len(llm.calls) == 2
        # the retry tells the model what was wrong
        assert "validation error" in llm.calls[1]["messages"][-1]["content"]

    async def test_gives_up_after_retries(self):
        llm = FakeLLM(["nope", "still nope"])
        with pytest.raises(ReflectError):
            await propose_reflections(llm, MESSAGES, max_retries=1)


class TestPromptContents:
    async def _system(self, **kwargs):
        llm = FakeLLM([reply()])
        await propose_reflections(llm, MESSAGES, **kwargs)
        return llm.calls[0]["messages"][0]["content"]

    async def test_anti_capture_rules_present(self):
        system = await self._system()
        # the self-poisoning modes this exists to prevent
        assert "does not work here" in system
        assert "retry" in system
        assert "stale within a week" in system

    async def test_declarative_rule_present(self):
        assert "declarative facts" in await self._system()

    async def test_nothing_to_save_is_endorsed(self):
        assert "Proposing nothing is a perfectly good outcome" in await self._system()

    async def test_new_skill_clause_gated(self):
        assert "skill_new" in await self._system(allow_new_skills=True)
        assert "skill_new" not in await self._system(allow_new_skills=False)

    async def test_new_skill_proposals_filtered_when_disabled(self):
        llm = FakeLLM(
            [reply({"kind": "skill_new", "text": "x", "skill_name": "align"})]
        )
        assert await propose_reflections(llm, MESSAGES, allow_new_skills=False) == []

    async def test_known_memories_passed_to_avoid_duplicates(self):
        system = await self._system(
            system_prompt_memories="Cluster is cubi.\nPrefers R."
        )
        assert "do NOT propose these again" in system
        assert "Cluster is cubi." in system
        assert "Prefers R." in system

    async def test_existing_skills_listed(self):
        assert "bam-subset" in await self._system(skills="- bam-subset: subset a bam")


class TestReflectionModel:
    def test_memory_text_without_keywords(self):
        proposal = Reflection(kind="memory", text="plain fact")
        assert proposal.memory_text() == "plain fact"

    def test_describe_new_skill(self):
        proposal = Reflection(kind="skill_new", text="x", skill_name="align")
        assert proposal.describe() == "new skill “align”"


class TestWholeSpanDigest:
    """The pre-eviction review sees a stretch that is about to be discarded,
    so both ends must stay verbatim — the goal and the site facts are stated
    at the beginning, and a recency bias would lose exactly those."""

    def messages(self, count):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(count)
        ]

    def test_keeps_both_ends(self):
        text = digest(self.messages(80), span="whole")
        assert "msg0" in text  # the oldest, which recency would drop
        assert "msg79" in text
        assert "messages omitted" in text

    def test_recent_span_drops_the_oldest(self):
        text = digest(self.messages(80), span="recent")
        assert "user: msg0" not in text
        assert "msg79" in text

    def test_short_stretch_is_whole_either_way(self):
        text = digest(self.messages(6), span="whole")
        assert "msg0" in text and "msg5" in text
        assert "omitted" not in text

    async def test_propose_uses_the_span(self):
        llm = FakeLLM([reply()])
        await propose_reflections(llm, self.messages(80), span="whole")
        sent = llm.calls[0]["messages"][1]["content"]
        assert "msg0" in sent


class TestRagDeduplication:
    """Found in a live run against Qwen3.6-27B: a retrieved RAG note is in
    the conversation when the reviewer reads it, so unless it is listed as
    known the reviewer re-proposes it — as system-prompt, promoting a
    situational note into the always-injected budget RAG keeps it out of."""

    async def test_rag_listed_as_known(self):
        llm = FakeLLM([reply()])
        await propose_reflections(
            llm, MESSAGES, rag_memories="DeepVariant needs the gpu partition."
        )
        system = llm.calls[0]["messages"][0]["content"]
        assert "do NOT propose these again" in system
        assert "DeepVariant needs the gpu partition." in system

    async def test_both_scopes_listed(self):
        llm = FakeLLM([reply()])
        await propose_reflections(
            llm, MESSAGES, system_prompt_memories="one\ntwo", rag_memories="three"
        )
        system = llm.calls[0]["messages"][0]["content"]
        assert "one" in system and "two" in system and "three" in system

    async def test_large_rag_is_capped(self):
        llm = FakeLLM([reply()])
        await propose_reflections(llm, MESSAGES, rag_memories="x" * 50_000)
        system = llm.calls[0]["messages"][0]["content"]
        assert len(system) < 12_000  # bounded, not the whole scope

    async def test_cap_keeps_the_most_recent_entries(self):
        llm = FakeLLM([reply()])
        rag = "\n\n".join(f"note number {i}" for i in range(500))
        await propose_reflections(llm, MESSAGES, rag_memories=rag)
        system = llm.calls[0]["messages"][0]["content"]
        assert "note number 499" in system  # newest kept
        assert "note number 0\n" not in system  # oldest dropped
