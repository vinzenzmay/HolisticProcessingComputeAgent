"""Tests for context compaction (redesign Phase 6)."""

import pytest

from hpca.agent import compact
from hpca.llm import ChatResponse


class FakeLLM:
    def __init__(self, summary="the user is aligning reads; STAR needs 40G"):
        self._summary = summary
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return ChatResponse(content=self._summary)


def history(count, chars=100):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * chars}
        for i in range(count)
    ]


class TestShouldCompact:
    def test_off_without_a_known_window(self):
        assert not compact.should_compact(history(100), max_model_len=None)

    def test_off_for_a_short_history(self):
        assert not compact.should_compact(history(4), max_model_len=100)

    def test_on_when_the_estimate_passes_the_threshold(self):
        # 60 messages * ~100 chars = ~1500 tokens, over 70% of 2000
        assert compact.should_compact(history(60), max_model_len=2000)

    def test_off_with_plenty_of_room(self):
        assert not compact.should_compact(history(60), max_model_len=100_000)


class TestSplit:
    def test_keeps_a_recent_tail_verbatim(self):
        older, keep = compact.split(history(60))
        assert len(keep) >= compact.KEEP_RECENT
        assert older + keep == history(60)
        assert keep[-1]["content"].startswith("m59")

    def test_short_history_is_all_kept(self):
        older, keep = compact.split(history(5))
        assert older == []
        assert len(keep) == 5


class TestSummarize:
    async def test_produces_one_user_message(self):
        summary = await compact.summarize(FakeLLM(), history(20))
        assert summary["role"] == "user"  # the shape the model already knows
        assert summary["content"].startswith(compact.SUMMARY_PREFIX)
        assert "STAR needs 40G" in summary["content"]

    async def test_prompt_asks_to_keep_identifiers(self):
        llm = FakeLLM()
        await compact.summarize(llm, history(20))
        system = llm.calls[0][0]["content"]
        assert "verbatim" in system
        assert "job ids" in system

    async def test_empty_summary_is_an_error(self):
        with pytest.raises(ValueError):
            await compact.summarize(FakeLLM(summary="   "), history(20))

    async def test_is_summary_recognizes_its_own_output(self):
        summary = await compact.summarize(FakeLLM(), history(20))
        assert compact.is_summary(summary)
        assert not compact.is_summary({"role": "user", "content": "hello"})

    async def test_summary_is_length_bounded(self):
        summary = await compact.summarize(FakeLLM(summary="x" * 9000), history(20))
        assert len(summary["content"]) < compact.MAX_SUMMARY_CHARS + 60


class TestSidecarAwareEstimate:
    """wire_messages sends api_content when present, so the estimate must
    count it — those are exactly the messages carrying extra payload."""

    def test_api_content_counted(self):
        plain = [{"role": "user", "content": "x" * 400}]
        with_recall = [
            {"role": "user", "content": "x" * 400, "api_content": "x" * 1200}
        ]
        assert compact.estimate_tokens(with_recall) == 300
        assert compact.estimate_tokens(plain) == 100

    def test_compaction_triggers_on_the_real_size(self):
        messages = [
            {"role": "user", "content": "short", "api_content": "x" * 900}
            for _ in range(20)
        ]
        # 20 * 900 chars = ~4500 tokens; only ~25 by content alone
        assert compact.should_compact(messages, max_model_len=2000)
