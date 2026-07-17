"""Tests for hpca.agent.titler: model-written session titles."""

import json

import pytest

from hpca.agent.titler import (
    TITLE_MAX_CHARS,
    TitleError,
    clean_title,
    propose_title,
)
from hpca.llm import ChatResponse

CONVERSATION = [
    {"role": "user", "content": "which BAMs are in the cohort dir?"},
    {"role": "assistant", "content": "Four BAMs match: three tumour, one normal."},
]


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return ChatResponse(content=self._outputs.pop(0))


def title_json(title):
    return json.dumps({"title": title})


class TestCleanTitle:
    def test_plain_title_kept(self):
        assert clean_title("Cohort BAM inventory") == "Cohort BAM inventory"

    def test_quotes_and_period_stripped(self):
        assert clean_title('"Cohort BAM inventory."') == "Cohort BAM inventory"

    def test_prefixes_stripped(self):
        assert clean_title("Session: Cohort BAM inventory") == "Cohort BAM inventory"
        assert clean_title("Chat about cohort BAMs") == "cohort BAMs"

    def test_whitespace_collapsed(self):
        assert clean_title("  Cohort   BAM\ninventory ") == "Cohort BAM inventory"

    def test_long_title_cut_at_a_word_boundary(self):
        title = clean_title(
            "Counting the binary alignment map files in the cohort directory today"
        )
        assert len(title) <= TITLE_MAX_CHARS + 1  # the ellipsis
        assert title.endswith("…")
        assert " fil…" not in title  # cut between words, not through one

    def test_long_single_word_still_cut(self):
        title = clean_title("x" * 100)
        assert len(title) == TITLE_MAX_CHARS + 1


class TestProposeTitle:
    async def test_returns_the_cleaned_title(self):
        llm = FakeLLM([title_json("Cohort BAM inventory.")])
        assert await propose_title(llm, CONVERSATION) == "Cohort BAM inventory"

    async def test_asks_with_a_schema_and_never_thinks(self):
        llm = FakeLLM([title_json("Cohort BAM inventory")])
        await propose_title(llm, CONVERSATION)
        call = llm.calls[0]
        assert call["json_schema"]["required"] == ["title"]
        # naming a chat must not cost a reasoning pass (~minutes on Qwen)
        assert call["enable_thinking"] is False

    async def test_transcript_carries_the_conversation(self):
        llm = FakeLLM([title_json("Cohort BAM inventory")])
        await propose_title(llm, CONVERSATION)
        prompt = llm.calls[0]["messages"][1]["content"]
        assert "which BAMs are in the cohort dir?" in prompt

    async def test_retries_with_feedback_then_succeeds(self):
        llm = FakeLLM(["not json at all", title_json("Cohort BAM inventory")])
        assert await propose_title(llm, CONVERSATION) == "Cohort BAM inventory"
        assert "validation error" in llm.calls[1]["messages"][-1]["content"]

    async def test_empty_title_is_rejected(self):
        llm = FakeLLM([title_json("   "), title_json("Cohort BAMs")])
        assert await propose_title(llm, CONVERSATION) == "Cohort BAMs"

    async def test_gives_up_after_the_budget(self):
        llm = FakeLLM(["garbage", "still garbage"])
        with pytest.raises(TitleError):
            await propose_title(llm, CONVERSATION)
        assert len(llm.calls) == 2


# --------------------------------------------------------- integration tests

from tests.live_backend import LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestTitleLive:
    async def test_live_model_names_the_conversation(self):
        from hpca.config import LLMSettings
        from hpca.llm import LLMClient

        async with LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                request_timeout_s=120,
                enable_thinking=True,  # the titler must override this itself
            )
        ) as llm:
            title = await propose_title(
                llm,
                [
                    {
                        "role": "user",
                        "content": "My STAR alignment job 27744534 was killed. Why?",
                    },
                    {
                        "role": "assistant",
                        "content": "It hit the 40G memory limit; request more.",
                    },
                ],
            )
        assert title
        assert len(title) <= TITLE_MAX_CHARS + 1
        assert "\n" not in title
        assert title.lower() != "untitled"
