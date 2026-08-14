"""The model's first draft for ``/skill-creator <what it should do>``: the
request (plus the conversation) goes in, a name/description/body draft comes
back, normalised into something the form can hold."""

import json

import pytest

from hpca.agent.skill_drafter import (
    DraftError,
    clean_body,
    clean_description,
    clean_name,
    propose_skill,
)
from hpca.llm import ChatResponse


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def draft_json(name="watch-notebook", description="Run a notebook", body="1. Go."):
    return json.dumps({"name": name, "description": description, "body": body})


class TestCleaners:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Watch Notebook", "watch-notebook"),
            ("submit_slurm_job", "submit-slurm-job"),
            ("  --Jupyter Runs!!  ", "jupyter-runs"),
            ("papermill/nbconvert", "papermill-nbconvert"),
        ],
    )
    def test_a_name_becomes_a_slash_command_handle(self, raw, expected):
        assert clean_name(raw) == expected

    def test_a_long_name_is_cut_on_a_word_boundary(self):
        name = clean_name("start and monitor jupyter notebook executions on slurm")
        assert len(name) <= 40
        assert not name.endswith("-")
        assert name.startswith("start-and-monitor")

    def test_a_name_of_pure_punctuation_is_empty_not_dashes(self):
        assert clean_name("!!!") == ""

    def test_a_description_is_one_bounded_line(self):
        text = clean_description("  when the user\n  asks to run   a notebook ")
        assert text == "when the user asks to run a notebook"
        assert clean_description("word " * 60).endswith("…")

    def test_a_fenced_body_is_unwrapped(self):
        assert clean_body("```markdown\n1. Step one.\n```") == "1. Step one."
        # A fence *inside* the procedure is content, not a wrapper.
        assert clean_body("1. Run:\n```\nsqueue\n```").startswith("1. Run:")


class TestProposeSkill:
    async def test_returns_a_normalised_draft(self):
        llm = FakeLLM([draft_json(name="Watch Notebook", body="```\n1. Go.\n```")])
        draft = await propose_skill(llm, "watch a notebook run")
        assert draft.name == "watch-notebook"
        assert draft.body == "1. Go."
        assert draft.description == "Run a notebook"

    async def test_the_request_reaches_the_model(self):
        llm = FakeLLM([draft_json()])
        await propose_skill(llm, "start and monitor jupyter runs")
        prompt = llm.calls[0]["messages"][-1]["content"]
        assert "start and monitor jupyter runs" in prompt

    async def test_drafting_is_schema_constrained_and_thinking_is_off(self):
        """A form the user is waiting on: seconds, not a minute of reasoning."""
        llm = FakeLLM([draft_json()])
        await propose_skill(llm, "watch a notebook")
        assert llm.calls[0]["json_schema"]["required"] == ["name", "description", "body"]
        assert llm.calls[0]["enable_thinking"] is False

    async def test_the_conversation_is_offered_as_context(self):
        llm = FakeLLM([draft_json()])
        await propose_skill(
            llm,
            "a skill from our conversation",
            messages=[
                {"role": "user", "content": "papermill keeps dying on OOM"},
                {"role": "assistant", "content": "raise --mem to 32G"},
            ],
        )
        prompt = llm.calls[0]["messages"][-1]["content"]
        assert "papermill keeps dying on OOM" in prompt
        assert "raise --mem to 32G" in prompt

    async def test_existing_skills_are_named_so_the_draft_does_not_clash(self):
        llm = FakeLLM([draft_json()])
        await propose_skill(llm, "watch a notebook", existing="- plan: plan a change")
        system = llm.calls[0]["messages"][0]["content"]
        assert "- plan: plan a change" in system

    async def test_a_malformed_reply_is_retried_with_the_error(self):
        llm = FakeLLM(["not json at all", draft_json()])
        draft = await propose_skill(llm, "watch a notebook")
        assert draft.name == "watch-notebook"
        assert "[validation error]" in llm.calls[1]["messages"][-1]["content"]

    async def test_an_unusable_name_is_retried_rather_than_written(self):
        """'!!!' sanitises to nothing — a skill with no handle is no skill."""
        llm = FakeLLM([draft_json(name="!!!"), draft_json()])
        draft = await propose_skill(llm, "watch a notebook")
        assert draft.name == "watch-notebook"

    async def test_gives_up_after_the_retry_budget(self):
        llm = FakeLLM(["nope", "still nope"])
        with pytest.raises(DraftError):
            await propose_skill(llm, "watch a notebook")
        assert len(llm.calls) == 2  # one attempt plus one retry


# --------------------------------------------------------- integration tests

import re  # noqa: E402

from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402
from tests.live_backend import LIVE_KEY, LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestLiveDrafting:
    """One real draft, checked for what the form needs rather than for words.
    The broader measurement — ten requests, five checks each — is
    ``evals/skill_draft_eval.py``; this is the regression guard that the call
    still works against a real backend at all (~5-10s on a free box)."""

    async def test_a_real_backend_drafts_a_usable_skill(self):
        settings = LLMSettings(
            base_url=LIVE_URL,
            model=LIVE_MODEL,
            api_key=LIVE_KEY,
            request_timeout_s=120,
            enable_thinking=False,
        )
        async with LLMClient(settings) as llm:
            draft = await propose_skill(
                llm,
                "a skill that submits a batch job to slurm and watches it",
            )
        # Invocable as /<skill>: kebab-case, and the form's field holds it.
        assert re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", draft.name)
        assert len(draft.name) <= 40
        assert draft.description and "\n" not in draft.description
        # A procedure, not a sentence — and it understood the request.
        assert len(draft.body.splitlines()) >= 3
        assert any(
            term in f"{draft.name} {draft.description} {draft.body}".lower()
            for term in ("sbatch", "squeue", "slurm")
        )
