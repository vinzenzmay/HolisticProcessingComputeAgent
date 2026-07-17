"""Tests for hpca.agent.explainer: structured failure explanations (§5.5 step 5)."""

import json

import pytest

from hpca.agent.explainer import FailureExplanation, explain_failure
from hpca.llm import ChatResponse
from hpca.triage import SignatureMatch, TriageReport


def oom_report():
    return TriageReport(
        job_id="27798067",
        state="OUT_OF_MEMORY",
        exit_info="OUT_OF_MEMORY, signal 125",
        elapsed_s=300,
        max_rss_bytes=25 * 1024**3,
        reqmem="25G",
        timelimit="5-00:00:00",
        script_key="align_bam",
        matches=[
            SignatureMatch(
                signature_id="oom_kill",
                title="Out of memory (OOM kill)",
                hint="Increase --mem or reduce the working set.",
                file="/logs/job.err",
                matched_line="slurmstepd: error: Detected 1 oom-kill event(s)",
                excerpt="loading BAM index\nslurmstepd: error: Detected 1 oom-kill event(s)",
            )
        ],
    )


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
        "why": "The job used all 25G of requested memory while loading the BAM.",
        "current_state": "Killed by the OOM killer; no output produced.",
        "suggested_fix": "Resubmit with --mem=50G or stream the BAM instead.",
        "finickiness": "easy",
        "justification": "Memory bumps are a routine, low-risk change.",
    }
)


class TestExplainFailure:
    async def test_returns_validated_explanation(self):
        llm = FakeLLM([VALID])
        result = await explain_failure(llm, oom_report())
        assert isinstance(result, FailureExplanation)
        assert result.finickiness == "easy"
        assert "25G" in result.why

    async def test_report_text_reaches_model_and_schema_used(self):
        llm = FakeLLM([VALID])
        await explain_failure(llm, oom_report())
        call = llm.calls[0]
        user_text = call["messages"][-1]["content"]
        assert "oom-kill" in user_text
        assert call["json_schema"] is not None
        assert "finickiness" in call["json_schema"]["properties"]

    async def test_invalid_json_retried_then_ok(self):
        llm = FakeLLM(["not json", VALID])
        result = await explain_failure(llm, oom_report())
        assert result.finickiness == "easy"
        assert len(llm.calls) == 2

    async def test_bad_enum_retried(self):
        bad = json.dumps(
            {
                "why": "x",
                "current_state": "x",
                "suggested_fix": "x",
                "finickiness": "impossible",
                "justification": "x",
            }
        )
        llm = FakeLLM([bad, VALID])
        result = await explain_failure(llm, oom_report())
        assert result.finickiness == "easy"

    async def test_exhausted_retries_raise(self):
        llm = FakeLLM(["junk"] * 5)
        with pytest.raises(Exception, match="explanation"):
            await explain_failure(llm, oom_report(), max_retries=2)


# --------------------------------------------------------- integration tests

from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

from tests.live_backend import LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestExplainLive:
    async def test_oom_explanation_from_live_model(self):
        llm = LLMClient(
            LLMSettings(base_url=LIVE_URL, model=LIVE_MODEL, request_timeout_s=120)
        )
        result = await explain_failure(llm, oom_report())
        assert result.why
        assert result.suggested_fix
        assert result.finickiness in ("trivial", "easy", "fiddly", "hard")
        # the explanation should engage with the actual failure mode
        combined = (result.why + result.suggested_fix).lower()
        assert "mem" in combined
        await llm.close()


class TestTier1Injection:
    async def test_standing_notes_in_system_prompt(self):
        llm = FakeLLM([VALID])
        await explain_failure(llm, oom_report(), tier1="Cluster is cubi.")
        system = llm.calls[0]["messages"][0]
        assert system["role"] == "system"
        assert "Cluster is cubi." in system["content"]
