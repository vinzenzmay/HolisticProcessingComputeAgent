"""Log-explainer (§5.5 step 5), honoring the context firewall (§4.2).

The triage report enters a *fresh* LLM conversation; only this bounded,
structured explanation flows back to the caller (the orchestrator's
``get_job_report`` tool). The fix is always framed as a suggestion — any
actual change re-enters the dry-run and HITL gates.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from hpca.triage import TriageReport, format_report

EXPLAIN_MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a triage assistant for HPC cluster jobs. You get a structured "
    "failure report: job metadata, matched error signatures with log "
    "excerpts, and deterministic hints. Base your answer ONLY on the report; "
    "if it is inconclusive, say so. Frame every fix as a suggestion."
)


class FailureExplanation(BaseModel):
    why: str
    current_state: str
    suggested_fix: str
    finickiness: Literal["trivial", "easy", "fiddly", "hard"]
    justification: str

    def render(self) -> str:
        return (
            f"why: {self.why}\n"
            f"state: {self.current_state}\n"
            f"suggested fix: {self.suggested_fix}\n"
            f"finickiness: {self.finickiness} — {self.justification}"
        )


EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {"type": "string", "description": "Why the job failed, 1-3 sentences"},
        "current_state": {"type": "string", "description": "Where things stand now"},
        "suggested_fix": {"type": "string", "description": "Suggested next step"},
        "finickiness": {"enum": ["trivial", "easy", "fiddly", "hard"]},
        "justification": {
            "type": "string",
            "description": "One sentence justifying the finickiness estimate",
        },
    },
    "required": ["why", "current_state", "suggested_fix", "finickiness",
                 "justification"],
    "additionalProperties": False,
}


class ExplainerError(Exception):
    """No valid explanation within the retry budget."""


async def explain_failure(
    llm,
    report: TriageReport,
    *,
    max_retries: int = EXPLAIN_MAX_RETRIES,
) -> FailureExplanation:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_report(report)},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=EXPLAIN_SCHEMA,
            schema_name="failure_explanation",
            max_tokens=1024,
        )
        try:
            return FailureExplanation.model_validate(json.loads(response.content))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
    raise ExplainerError(
        f"No valid failure explanation after {max_retries + 1} attempts: {last_error}"
    )
