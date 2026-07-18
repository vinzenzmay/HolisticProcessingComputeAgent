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


# ---------------------------------------------------- tier 3: local failures

PROCESS_SYSTEM_PROMPT = (
    "You diagnose failed command-line runs on an HPC system. You are given a "
    "process's exit code and candidate error lines pulled from its log by a "
    "keyword scan — some of them are noise (progress counters, statistics, "
    "cleanup messages), so decide which one actually caused the failure.\n\n"
    "Rules you must follow:\n"
    "1. Quote the line you believe is the cause, copied exactly from the "
    "candidates. Never write a line that is not there.\n"
    "2. If none of the candidates explains the failure, set cause_line to an "
    "empty string and say so in `why`. An honest 'cannot tell from this log' "
    "is correct and useful; a confident guess is not.\n"
    "3. Frame the fix as a suggestion.\n"
    "4. If — and only if — the cause is a recognisable, recurring failure "
    "mode, propose a signature so the next occurrence is caught without a "
    "model call. Its regex must match the cause line and must be specific "
    "enough not to match ordinary progress output. Otherwise leave "
    "proposed_signature null."
)


class ProposedSignature(BaseModel):
    id: str
    title: str
    patterns: list[str]
    hint: str


class ProcessExplanation(BaseModel):
    why: str
    cause_line: str  # "" when the log does not say
    suggested_fix: str
    proposed_signature: ProposedSignature | None = None

    @property
    def conclusive(self) -> bool:
        return bool(self.cause_line.strip())

    def render(self) -> str:
        parts = [f"why: {self.why}"]
        if self.cause_line.strip():
            parts.append(f"cause line: {self.cause_line.strip()}")
        parts.append(f"suggested fix: {self.suggested_fix}")
        return "\n".join(parts)


PROCESS_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {
            "type": "string",
            "description": "Why the process failed, 1-3 sentences; say so if unclear",
        },
        "cause_line": {
            "type": "string",
            "description": "The causing line copied verbatim, or '' if none does",
        },
        "suggested_fix": {"type": "string", "description": "Suggested next step"},
        "proposed_signature": {
            "type": ["object", "null"],
            "properties": {
                "id": {"type": "string", "description": "snake_case identifier"},
                "title": {"type": "string"},
                "patterns": {"type": "array", "items": {"type": "string"}},
                "hint": {"type": "string"},
            },
            "required": ["id", "title", "patterns", "hint"],
            "additionalProperties": False,
        },
    },
    "required": ["why", "cause_line", "suggested_fix", "proposed_signature"],
    "additionalProperties": False,
}


def format_candidates(name: str, exit_code: int | None, candidates) -> str:
    lines = [f"process {name} exited {exit_code}.", "", "candidate log lines:"]
    for candidate in candidates:
        lines.append(f"  line {candidate.line_no}: {candidate.line}")
    if candidates:
        lines += ["", "context around the highest-scoring candidate:", "```",
                  candidates[0].excerpt[:2000], "```"]
    return "\n".join(lines)


async def explain_process_failure(
    llm,
    *,
    name: str,
    exit_code: int | None,
    candidates,
    tier1: str = "",
    max_retries: int = EXPLAIN_MAX_RETRIES,
) -> ProcessExplanation:
    """Pick the causing line out of scored candidates (§5.5 tier 3).

    Runs only when the signature library and the keyword scan both failed to
    settle it — the model is the expensive layer, so it is the last one. It
    chooses among lines that deterministic code retrieved rather than
    searching itself: grep is better at finding, a model is better at judging
    which of five error-ish lines actually stopped the run.
    """
    system = PROCESS_SYSTEM_PROMPT
    if tier1:
        system += f"\n\nStanding site notes:\n{tier1}"
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": format_candidates(name, exit_code, candidates)},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=PROCESS_SCHEMA,
            schema_name="process_explanation",
            max_tokens=1024,
        )
        try:
            explanation = ProcessExplanation.model_validate(
                json.loads(response.content)
            )
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
            continue
        # Quote-or-admit: a cause line that is not in the candidates was
        # invented, which is the failure mode this whole tier risks.
        known = {c.line.strip() for c in candidates}
        if explanation.cause_line.strip() and explanation.cause_line.strip() not in known:
            explanation.cause_line = ""
            explanation.why = (
                f"{explanation.why} [unverified — the quoted line is not in the log]"
            )
        return explanation
    raise ExplainerError(
        f"No valid process explanation after {max_retries + 1} attempts: {last_error}"
    )


async def explain_failure(
    llm,
    report: TriageReport,
    *,
    tier1: str = "",
    max_retries: int = EXPLAIN_MAX_RETRIES,
) -> FailureExplanation:
    system = SYSTEM_PROMPT
    if tier1:  # tier 1 standing notes go into every agent's prompt (§6.1)
        system += f"\n\nStanding site notes:\n{tier1}"
    conversation = [
        {"role": "system", "content": system},
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
