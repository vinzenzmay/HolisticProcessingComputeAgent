"""Prompt building blocks (§4.3 prompt assembly).

Empirically validated against the live Qwen3.6 backend: without the
respond-vs-tool guidance the model routes its own words through action tools
(e.g. wrapping a greeting in an ``echo`` call); with it, conversational turns
reliably become direct responses.
"""

RESPOND_VS_TOOL_GUIDANCE = (
    "You are an assistant with tools. Tools perform real actions. "
    "Use a tool only when the user asks for an action a tool performs. "
    "For conversation, questions, and greetings, always answer directly "
    'with {"action": "respond", ...} — never route your own words through a tool.'
)


def orchestrator_system_prompt(*, environment: str = "") -> str:
    """System prompt for the orchestrator; dynamic facts injected per render."""
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
    ]
    if environment:
        parts.append(environment)
    return "\n\n".join(parts)
