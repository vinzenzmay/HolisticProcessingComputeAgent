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


PATH_WORKFLOW_GUIDANCE = (
    "Tools take registry KEYS, never literal paths. When the user mentions a "
    "path that is not registered yet, first call register_path (copy the "
    "path from the user's message exactly), then call the actual tool with "
    "the new key. Do not ask the user to register paths — that is your job."
)


def orchestrator_system_prompt(*, environment: str = "") -> str:
    """System prompt for the orchestrator; dynamic facts injected per render."""
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
        PATH_WORKFLOW_GUIDANCE,
    ]
    if environment:
        parts.append(environment)
    return "\n\n".join(parts)
