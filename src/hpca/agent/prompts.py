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


GROUNDED_ANSWERING_GUIDANCE = (
    "Never answer technical questions about tools, libraries, APIs, CLI "
    "flags, file formats, or error messages from memory — call ask_docs and "
    "relay its cited answer. Answer directly only for conversation and for "
    "information already present in this conversation."
)

PATH_WORKFLOW_GUIDANCE = (
    "Tools take registry KEYS, never literal paths. When the user mentions a "
    "path that is not registered yet, first call register_path (copy the "
    "path from the user's message exactly), then call the actual tool with "
    "the new key. Do not ask the user to register paths — that is your job."
)


def environment_facts() -> str:
    """Dynamic facts, rendered per call — never stored as memories (§4.3)."""
    from datetime import datetime

    return f"Current date and time: {datetime.now():%Y-%m-%d %H:%M} (local)."


def orchestrator_system_prompt(
    *, environment: str = "", tier1: str = "", tier2: str = ""
) -> str:
    """System prompt for the orchestrator; dynamic facts injected per render.

    Tier 1 memories go into *every* agent's prompt, tier 2 only here (§6.1).
    """
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
        PATH_WORKFLOW_GUIDANCE,
        GROUNDED_ANSWERING_GUIDANCE,
    ]
    if tier1:
        parts.append(f"Standing site notes:\n{tier1}")
    if tier2:
        parts.append(f"Learnings and preferences from earlier sessions:\n{tier2}")
    parts.append(environment or environment_facts())
    return "\n\n".join(parts)
