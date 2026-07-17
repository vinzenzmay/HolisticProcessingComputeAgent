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
    "the new key. When the user does NOT give you a path, find it yourself "
    "(see below) and register what you found. Do not ask the user to "
    "register paths — that is your job."
)

# Without this the model has no idea it may look around: every file tool takes
# a key, and keys came from the user's message, so "find X somewhere on this
# system" looked impossible and it narrated instead of acting.
DISCOVERY_GUIDANCE = (
    "You can look around this system, and you should rather than guess or ask. "
    "To find a file, learn whether a program is installed, or see what "
    "environments exist: write a small bash script with create_script and run "
    "it with run_script, which waits and gives you its output. Useful lines: "
    "`find <dirs> -maxdepth <n> -iname '<pattern>' 2>/dev/null | head -20` to "
    "locate files; `command -v samtools` and `samtools --version` to check a "
    "program; `conda env list`, `module avail <name> 2>&1` to find one that is "
    "not on PATH. Keep searches bounded so they finish in seconds: start from "
    "likely roots rather than /, cap the depth, pipe through head. Then "
    "register_path the paths it printed and use them by key."
)


def environment_facts() -> str:
    """Dynamic facts, rendered per call — never stored as memories (§4.3)."""
    from datetime import datetime

    return f"Current date and time: {datetime.now():%Y-%m-%d %H:%M} (local)."


SKILLS_GUIDANCE = (
    "The user has defined skills: written procedures for specific tasks. "
    "When a request matches one, call read_skill to get the procedure and "
    "follow it. Available skills:"
)


def orchestrator_system_prompt(
    *,
    environment: str = "",
    tier1: str = "",
    tier2: str = "",
    skills: str = "",
) -> str:
    """System prompt for the orchestrator; dynamic facts injected per render.

    Tier 1 memories go into *every* agent's prompt, tier 2 only here (§6.1).
    Skills are listed by name/description only; bodies are fetched on demand.
    """
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
        PATH_WORKFLOW_GUIDANCE,
        DISCOVERY_GUIDANCE,
        GROUNDED_ANSWERING_GUIDANCE,
    ]
    if skills:
        parts.append(f"{SKILLS_GUIDANCE}\n{skills}")
    if tier1:
        parts.append(f"Standing site notes:\n{tier1}")
    if tier2:
        parts.append(f"Learnings and preferences from earlier sessions:\n{tier2}")
    parts.append(environment or environment_facts())
    return "\n\n".join(parts)
