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


# Scoped deliberately: the earlier blanket "never answer from memory" made the
# agent research its OWN tools before calling them, which is pure waste — their
# schemas are already in this prompt. The hallucination risk is in the *external*
# programs it drives from scripts (samtools, minimap2, ...), not in its toolbox.
GROUNDED_ANSWERING_GUIDANCE = (
    "Your own tools need no research: you already have their schemas, so never "
    "look up documentation before calling one — just call it. Grounding applies "
    "to EXTERNAL software instead: command-line programs you invoke from scripts "
    "(samtools, minimap2, bcftools, ...), libraries, file formats, and error "
    "messages. Never state their flags or syntax from memory — call ask_docs and "
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
    "Use run_bash for a one-shot check — it writes, runs and returns the "
    "output of a small bash script in one step (create_script + run_script is "
    "for scripts worth keeping, e.g. submitted to Slurm). Useful lines: "
    "`find <dirs> -maxdepth <n> -iname '<pattern>' 2>/dev/null | head -20` to "
    "locate files; `command -v samtools` to check a program. For a tool that "
    "is not on PATH, first see which package managers this site actually has "
    "(`command -v conda mamba micromamba spack module apptainer`) and query "
    "only the ones that answered, or search the likely install roots directly "
    "with find. Keep searches "
    "bounded so they finish in seconds: start from likely roots rather than /, "
    "cap the depth, pipe through head. Then register_path the paths it printed "
    "and use them by key."
)

# The single most common way the agent burns its tool budget: it cannot run an
# environment-managed tool and thrashes trying to activate one. The idiom that
# ends that thrashing lives in ENVIRONMENT_TOOL_GUIDANCE below.
SCRIPT_GUIDANCE = (
    "Run real work (a tool, a pipeline) with create_script then start_script — "
    "those bash scripts run fail-fast (`set -euo pipefail` is added), so a "
    "failed command stops the script and is reported as failed. Because of "
    "that, never end a script with an unconditional `echo \"Done\"`: let the "
    "exit code report success. run_bash is for quick look-around checks only. "
    "Build each command plainly on one line — real flags and paths separated "
    "by single spaces, nothing else. Do NOT insert quotes, commas or `\\` "
    "line-continuations between arguments; a stray `\",` turns your command "
    "into garbage the tool rejects. Reference paths by their registered value "
    "or write the literal path; do not leave a shell variable unset."
)


# Retired the conda-specific idioms. They were wrong on the very machine this
# runs on, where conda is not installed at all (mamba is), and wrong again on
# any site using modules, Spack or containers. A live session followed them and
# failed twice — `conda run` then `mamba run`, both "command not found" — and
# only succeeded once it called the binary by its full path. So teach the form
# that works whatever installed the tool, and teach checking before assuming.
# The real fix is a provider registry rendering these facts per site; until
# then this says only what is true regardless of packaging system.
ENVIRONMENT_TOOL_GUIDANCE = (
    "Scientific tools here are usually NOT on PATH: they live in package "
    "environments or module trees, and which system manages them differs per "
    "site (conda, mamba, micromamba, Spack, Lmod, containers). Assume nothing "
    "— neither that a tool name works as typed, nor that any particular "
    "package manager exists. Find the executable first (see below), then call "
    "it by its full binary path in your script: that is the one form that "
    "works whatever installed it. A wrapper like `conda run -n <env> <tool>` "
    "only works if that wrapper is itself installed, so check with "
    "`command -v conda` before relying on one. Do not try to `source` an "
    "activate script: it is rarely where you expect and wastes a step."
)


def environment_facts() -> str:
    """Volatile facts, rendered per turn — never stored as memories (§4.3).

    These ride the *tail* of the turn (the user message's ``api_content``
    sidecar), NOT the system prompt: the minute-granularity timestamp used to
    sit at the front of the prompt and invalidated the backend's prefix KV
    cache for the whole conversation every time the clock ticked over a minute.
    Keeping volatile content after the stable history preserves the cache.
    """
    from datetime import datetime

    return f"Current date and time: {datetime.now():%Y-%m-%d %H:%M} (local)."


# The tool schema alone does not teach *when* to reach for recall; small
# models ask the user to repeat themselves instead. The second half is the
# Hermes "source-first limit": recall is what was SAID, not what currently IS.
SESSION_SEARCH_GUIDANCE = (
    "Past sessions are searchable with session_search. When the user refers "
    "to something from an earlier conversation ('like last time', 'the "
    "pipeline we set up'), search before asking them to repeat it. "
    "session_search shows what was said back then — never treat it as "
    "evidence about the current state of files, jobs, or the cluster; check "
    "the system itself for that."
)


# Adapted from Hermes' MEMORY_GUIDANCE. The declarative-vs-imperative rule is
# the load-bearing one: an imperative memory ("always use --no-mmap") is
# re-read as a standing order in later sessions and overrides what the user is
# actually asking for now — a failure mode small models are especially prone
# to, since they weight instructions in context over the current request.
MEMORY_GUIDANCE = (
    "You have memory that persists across sessions, and the memory tool "
    "writes to it. Save proactively when the user states a preference, "
    "corrects you, or tells you a durable fact about this site — the best "
    "memory is one that stops the user having to repeat themselves. "
    "Priority: corrections and preferences first, then site facts, then "
    "workarounds. Write memories as declarative FACTS, not instructions to "
    "yourself: “the user prefers R over Python” is right, “always answer in "
    "R” is wrong — an instruction gets re-read as a standing order in a "
    "later session and overrides what is being asked then. Do NOT save task "
    "progress, what you did this session, file names, job ids, or anything "
    "that will be stale in a week; past sessions are searchable instead. "
    "The user approves every write, so propose rather than agonize."
)


SKILLS_GUIDANCE = (
    "The user has defined skills: written procedures for specific tasks. "
    "When a request matches one, call read_skill to get the procedure and "
    "follow it. Available skills:"
)


def orchestrator_system_prompt(
    *,
    tier1: str = "",
    tier2: str = "",
    tier1_meter: str = "",
    tier2_meter: str = "",
    skills: str = "",
    session_search: bool = False,
    memory_tool: bool = False,
) -> str:
    """System prompt for the orchestrator; the cacheable prompt *prefix*.

    Everything here is stable across a session so the backend's prefix KV
    cache survives from turn to turn: only memory approvals, a skill change or
    a mode switch move it. Volatile facts (the date/time) deliberately live at
    the tail instead — see ``environment_facts``.

    Tier 1 memories go into *every* agent's prompt, tier 2 only here (§6.1).
    Skills are listed by name/description only; bodies are fetched on demand.
    The meters show how full each memory tier is — groundwork for the model
    managing its own memory under a hard budget (redesign Phase 3).
    """
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
        PATH_WORKFLOW_GUIDANCE,
        DISCOVERY_GUIDANCE,
        ENVIRONMENT_TOOL_GUIDANCE,
        SCRIPT_GUIDANCE,
        GROUNDED_ANSWERING_GUIDANCE,
    ]
    if session_search:
        parts.append(SESSION_SEARCH_GUIDANCE)
    if memory_tool:
        parts.append(MEMORY_GUIDANCE)
    if skills:
        parts.append(f"{SKILLS_GUIDANCE}\n{skills}")
    if tier1:
        label = _metered("Standing site notes", tier1_meter)
        parts.append(f"{label}:\n{tier1}")
    if tier2:
        label = _metered(
            "Learnings and preferences from earlier sessions", tier2_meter
        )
        parts.append(f"{label}:\n{tier2}")
    return "\n\n".join(parts)


def _metered(label: str, meter: str) -> str:
    return f"{label} [{meter}]" if meter else label
