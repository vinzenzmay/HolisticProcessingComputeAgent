"""Doc-researcher, exact-lookup mode (§4.2): grounded answers with citations.

A firewalled sub-loop: the question enters a fresh conversation with only the
read-only doc tools; raw lookups and man-page text stay inside; the bounded
answer flows back to the orchestrator. Per the grounded answering policy,
claims carry their source, and anything retrieval could not confirm is marked
``[ungrounded — not in indexed docs]``.
"""

from __future__ import annotations

from hpca.agent.context import ToolContext
from hpca.agent.middleware import DecisionError, DirectResponse, decide

MAX_ROUNDS = 6

RESEARCHER_SYSTEM = (
    "You are a documentation researcher for HPC tools and libraries. Answer "
    "the question using ONLY your tools (symbol lookup, man pages, indexed "
    "source). Cite the source of every claim, e.g. (man:grep) or the indexed "
    "signature. If the tools cannot confirm an answer, say what you did find "
    "and mark unverified statements with the exact text "
    "[ungrounded — not in indexed docs]. Be concise: a few sentences."
)


def _research_tools():
    from hpca.agent.doc_tools import RESEARCH_TOOL_NAMES, add_doc_tools
    from hpca.agent.tools import ToolRegistry

    return add_doc_tools(ToolRegistry()).subset(RESEARCH_TOOL_NAMES)


async def research(
    llm,
    question: str,
    ctx: ToolContext,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> str:
    tools = _research_tools()
    system = RESEARCHER_SYSTEM
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    for _ in range(max_rounds):
        try:
            decision = await decide(llm, conversation, tools)
        except DecisionError as e:
            return f"[doc-researcher failed: {e}]"
        if isinstance(decision, DirectResponse):
            return decision.text
        try:
            output = await decision.execute(ctx)
            result = f"[tool result] {decision.tool.name}: {output}"
        except Exception as e:
            result = f"[tool error] {decision.tool.name}: {e}"
        conversation = conversation + [
            {
                "role": "assistant",
                "content": f"(called {decision.tool.name})",
            },
            {"role": "user", "content": result},
        ]
    return (
        "[doc-researcher exhausted its lookup budget without a final answer; "
        "partial findings are unavailable]"
    )
