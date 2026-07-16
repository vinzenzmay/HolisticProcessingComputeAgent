"""Tests for hpca.agent.doc_tools and the doc-researcher (§5.6, §4.2)."""

import json
import shutil

import pytest

from hpca.agent import doc_tools as doc_tools_module
from hpca.agent.context import ToolContext
from hpca.agent.doc_tools import RESEARCH_TOOL_NAMES, add_ask_docs, add_doc_tools
from hpca.agent.researcher import research
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.llm import ChatResponse
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.symbols import Symbol, SymbolIndex

MAN_PAGE = """\
NAME
       samtools view - views SAM/BAM/CRAM files

OPTIONS
       -b      Output in the BAM format.

       -q INT Skip alignments with MAPQ smaller than INT [0].
"""


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    context = ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        symbols=SymbolIndex(conn),
    )
    yield context
    conn.close()


@pytest.fixture
def tools():
    return add_doc_tools(ToolRegistry())


async def call(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    return await tool.handler(tool.params.model_validate(kwargs), ctx)


class TestLookupSymbol:
    async def test_found_with_signature_and_source(self, tools, ctx):
        ctx.symbols.add(
            [
                Symbol(
                    name="align",
                    kind="function",
                    parent="tools",
                    signature="align(reads, ref, *, threads=1)",
                    params=["reads", "ref", "threads"],
                    source="tools.py",
                    doc="Align reads.",
                )
            ]
        )
        result = await call(tools, "lookup_symbol", ctx, name="align")
        assert "align(reads, ref" in result
        assert "tools.py" in result

    async def test_missing_is_explicit_about_absence(self, tools, ctx):
        result = await call(tools, "lookup_symbol", ctx, name="frobnicate")
        assert "NOT in the symbol index" in result


class TestReadManpage:
    async def test_uses_fetch_and_bounds(self, tools, ctx, monkeypatch):
        async def fake_fetch(name):
            return "\n".join(f"line {i}" for i in range(1000))

        monkeypatch.setattr(doc_tools_module, "fetch_manpage", fake_fetch)
        result = await call(tools, "read_manpage", ctx, name="grep")
        assert "omitted" in result
        assert len(result.splitlines()) <= 401

    async def test_missing_manpage(self, tools, ctx, monkeypatch):
        async def fake_fetch(name):
            return None

        monkeypatch.setattr(doc_tools_module, "fetch_manpage", fake_fetch)
        result = await call(tools, "read_manpage", ctx, name="nosuchtool")
        assert "No man page" in result


class TestReadSource:
    async def test_numbered_range(self, tools, ctx, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("\n".join(f"code line {i}" for i in range(1, 51)))
        ctx.registry.register("code", f)
        result = await call(
            tools, "read_source", ctx, registry_key="code", start_line=10, end_line=12
        )
        assert result.splitlines() == [
            "10: code line 10",
            "11: code line 11",
            "12: code line 12",
        ]


class TestIndexDocs:
    async def test_python_source(self, tools, ctx, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text("def fn(a, b=2):\n    pass\n")
        ctx.registry.register("pkg", pkg)
        result = await call(
            tools, "index_docs", ctx, what="python_source", target="pkg"
        )
        assert "1 Python files" in result
        assert ctx.symbols.kwargs_for("fn") == ["a", "b"]

    async def test_manpages_via_fake_fetch(self, tools, ctx, monkeypatch):
        async def fake_fetch(name):
            return MAN_PAGE if name == "samtools-view" else None

        monkeypatch.setattr(doc_tools_module, "fetch_manpage", fake_fetch)
        result = await call(
            tools, "index_docs", ctx, what="manpages",
            target="samtools-view bcftools",
        )
        assert "samtools-view (2 flags)" in result
        assert "bcftools" in result  # reported missing
        assert ctx.symbols.flags_for("samtools-view") == ["-b", "-q"]

    @pytest.mark.skipif(shutil.which("man") is None, reason="man not installed")
    async def test_real_manpage_indexing(self, tools, ctx):
        result = await call(tools, "index_docs", ctx, what="manpages", target="grep")
        assert "grep (" in result
        assert "-v" in ctx.symbols.flags_for("grep")


# ------------------------------------------------------------ researcher


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


class TestResearcher:
    async def test_lookup_then_cited_answer(self, ctx):
        ctx.symbols.add(
            [Symbol(name="-q", kind="cli-flag", parent="samtools-view",
                    source="man:samtools-view", doc="Skip low MAPQ.")]
        )
        llm = FakeLLM(
            [
                tool_json("lookup_symbol", name="-q"),
                respond_json("Yes, -q exists (man:samtools-view)."),
            ]
        )
        answer = await research(llm, "Does samtools view have -q?", ctx)
        assert "man:samtools-view" in answer
        # the tool result reached the model in the second round
        assert any("Skip low MAPQ" in m["content"] for m in llm.calls[1])

    async def test_round_budget_exhaustion(self, ctx):
        llm = FakeLLM([tool_json("lookup_symbol", name="x")] * 10)
        answer = await research(llm, "q", ctx, max_rounds=3)
        assert "exhausted" in answer

    async def test_research_tools_are_read_only_subset(self):
        tools = add_doc_tools(ToolRegistry()).subset(RESEARCH_TOOL_NAMES)
        assert set(tools.names()) == {"lookup_symbol", "read_manpage", "read_source"}


class TestAskDocsTool:
    async def test_firewall_returns_only_answer(self, ctx):
        ctx.symbols.add(
            [Symbol(name="-b", kind="cli-flag", parent="samtools-view",
                    source="man:samtools-view", doc="BAM output.")]
        )
        ctx.llm = FakeLLM(
            [
                tool_json("lookup_symbol", name="-b"),
                respond_json("-b outputs BAM (man:samtools-view)."),
            ]
        )
        registry = add_ask_docs(ToolRegistry())
        result = await call(
            registry, "ask_docs", ctx, question="What does samtools view -b do?"
        )
        assert result == "-b outputs BAM (man:samtools-view)."
