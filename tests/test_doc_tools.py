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
        # no index_docs / ask_docs: the researcher reads, never writes or recurses
        tools = add_doc_tools(ToolRegistry()).subset(RESEARCH_TOOL_NAMES)
        assert set(tools.names()) == {
            "lookup_symbol",
            "read_manpage",
            "read_source",
            "search_docs",
        }


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


# ------------------------------------------------------ semantic search (11)

from hpca.embeddings import EmbeddingClient  # noqa: E402
from hpca.rag import RagStore  # noqa: E402


class FakeEmbedder:
    """Deterministic keyword-space embeddings; no backend needed."""

    KEYWORDS = ["bam", "slurm", "memory", "quota"]

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def embed(self, texts):
        from hpca.embeddings import EmbeddingError

        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("backend down")
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [1.0 if kw in lowered else 0.0 for kw in self.KEYWORDS] + [0.1]
            )
        return vectors


@pytest.fixture
def rag_ctx(ctx, tmp_path):
    ctx.rag = RagStore(tmp_path / "rag.db")
    ctx.embedder = FakeEmbedder()
    yield ctx
    ctx.rag.close()


class TestSearchDocs:
    async def test_returns_nearest_chunks_with_sources(self, tools, rag_ctx):
        rag_ctx.rag.add(
            "guide.md",
            ["How to subset a BAM by region", "How to request more memory"],
            await rag_ctx.embedder.embed(
                ["How to subset a BAM by region", "How to request more memory"]
            ),
        )
        result = await call(tools, "search_docs", rag_ctx, query="subset bam file")
        assert "subset a BAM" in result
        assert "guide.md" in result

    async def test_empty_index_is_explicit(self, tools, rag_ctx):
        result = await call(tools, "search_docs", rag_ctx, query="anything")
        assert "empty" in result.lower()

    async def test_unconfigured_points_at_exact_tools(self, tools, ctx):
        result = await call(tools, "search_docs", ctx, query="anything")
        assert "lookup_symbol" in result

    async def test_embedding_failure_degrades_gracefully(self, tools, rag_ctx):
        rag_ctx.rag.add("d.md", ["bam stuff"], [[1.0, 0, 0, 0, 0.1]])
        rag_ctx.embedder = FakeEmbedder(fail=True)
        result = await call(tools, "search_docs", rag_ctx, query="bam")
        assert "unavailable" in result.lower()


class TestIndexDocsDir:
    async def test_indexes_markdown_and_text(self, tools, rag_ctx, tmp_path):
        docs = tmp_path / "docs"
        (docs / "sub").mkdir(parents=True)
        (docs / "a.md").write_text("Slurm job submission guide.")
        (docs / "sub" / "b.txt").write_text("Memory limits on the cluster.")
        (docs / "ignore.png").write_bytes(b"\x89PNG")
        rag_ctx.registry.register("docs", docs)
        result = await call(
            tools, "index_docs", rag_ctx, what="docs_dir", target="docs"
        )
        assert "Indexed 2 documents" in result
        assert rag_ctx.rag.count() == 2

    async def test_reindex_replaces_not_duplicates(self, tools, rag_ctx, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("Slurm guide.")
        rag_ctx.registry.register("docs", docs)
        await call(tools, "index_docs", rag_ctx, what="docs_dir", target="docs")
        await call(tools, "index_docs", rag_ctx, what="docs_dir", target="docs")
        assert rag_ctx.rag.count() == 1

    async def test_without_embedder_refuses_clearly(self, tools, ctx, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("x")
        ctx.registry.register("docs", docs)
        result = await call(tools, "index_docs", ctx, what="docs_dir", target="docs")
        assert "not configured" in result

    async def test_manpage_indexing_also_feeds_rag(self, tools, rag_ctx, monkeypatch):
        async def fake_fetch(name):
            return MAN_PAGE if name == "samtools-view" else None

        monkeypatch.setattr(doc_tools_module, "fetch_manpage", fake_fetch)
        result = await call(
            tools, "index_docs", rag_ctx, what="manpages", target="samtools-view"
        )
        assert "2 flags" in result  # symbol table
        assert "semantic search" in result  # vector store
        assert rag_ctx.rag.count() >= 1


class TestResearchToolset:
    def test_search_docs_available_to_researcher(self):
        tools = add_doc_tools(ToolRegistry()).subset(RESEARCH_TOOL_NAMES)
        assert "search_docs" in tools.names()
