"""Tests for hpca.doc_index: incremental, backgrounded docs_dir indexing (§5.6.2)."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from hpca import doc_index
from hpca.agent.context import ToolContext
from hpca.agent.doc_tools import add_doc_tools
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.doc_index import DocIndexer
from hpca.embeddings import EmbeddingError
from hpca.protocol import Notify
from hpca.rag import RagStore, Recorded
from hpca.runner import ProcessRunner
from hpca.symbols import SymbolIndex


class Embedder:
    """Counts what it is asked to embed; fails on demand."""

    def __init__(self, model="model-a"):
        self.model = model
        self.texts: list[str] = []
        self.fail_on: set[str] = set()
        self.down = False
        self.gate: asyncio.Event | None = None

    async def embed(self, texts):
        if self.gate is not None:
            await self.gate.wait()
        if self.down or any(word in t for t in texts for word in self.fail_on):
            raise EmbeddingError("backend down")
        self.texts += texts
        return [[float(len(t)), 1.0, 0.5] for t in texts]


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    rag = RagStore(tmp_path / "rag.db")
    context = ToolContext(
        workdir=tmp_path,
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        symbols=SymbolIndex(conn),
        session_id="s1",
        rag=rag,
        embedder=Embedder(),
    )
    yield context
    rag.close()
    conn.close()


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "papers"
    root.mkdir()
    return root


def write(root, count, start=0):
    for i in range(start, start + count):
        (root / f"paper{i:04d}.txt").write_text(f"Paper {i}.\n\nIts findings, part {i}.")


async def index(ctx, root):
    tools = add_doc_tools(ToolRegistry())
    tool = tools.get("index_docs")
    return await tool.handler(
        tool.params(what="docs_dir", target=str(root)), ctx
    )


def sources(ctx) -> set[str]:
    rows = ctx.rag._conn.execute("SELECT DISTINCT source FROM chunks").fetchall()
    return {row[0] for row in rows}


class Recorder:
    def __init__(self):
        self.events = []
        self.submitted = []

    def indexer(self):
        return DocIndexer(
            emit=self.events.append,
            submit_event=lambda sid, text: self.submitted.append((sid, text)),
        )

    def notes(self):
        return [e.text for e in self.events if isinstance(e, Notify)]


async def finish(indexer):
    await asyncio.wait_for(indexer._task, 10)


class TestInline:
    async def test_a_small_directory_is_indexed_in_the_call(self, ctx, docs):
        write(docs, 3)
        result = await index(ctx, docs)
        assert "Indexed 3 documents" in result
        assert len(sources(ctx)) == 3

    async def test_a_second_run_embeds_nothing_that_did_not_change(self, ctx, docs):
        write(docs, 3)
        await index(ctx, docs)
        ctx.embedder.texts.clear()
        result = await index(ctx, docs)
        assert ctx.embedder.texts == []
        assert "Indexed 0 documents" in result
        assert "3 unchanged" in result

    async def test_a_changed_file_is_embedded_again_in_place(self, ctx, docs):
        write(docs, 3)
        await index(ctx, docs)
        ctx.embedder.texts.clear()
        (docs / "paper0001.txt").write_text("Rewritten entirely, and longer than before.")
        result = await index(ctx, docs)
        assert "Indexed 1 documents" in result and "2 unchanged" in result
        assert ctx.embedder.texts == ["Rewritten entirely, and longer than before."]
        assert ctx.rag.count() == 3  # replaced, not added to

    async def test_a_touched_file_is_read_but_not_embedded(self, ctx, docs):
        write(docs, 1)
        await index(ctx, docs)
        ctx.embedder.texts.clear()
        path = docs / "paper0000.txt"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert "1 unchanged" in await index(ctx, docs)
        assert ctx.embedder.texts == []
        # And the new mtime is remembered, so the next scan does not read it.
        recorded = ctx.rag.recorded_under(str(docs))[str(path)]
        assert recorded.mtime_ns == stat.st_mtime_ns + 10**9

    async def test_a_file_gone_from_the_directory_leaves_the_index(self, ctx, docs):
        write(docs, 3)
        await index(ctx, docs)
        (docs / "paper0002.txt").unlink()
        result = await index(ctx, docs)
        assert "Removed 1" in result
        assert str(docs / "paper0002.txt") not in sources(ctx)
        assert len(sources(ctx)) == 2

    async def test_a_different_model_embeds_everything_again(self, ctx, docs):
        write(docs, 2)
        await index(ctx, docs)
        ctx.embedder = Embedder(model="model-b")
        result = await index(ctx, docs)
        assert "Indexed 2 documents" in result
        assert len(ctx.embedder.texts) == 2

    async def test_other_sources_are_left_alone(self, ctx, docs):
        ctx.rag.add("man:grep", ["grep searches"], [[1.0, 1.0, 1.0]])
        write(docs, 1)
        await index(ctx, docs)
        (docs / "paper0000.txt").unlink()
        await index(ctx, docs)
        assert sources(ctx) == {"man:grep"}

    async def test_an_emptied_file_takes_its_old_chunks_with_it(self, ctx, docs):
        write(docs, 1)
        await index(ctx, docs)
        (docs / "paper0000.txt").write_text("   \n\n  ")
        result = await index(ctx, docs)
        assert "1 had no text" in result
        assert ctx.rag.count() == 0

    async def test_not_a_directory_is_said(self, ctx, tmp_path):
        result = await index(ctx, tmp_path / "nowhere")
        assert "not a directory" in result

    async def test_one_failed_document_is_reported_and_the_rest_indexed(self, ctx, docs):
        write(docs, 3)
        (docs / "paper0001.txt").write_text("poison")
        ctx.embedder.fail_on = {"poison"}
        result = await index(ctx, docs)
        assert "Indexed 2 documents" in result
        assert "1 could NOT be indexed" in result and "paper0001.txt" in result
        # Not recorded, so the next run tries it again.
        ctx.embedder.fail_on = set()
        assert "Indexed 1 documents" in await index(ctx, docs)

    async def test_a_backend_that_is_down_stops_the_job(self, ctx, docs):
        write(docs, 10)
        ctx.embedder.down = True
        result = await index(ctx, docs)
        assert result.startswith("Stopped after 3 of 10 documents")
        assert "picks up where it stopped" in result

    async def test_and_the_next_run_picks_up_where_it_stopped(self, ctx, docs):
        write(docs, 10)
        real = ctx.embedder.embed

        async def dies_after_four(texts):
            if len(ctx.embedder.texts) >= 4:
                raise EmbeddingError("backend down")
            return await real(texts)

        ctx.embedder.embed = dies_after_four
        assert (await index(ctx, docs)).startswith("Stopped after 7 of 10")
        ctx.embedder.embed = real
        ctx.embedder.texts.clear()
        result = await index(ctx, docs)
        assert "Indexed 6 documents" in result and "4 unchanged" in result
        assert len(ctx.embedder.texts) == 6

    async def test_papers_added_later_are_the_only_ones_embedded(self, ctx, docs):
        write(docs, 6)
        await index(ctx, docs)
        write(docs, 4, start=6)
        ctx.embedder.texts.clear()
        assert "Indexed 4 documents" in await index(ctx, docs)
        assert len(ctx.embedder.texts) == 4


class TestBackground:
    @pytest.fixture
    def recorder(self, ctx):
        recorder = Recorder()
        ctx.doc_indexer = recorder.indexer()
        return recorder

    async def test_a_large_directory_returns_at_once(self, ctx, docs, recorder):
        write(docs, doc_index.INLINE_LIMIT + 5)
        ctx.embedder.gate = asyncio.Event()
        result = await index(ctx, docs)
        assert "in the background" in result
        assert ctx.embedder.texts == []
        assert ctx.doc_indexer.busy() is not None
        ctx.embedder.gate.set()
        await finish(ctx.doc_indexer)
        assert len(sources(ctx)) == doc_index.INLINE_LIMIT + 5

    async def test_the_session_that_asked_is_told_when_it_is_done(
        self, ctx, docs, recorder
    ):
        write(docs, doc_index.INLINE_LIMIT + 1)
        await index(ctx, docs)
        await finish(ctx.doc_indexer)
        [(session_id, text)] = recorder.submitted
        assert session_id == "s1"
        assert text.startswith("[indexing finished]")
        assert f"Indexed {doc_index.INLINE_LIMIT + 1} documents" in text

    async def test_the_user_sees_progress_and_the_outcome(self, ctx, docs, recorder):
        write(docs, 40)
        await index(ctx, docs)
        await finish(ctx.doc_indexer)
        notes = recorder.notes()
        # One at each quarter, then the outcome.
        progress = [n for n in notes if "of 40 documents done" in n]
        assert [n.split(":")[1].split(" of")[0] for n in progress] == [" 10", " 20", " 30"]
        assert "Indexed 40 documents" in notes[-1]

    async def test_asking_again_reports_progress(self, ctx, docs, recorder):
        write(docs, doc_index.INLINE_LIMIT + 1)
        ctx.embedder.gate = asyncio.Event()
        await index(ctx, docs)
        again = await index(ctx, docs)
        assert "Still running" in again and "0 of" in again
        ctx.embedder.gate.set()
        await finish(ctx.doc_indexer)

    async def test_a_second_directory_waits_its_turn(self, ctx, docs, tmp_path, recorder):
        write(docs, doc_index.INLINE_LIMIT + 1)
        other = tmp_path / "other"
        other.mkdir()
        write(other, 1)
        ctx.embedder.gate = asyncio.Event()
        await index(ctx, docs)
        assert "Already indexing another directory" in await index(ctx, other)
        ctx.embedder.gate.set()
        await finish(ctx.doc_indexer)

    async def test_stopping_keeps_what_was_finished(self, ctx, docs, recorder):
        write(docs, doc_index.INLINE_LIMIT + 10)
        stop_after = 5

        real = ctx.embedder.embed

        async def slow(texts):
            if len(ctx.embedder.texts) >= stop_after:
                await asyncio.Event().wait()  # until cancelled
            return await real(texts)

        ctx.embedder.embed = slow
        await index(ctx, docs)
        while len(ctx.embedder.texts) < stop_after:
            await asyncio.sleep(0.01)
        await ctx.doc_indexer.stop()
        assert ctx.doc_indexer.busy() is None
        assert len(ctx.rag.recorded_under(str(docs))) == stop_after
        assert recorder.submitted == []  # a cancelled job says nothing


class TestRecordedUnder:
    def test_a_sibling_with_the_same_prefix_is_not_inside(self, tmp_path):
        rag = RagStore(tmp_path / "rag.db")
        mark = Recorded(1, 1, "d", "m")
        rag.record("/data/docs/a.txt", mark)
        rag.record("/data/docs2/b.txt", mark)
        rag.record("/data/do%s/c.txt", mark)
        assert set(rag.recorded_under("/data/docs")) == {"/data/docs/a.txt"}
        assert set(rag.recorded_under("/data/do%s/")) == {"/data/do%s/c.txt"}
        rag.close()


class TestTheService:
    @pytest.fixture
    def service(self, tmp_path, monkeypatch):
        from langgraph.checkpoint.memory import InMemorySaver

        from hpca.core.service import build_service

        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)

        async def db(fn):
            return fn(conn)

        service = build_service(
            settings=Settings(),
            app_dir=tmp_path,
            db=db,
            conn=conn,
            checkpointer=InMemorySaver(),
            llm=object(),
            project_root=tmp_path,
        )
        yield service
        service._deps.extras["rag_stores"].close()
        conn.close()

    def ctx_for(self, service, profile):
        from hpca.core.service import _make_tool_ctx

        backends = SimpleNamespace(client_for=lambda sid: None, embedder=None)
        session = SimpleNamespace(session_id="s1", profile=profile)
        return _make_tool_ctx(
            service._deps, session, None, skills=[], backends=backends, tools=None
        )

    def test_a_tool_context_gets_the_core_s_indexer(self, service):
        indexer = service._deps.extras["doc_indexer"]
        assert isinstance(indexer, DocIndexer)
        assert self.ctx_for(service, "default").doc_indexer is indexer

    async def test_each_profile_searches_its_own_index(self, service, tmp_path):
        godot = await self.ctx_for(service, "godot").rag_store()
        papers = await self.ctx_for(service, "papers").rag_store()
        assert godot is not papers
        assert godot.path == tmp_path / "rag" / "godot.db"
        assert papers.path == tmp_path / "rag" / "papers.db"

    async def test_the_index_is_not_opened_until_a_tool_asks(self, service, tmp_path):
        ctx = self.ctx_for(service, "godot")
        assert ctx.rag is None
        assert not (tmp_path / "rag").exists()
        await ctx.rag_store()
        assert (tmp_path / "rag" / "godot.db").exists()

    async def test_the_same_profile_shares_one_open_store(self, service):
        first = await self.ctx_for(service, "godot").rag_store()
        assert await self.ctx_for(service, "godot").rag_store() is first
