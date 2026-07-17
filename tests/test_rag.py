"""Tests for hpca.embeddings and hpca.rag (§5.6.2): vector retrieval."""

import json

import httpx
import pytest

from hpca.embeddings import EmbeddingClient, EmbeddingError
from hpca.rag import RagStore, chunk_text

# ---------------------------------------------------------------- chunking


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_chunks_bounded(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
        chunks = chunk_text(text, max_chars=800)
        assert len(chunks) > 1
        assert all(len(c) <= 800 for c in chunks)

    def test_splits_on_paragraphs_when_possible(self):
        text = "First paragraph.\n\nSecond paragraph.\n\n" + "x" * 720
        chunks = chunk_text(text, max_chars=750)
        assert "First paragraph." in chunks[0]
        assert chunks[0].endswith("Second paragraph.")

    def test_oversized_single_paragraph_hard_split(self):
        chunks = chunk_text("y" * 2000, max_chars=800)
        assert all(len(c) <= 800 for c in chunks)
        assert "".join(chunks).count("y") == 2000

    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []


# --------------------------------------------------------- embedding client


def make_client(handler, **kwargs) -> EmbeddingClient:
    return EmbeddingClient(
        base_url="http://test/v1",
        model="mini",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def embedding_response(inputs):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [float(i), 1.0]}
            for i in range(len(inputs))
        ],
        "model": "mini",
    }


class TestEmbeddingClient:
    async def test_embed_shapes_request_and_parses(self):
        seen = {}

        def handler(request):
            body = json.loads(request.content)
            seen.update(body)
            return httpx.Response(200, json=embedding_response(body["input"]))

        client = make_client(handler)
        vectors = await client.embed(["a", "b"])
        assert seen["model"] == "mini"
        assert seen["input"] == ["a", "b"]
        assert vectors == [[0.0, 1.0], [1.0, 1.0]]

    async def test_batches_large_input(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(len(body["input"]))
            return httpx.Response(200, json=embedding_response(body["input"]))

        client = make_client(handler)
        vectors = await client.embed([f"t{i}" for i in range(150)])
        assert len(vectors) == 150
        assert calls == [64, 64, 22]

    async def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        client = make_client(handler)
        with pytest.raises(EmbeddingError):
            await client.embed(["x"])

    async def test_empty_input_no_request(self):
        def handler(request):
            raise AssertionError("no request expected")

        client = make_client(handler)
        assert await client.embed([]) == []


# ------------------------------------------------------------- vector store


@pytest.fixture
def store(tmp_path):
    s = RagStore(tmp_path / "rag.db")
    yield s
    s.close()


V = {
    "cats": [1.0, 0.0, 0.0],
    "dogs": [0.9, 0.1, 0.0],
    "slurm": [0.0, 1.0, 0.0],
    "quota": [0.0, 0.0, 1.0],
}


class TestRagStore:
    def test_add_and_query_nearest(self, store):
        store.add("doc.md", ["about cats", "about slurm"], [V["cats"], V["slurm"]])
        hits = store.query(V["dogs"], k=1)
        assert len(hits) == 1
        assert hits[0].text == "about cats"
        assert hits[0].source == "doc.md"

    def test_query_ranked_by_distance(self, store):
        store.add(
            "d", ["cats", "slurm", "quota"], [V["cats"], V["slurm"], V["quota"]]
        )
        hits = store.query(V["dogs"], k=3)
        assert [h.text for h in hits] == ["cats", "slurm", "quota"]

    def test_clear_source(self, store):
        store.add("a.md", ["one"], [V["cats"]])
        store.add("b.md", ["two"], [V["slurm"]])
        store.clear_source("a.md")
        assert store.count() == 1
        assert store.query(V["cats"], k=5)[0].text == "two"

    def test_dimension_mismatch_raises(self, store):
        store.add("a.md", ["one"], [V["cats"]])
        with pytest.raises(ValueError, match="dimension"):
            store.add("b.md", ["two"], [[1.0, 0.0]])

    def test_persists_across_reopen(self, tmp_path):
        s1 = RagStore(tmp_path / "rag.db")
        s1.add("a.md", ["persistent"], [V["cats"]])
        s1.close()
        s2 = RagStore(tmp_path / "rag.db")
        assert s2.query(V["cats"], k=1)[0].text == "persistent"
        s2.close()

    def test_empty_store_query(self, store):
        assert store.query(V["cats"], k=5) == []
