"""Tests for hpca.embeddings and hpca.rag (§5.6.2): vector retrieval."""

import json

import httpx
import pytest

from hpca.embeddings import EmbeddingClient, EmbeddingError, InputTooLong
from hpca.rag import RagStore, RagStores, chunk_text, embed_fitting, migrate_shared_index

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

    async def test_llama_cpp_batch_refusal_is_too_long(self):
        """llama-server refuses an over-long sequence with 500, not 400, and in
        wording none of the OpenAI markers match. Taken at face value that is
        an EmbeddingError, and the caller drops the whole document instead of
        splitting it and retrying - which is how 941 of 1609 Godot doc pages
        came to be missing from an index that reported no failures."""

        def handler(request):
            return httpx.Response(
                500,
                text=(
                    "input (733 tokens) is too large to process. "
                    "increase the physical batch size"
                ),
            )

        client = make_client(handler)
        with pytest.raises(InputTooLong):
            await client.embed(["x"])

    async def test_openai_context_overrun_is_still_too_long(self):
        """The status code stopped being part of the test; the OpenAI shape
        must still be recognised on the 400 it has always arrived with."""

        def handler(request):
            return httpx.Response(
                400, text="This model's maximum context length is 8192 tokens"
            )

        client = make_client(handler)
        with pytest.raises(InputTooLong):
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


# ------------------------------------------------- oversized-chunk splitting


class LengthCappedEmbedder:
    """Stands in for a 256-token embedder: refuses any text over ``cap`` chars.

    The refusal carries the wording a vLLM server uses, because that string is
    what tells an over-length input apart from a dead backend.
    """

    def __init__(self, cap: int):
        self.cap = cap
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        for text in texts:
            if len(text) > self.cap:
                raise InputTooLong(
                    "Embedding request failed (400): This model's maximum "
                    "context length is 256 tokens. However, you requested 0 "
                    "output tokens and your prompt contains at least 257 "
                    "input tokens. Please reduce the length of the messages."
                )
        return [[float(len(t)), 0.0, 0.0] for t in texts]


class DeadEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        raise EmbeddingError("All connection attempts failed")


class TestEmbedFitting:
    async def test_short_chunks_pass_through_untouched(self):
        embedder = LengthCappedEmbedder(cap=100)
        chunks = ["one", "two", "three"]
        text, vectors = await embed_fitting(embedder, chunks)
        assert text == chunks
        assert len(vectors) == 3
        assert embedder.calls == [chunks]  # a single request, no splitting

    async def test_one_oversized_chunk_does_not_lose_the_document(self):
        """The regression: a whole document was dropped because one of its
        chunks (a code block, an API table) overran the token window."""
        embedder = LengthCappedEmbedder(cap=200)
        text, vectors = await embed_fitting(
            embedder, ["fine", "x" * 800, "also fine"]
        )
        assert len(text) == len(vectors)
        assert "fine" in text and "also fine" in text  # neighbours survive
        assert all(len(t) <= 200 for t in text)  # the big one came back split
        assert "".join(t for t in text if set(t) == {"x"}) == "x" * 800

    async def test_every_chunk_oversized_still_indexes(self):
        embedder = LengthCappedEmbedder(cap=150)
        text, vectors = await embed_fitting(embedder, ["y" * 900, "z" * 700])
        assert len(text) == len(vectors) and text
        assert all(len(t) <= 150 for t in text)

    async def test_a_dead_backend_is_not_retried(self):
        """Only InputTooLong recurses; anything else propagates at once, so a
        backend that is down is asked once rather than 2^n times."""
        embedder = DeadEmbedder()
        with pytest.raises(EmbeddingError, match="connection attempts"):
            await embed_fitting(embedder, ["a", "b", "c", "d"])
        assert embedder.calls == 1

    async def test_unsplittable_chunk_is_dropped_not_looped(self):
        """A chunk under the floor that is still refused cannot be usefully
        halved; it is dropped so indexing terminates."""
        embedder = LengthCappedEmbedder(cap=1)
        text, vectors = await embed_fitting(embedder, ["still too long"])
        assert text == [] and vectors == []

    async def test_empty_input(self):
        assert await embed_fitting(LengthCappedEmbedder(cap=10), []) == ([], [])


class TestThereIsOneIndexPerProfile:
    """``rag/<profile>.db``, and the shared ``rag.db`` of earlier versions."""

    def test_the_shared_index_becomes_the_default_profile_s(self, tmp_path):
        RagStore(tmp_path / "rag.db").close()
        (tmp_path / "rag.db-wal").write_bytes(b"")
        notices = migrate_shared_index(tmp_path)
        assert not (tmp_path / "rag.db").exists()
        assert (tmp_path / "rag" / "default.db").exists()
        assert (tmp_path / "rag" / "default.db-wal").exists()
        assert "per profile" in notices[0]

    def test_nothing_to_move_says_nothing(self, tmp_path):
        assert migrate_shared_index(tmp_path) == []

    def test_an_existing_default_index_is_never_overwritten(self, tmp_path):
        RagStore(tmp_path / "rag.db").close()
        (tmp_path / "rag").mkdir()
        (tmp_path / "rag" / "default.db").write_bytes(b"keep me")
        notices = migrate_shared_index(tmp_path)
        assert (tmp_path / "rag.db").exists()
        assert (tmp_path / "rag" / "default.db").read_bytes() == b"keep me"
        assert "left rag.db where it is" in notices[0]

    async def test_profiles_get_separate_stores(self, tmp_path):
        stores = RagStores(tmp_path)
        a = await stores.open("a")
        a.add("doc", ["only in a"], [[1.0, 0.0]])
        assert (await stores.open("b")).count() == 0
        assert await stores.open("a") is a
        stores.close()

    async def test_through_the_cache_they_are_worked_on_locally(self, tmp_path):
        from hpca.dbcache import DbCache

        home, local = tmp_path / "home", tmp_path / "local"
        home.mkdir()
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        stores = RagStores(home, cache)
        store = await stores.open("lab")
        assert store.path == local / "rag" / "lab.db"
        store.add("doc", ["x"], [[1.0, 0.0]])
        cache.sync()
        assert (home / "rag" / "lab.db").exists()
        await stores.remove("lab")
        assert not (home / "rag" / "lab.db").exists()
        assert not (local / "rag" / "lab.db").exists()
        cache.sync(force=True)
        assert not (home / "rag" / "lab.db").exists()  # not synced back
        stores.close()
        cache.release()

    async def test_a_copy_of_nothing_is_nothing(self, tmp_path):
        stores = RagStores(tmp_path)
        assert await stores.copy("never-indexed", "copy") is False
        assert not (tmp_path / "rag" / "copy.db").exists()
