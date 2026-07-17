#!/usr/bin/env python3
"""Validate the embedding sidecar and the RAG path end to end.

Run with both vLLM servers reachable (chat + embeddings):

    uv run python scripts/validate_rag.py [--chat-url URL] [--embed-url URL]

Checks the embedding endpoint, dimension consistency, that semantically
related text ranks above unrelated text, and — if the chat backend is up —
that the doc-researcher answers a prose question from indexed docs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hpca.agent.context import ToolContext  # noqa: E402
from hpca.agent.doc_tools import IndexDocsParams, index_docs  # noqa: E402
from hpca.agent.researcher import research  # noqa: E402
from hpca.config import LLMSettings, Settings  # noqa: E402
from hpca.db import connect, init_db  # noqa: E402
from hpca.embeddings import EmbeddingClient, EmbeddingError  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402
from hpca.rag import RagStore  # noqa: E402
from hpca.registry import PathRegistry  # noqa: E402
from hpca.runner import ProcessRunner  # noqa: E402
from hpca.symbols import SymbolIndex  # noqa: E402

# Two documents, so retrieval has to discriminate rather than return the
# only chunk that exists.
BAM_DOC = """\
# Cluster BAM handling

To subset a BAM file by genomic region, use samtools view with a region
argument. Indexing the BAM first with samtools index is required for
region queries to work.
"""

MEMORY_DOC = """\
# Memory limits

Jobs that exceed their requested memory are killed by the OOM killer.
Request more memory with the sbatch --mem option, for example --mem=40G.
"""

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-url", default="http://localhost:51941/v1")
    parser.add_argument("--embed-url", default="http://localhost:51943/v1")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--chat-model", default=None)
    args = parser.parse_args()

    # discover model names so this works regardless of what is served
    import httpx

    try:
        embed_models = httpx.get(f"{args.embed_url}/models", timeout=5).json()["data"]
        embed_model = args.embed_model or embed_models[0]["id"]
        record("embedding server reachable", True, embed_model)
    except Exception as e:
        record("embedding server reachable", False, str(e))
        print("\nAborting: start the sidecar and forward its port.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="hpca-rag-validate-"))
    conn = connect(tmp / "hpca.db")
    init_db(conn)
    rag = RagStore(tmp / "rag.db")
    embedder = EmbeddingClient(base_url=args.embed_url, model=embed_model)

    # 1. raw embedding call
    try:
        vectors = await embedder.embed(["hello world", "goodbye world"])
        dim = len(vectors[0])
        record("embeddings returned", len(vectors) == 2 and dim > 0, f"dim={dim}")
        record(
            "dimensions consistent",
            all(len(v) == dim for v in vectors),
            f"{[len(v) for v in vectors]}",
        )
    except EmbeddingError as e:
        record("embeddings returned", False, str(e))
        return

    # 2. index a document and search it
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "bam.md").write_text(BAM_DOC)
    (docs / "memory.md").write_text(MEMORY_DOC)
    ctx = ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp / "logs"),
        settings=Settings(),
        scripts_dir=tmp / "scripts",
        symbols=SymbolIndex(conn),
        rag=rag,
        embedder=embedder,
    )
    ctx.registry.register("docs", docs)
    message = await index_docs(
        IndexDocsParams(what="docs_dir", target="docs"), ctx
    )
    record("docs_dir indexed", rag.count() > 0, f"{rag.count()} chunks; {message}")

    query_vector = (await embedder.embed(["how do I subset a BAM by region?"]))[0]
    hits = rag.query(query_vector, k=1)
    top = hits[0] if hits else None
    record(
        "BAM question retrieves the BAM doc",
        top is not None and top.source.endswith("bam.md"),
        f"{Path(top.source).name}: {top.text[:60]}".replace("\n", " ") if top else "",
    )

    memory_vector = (
        await embedder.embed(["my job was killed for using too much RAM"])
    )[0]
    memory_hits = rag.query(memory_vector, k=1)
    memory_top = memory_hits[0] if memory_hits else None
    record(
        "memory question retrieves the memory doc (discriminates)",
        memory_top is not None and memory_top.source.endswith("memory.md"),
        f"{Path(memory_top.source).name}: {memory_top.text[:60]}".replace("\n", " ")
        if memory_top
        else "",
    )

    # 3. full doc-researcher path (needs the chat backend)
    try:
        chat_models = httpx.get(f"{args.chat_url}/models", timeout=5).json()["data"]
        chat_model = args.chat_model or chat_models[0]["id"]
    except Exception as e:
        record("chat server reachable (for researcher test)", False, str(e))
        chat_model = None

    if chat_model:
        record("chat server reachable (for researcher test)", True, chat_model)
        llm = LLMClient(
            LLMSettings(base_url=args.chat_url, model=chat_model, request_timeout_s=120)
        )
        ctx.llm = llm
        answer = await research(
            llm, "How do I subset a BAM file by region on this cluster?", ctx
        )
        print(f"\n  researcher answer: {answer}\n")
        record(
            "researcher answers from indexed docs",
            "samtools" in answer.lower(),
            "cited" if "cluster.md" in answer or "man:" in answer else "no citation",
        )
        await llm.close()

    await embedder.close()
    rag.close()
    conn.close()

    failed = [r for r in results if not r[1]]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} checks passed ===")
    print(f"(work dir: {tmp})")


if __name__ == "__main__":
    asyncio.run(main())
