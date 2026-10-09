"""Embedded vector store on sqlite-vec (§5.6.2) — daemon-free, one file.

``chunk_text`` splits documents on paragraph boundaries under a character
budget (the MiniLM-class embedders cap at ~256 tokens). ``RagStore`` pairs a
plain ``chunks`` table with a ``vec0`` virtual table for KNN queries; the
vector dimension is fixed by the first insert and recorded in ``meta``.

``indexed_files`` remembers, per file indexed from a directory, what it looked
like when it was embedded and with which model — so indexing the same
directory again embeds only what changed (`hpca.doc_index`).

Each agent profile has its own index, ``<app_dir>/rag/<profile>.db``
(`RagStores`): what was indexed for one line of work is not searched in
another. There used to be one, ``rag.db``, shared by every profile; boot moves
it to the default profile's (`migrate_shared_index`).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpca.dbcache import DbCache

import sqlite_vec

from hpca.profiles import DEFAULT_PROFILE

DEFAULT_MAX_CHARS = 1000


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Greedy paragraph packing; oversized paragraphs are hard-split."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph]
            if len(paragraph) <= max_chars
            else [
                paragraph[i : i + max_chars]
                for i in range(0, len(paragraph), max_chars)
            ]
        )
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= max_chars:
                current += "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


# A chunk this short that the model still refuses is not a length problem any
# more, so splitting it further only multiplies requests. Dropped instead.
MIN_CHARS = 100


async def embed_fitting(embedder, chunks: list[str]) -> tuple[list[str], list[list]]:
    """Embed ``chunks``, splitting any the model finds too long, in step.

    ``chunk_text`` budgets in characters while the model budgets in tokens, and
    the ratio between them is a property of the text: ~4 chars/token for prose,
    barely 2 for the code blocks and API tables that make up most of a
    reference manual. So a character budget that fits an English paragraph
    overruns a 256-token embedder on a page of GDScript — and because one
    request carries a whole document's chunks, a single overlong chunk used to
    raise and discard the entire document. Indexing the Godot manual that way
    kept 100 files out of 1597 and reported success.

    Splitting is halving, driven by the server's own refusal rather than by a
    guessed chars-per-token constant: divide and conquer over the list narrows
    to the offending chunks in log time, then those are cut in half and retried
    until they fit. Only ``InputTooLong`` recurses; a backend that is down or
    broken propagates on the first try instead of being asked 2^n times.

    Returns the chunks actually embedded — the split ones in place of their
    oversized original — paired with their vectors, so callers store text and
    vector that agree.
    """
    from hpca.embeddings import InputTooLong

    if not chunks:
        return [], []
    try:
        return chunks, await embedder.embed(chunks)
    except InputTooLong:
        pass
    if len(chunks) > 1:
        middle = len(chunks) // 2
        left_text, left_vectors = await embed_fitting(embedder, chunks[:middle])
        right_text, right_vectors = await embed_fitting(embedder, chunks[middle:])
        return left_text + right_text, left_vectors + right_vectors
    text = chunks[0]
    if len(text) <= MIN_CHARS:
        return [], []  # unsplittable and still refused: drop this one chunk
    halves = chunk_text(text, max_chars=max(MIN_CHARS, len(text) // 2))
    if len(halves) < 2:  # nothing to split on: cut the string itself
        cut = len(text) // 2
        halves = [text[:cut], text[cut:]]
    return await embed_fitting(embedder, halves)


async def index_text(rag: "RagStore", embedder, source: str, text: str) -> int:
    """Chunk, embed and store one document in place of what `source` had.

    Returns how many chunks were stored; 0 when nothing in it could be
    embedded, in which case what was stored before is left alone. Raises
    `EmbeddingError` when the backend fails, also leaving the store untouched.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0
    chunks, vectors = await embed_fitting(embedder, chunks)
    if not chunks:
        return 0
    rag.clear_source(source)
    rag.add(source, chunks, vectors)
    return len(chunks)


@dataclass(frozen=True)
class Recorded:
    """A file as it was when it was last indexed, and by which model."""

    size: int
    mtime_ns: int
    digest: str
    model: str


@dataclass
class Hit:
    text: str
    source: str
    distance: float


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class RagStore:
    def __init__(self, path: Path) -> None:
        path = Path(path)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS indexed_files (
                source TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                digest TEXT NOT NULL,
                model TEXT NOT NULL
            );
            """
        )
        self._dim = self._load_dim()

    def close(self) -> None:
        self._conn.close()

    def _load_dim(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'dim'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors "
                f"USING vec0(embedding float[{dim}])"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dim', ?)",
                (str(dim),),
            )
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: store has {self._dim}, got {dim}"
            )

    def add(
        self, source: str, texts: list[str], embeddings: list[list[float]]
    ) -> None:
        if not texts:
            return
        self._ensure_dim(len(embeddings[0]))
        for text, embedding in zip(texts, embeddings):
            if len(embedding) != self._dim:
                raise ValueError(
                    f"Embedding dimension mismatch: store has {self._dim}, "
                    f"got {len(embedding)}"
                )
            cursor = self._conn.execute(
                "INSERT INTO chunks (source, text) VALUES (?, ?)", (source, text)
            )
            self._conn.execute(
                "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, _serialize(embedding)),
            )
        self._conn.commit()

    def query(self, embedding: list[float], *, k: int = 5) -> list[Hit]:
        if self._dim is None:
            return []
        rows = self._conn.execute(
            "SELECT chunks.text, chunks.source, distance "
            "FROM chunk_vectors JOIN chunks ON chunks.id = chunk_vectors.rowid "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_serialize(embedding), k),
        ).fetchall()
        return [
            Hit(text=row["text"], source=row["source"], distance=row["distance"])
            for row in rows
        ]

    def clear_source(self, source: str) -> None:
        ids = [
            row["id"]
            for row in self._conn.execute(
                "SELECT id FROM chunks WHERE source = ?", (source,)
            )
        ]
        if self._dim is not None:
            self._conn.executemany(
                "DELETE FROM chunk_vectors WHERE rowid = ?", [(i,) for i in ids]
            )
        self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._conn.execute("DELETE FROM indexed_files WHERE source = ?", (source,))
        self._conn.commit()

    def record(self, source: str, recorded: Recorded) -> None:
        """Remember what `source` looked like when it was just indexed."""
        self._conn.execute(
            "INSERT OR REPLACE INTO indexed_files "
            "(source, size, mtime_ns, digest, model) VALUES (?, ?, ?, ?, ?)",
            (source, recorded.size, recorded.mtime_ns, recorded.digest, recorded.model),
        )
        self._conn.commit()

    def recorded_under(self, directory: str) -> dict[str, Recorded]:
        """Every file recorded as indexed from inside `directory`."""
        prefix = directory.rstrip("/") + "/"
        # substr rather than LIKE: a path may hold % or _, which LIKE would
        # read as wildcards.
        rows = self._conn.execute(
            "SELECT source, size, mtime_ns, digest, model FROM indexed_files "
            "WHERE substr(source, 1, ?) = ?",
            (len(prefix), prefix),
        ).fetchall()
        return {
            row["source"]: Recorded(
                row["size"], row["mtime_ns"], row["digest"], row["model"]
            )
            for row in rows
        }

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# Where the indexes live, relative to the app dir. The same value as
# `hpca.dbcache.RAG_DIR`, which is the cache's half of the same layout.
RAG_DIR = "rag"
SHARED_INDEX = "rag.db"
SIDECARS = ("", "-wal", "-shm")


def index_name(profile: str) -> str:
    """A profile's index, relative to the app dir."""
    return f"{RAG_DIR}/{profile}.db"


def migrate_shared_index(app_dir: Path) -> list[str]:
    """Move the index every profile used to share to the default profile's.

    Moved rather than copied into every profile: an index can be hundreds of
    megabytes, and what was indexed into it was indexed from whichever profile
    happened to be open. Returns what the user should be told.
    """
    app_dir = Path(app_dir)
    shared = app_dir / SHARED_INDEX
    if not shared.exists():
        return []
    target = app_dir / index_name(DEFAULT_PROFILE)
    if target.exists():
        return [
            f"Both {SHARED_INDEX} and {index_name(DEFAULT_PROFILE)} exist in "
            f"{app_dir}; left {SHARED_INDEX} where it is. Move it to "
            f"{RAG_DIR}/<profile>.db for the profile it belongs to, or delete it."
        ]
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in SIDECARS:
        sidecar = app_dir / (SHARED_INDEX + suffix)
        if sidecar.exists():
            os.replace(sidecar, Path(str(target) + suffix))
    return [
        "Document indexes are per profile now. The shared one became the "
        f"default profile's ({index_name(DEFAULT_PROFILE)}); to give it to "
        f"another profile, rename it to {RAG_DIR}/<profile>.db."
    ]


class RagStores:
    """One `RagStore` per profile, opened when that profile first needs it.

    Opening may first copy the index from home to the node-local working dir
    (`DbCache.adopt`), which for a large one is seconds over NFS — so it is
    done off the loop, and the connection itself is opened on the loop, whose
    thread every later call comes from.
    """

    def __init__(self, app_dir: Path, cache: "DbCache | None" = None) -> None:
        self.app_dir = Path(app_dir)
        self.cache = cache
        self._open: dict[str, RagStore] = {}
        self._lock = asyncio.Lock()

    def _locate(self, profile: str) -> Path:
        name = index_name(profile)
        if self.cache is not None:
            return self.cache.adopt(name)
        return self.app_dir / name

    async def open(self, profile: str) -> RagStore:
        """`profile`'s index, opened on first use."""
        async with self._lock:
            store = self._open.get(profile)
            if store is None:
                path = await asyncio.to_thread(self._locate, profile)
                store = self._open[profile] = RagStore(path)
            return store

    async def copy(self, source: str, target: str) -> bool:
        """Start `target`'s index as a copy of `source`'s. Whether there was
        anything to copy."""
        from hpca.dbcache import copy_database

        async with self._lock:
            open_source = self._open.get(source)
            src = (
                open_source.path
                if open_source is not None
                else await asyncio.to_thread(self._locate, source)
            )
            if not src.exists():
                return False
            dst = await asyncio.to_thread(self._locate, target)
            await asyncio.to_thread(copy_database, src, dst)
            return True

    async def remove(self, profile: str) -> None:
        """Close and delete `profile`'s index, working copy and home copy."""
        async with self._lock:
            store = self._open.pop(profile, None)
            if store is not None:
                store.close()
            name = index_name(profile)

            def delete() -> None:
                if self.cache is not None:
                    self.cache.forget(name)
                for suffix in SIDECARS:
                    (self.app_dir / (name + suffix)).unlink(missing_ok=True)

            await asyncio.to_thread(delete)

    def close(self) -> None:
        for store in self._open.values():
            store.close()
        self._open.clear()
