"""Embedded vector store on sqlite-vec (§5.6.2) — daemon-free, one file.

``chunk_text`` splits documents on paragraph boundaries under a character
budget (the MiniLM-class embedders cap at ~256 tokens). ``RagStore`` pairs a
plain ``chunks`` table with a ``vec0`` virtual table for KNN queries; the
vector dimension is fixed by the first insert and recorded in ``meta``.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

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
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
