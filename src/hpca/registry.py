"""Path registry (§4.3): the model refers to paths only by key.

Tools accept registry keys; middleware resolves them to absolute paths and
errors out on unknown keys. The error text is fed back to the model for a
retry, so it names the keys that *do* exist.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

KEY_RE = re.compile(r"^[a-z0-9_.-]+$")
MAX_KEYS_IN_ERROR = 30


class RegistryError(Exception):
    """Invalid registration (bad key, relative path, key conflict)."""


class UnknownKeyError(RegistryError):
    """Key not in the registry; message lists available keys for model retry."""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "_", text.lower()).strip("_")
    return slug or "path"


class PathRegistry:
    def __init__(
        self, conn: sqlite3.Connection, *, profile: str, session_id: str
    ) -> None:
        self._conn = conn
        self._profile = profile
        self._session_id = session_id

    def register(self, key: str, path: Path | str) -> None:
        if not KEY_RE.match(key):
            raise RegistryError(
                f"Invalid registry key {key!r}: use lowercase letters, digits, "
                "'_', '-' and '.'"
            )
        path = Path(path)
        if not path.is_absolute():
            raise RegistryError(f"Registry paths must be absolute, got {path!r}")
        existing = self._get(key)
        if existing is not None:
            if existing == path:
                return
            raise RegistryError(
                f"Key {key!r} is already registered for {existing}; "
                "pick a different key"
            )
        self._conn.execute(
            "INSERT INTO path_registry (profile, session_id, key, path) "
            "VALUES (?, ?, ?, ?)",
            (self._profile, self._session_id, key, str(path)),
        )
        self._conn.commit()

    def register_auto(self, path: Path | str, hint: str | None = None) -> str:
        """Register a tool-discovered path under a generated key (§4.3).

        Returns the existing key if this exact path is already registered.
        """
        path = Path(path)
        row = self._conn.execute(
            "SELECT key FROM path_registry "
            "WHERE profile = ? AND session_id = ? AND path = ?",
            (self._profile, self._session_id, str(path)),
        ).fetchone()
        if row is not None:
            return row["key"]
        base = slugify(hint if hint else path.name)
        key = base
        suffix = 2
        while self._get(key) is not None:
            key = f"{base}_{suffix}"
            suffix += 1
        self.register(key, path)
        return key

    def resolve(self, key: str) -> Path:
        path = self._get(key)
        if path is None:
            known = ", ".join(sorted(self.list())[:MAX_KEYS_IN_ERROR]) or "(none)"
            raise UnknownKeyError(
                f"Unknown registry key {key!r}. Available keys: {known}"
            )
        return path

    def list(self) -> dict[str, Path]:
        rows = self._conn.execute(
            "SELECT key, path FROM path_registry "
            "WHERE profile = ? AND session_id = ?",
            (self._profile, self._session_id),
        ).fetchall()
        return {row["key"]: Path(row["path"]) for row in rows}

    def _get(self, key: str) -> Path | None:
        row = self._conn.execute(
            "SELECT path FROM path_registry "
            "WHERE profile = ? AND session_id = ? AND key = ?",
            (self._profile, self._session_id, key),
        ).fetchone()
        return Path(row["path"]) if row else None
