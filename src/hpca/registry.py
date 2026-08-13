"""Path registry (§4.3): the model refers to paths by key.

Tools accept registry keys; middleware resolves them to absolute paths and
errors out on unknown keys. The error text is fed back to the model for a
retry, so it names the keys that *do* exist.

Keys are the *preferred* way in, not the only one: a key stops a small model
from re-typing a 90-character cluster path (and getting it wrong), but every
file tool in every agent corpus the model was trained on takes a path, so
*requiring* a key means fighting that prior on every call and paying a
round-trip to mint the key first. ``resolve_or_register`` is the compromise —
a key resolves as before, a literal absolute path is used as given and gains a
key on the way through, and anything else is still the unknown-key error.
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


def looks_like_path(value: str) -> bool:
    """Whether a tool argument is meant as a literal path, not a key.

    One character decides it: KEY_RE keys can never start with '/' or '~', so
    the two namespaces are disjoint by construction and no value is ever
    ambiguous. A *relative* path deliberately does not count — "results/x.txt"
    is as likely a mistyped key as a path, and the unknown-key error (which
    lists the keys) is the more useful answer to both.
    """
    return value.startswith(("/", "~"))


def registered_note(value: str, key: str) -> str:
    """The note a result carries when a literal path argument gained a key.

    Empty when ``value`` was already a key — the caller passes both and gets
    "" or the parenthetical, so a tool never has to branch on which form it
    was given.
    """
    return "" if key == value else f" ({value} is registered as {key!r})"


class PathRegistry:
    def __init__(
        self, conn: sqlite3.Connection, *, profile: str, session_id: str
    ) -> None:
        self._conn = conn
        self._profile = profile
        self._session_id = session_id

    def register(self, key: str, path: Path | str) -> Path | None:
        """Bind a key to an absolute path; returns the dead path it replaced.

        A key whose registered path exists is a real handle and stays
        protected: pointing it somewhere else raises, so the model cannot
        quietly rename someone else's file out from under a later call.

        A key whose registered path does *not* exist names nothing, so this
        overwrites it and returns the path that was there. That is the only
        way back from a mistyped registration: register_path takes a path
        before it exists (a typo looks exactly like an output that is not
        written yet), there is no unregister tool, and without this the typo
        would burn the key for the rest of the session.

        Returns None when the key was free, or already pointed at this exact
        path (an idempotent re-registration, not a repoint).
        """
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
                return None
            if existing.exists():
                raise RegistryError(
                    f"Key {key!r} is already registered for {existing}; "
                    "pick a different key"
                )
            self._point(key, path)
            return existing
        self._conn.execute(
            "INSERT INTO path_registry (profile, session_id, key, path) "
            "VALUES (?, ?, ?, ?)",
            (self._profile, self._session_id, key, str(path)),
        )
        self._conn.commit()
        return None

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

    def resolve_or_register(
        self, value: str, *, hint: str | None = None, register: bool = True
    ) -> tuple[Path, str]:
        """Resolve a tool argument that is either a key or a literal path.

        The order matters: a known key always wins, so nothing a session has
        already named changes meaning. Failing that, a value shaped like an
        absolute path ('~' expanded) is taken at face value and auto-registered,
        which is what lets a model call read_file with the path it just saw in
        `ls` output. Everything else falls through to ``resolve``'s
        UnknownKeyError, keys listed, unchanged.

        Returns ``(path, key)``. The key is ``value`` itself when ``value`` was
        a key, so ``key != value`` is how a caller knows a path just gained one
        (see ``registered_note``).

        ``register=False`` is for the gating predicates and describe helpers,
        which must stay side-effect free — the graph re-runs them when a parked
        turn resumes, and a call the user then declines must not leave a key
        behind. They get the path and ``value`` back unchanged.
        """
        path = self._get(value)
        if path is not None:
            return path, value
        if looks_like_path(value):
            literal = Path(value).expanduser()
            if literal.is_absolute():  # '~unknownuser' stays unexpanded, and errors
                if not register:
                    return literal, value
                return literal, self.register_auto(literal, hint=hint)
        return self.resolve(value), value  # raises UnknownKeyError

    def get(self, key: str) -> Path | None:
        """The path a key names, or None when the key is free.

        ``key in registry`` only answers whether the key is taken. A caller
        that has to decide like :meth:`register` does — a key whose path no
        longer exists names nothing and may be repointed — needs the path
        itself to test.
        """
        return self._get(key)

    def __contains__(self, key: str) -> bool:
        return self._get(key) is not None

    def remove(self, key: str) -> None:
        """Drop a key (e.g. after its file was deleted); missing keys error."""
        self.resolve(key)  # raises UnknownKeyError with the available keys
        self._conn.execute(
            "DELETE FROM path_registry "
            "WHERE profile = ? AND session_id = ? AND key = ?",
            (self._profile, self._session_id, key),
        )
        self._conn.commit()

    def reassign(self, key: str, path: Path | str) -> None:
        """Point an existing key at a new absolute path (e.g. after a move).

        Stays separate from ``register`` because it asserts the opposite
        things: the key must already exist (a move that repoints a key nobody
        registered is a bug, not a fresh registration), and the file behind it
        is expected to be real — the mover knows the old path is stale, so the
        does-it-exist check ``register`` applies would be wrong here.
        """
        self.resolve(key)
        path = Path(path)
        if not path.is_absolute():
            raise RegistryError(f"Registry paths must be absolute, got {path!r}")
        self._point(key, path)

    def list(self) -> dict[str, Path]:
        rows = self._conn.execute(
            "SELECT key, path FROM path_registry "
            "WHERE profile = ? AND session_id = ?",
            (self._profile, self._session_id),
        ).fetchall()
        return {row["key"]: Path(row["path"]) for row in rows}

    def _point(self, key: str, path: Path) -> None:
        self._conn.execute(
            "UPDATE path_registry SET path = ? "
            "WHERE profile = ? AND session_id = ? AND key = ?",
            (str(path), self._profile, self._session_id, key),
        )
        self._conn.commit()

    def _get(self, key: str) -> Path | None:
        row = self._conn.execute(
            "SELECT path FROM path_registry "
            "WHERE profile = ? AND session_id = ? AND key = ?",
            (self._profile, self._session_id, key),
        ).fetchone()
        return Path(row["path"]) if row else None
