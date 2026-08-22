"""Tests for hpca.db: schema init and connection settings (§5.4)."""

from hpca.db import connect, init_db
from hpca.sessions import SessionStore
from hpca.watches import KIND_LOG, WatchStore

EXPECTED_TABLES = {"jobs", "job_logs", "sessions", "processes"}


def table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


class TestConnect:
    def test_creates_file_and_parents(self, tmp_path):
        path = tmp_path / "deep" / "hpca.db"
        conn = connect(path)
        assert path.exists()
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        conn.close()

    def test_foreign_keys_enabled(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.close()

    def test_rows_accessible_by_name(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        row = conn.execute("SELECT 1 AS x").fetchone()
        assert row["x"] == 1
        conn.close()

    def test_default_location_uses_app_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        conn = connect()
        assert (tmp_path / "hpca.db").exists()
        conn.close()


class TestInitDb:
    def test_creates_all_tables(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        assert EXPECTED_TABLES <= table_names(conn)
        conn.close()

    def test_idempotent(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        init_db(conn)  # must not raise
        assert EXPECTED_TABLES <= table_names(conn)
        conn.close()


class TestTheLastActiveBackfill:
    """A session made before the column existed still has to say something.

    Its messages have no stamps either — those arrived in the same release —
    so its creation is the one honest thing left, and it is at least the right
    order of magnitude for a sidebar people read as "old / recent".
    """

    def old_database(self, tmp_path):
        """A sessions table exactly as it was before `last_active`."""
        conn = connect(tmp_path / "hpca.db")
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, profile TEXT "
            "NOT NULL, title TEXT, created_at TEXT, checkpoint_ref TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'default', 'old one', "
            "'2026-01-02T03:04:05+00:00', 's1')"
        )
        conn.commit()
        return conn

    def test_an_old_row_is_seeded_from_its_creation(self, tmp_path):
        conn = self.old_database(tmp_path)
        init_db(conn)
        row = conn.execute("SELECT last_active FROM sessions").fetchone()
        assert row["last_active"] == "2026-01-02T03:04:05+00:00"
        conn.close()

    def test_and_a_second_start_does_not_undo_a_touch(self, tmp_path):
        # The backfill runs on every start, not once, so it has to be written
        # to only touch rows that have never been touched.
        conn = self.old_database(tmp_path)
        init_db(conn)
        conn.execute("UPDATE sessions SET last_active = '2030-01-01T00:00:00+00:00'")
        conn.commit()
        init_db(conn)
        row = conn.execute("SELECT last_active FROM sessions").fetchone()
        assert row["last_active"] == "2030-01-01T00:00:00+00:00"
        conn.close()


class TestTheSessionOrderBackfill:
    """A sidebar that predates `position` has to open in the order it closed.

    Every row carries 0 until the migration runs, and 0 is not an order: the
    seed has to reproduce newest-first, which is the only arrangement those
    rows ever had, and it has to leave every one of them at or above 1 so that
    "position = 0" goes on meaning "never assigned".
    """

    def old_database(self, tmp_path):
        """A sessions table exactly as it was before `position`, three rows
        deep so an order is something the assertions can actually see."""
        conn = connect(tmp_path / "hpca.db")
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, profile TEXT "
            "NOT NULL, title TEXT, created_at TEXT, checkpoint_ref TEXT)"
        )
        for made, title in (
            ("2026-01-01T00:00:00+00:00", "oldest"),
            ("2026-02-01T00:00:00+00:00", "middle"),
            ("2026-03-01T00:00:00+00:00", "newest"),
        ):
            conn.execute(
                "INSERT INTO sessions VALUES (?, 'default', ?, ?, ?)",
                (title, title, made, title),
            )
        conn.commit()
        return conn

    def test_the_rows_survive_and_keep_the_order_they_had(self, tmp_path):
        conn = self.old_database(tmp_path)
        init_db(conn)
        store = SessionStore(conn)
        assert [s.title for s in store.list_all()] == [
            "newest",
            "middle",
            "oldest",
        ]
        # Numbered densely from the top, not left at 0: an unnumbered row
        # would sort above every row the user goes on to arrange.
        assert [s.position for s in store.list_all()] == [1, 2, 3]
        conn.close()

    def test_and_a_second_start_does_not_undo_an_arrangement(self, tmp_path):
        # The backfill runs on every start, not once, so it has to be written
        # to only touch rows that have never been placed.
        conn = self.old_database(tmp_path)
        init_db(conn)
        store = SessionStore(conn)
        store.move("oldest", -1)
        init_db(conn)
        assert [s.title for s in store.list_all()] == [
            "newest",
            "oldest",
            "middle",
        ]
        conn.close()

    def test_an_old_row_can_be_moved_at_all(self, tmp_path):
        # Swapping two zeroes changes nothing, which is why the move
        # renumbers the whole list instead — the case that motivated it.
        conn = self.old_database(tmp_path)
        init_db(conn)
        conn.execute("UPDATE sessions SET position = 0")  # as if never seeded
        conn.commit()
        store = SessionStore(conn)
        assert store.move("middle", -1) is True
        assert [s.title for s in store.list_all()] == [
            "middle",
            "newest",
            "oldest",
        ]
        conn.close()


class TestWatchUniqueness:
    """Watches are session-scoped; the index that enforced profile-scoping has
    to actually go, or an existing database keeps applying the old rule."""

    def index_names(self, conn):
        return {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    def test_the_profile_scoped_index_is_dropped_on_an_existing_database(
        self, tmp_path
    ):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        # Recreate the pre-migration state, as an older install has it.
        conn.execute("DROP INDEX IF EXISTS idx_watches_session_target")
        conn.execute(
            "CREATE UNIQUE INDEX idx_watches_target "
            "ON watches(profile, kind, target)"
        )
        conn.commit()

        init_db(conn)
        names = self.index_names(conn)
        assert "idx_watches_target" not in names
        assert "idx_watches_session_target" in names
        conn.close()

    def test_an_existing_database_gains_the_position_column_backfilled(
        self, tmp_path
    ):
        """A DEFAULT only describes new rows, so without the backfill every
        watch made before this release would sort above every one made after
        it — the column would look shuffled on the first start."""
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        store = WatchStore(conn)
        for name in ("a", "b"):
            store.add(kind=KIND_LOG, target=f"/{name}.log", session_id="s1")
        # Recreate the pre-migration state: the column gone, and with it the
        # positions the two rows were given on insert.
        conn.execute("ALTER TABLE watches DROP COLUMN position")
        conn.commit()

        init_db(conn)
        rows = conn.execute("SELECT target, position FROM watches ORDER BY id")
        placed = list(rows)
        assert [row["target"] for row in placed] == ["/a.log", "/b.log"]
        assert all(row["position"] > 0 for row in placed)
        assert [w.target for w in store.list(session_id="s1")] == [
            "/a.log",
            "/b.log",
        ]
        conn.close()

    def test_two_sessions_can_watch_the_same_log(self, tmp_path):
        # The whole reason the index had to move: under the old one the second
        # of these was a constraint violation.
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        store = WatchStore(conn)
        first = store.add(
            kind=KIND_LOG, target="/scratch/run.log", profile="default",
            session_id="s1",
        )
        second = store.add(
            kind=KIND_LOG, target="/scratch/run.log", profile="default",
            session_id="s2",
        )
        assert first.id != second.id
        assert [w.id for w in store.list(session_id="s1")] == [first.id]
        assert [w.id for w in store.list(session_id="s2")] == [second.id]
        conn.close()

    def test_one_session_still_gets_a_single_box_per_target(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        store = WatchStore(conn)
        first = store.add(
            kind=KIND_LOG, target="/scratch/run.log", profile="default",
            session_id="s1",
        )
        again = store.add(
            kind=KIND_LOG, target="/scratch/run.log", label="renamed",
            profile="default", session_id="s1",
        )
        assert again.id == first.id and again.label == "renamed"
        conn.close()


class TestConcurrencySettings:
    def test_busy_timeout_set(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        conn.close()
