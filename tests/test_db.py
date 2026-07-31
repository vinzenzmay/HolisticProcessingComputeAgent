"""Tests for hpca.db: schema init and connection settings (§5.4)."""

from hpca.db import connect, init_db
from hpca.watches import KIND_LOG, WatchStore

EXPECTED_TABLES = {"jobs", "job_logs", "sessions", "path_registry", "processes"}


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
