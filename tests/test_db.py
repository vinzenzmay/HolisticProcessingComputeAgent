"""Tests for hpca.db: schema init and connection settings (§5.4)."""

from hpca.db import connect, init_db

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
