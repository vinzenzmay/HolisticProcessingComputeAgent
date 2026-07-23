"""Tests for hpca.db.DbIO: the dedicated sqlite worker thread.

DbIO exists because sync sqlite on an NFS home blocked the TUI event loop
(the 2s poll timers ate keystrokes on cluster nodes). These tests pin the
contract the TUI relies on: work runs off the calling thread on one reused
connection, commits are visible to other connections, and a closed DbIO
refuses new work instead of hanging.
"""

import threading

import pytest

from hpca.db import DbIO, connect, init_db


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "hpca.db"
    conn = connect(path)
    init_db(conn)
    conn.close()
    return path


class TestRun:
    async def test_runs_off_the_calling_thread(self, db_file):
        dbio = DbIO(db_file)
        try:
            thread = await dbio.run(lambda conn: threading.current_thread())
            assert thread is not threading.current_thread()
        finally:
            await dbio.close()

    async def test_reuses_one_connection(self, db_file):
        dbio = DbIO(db_file)
        try:
            first = await dbio.run(id)
            second = await dbio.run(id)
            assert first == second
        finally:
            await dbio.close()

    async def test_returns_result(self, db_file):
        dbio = DbIO(db_file)
        try:
            count = await dbio.run(
                lambda conn: conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            )
            assert count == 0
        finally:
            await dbio.close()

    async def test_committed_writes_visible_to_other_connections(self, db_file):
        dbio = DbIO(db_file)
        try:

            def _insert(conn):
                conn.execute(
                    "INSERT INTO command_usage (name, count) VALUES ('x', 1)"
                )
                conn.commit()

            await dbio.run(_insert)
            other = connect(db_file)
            try:
                rows = other.execute("SELECT name FROM command_usage").fetchall()
                assert [r["name"] for r in rows] == ["x"]
            finally:
                other.close()
        finally:
            await dbio.close()

    async def test_exception_propagates(self, db_file):
        dbio = DbIO(db_file)
        try:
            with pytest.raises(ValueError):
                await dbio.run(lambda conn: (_ for _ in ()).throw(ValueError("boom")))
        finally:
            await dbio.close()


class TestClose:
    async def test_run_after_close_raises(self, db_file):
        dbio = DbIO(db_file)
        await dbio.close()
        assert dbio.closed
        with pytest.raises(RuntimeError):
            await dbio.run(lambda conn: None)

    async def test_close_is_idempotent(self, db_file):
        dbio = DbIO(db_file)
        await dbio.close()
        await dbio.close()

    async def test_close_before_any_work_opens_no_connection(self, db_file):
        # The connection is lazy; closing an unused DbIO must not create it.
        dbio = DbIO(db_file)
        await dbio.close()
        assert dbio._conn is None
